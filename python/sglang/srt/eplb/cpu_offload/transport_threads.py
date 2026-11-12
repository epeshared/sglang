"""初始化末尾核对 GPU 预留核上的线程；只记录观察，不猜测线程所属运行库。

11-09 用户批准的判据：采样 1 秒；与预留核相交的非主线程，占用超过采样时长一半
才判为空转并拒绝。更少的 CPU 活动只记入告警，原始快照照样保留。
"""

import json
import logging
import os
import threading
import time
from collections import Counter
from pathlib import Path

SAMPLE_SECONDS = 1.0
MIN_SAMPLE_SECONDS = 0.9
SPIN_FRACTION = 0.5


def thread_snapshot():
    rows = []
    for task in sorted(Path("/proc/self/task").iterdir(), key=lambda p: int(p.name)):
        try:
            fields = (task / "stat").read_text().rsplit(")", 1)[1].split()
            rows.append(
                {
                    "tid": int(task.name),
                    "name": (task / "comm").read_text().strip(),
                    "cores": sorted(os.sched_getaffinity(int(task.name))),
                    "cpu_ticks": int(fields[11]) + int(fields[12]),
                }
            )
        except (FileNotFoundError, ProcessLookupError):
            # 退出中的线程不补零；两次快照的差集会使检查不通过。
            continue
    return rows


def reserved_core_check(before, after, main_tid, main_cores, elapsed, clock_ticks):
    """保留原始快照；空转按占用比例判，轻微活动列为告警而不放过记录。"""
    old = {row["tid"]: row for row in before}
    new = {row["tid"]: row for row in after}
    changed = sorted(set(old) ^ set(new))
    busy, active, overlapping = [], [], []
    spin_ticks = SPIN_FRACTION * elapsed * clock_ticks
    for tid in sorted(set(old) & set(new)):
        first, last = old[tid], new[tid]
        if first["cores"] != last["cores"] or last["cpu_ticks"] < first["cpu_ticks"]:
            changed.append(tid)
        if tid == main_tid or not set(last["cores"]) & set(main_cores):
            continue
        overlapping.append(tid)
        delta = last["cpu_ticks"] - first["cpu_ticks"]
        if delta > spin_ticks:
            busy.append(tid)
        elif delta > 0:
            active.append(tid)
    groups = Counter(tuple(row["cores"]) for row in after)
    valid_main = (
        main_tid in old
        and main_tid in new
        and old[main_tid]["cores"] == new[main_tid]["cores"] == main_cores
        and len(old) == len(before)
        and len(new) == len(after)
    )
    return {
        "schema_version": 2,
        "spin_fraction": SPIN_FRACTION,
        "main_tid": main_tid,
        "main_cores": main_cores,
        "sample_seconds": elapsed,
        "clock_ticks_per_second": clock_ticks,
        "before": before,
        "after": after,
        "thread_groups": [
            {"cores": list(cores), "count": n} for cores, n in sorted(groups.items())
        ],
        "overlapping_nonmain_tids": overlapping,
        "busy_nonmain_tids": busy,
        "warning_active_nonmain_tids": active,
        "changed_tids": sorted(set(changed)),
        "passed": bool(
            valid_main
            and main_cores
            and elapsed >= MIN_SAMPLE_SECONDS
            and clock_ticks > 0
            and not busy
            and not changed
        ),
    }


def check_problems(record):
    """离线检查复算原始采样；缺字段不能默认通过。"""
    try:
        expected = reserved_core_check(
            record["before"],
            record["after"],
            record["main_tid"],
            record["main_cores"],
            record["sample_seconds"],
            record["clock_ticks_per_second"],
        )
    except (KeyError, TypeError, ValueError):
        return ["缺少完整的 GPU 预留核线程采样"]
    if record != expected:
        return ["GPU 预留核线程摘要与原始采样不符"]
    if not expected["passed"]:
        return ["GPU 预留核上有非主线程空转（占用超过采样时长一半），或采样不完整"]
    return []


def startup_problems(record):
    """校验启动回执和采样属于同一个进程、同一个主线程。"""
    if not isinstance(record, dict):
        return ["缺少 rank 启动线程回执"]
    early = record.get("early_bootstrap", {})
    check = record.get("gpu_reserved_check", {})
    problems = check_problems(check)
    if (
        early.get("stage") != "sglang_package_before_public_imports"
        or early.get("main_threads") != 1
        or early.get("omp_num_threads") != "1"
        or not isinstance(record.get("pid"), int)
        or record["pid"] <= 0
        or early.get("pid") != record.get("pid")
        or early.get("tid") != check.get("main_tid")
    ):
        problems.append("缺少本 rank 在公共导入之前的单线程回执")
    return problems


def record_rank_startup(rank):
    """所有 rank 都检查，包括不参与 CPU 专家计算的 rank；失败先留底再报错。"""
    from sglang._pinned_cpu_threads import require_bootstrap

    early = require_bootstrap()
    main_tid = threading.get_native_id()
    main_cores = sorted(os.sched_getaffinity(0))
    before = thread_snapshot()
    start = time.monotonic()
    time.sleep(SAMPLE_SECONDS)
    after = thread_snapshot()
    check = reserved_core_check(
        before,
        after,
        main_tid,
        main_cores,
        time.monotonic() - start,
        os.sysconf("SC_CLK_TCK"),
    )
    record = {
        "rank": rank,
        "pid": os.getpid(),
        "early_bootstrap": early,
        "gpu_reserved_check": check,
        "wait_environment": {
            key: os.environ.get(key) for key in ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT")
        },
    }
    text = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    directory = os.environ.get("CPU_OFFLOAD_TRANSPORT_METADATA_DIR")
    if directory:
        path = Path(directory) / "startup"
        path.mkdir(parents=True, exist_ok=True)
        (path / f"rank-{rank}.json").write_text(text)
    logging.getLogger(__name__).info("rank 启动线程核验：%s", text)
    problems = startup_problems(record)
    if problems:
        raise RuntimeError("；".join(problems))
    return record
