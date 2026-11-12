"""逐个 decode 步记录 MoE 异构执行明细。

这套记录与 :mod:`trace` 的区间汇总彼此独立。只有显式设置
``SGLANG_MOE_STEP_TRACE=1`` 时才会创建 CUDA event、读取 daemon 快照并写
JSONL；关闭时热路径只剩一个模块级布尔判断。

CUDA Graph 路径在捕获时把 ``external=True`` 的 CUDA event 嵌入图中，
随后每次 replay 都会刷新这些 event。Python 只在整个 decode 步结束后同步
一次，再统一读取所有层和 CPU 任务的结果，避免逐层同步破坏 CPU/GPU 重叠。

长输出时每步都记，数据量和同步开销都会变大，所以支持三种控制：

* 抽样：``SGLANG_MOE_STEP_TRACE_SAMPLE_EVERY=N`` 只记第 1、1+N、1+2N…个
  decode 步。``batch_step_id`` 仍按实际 decode 次数编号，所以抽样后步号不
  连续，而且同一步号在各 rank 上指的是同一个 decode 步。
* 分段：基准脚本按（batch、输入长度、输出长度）组合调用 :func:`start_segment`，
  每段写进自己的子目录、步号从 1 重新数，一段出一份报告。
* 暂停：预热这类不想入账的 decode 用 :func:`pause` / :func:`resume` 跳过。

被记录的步若紧跟在没记录的步后面，开始前会先等 GPU 把之前的活全部做完；
否则上一步还没做完的 CPU 任务会被错算到这一步。基准脚本本来每步都同步，
这次额外同步对它没有影响；在线服务开了抽样才会多出这一次。
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, TextIO

import torch

logger = logging.getLogger(__name__)


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


ENABLED = _env_enabled("SGLANG_MOE_STEP_TRACE")
SCHEMA_VERSION = 1
SAMPLE_EVERY_ENV = "SGLANG_MOE_STEP_TRACE_SAMPLE_EVERY"
_SEGMENT_NAME = re.compile(r"[A-Za-z0-9_.-]+")


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _env_sample_every() -> int:
    raw = os.environ.get(SAMPLE_EVERY_ENV, "").strip()
    if not raw:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "[moe_step_trace] %s=%r 不是整数，按每步都记处理", SAMPLE_EVERY_ENV, raw
        )
        return 1


def _sampling_label(sample_every: int) -> str:
    if sample_every <= 1:
        return "every_decode_forward"
    return f"every_{sample_every}_decode_forwards"


@dataclass
class _Segment:
    name: str
    run_id: str
    path: Path
    fh: TextIO
    sample_every: int
    saved_decode_forwards: int


@dataclass
class LayerEvents:
    layer_id: int
    start: Any
    dispatch_end: Any = None
    local_moe_end: Any = None
    end: Any = None
    branches_received: Any = None
    unsupported_reasons: list[str] = field(default_factory=list)


@dataclass
class CpuGpuEvents:
    layer_id: int
    d2h_start: Any
    d2h_end: Any
    wait_start: Any
    wait_end: Any
    h2d_start: Any
    h2d_end: Any
    visible_wait_start: Any
    visible_wait_end: Any
    wait_status: Any = None
    business_offloaded_branches: Any = None
    unsupported_reasons: list[str] = field(default_factory=list)

    def submit_events(self) -> dict[str, Any]:
        return {
            "d2h_start": self.d2h_start,
            "d2h_end": self.d2h_end,
        }

    def wait_events(self) -> dict[str, Any]:
        return {
            "wait_start": self.wait_start,
            "wait_end": self.wait_end,
            "h2d_start": self.h2d_start,
            "h2d_end": self.h2d_end,
        }


@dataclass
class GraphTrace:
    key: tuple[int, int]
    graph_batch_size: int
    graph_num_tokens: int
    active_tokens: Any
    row_indices: Any
    layers: list[LayerEvents] = field(default_factory=list)
    cpu_events: list[CpuGpuEvents] = field(default_factory=list)


@dataclass
class StepTrace:
    batch_step_id: int
    request_ids: list[str]
    request_decode_steps: list[Optional[int]]
    request_id_source: str
    active_batch_size: int
    start: Any
    end: Any
    cpu_baseline_task_id: int
    cpu_baseline_discarded_records: int
    cpu_baseline_lost_records: int
    cpu_baseline_valid: bool
    cpu_baseline_supported: bool
    cpu_baseline_enabled: bool
    cpu_baseline_cursor_reset: bool
    cpu_slot_generation: int
    plan_by_layer: dict[int, tuple[int, int]]
    execution_mode: str = "eager"
    graph_batch_size: Optional[int] = None
    graph_trace: Optional[GraphTrace] = None
    layers: list[LayerEvents] = field(default_factory=list)
    cpu_events: list[CpuGpuEvents] = field(default_factory=list)
    unsupported_reasons: list[str] = field(default_factory=list)


class StepTraceRecorder:
    """单个模型 rank 的 JSONL 写入器和 CUDA event 注册表。"""

    def __init__(
        self,
        *,
        rank_id: int,
        hetero_mode: str,
        cpu_trace_applicable: bool,
        output_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        event_factory: Optional[Callable[[], Any]] = None,
        snapshot_fn: Optional[Callable[[int], dict[str, Any]]] = None,
        expected_rank_ids: Optional[list[int]] = None,
        expected_layer_ids: Optional[list[int]] = None,
        data_origin: str = "measured",
        write_run_record: bool = True,
        sample_every: Optional[int] = None,
        sync_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        self.rank_id = int(rank_id)
        self.hetero_mode = hetero_mode
        self.cpu_trace_applicable = bool(cpu_trace_applicable)
        self.run_id = run_id or os.environ.get("SGLANG_MOE_STEP_TRACE_RUN_ID")
        if not self.run_id:
            # 同一台机器由同一个 launcher 拉起的各 rank 通常共享父进程；这样
            # 即使操作者没给 run ID，也能先得到可合并的一组文件。正式实验仍
            # 应显式设置环境变量，跨节点时父进程号没有全局唯一性。
            self.run_id = f"local-{os.getppid()}"

        trace_dir = output_dir or os.environ.get(
            "SGLANG_MOE_STEP_TRACE_DIR", "/tmp/sglang-moe-step-trace"
        )
        self.output_dir = Path(trace_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / f"step_trace_rank_{self.rank_id}.jsonl"
        # 顶层文件等第一次写入才打开。基准脚本把记录全部写进分段子目录时，
        # 顶层不能再留一个只有 run 记录的空壳，否则对顶层目录出报告会混进去。
        self._fh: Optional[TextIO] = None
        self._write_lock = threading.Lock()
        self._write_run_record = bool(write_run_record)
        self._top_run_written = False
        self._segment: Optional[_Segment] = None
        self._segments_started = 0
        self._paused = False
        self.sample_every = (
            _positive_int(sample_every, 1)
            if sample_every is not None
            else _env_sample_every()
        )
        # 当前分段（没分段时是整个进程）里经过的 decode 次数，含没抽中的步。
        self._decode_forwards = 0
        self._last_decode_traced = True
        self._sync_fn = sync_fn or self._cuda_synchronize

        self._event_factory = event_factory or self._new_cuda_event
        self._snapshot_fn = snapshot_fn or self._read_cpu_snapshot
        self._cpu_cursor = 0
        self._active_step: Optional[StepTrace] = None
        self._capture_trace: Optional[GraphTrace] = None
        self._graphs: dict[tuple[int, int], GraphTrace] = {}
        self._plan_epoch = 0
        self._plan_by_layer: dict[int, tuple[int, int]] = {}
        self._slot_generation = 0
        self._task_id_offset = 0
        self._max_exported_task_id = 0
        self._closed = False
        expected_rank_ids = [int(x) for x in (expected_rank_ids or [self.rank_id])]
        expected_layer_ids = [int(x) for x in (expected_layer_ids or [])]

        self._run_fields: dict[str, Any] = {
            "event": "start",
            "trace_enabled": True,
            "data_origin": str(data_origin),
            "pid": os.getpid(),
            "world_size": len(expected_rank_ids),
            "expected_rank_ids": expected_rank_ids,
            "num_layers": len(expected_layer_ids),
            "expected_layer_ids": expected_layer_ids,
            "clock": {
                "gpu": "cuda_event",
                "cpu": "CLOCK_MONOTONIC",
            },
            "cuda_event_external": True,
            "decode_latency_scope": "model_decode_gpu",
            "batch_step_id_scope": "decode_forwards_since_segment_start",
            "pre_step_sync": "device_synchronize_when_previous_decode_untraced",
            "request_decode_step_scope": (
                "tokens_committed_for_request_before_model_decode"
            ),
            "trace_collect_overhead_scope": (
                "host_after_trace_sync_until_records_ready_excluding_jsonl_write"
            ),
            "trace_sync_wall_scope": "host_wait_for_end_cuda_event",
            "branch_definition": "token_expert_pair",
            "branches_received_scope": (
                "actual_non_padding_gpu_and_cpu_pairs_received_by_rank"
            ),
            "executed_offloaded_branches_scope": (
                "daemon_actual_pairs_including_cuda_graph_padding"
            ),
            "gpu_wait_cpu_total_scope": (
                "union_of_main_stream_visible_wait_intervals_on_this_rank"
            ),
            "cpu_path_scope": "cuda_d2h_start_to_h2d_end",
            "cpu_compute_scope": "daemon_forwardExperts_CLOCK_MONOTONIC",
            "cpu_task_record_scope": "real_FORWARD_EXPERTS_only",
            "noop_policy": "NOOP_submissions_are_not_emitted_as_cpu_task",
            "layer_phase_scope": {
                "dispatch_us": "layer_entry_through_rank_dispatch",
                "local_moe_us": "rank_dispatch_end_through_local_expert_compute",
                "combine_us": (
                    "local_expert_end_through_dispatch_combine_cpu_join_and_rank_reduce"
                ),
            },
        }
        atexit.register(self.close)

    @staticmethod
    def _new_cuda_event():
        # external=True 才会在 CUDA Graph 中生成可由 replay 刷新的 event 节点。
        return torch.cuda.Event(enable_timing=True, external=True)

    @staticmethod
    def _cuda_synchronize() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _current_run_id(self) -> str:
        return self._segment.run_id if self._segment is not None else self.run_id

    def _current_sample_every(self) -> int:
        if self._segment is not None:
            return self._segment.sample_every
        return self.sample_every

    def _base_record(self, kind: str, batch_step_id: Optional[int]) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "run_id": self._current_run_id(),
            "hetero_mode": self.hetero_mode,
            "batch_step_id": batch_step_id,
            "rank_id": self.rank_id,
        }

    def _run_record(self, output_file: str) -> dict[str, Any]:
        sample_every = self._current_sample_every()
        return (
            self._base_record("run", batch_step_id=None)
            | self._run_fields
            | {
                "output_file": output_file,
                "sampling": _sampling_label(sample_every),
                "sample_every": sample_every,
                "diagnostic_sync": (
                    "once_per_decode_step"
                    if sample_every == 1
                    else "once_per_sampled_decode_step"
                ),
            }
        )

    @staticmethod
    def _dumps(record: dict[str, Any]) -> str:
        return json.dumps(record, ensure_ascii=False, separators=(",", ":"))

    def _sink(self) -> TextIO:
        """返回当前写入目标；调用方必须持有写锁。"""
        if self._segment is not None:
            return self._segment.fh
        if self._fh is None:
            self._fh = self.path.open("a", encoding="utf-8", buffering=1)
            if self._write_run_record and not self._top_run_written:
                self._top_run_written = True
                self._fh.write(self._dumps(self._run_record(self.path.name)) + "\n")
        return self._fh

    def _write(self, record: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            line = self._dumps(record)
            with self._write_lock:
                self._sink().write(line + "\n")
        except Exception as exc:  # 诊断功能不能拖垮服务。
            logger.warning("[moe_step_trace] JSONL 写入失败：%s", exc)

    def start_segment(
        self,
        name: str,
        fields: Optional[dict[str, Any]] = None,
        *,
        sample_every: Optional[int] = None,
    ) -> Optional[Path]:
        """把之后的记录写进 ``<输出目录>/<name>/``，步号从 1 重新数。

        CPU 任务游标、方案版本和已捕获的 CUDA Graph 都跨段保留：它们描述的
        是进程状态，不属于某一段。``fields`` 原样写进本段 run 记录的
        ``workload`` 字段，用来说明这一段跑的是什么负载。
        """
        if self._closed:
            return None
        name = str(name)
        if not _SEGMENT_NAME.fullmatch(name) or name in {".", ".."}:
            raise ValueError(f"分段名只能含字母、数字、点、下划线和连字符：{name!r}")
        self.end_segment()
        segment_dir = self.output_dir / name
        segment_dir.mkdir(parents=True, exist_ok=True)
        path = segment_dir / self.path.name
        segment = _Segment(
            name=name,
            run_id=f"{self.run_id}/{name}",
            path=path,
            fh=path.open("a", encoding="utf-8", buffering=1),
            sample_every=(
                _positive_int(sample_every, self.sample_every)
                if sample_every is not None
                else self.sample_every
            ),
            saved_decode_forwards=self._decode_forwards,
        )
        with self._write_lock:
            self._segment = segment
        self._segments_started += 1
        self._decode_forwards = 0
        if self._write_run_record:
            self._write(
                self._run_record(path.name)
                | {
                    "segment": name,
                    "base_run_id": self.run_id,
                    "workload": dict(fields or {}),
                }
            )
        return segment_dir

    def end_segment(self) -> None:
        segment = self._segment
        if segment is None:
            return
        if self._active_step is not None:
            logger.warning(
                "[moe_step_trace] 分段 %s 结束时还有没收尾的 decode 步，丢弃该步",
                segment.name,
            )
            self._active_step = None
            # 被丢弃那一步的 CPU 任务没人读过，下一个记录步要先同步、
            # 把它们当作基线丢掉。
            self._last_decode_traced = False
        with self._write_lock:
            self._segment = None
            try:
                segment.fh.flush()
                segment.fh.close()
            except Exception:
                pass
        self._decode_forwards = segment.saved_decode_forwards

    def pause(self) -> None:
        """暂停记录；暂停期间的 decode 不计入步号。"""
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def close(self) -> None:
        if self._closed:
            return
        self.end_segment()
        if self._fh is None and self._segments_started == 0 and self._write_run_record:
            # 一步没记、也没分过段：照旧留下 run 记录，表明这个 rank 开过 trace。
            try:
                with self._write_lock:
                    self._sink()
            except Exception:
                pass
        self._closed = True
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass

    def note_plan_update(
        self, layer_id: int, expert_id: int, plan_epoch: Optional[int] = None
    ) -> None:
        """记录真正写入 device slot 的方案版本。"""
        layer_id = int(layer_id)
        expert_id = int(expert_id)
        if plan_epoch is None or int(plan_epoch) <= 0:
            previous = self._plan_by_layer.get(layer_id)
            if previous is not None and previous[0] == expert_id:
                return
            self._plan_epoch += 1
            plan_epoch = self._plan_epoch
        else:
            plan_epoch = int(plan_epoch)
            self._plan_epoch = max(self._plan_epoch, plan_epoch)
        self._plan_by_layer[layer_id] = (expert_id, plan_epoch)

    def _snapshot_plan(self) -> dict[int, tuple[int, int]]:
        plan = dict(self._plan_by_layer)
        # eager 路径或旧代码可能尚未走 update_slot；此时只能把 registry 中的
        # 专家记为“版本未知”，不能凭空编一个 epoch。
        try:
            from sglang.srt.eplb.cpu_offload import get_cpu_offload_runtime

            runtime = get_cpu_offload_runtime()
            state = runtime.registry.local_state() if runtime is not None else None
            if state is not None:
                for layer_id, expert_id in state.cpu_slot_by_layer.items():
                    plan.setdefault(int(layer_id), (int(expert_id), 0))
        except Exception:
            pass
        return plan

    def _read_cpu_snapshot(self, after_task_id: int) -> dict[str, Any]:
        if not self.cpu_trace_applicable:
            return {
                "valid": True,
                "supported": True,
                "enabled": False,
                "write_task_id": int(after_task_id),
                "lost_records": 0,
                "inconsistent_records": 0,
                "records": [],
            }
        try:
            from sglang.srt.eplb.cpu_offload import get_cpu_offload_runtime

            runtime = get_cpu_offload_runtime()
            if runtime is None:
                return {
                    "valid": False,
                    "supported": False,
                    "enabled": False,
                    "write_task_id": int(after_task_id),
                    "records": [],
                    "reason": "cpu_offload_runtime_unavailable",
                }
            fn = getattr(runtime.engine, "read_task_trace_snapshot", None)
            if fn is None:
                return {
                    "valid": False,
                    "supported": False,
                    "enabled": False,
                    "write_task_id": int(after_task_id),
                    "records": [],
                    "reason": "task_trace_snapshot_api_unavailable",
                }
            return fn(after_task_id=int(after_task_id)) or {}
        except Exception as exc:
            return {
                "valid": False,
                "supported": False,
                "enabled": False,
                "write_task_id": int(after_task_id),
                "records": [],
                "reason": f"snapshot_error:{type(exc).__name__}",
            }

    @staticmethod
    def _snapshot_int(snapshot: dict[str, Any], name: str, default: int = 0) -> int:
        try:
            return int(snapshot.get(name, default) or 0)
        except (TypeError, ValueError):
            return default

    def _accept_snapshot_cursor(self, snapshot: dict[str, Any]) -> int:
        """接纳 daemon 游标；slot 重建后允许游标回退并切换任务代际。"""
        write_task_id = self._snapshot_int(snapshot, "write_task_id", self._cpu_cursor)
        if bool(snapshot.get("cursor_reset", False)):
            self._slot_generation += 1
            self._task_id_offset = self._max_exported_task_id
            self._cpu_cursor = write_task_id
        elif write_task_id >= self._cpu_cursor:
            self._cpu_cursor = write_task_id
        return write_task_id

    def begin_step(self, forward_batch: Any) -> Optional[StepTrace]:
        if self._active_step is not None:
            # 两批重叠目前不允许共用一个进程级 trace 上下文；明确标出而不是
            # 把两步事件混写到一起。
            logger.warning("[moe_step_trace] 检测到嵌套 decode forward，本步不记录")
            return None
        if self._paused or self._closed:
            self._last_decode_traced = False
            return None

        self._decode_forwards += 1
        if (self._decode_forwards - 1) % self._current_sample_every() != 0:
            self._last_decode_traced = False
            return None
        if not self._last_decode_traced:
            # 上一个 decode 步没记录，它提交的 CPU 任务可能还没做完。先等 GPU
            # 把之前的活全部做完，这些任务才会落进下面的基线快照而被丢弃；
            # 不等的话，它们会在本步结束时被读到，并按层号错配到本步。
            try:
                self._sync_fn()
            except Exception as exc:
                logger.warning("[moe_step_trace] 抽样前同步失败：%s", exc)
        self._last_decode_traced = True

        baseline = self._snapshot_fn(self._cpu_cursor)
        baseline_write = self._accept_snapshot_cursor(baseline)
        discarded = len(baseline.get("records", []) or [])
        baseline_lost = self._snapshot_int(baseline, "lost_records")
        baseline_valid = bool(baseline.get("valid", False))
        baseline_supported = bool(baseline.get("supported", False))
        baseline_enabled = bool(baseline.get("enabled", False))
        baseline_cursor_reset = bool(baseline.get("cursor_reset", False))

        request_ids = [
            str(x) for x in (getattr(forward_batch, "request_ids", None) or [])
        ]
        raw_decode_steps = getattr(forward_batch, "request_decode_steps", None) or []
        request_decode_steps: list[Optional[int]] = []
        for value in raw_decode_steps:
            try:
                request_decode_steps.append(int(value) if value is not None else None)
            except (TypeError, ValueError):
                request_decode_steps.append(None)

        try:
            start = self._event_factory()
            end = self._event_factory()
            start.record()
        except Exception as exc:
            logger.warning("[moe_step_trace] decode CUDA event 创建失败：%s", exc)
            start = end = None

        step = StepTrace(
            batch_step_id=self._decode_forwards,
            request_ids=request_ids,
            request_decode_steps=request_decode_steps,
            request_id_source=str(
                getattr(forward_batch, "request_id_source", "unavailable")
            ),
            active_batch_size=int(getattr(forward_batch, "batch_size", 0)),
            start=start,
            end=end,
            cpu_baseline_task_id=baseline_write,
            cpu_baseline_discarded_records=discarded,
            cpu_baseline_lost_records=baseline_lost,
            cpu_baseline_valid=baseline_valid,
            cpu_baseline_supported=baseline_supported,
            cpu_baseline_enabled=baseline_enabled,
            cpu_baseline_cursor_reset=baseline_cursor_reset,
            cpu_slot_generation=self._slot_generation,
            plan_by_layer=self._snapshot_plan(),
        )
        if start is None:
            step.unsupported_reasons.append("decode_cuda_event_unavailable")
        self._active_step = step
        return step

    def prepare_graph_capture(
        self,
        runner_id: int,
        graph_batch_size: int,
        graph_num_tokens: int,
        device: Any,
    ) -> Optional[GraphTrace]:
        key = (int(runner_id), int(graph_batch_size))
        try:
            active_tokens = torch.tensor(
                [int(graph_num_tokens)], dtype=torch.int32, device=device
            )
            row_indices = torch.arange(
                int(graph_num_tokens), dtype=torch.int32, device=device
            )
        except Exception as exc:
            logger.warning("[moe_step_trace] 图计数缓冲区创建失败：%s", exc)
            return None
        trace = GraphTrace(
            key=key,
            graph_batch_size=int(graph_batch_size),
            graph_num_tokens=int(graph_num_tokens),
            active_tokens=active_tokens,
            row_indices=row_indices,
        )
        self._graphs[key] = trace
        return trace

    def enter_graph_capture(self, trace: Optional[GraphTrace]) -> None:
        self._capture_trace = trace

    def leave_graph_capture(self, trace: Optional[GraphTrace]) -> None:
        if self._capture_trace is trace:
            self._capture_trace = None

    def mark_graph_replay(
        self,
        runner_id: int,
        graph_batch_size: int,
        active_num_tokens: int,
    ) -> None:
        step = self._active_step
        if step is None:
            return
        key = (int(runner_id), int(graph_batch_size))
        graph_trace = self._graphs.get(key)
        step.execution_mode = "cuda_graph_replay"
        step.graph_batch_size = int(graph_batch_size)
        step.graph_trace = graph_trace
        if graph_trace is None:
            step.unsupported_reasons.append("cuda_graph_events_not_registered")
            return
        try:
            graph_trace.active_tokens.fill_(int(active_num_tokens))
        except Exception:
            step.unsupported_reasons.append("active_token_buffer_update_failed")

    def _event_target(self) -> Optional[Any]:
        return (
            self._capture_trace
            if self._capture_trace is not None
            else self._active_step
        )

    def begin_layer(self, layer_id: int) -> Optional[LayerEvents]:
        target = self._event_target()
        if target is None:
            return None
        try:
            start = self._event_factory()
            start.record()
        except Exception as exc:
            logger.debug("[moe_step_trace] layer event 创建失败：%s", exc)
            return None
        events = LayerEvents(layer_id=int(layer_id), start=start)
        target.layers.append(events)
        return events

    def mark_layer_phase(self, events: Optional[LayerEvents], phase: str) -> None:
        if events is None:
            return
        try:
            event = self._event_factory()
            event.record()
            setattr(events, phase, event)
        except Exception:
            events.unsupported_reasons.append(f"{phase}_event_unavailable")

    def record_branches_received(
        self,
        events: Optional[LayerEvents],
        branch_ids: Any,
        *,
        local_expert_lo: Optional[int] = None,
        local_expert_hi: Optional[int] = None,
    ) -> None:
        if events is None or branch_ids is None:
            return
        try:
            valid = branch_ids >= 0
            if local_expert_lo is not None and local_expert_hi is not None:
                valid = valid & (branch_ids >= int(local_expert_lo))
                valid = valid & (branch_ids < int(local_expert_hi))
            graph_trace = self._capture_trace
            if graph_trace is not None and branch_ids.ndim >= 1:
                rows = graph_trace.row_indices[: branch_ids.shape[0]]
                active = rows < graph_trace.active_tokens[0]
                while active.ndim < valid.ndim:
                    active = active.unsqueeze(-1)
                valid = valid & active
            events.branches_received = valid.to(torch.int64).sum()
        except Exception:
            events.unsupported_reasons.append("branches_received_unavailable")

    def limit_to_active_rows(self, mask: Any) -> Any:
        """仅给统计量剔除 CUDA Graph padding 行，不能改实际执行掩码。"""
        graph_trace = self._capture_trace
        if graph_trace is None or getattr(mask, "ndim", 0) < 1:
            return mask
        try:
            rows = graph_trace.row_indices[: mask.shape[0]]
            active = rows < graph_trace.active_tokens[0]
            while active.ndim < mask.ndim:
                active = active.unsqueeze(-1)
            return mask & active
        except Exception:
            return mask

    def record_cpu_business_branches(
        self,
        events: Optional[CpuGpuEvents],
        token_mask: Any,
        *,
        max_branches: Optional[int] = None,
    ) -> None:
        """记录真实请求最终交给 daemon 的分支数，不改变 offload 执行。"""
        if events is None or token_mask is None:
            return
        try:
            business_mask = self.limit_to_active_rows(token_mask)
            branches = business_mask.to(torch.int64).sum()
            if max_branches is not None:
                branches = torch.clamp(branches, max=int(max_branches))
            events.business_offloaded_branches = branches
        except Exception:
            events.unsupported_reasons.append("business_offloaded_branches_unavailable")

    def begin_cpu_gpu_task(self, layer_id: int) -> Optional[CpuGpuEvents]:
        target = self._event_target()
        if target is None:
            return None
        try:
            events = CpuGpuEvents(
                layer_id=int(layer_id),
                d2h_start=self._event_factory(),
                d2h_end=self._event_factory(),
                wait_start=self._event_factory(),
                wait_end=self._event_factory(),
                h2d_start=self._event_factory(),
                h2d_end=self._event_factory(),
                visible_wait_start=self._event_factory(),
                visible_wait_end=self._event_factory(),
            )
        except Exception as exc:
            logger.debug("[moe_step_trace] CPU 路径 CUDA event 创建失败：%s", exc)
            return None
        target.cpu_events.append(events)
        return events

    @staticmethod
    def record_visible_wait_start(events: Optional[CpuGpuEvents]) -> None:
        if events is not None:
            events.visible_wait_start.record()

    @staticmethod
    def record_visible_wait_end(events: Optional[CpuGpuEvents]) -> None:
        if events is not None:
            events.visible_wait_end.record()

    @staticmethod
    def record_wait_status(events: Optional[CpuGpuEvents], status: Any) -> None:
        if events is None:
            return
        try:
            # 调用点已在主流等待 side stream，因此这里的副本不会与 wait kernel 竞争。
            events.wait_status = status.clone()
        except Exception:
            events.unsupported_reasons.append("gpu_wait_status_unavailable")

    @staticmethod
    def _elapsed_us(start: Any, end: Any) -> Optional[float]:
        if start is None or end is None:
            return None
        try:
            return float(start.elapsed_time(end)) * 1000.0
        except Exception:
            return None

    @staticmethod
    def _tensor_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value.item())
        except Exception:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

    @staticmethod
    def _ns_to_us(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value) / 1000.0
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _completeness(
        values: list[Any], reasons: list[str]
    ) -> tuple[str, Optional[str]]:
        if reasons and all(value is None for value in values):
            return "unsupported", ",".join(sorted(set(reasons)))
        if reasons or any(value is None for value in values):
            all_reasons = list(reasons)
            if any(value is None for value in values):
                all_reasons.append("missing_measurement")
            return "partial", ",".join(sorted(set(all_reasons)))
        return "complete", None

    def _layer_records(
        self,
        step: StepTrace,
        layers: list[LayerEvents],
        cpu_events: list[CpuGpuEvents],
    ) -> list[dict[str, Any]]:
        cpu_branches_by_layer: dict[int, list[Optional[int]]] = {}
        for cpu_event in cpu_events:
            cpu_branches_by_layer.setdefault(cpu_event.layer_id, []).append(
                self._tensor_int(cpu_event.business_offloaded_branches)
            )
        records: list[dict[str, Any]] = []
        for events in layers:
            dispatch_us = self._elapsed_us(events.start, events.dispatch_end)
            local_moe_us = self._elapsed_us(events.dispatch_end, events.local_moe_end)
            combine_us = self._elapsed_us(events.local_moe_end, events.end)
            total_us = self._elapsed_us(events.start, events.end)
            gpu_branches = self._tensor_int(events.branches_received)
            layer_cpu_branches = cpu_branches_by_layer.get(events.layer_id, [])
            cpu_branches = layer_cpu_branches.pop(0) if layer_cpu_branches else 0
            branches = (
                gpu_branches + cpu_branches
                if gpu_branches is not None and cpu_branches is not None
                else None
            )
            values = [dispatch_us, local_moe_us, combine_us, total_us, branches]
            completeness, reason = self._completeness(
                values, events.unsupported_reasons
            )
            records.append(
                self._base_record("layer_rank", step.batch_step_id)
                | {
                    "layer_id": events.layer_id,
                    "dispatch_us": dispatch_us,
                    "local_moe_us": local_moe_us,
                    "combine_us": combine_us,
                    "layer_rank_total_us": total_us,
                    "branches_received": branches,
                    "gpu_branches_received": gpu_branches,
                    "cpu_branches_received": cpu_branches,
                    "active_batch_size": step.active_batch_size,
                    "graph_batch_size": step.graph_batch_size,
                    "completeness": completeness,
                    "unsupported_reason": reason,
                }
            )
        return records

    def _cpu_event_values(self, events: CpuGpuEvents) -> dict[str, Any]:
        status = self._tensor_int(events.wait_status)
        business_offloaded_branches = self._tensor_int(
            events.business_offloaded_branches
        )
        ready_after_join_start_us = self._elapsed_us(
            events.visible_wait_start, events.h2d_end
        )
        visible_wait_us = (
            max(0.0, ready_after_join_start_us)
            if ready_after_join_start_us is not None
            else None
        )
        return {
            "d2h_us": self._elapsed_us(events.d2h_start, events.d2h_end),
            "sync_us": self._elapsed_us(events.wait_start, events.wait_end),
            "h2d_us": self._elapsed_us(events.h2d_start, events.h2d_end),
            "cpu_path_us": self._elapsed_us(events.d2h_start, events.h2d_end),
            # wait_stream 前后 event 会把 event 节点和同步指令本身约 4 微秒的
            # 固定税也算进去。visible_wait 只取 side 结果晚于主流进入 join 的
            # 部分；完整包围区间另存 join_span_us 供诊断。
            "visible_wait_us": visible_wait_us,
            "join_span_us": self._elapsed_us(
                events.visible_wait_start, events.visible_wait_end
            ),
            "gpu_wait_status": status,
            "business_offloaded_branches": business_offloaded_branches,
        }

    def _cpu_records(
        self,
        step: StepTrace,
        snapshot: dict[str, Any],
        cpu_events: list[CpuGpuEvents],
    ) -> tuple[list[dict[str, Any]], float]:
        by_layer: dict[int, list[CpuGpuEvents]] = {}
        for events in cpu_events:
            by_layer.setdefault(events.layer_id, []).append(events)

        records: list[dict[str, Any]] = []
        visible_intervals: list[float] = []
        for raw in snapshot.get("records", []) or []:
            try:
                layer_id = int(raw.get("layer_id", -1))
            except (TypeError, ValueError):
                layer_id = -1
            layer_queue = by_layer.get(layer_id, [])
            gpu_events = layer_queue.pop(0) if layer_queue else None
            gpu_values = (
                self._cpu_event_values(gpu_events)
                if gpu_events is not None
                else {
                    "d2h_us": None,
                    "sync_us": None,
                    "h2d_us": None,
                    "cpu_path_us": None,
                    "visible_wait_us": None,
                    "join_span_us": None,
                    "gpu_wait_status": None,
                    "business_offloaded_branches": None,
                }
            )
            if gpu_values["visible_wait_us"] is not None:
                visible_intervals.append(float(gpu_values["visible_wait_us"]))

            try:
                expert_id = int(raw.get("expert_id", -1))
            except (TypeError, ValueError):
                expert_id = -1
            plan = step.plan_by_layer.get(layer_id)
            plan_epoch = None
            reasons: list[str] = []
            if plan is None:
                reasons.append("plan_epoch_unavailable")
            elif plan[0] != expert_id:
                reasons.append("task_expert_does_not_match_step_plan")
            elif plan[1] <= 0:
                reasons.append("plan_epoch_unavailable")
            else:
                plan_epoch = plan[1]

            error_code = int(raw.get("error_code", 0) or 0)
            did_compute = bool(raw.get("did_compute", False))
            daemon_status = str(raw.get("status") or "unknown")
            gpu_wait_status = gpu_values["gpu_wait_status"]
            executed_offloaded_branches = int(raw.get("offloaded_branches", 0) or 0)
            business_offloaded_branches = gpu_values["business_offloaded_branches"]
            if business_offloaded_branches is not None:
                business_offloaded_branches = int(business_offloaded_branches)
            padding_offloaded_branches = (
                max(
                    0,
                    executed_offloaded_branches - business_offloaded_branches,
                )
                if business_offloaded_branches is not None
                else None
            )
            if error_code != 0:
                status = "daemon_error"
            elif gpu_wait_status not in (None, 0):
                status = "wait_timeout"
            elif daemon_status == "no_compute" or not did_compute:
                status = (
                    "skipped_empty"
                    if executed_offloaded_branches == 0
                    else "no_compute"
                )
            else:
                status = "completed"

            if gpu_events is None:
                reasons.append("gpu_phase_events_unmatched")
            elif gpu_events.unsupported_reasons:
                reasons.extend(gpu_events.unsupported_reasons)
            values = [
                gpu_values["d2h_us"],
                self._ns_to_us(raw.get("cpu_compute_ns")),
                gpu_values["h2d_us"],
                gpu_values["sync_us"],
                gpu_values["cpu_path_us"],
                gpu_values["visible_wait_us"],
                plan_epoch,
                business_offloaded_branches,
            ]
            completeness, reason = self._completeness(values, reasons)

            daemon_task_id = int(raw.get("task_id", 0) or 0)
            task_id = self._task_id_offset + daemon_task_id
            self._max_exported_task_id = max(self._max_exported_task_id, task_id)
            records.append(
                self._base_record("cpu_task", step.batch_step_id)
                | {
                    "task_id": task_id,
                    "daemon_task_id": daemon_task_id,
                    "slot_generation": step.cpu_slot_generation,
                    "req_seq": int(raw.get("req_seq", 0) or 0),
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "worker_id": (
                        int(raw["worker_id"])
                        if raw.get("worker_id") is not None
                        else None
                    ),
                    "socket_id": raw.get("socket_id", snapshot.get("socket_id")),
                    "slot_id": raw.get("slot_id", snapshot.get("slot_id")),
                    # 主字段保持 daemon 的真实 fwd_pair_count，和这条任务的
                    # CPU 耗时口径一致；业务行与图 padding 另列。
                    "offloaded_branches": executed_offloaded_branches,
                    "executed_offloaded_branches": executed_offloaded_branches,
                    "business_offloaded_branches": business_offloaded_branches,
                    "padding_offloaded_branches": padding_offloaded_branches,
                    "active_tokens": int(raw.get("active_tokens", 0) or 0),
                    "cpu_pairs": int(raw.get("cpu_pairs", 0) or 0),
                    "d2h_us": gpu_values["d2h_us"],
                    "cpu_compute_us": self._ns_to_us(raw.get("cpu_compute_ns")),
                    "h2d_us": gpu_values["h2d_us"],
                    "sync_us": gpu_values["sync_us"],
                    "cpu_path_us": gpu_values["cpu_path_us"],
                    "visible_wait_us": gpu_values["visible_wait_us"],
                    "join_span_us": gpu_values["join_span_us"],
                    "daemon_service_us": self._ns_to_us(raw.get("daemon_service_ns")),
                    "daemon_dispatch_us": self._ns_to_us(raw.get("dispatch_ns")),
                    "daemon_flush_us": self._ns_to_us(raw.get("flush_ns")),
                    "daemon_queue_us": self._ns_to_us(raw.get("queue_ns")),
                    "daemon_protocol_us": self._ns_to_us(raw.get("protocol_ns")),
                    "request_observed_ns": raw.get("request_observed_ns"),
                    "compute_start_ns": raw.get("compute_start_ns"),
                    "compute_end_ns": raw.get("compute_end_ns"),
                    "trace_ready_ns": raw.get("trace_ready_ns"),
                    "plan_epoch": plan_epoch,
                    "error_code": error_code,
                    "did_compute": did_compute,
                    "daemon_status": daemon_status,
                    "is_noop": False,
                    "gpu_wait_status": gpu_wait_status,
                    "status": status,
                    "completeness": completeness,
                    "unsupported_reason": reason,
                }
            )

        # 主流上的每段 visible wait 按层串行，不会彼此重叠，因此并集就是总和。
        return records, sum(visible_intervals)

    def end_step(self, step: Optional[StepTrace], *, failed: bool = False) -> None:
        if step is None or self._active_step is not step:
            return
        sync_start_ns = time.monotonic_ns()
        try:
            if step.end is not None:
                step.end.record()
                step.end.synchronize()
        except Exception:
            step.unsupported_reasons.append("decode_step_sync_failed")
        trace_sync_wall_us = (time.monotonic_ns() - sync_start_ns) / 1000.0
        collect_start_ns = time.monotonic_ns()

        # 同步完成后才读 event 和 daemon ring；后续 decode 尚未开始，不会覆盖本步。
        snapshot = self._snapshot_fn(step.cpu_baseline_task_id)
        snapshot_valid = bool(snapshot.get("valid", False))
        snapshot_supported = bool(snapshot.get("supported", False))
        snapshot_enabled = bool(snapshot.get("enabled", False))
        cursor_reset = bool(snapshot.get("cursor_reset", False))
        write_task_id = self._accept_snapshot_cursor(snapshot)
        if cursor_reset:
            # daemon 在本步中途重建时，快照中的任务属于新代际，无法保证
            # 与本步起点严格对应，因此不把它们强行绑定到本步 GPU event。
            step.cpu_slot_generation = self._slot_generation

        layers = (
            step.graph_trace.layers if step.graph_trace is not None else step.layers
        )
        cpu_events = (
            step.graph_trace.cpu_events
            if step.graph_trace is not None
            else step.cpu_events
        )
        layer_records = self._layer_records(step, layers, cpu_events)
        task_records_can_be_assigned = (
            step.cpu_baseline_valid
            and step.cpu_baseline_supported
            and step.cpu_baseline_enabled
            and snapshot_valid
            and snapshot_supported
            and snapshot_enabled
            and not cursor_reset
        )
        unassigned_task_records = (
            0
            if task_records_can_be_assigned
            else len(snapshot.get("records", []) or [])
        )
        task_snapshot = snapshot
        if not task_records_can_be_assigned:
            task_snapshot = dict(snapshot)
            task_snapshot["records"] = []
        cpu_records, visible_wait_total = self._cpu_records(
            step, task_snapshot, cpu_events
        )

        decode_latency_us = self._elapsed_us(step.start, step.end)
        lost_records = self._snapshot_int(snapshot, "lost_records")
        inconsistent_records = self._snapshot_int(snapshot, "inconsistent_records")
        cpu_totals_reliable = not self.cpu_trace_applicable or (
            step.cpu_baseline_valid
            and step.cpu_baseline_supported
            and step.cpu_baseline_enabled
            and snapshot_valid
            and snapshot_supported
            and snapshot_enabled
            and not cursor_reset
            and lost_records == 0
            and inconsistent_records == 0
        )
        if not self.cpu_trace_applicable:
            total_offloaded = 0
            total_business_offloaded = 0
            total_padding_offloaded = 0
            visible_wait_total_value: Optional[float] = 0.0
        elif cpu_totals_reliable:
            total_offloaded = sum(
                int(record.get("offloaded_branches", 0) or 0) for record in cpu_records
            )
            business_values = [
                record.get("business_offloaded_branches") for record in cpu_records
            ]
            padding_values = [
                record.get("padding_offloaded_branches") for record in cpu_records
            ]
            total_business_offloaded = (
                sum(int(value) for value in business_values)
                if all(value is not None for value in business_values)
                else None
            )
            total_padding_offloaded = (
                sum(int(value) for value in padding_values)
                if all(value is not None for value in padding_values)
                else None
            )
            visible_wait_total_value = visible_wait_total
        else:
            total_offloaded = None
            total_business_offloaded = None
            total_padding_offloaded = None
            visible_wait_total_value = None
        reasons = list(step.unsupported_reasons)
        if failed:
            reasons.append("model_forward_failed")
        if self.cpu_trace_applicable:
            if not step.cpu_baseline_valid:
                reasons.append("cpu_trace_baseline_invalid")
            elif not step.cpu_baseline_supported:
                reasons.append("cpu_trace_baseline_unsupported")
            elif not step.cpu_baseline_enabled:
                reasons.append("cpu_trace_baseline_not_enabled")
            if not snapshot_valid:
                reasons.append(
                    str(snapshot.get("reason") or "cpu_trace_snapshot_invalid")
                )
            elif not snapshot_supported:
                reasons.append(str(snapshot.get("reason") or "cpu_trace_unsupported"))
            elif not snapshot_enabled:
                reasons.append("cpu_trace_not_enabled")
        if cursor_reset:
            reasons.append("cpu_trace_cursor_reset")
        if lost_records > 0:
            reasons.append("cpu_trace_ring_overflow")
        if inconsistent_records > 0:
            reasons.append("cpu_trace_inconsistent_record")
        if not layers:
            reasons.append("no_layer_events")
        child_incomplete = any(
            record.get("completeness") != "complete"
            for record in layer_records + cpu_records
        )
        if child_incomplete:
            reasons.append("child_record_incomplete")

        required_step_values = [decode_latency_us]
        completeness, reason = self._completeness(required_step_values, reasons)
        collect_overhead_us = (time.monotonic_ns() - collect_start_ns) / 1000.0

        requests = []
        for index, request_id in enumerate(step.request_ids):
            decode_step_id = (
                step.request_decode_steps[index]
                if index < len(step.request_decode_steps)
                else None
            )
            requests.append(
                {
                    "request_id": request_id,
                    "decode_step_id": decode_step_id,
                    "identity": step.request_id_source,
                }
            )

        step_record = self._base_record("step", step.batch_step_id) | {
            "request_ids": step.request_ids,
            "request_decode_steps": step.request_decode_steps,
            "request_id_source": step.request_id_source,
            "requests": requests,
            "decode_latency_us": decode_latency_us,
            "decode_latency_scope": "model_decode_gpu",
            "active_batch_size": step.active_batch_size,
            "graph_batch_size": step.graph_batch_size,
            "execution_mode": step.execution_mode,
            "total_offloaded_branches": total_offloaded,
            "total_business_offloaded_branches": total_business_offloaded,
            "total_padding_offloaded_branches": total_padding_offloaded,
            "gpu_wait_cpu_total_us": visible_wait_total_value,
            "gpu_wait_cpu_total_scope": "main_stream_visible_wait_union",
            "trace_collect_overhead_us": collect_overhead_us,
            "trace_collect_overhead_scope": (
                "host_after_trace_sync_until_records_ready_excluding_jsonl_write"
            ),
            "trace_sync_wall_us": trace_sync_wall_us,
            "trace_sync_wall_scope": "host_wait_for_end_cuda_event",
            "cpu_trace_lost_records": lost_records,
            "cpu_trace_inconsistent_records": inconsistent_records,
            "cpu_trace_baseline_task_id": step.cpu_baseline_task_id,
            "cpu_trace_baseline_discarded_records": (
                step.cpu_baseline_discarded_records
            ),
            "cpu_trace_baseline_lost_records": step.cpu_baseline_lost_records,
            "cpu_trace_baseline_valid": step.cpu_baseline_valid,
            "cpu_trace_baseline_supported": step.cpu_baseline_supported,
            "cpu_trace_baseline_enabled": step.cpu_baseline_enabled,
            "cpu_trace_baseline_cursor_reset": step.cpu_baseline_cursor_reset,
            "cpu_trace_cursor_reset": cursor_reset,
            "cpu_trace_write_task_id": write_task_id,
            "cpu_trace_slot_generation": step.cpu_slot_generation,
            "cpu_trace_unassigned_task_records": unassigned_task_records,
            "completeness": completeness,
            "unsupported_reason": reason,
        }

        self._write(step_record)
        for record in layer_records:
            self._write(record)
        for record in cpu_records:
            self._write(record)
        self._active_step = None


_recorder_lock = threading.Lock()
_recorder: Optional[StepTraceRecorder] = None


def init_step_trace(model_runner: Any) -> Optional[StepTraceRecorder]:
    """按 ModelRunner 的 rank 和模式创建进程级记录器。"""
    global _recorder
    if not ENABLED:
        return None
    if _recorder is not None:
        return _recorder
    with _recorder_lock:
        if _recorder is not None:
            return _recorder
        args = model_runner.server_args
        dynamic = bool(getattr(args, "enable_cpu_offload_fast_loop", False))
        static = bool(getattr(args, "enable_ep_moe_heto", False))
        hetero_mode = (
            "dynamic_dual_loop" if dynamic else "static_heto" if static else "gpu_only"
        )
        expected_layer_ids: list[int] = []
        runtime = None
        # static_heto 也可能真实走 CPU；本轮没有它的逐任务快照接口时必须
        # 输出 unsupported/null，不能把“未知”伪装成纯 GPU 的 0。
        cpu_trace_applicable = static
        try:
            routed_weights = getattr(
                model_runner.model, "routed_experts_weights_of_layer", None
            )
            if routed_weights is not None:
                expected_layer_ids = sorted(int(x) for x in routed_weights.keys())
        except Exception:
            pass
        try:
            from sglang.srt.eplb.cpu_offload import get_cpu_offload_runtime

            runtime = get_cpu_offload_runtime()
            if runtime is not None:
                registered = [int(x) for x in runtime.engine.registered_layers()]
                if registered:
                    expected_layer_ids = sorted(registered)
                cpu_trace_applicable = static or bool(runtime.is_active_locally())
        except Exception:
            pass
        if not expected_layer_ids:
            expected_layer_ids = list(
                range(int(getattr(model_runner.model_config, "num_hidden_layers", 0)))
            )
        _recorder = StepTraceRecorder(
            rank_id=int(model_runner.tp_rank),
            hetero_mode=hetero_mode,
            cpu_trace_applicable=cpu_trace_applicable,
            expected_rank_ids=list(range(int(model_runner.tp_size))),
            expected_layer_ids=expected_layer_ids,
        )
        return _recorder


def get_step_trace() -> Optional[StepTraceRecorder]:
    return _recorder if ENABLED else None


def begin_step(forward_batch: Any) -> Optional[StepTrace]:
    recorder = get_step_trace()
    return recorder.begin_step(forward_batch) if recorder is not None else None


def end_step(step: Optional[StepTrace], *, failed: bool = False) -> None:
    recorder = get_step_trace()
    if recorder is not None:
        recorder.end_step(step, failed=failed)


def start_segment(
    name: str,
    fields: Optional[dict[str, Any]] = None,
    *,
    sample_every: Optional[int] = None,
) -> Optional[Path]:
    """trace 没开时什么都不做，返回 None。"""
    recorder = get_step_trace()
    if recorder is None:
        return None
    return recorder.start_segment(name, fields, sample_every=sample_every)


def end_segment() -> None:
    recorder = get_step_trace()
    if recorder is not None:
        recorder.end_segment()


def pause() -> None:
    recorder = get_step_trace()
    if recorder is not None:
        recorder.pause()


def resume() -> None:
    recorder = get_step_trace()
    if recorder is not None:
        recorder.resume()


def prepare_graph_capture(
    runner_id: int,
    graph_batch_size: int,
    graph_num_tokens: int,
    device: Any,
) -> Optional[GraphTrace]:
    recorder = get_step_trace()
    if recorder is None:
        return None
    return recorder.prepare_graph_capture(
        runner_id, graph_batch_size, graph_num_tokens, device
    )


def enter_graph_capture(trace: Optional[GraphTrace]) -> None:
    recorder = get_step_trace()
    if recorder is not None:
        recorder.enter_graph_capture(trace)


def leave_graph_capture(trace: Optional[GraphTrace]) -> None:
    recorder = get_step_trace()
    if recorder is not None:
        recorder.leave_graph_capture(trace)


def mark_graph_replay(
    runner_id: int, graph_batch_size: int, active_num_tokens: int
) -> None:
    recorder = get_step_trace()
    if recorder is not None:
        recorder.mark_graph_replay(runner_id, graph_batch_size, active_num_tokens)


def begin_layer(layer_id: int) -> Optional[LayerEvents]:
    recorder = get_step_trace()
    return recorder.begin_layer(layer_id) if recorder is not None else None


def mark_dispatch_done(events: Optional[LayerEvents]) -> None:
    recorder = get_step_trace()
    if recorder is not None:
        recorder.mark_layer_phase(events, "dispatch_end")


def mark_local_moe_done(events: Optional[LayerEvents]) -> None:
    recorder = get_step_trace()
    if recorder is not None:
        recorder.mark_layer_phase(events, "local_moe_end")


def mark_layer_done(events: Optional[LayerEvents]) -> None:
    recorder = get_step_trace()
    if recorder is not None:
        recorder.mark_layer_phase(events, "end")


def record_branches_received(
    events: Optional[LayerEvents],
    branch_ids: Any,
    *,
    local_expert_lo: Optional[int] = None,
    local_expert_hi: Optional[int] = None,
) -> None:
    recorder = get_step_trace()
    if recorder is not None:
        recorder.record_branches_received(
            events,
            branch_ids,
            local_expert_lo=local_expert_lo,
            local_expert_hi=local_expert_hi,
        )


def limit_to_active_rows(mask: Any) -> Any:
    recorder = get_step_trace()
    return recorder.limit_to_active_rows(mask) if recorder is not None else mask


def begin_cpu_gpu_task(layer_id: int) -> Optional[CpuGpuEvents]:
    recorder = get_step_trace()
    return recorder.begin_cpu_gpu_task(layer_id) if recorder is not None else None


def record_cpu_business_branches(
    events: Optional[CpuGpuEvents],
    token_mask: Any,
    *,
    max_branches: Optional[int] = None,
) -> None:
    recorder = get_step_trace()
    if recorder is not None:
        recorder.record_cpu_business_branches(
            events, token_mask, max_branches=max_branches
        )


def record_visible_wait_start(events: Optional[CpuGpuEvents]) -> None:
    StepTraceRecorder.record_visible_wait_start(events)


def record_visible_wait_end(events: Optional[CpuGpuEvents]) -> None:
    StepTraceRecorder.record_visible_wait_end(events)


def record_wait_status(events: Optional[CpuGpuEvents], status: Any) -> None:
    StepTraceRecorder.record_wait_status(events, status)


def note_plan_update(
    layer_id: int, expert_id: int, plan_epoch: Optional[int] = None
) -> None:
    recorder = get_step_trace()
    if recorder is not None:
        recorder.note_plan_update(layer_id, expert_id, plan_epoch)


def reset_for_tests() -> None:
    global _recorder
    if _recorder is not None:
        _recorder.close()
    _recorder = None
