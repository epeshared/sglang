"""初始化时选择通信实现；原 daemon 和全部逐请求方法不加 wrapper。"""

import json
import logging
import os
from pathlib import Path

from .engine_daemon import DaemonBackend


def transport_name(config):
    name = os.environ.get("CPU_OFFLOAD_TRANSPORT", "shm")
    if name == "anonymous":
        name = "pinned"  # 兼容旧配置；只在初始化时归一化名称。
    if name not in ("shm", "pinned"):
        raise ValueError(f"未知 CPU_OFFLOAD_TRANSPORT：{name!r}")
    if name != "shm" and config.cpu_offload_backend != "daemon":
        raise ValueError("pinned 通信要求 daemon 接口，不能与 legacy fork 路径混用")
    return name


def make_backend(config, **kwargs):
    cls = DaemonBackend if transport_name(config) == "shm" else AnonymousBackend
    return cls(config, **kwargs)


def _boolean(name, default):
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} 只能为 0 或 1")
    return value == "1"


class AnonymousBackend(DaemonBackend):
    """只重写初始化和关闭；层注册及计算方法沿用原 DaemonBackend。"""

    def init_runtime(self):
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            cfg = self._config
            self._owner = None
            # 必须先排除不参与的 rank，避免八卡全部载入 CPU 权重。
            if not cfg.rank_is_active(self._host_rank):
                self._slot_id = -1
                self._initialized = True
                return
            slot = cfg.compute_worker_slot(
                self._host_rank,
                world_size=self._world_size,
                num_numa_nodes=self._num_numa_nodes,
                num_workers_per_numa=cfg.cpu_offload_num_workers_per_numa,
            )
            if slot is None:
                raise ValueError(
                    "参与 offload 的 rank 没有 worker，请显式配置有效 worker 数和映射"
                )
            if cfg.cpu_offload_daemon_launcher:
                raise ValueError(
                    "pinned 后端不启动外部 daemon，请清空 CPU_OFFLOAD_DAEMON_LAUNCHER"
                )
            if _boolean("SGLANG_MOE_STEP_TRACE", "0"):
                raise ValueError(
                    "本后端基于协议 12，不支持附加 trace 协议；请关闭 STEP_TRACE"
                )
            mapping = json.loads(os.environ.get("CPU_OFFLOAD_TRANSPORT_RANKS", "{}"))
            if not isinstance(mapping, dict) or str(self._host_rank) not in mapping:
                raise ValueError(
                    "每个参与 rank 必须在 CPU_OFFLOAD_TRANSPORT_RANKS 声明 node、cores、threads"
                )
            entry = mapping[str(self._host_rank)]
            if not isinstance(entry, dict) or set(entry) != {
                "node",
                "cores",
                "threads",
            }:
                raise ValueError("rank 配置必须且只能包含 node、cores、threads")
            local = cfg.numa_for_rank(
                self._host_rank,
                world_size=self._world_size,
                num_numa_nodes=self._num_numa_nodes,
            )
            nodes = os.environ.get("CPU_NUMA_NODE_IDS", "").strip()
            actual = [int(x) for x in nodes.split(",")][local] if nodes else local
            if int(entry["node"]) != actual:
                raise ValueError("新后端 NUMA 与原 rank 到节点映射不一致")
            model_path = os.environ.get("CPU_OFFLOAD_MODEL_PATH")
            if not model_path:
                raise ValueError(
                    "真实模型入口要求 CPU_OFFLOAD_MODEL_PATH，复用原完整专家预载"
                )
            from moe_transport import TransportOwner
            from moe_transport.placement import parse_cores

            from sglang._pinned_cpu_threads import require_bootstrap

            early_bootstrap = require_bootstrap()
            cores = parse_cores(entry["cores"])
            if set(cores) & set(os.sched_getaffinity(0)):
                raise ValueError("专家计算核与 GPU rank 主线程预留核重叠")
            owner = TransportOwner(
                node=actual,
                slot=int(slot),
                cores=cores,
                threads=int(entry["threads"]),
                model_path=model_path,
                layer_range=os.environ.get("CPU_OFFLOAD_LAYER_RANGE"),
                use_thp=_boolean("CPU_OFFLOAD_TRANSPORT_USE_THP", "1"),
                force_thp_failure=_boolean(
                    "CPU_OFFLOAD_TRANSPORT_FORCE_THP_FAILURE", "0"
                ),
            )
            try:
                self._owner = owner
                self._engine_mod = owner.engine
                self._remote_moe_cls = owner.make_remote
                self._slot_handle = owner.handle
                self._socket_id, self._slot_id = actual, int(slot)
                self._finished = False
                metadata = dict(owner.metadata, rank=self._host_rank)
                # 初始化时保存实际运行条件；不向每层请求插入观测。
                libraries = sorted(
                    {
                        line.split()[-1]
                        for line in Path("/proc/self/maps").read_text().splitlines()
                        if any(
                            name in line for name in ("libgomp", "libiomp", "libomp.so")
                        )
                    }
                )
                metadata["thread_runtime"] = {
                    "main_threads": int(owner.torch.get_num_threads()),
                    "worker_team": owner.metadata.get("startup_compute_team"),
                    "wait_environment": {
                        name: os.environ.get(name)
                        for name in ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT")
                    },
                    "openmp_libraries": libraries,
                }
                metadata["thread_runtime"]["early_bootstrap"] = early_bootstrap
                metadata["计算与预取环境"] = {
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith("CPU_OFFLOAD_PREFETCH")
                    or k
                    in (
                        "MOE_DTYPE",
                        "MOE_KERNEL_BACKEND",
                        "TOKS_REORDER",
                        "OMP_NUM_THREADS",
                        "CPU_OFFLOAD_WARMUP_LEVEL",
                        "CPU_OFFLOAD_GPU_QUANT",
                    )
                }
                text = json.dumps(metadata, ensure_ascii=False, indent=2)
                directory = os.environ.get("CPU_OFFLOAD_TRANSPORT_METADATA_DIR")
                if directory:
                    path = Path(directory)
                    path.mkdir(parents=True, exist_ok=True)
                    self._metadata_path = path / f"rank-{self._host_rank}.json"
                    self._metadata_path.write_text(text + "\n")
                else:
                    self._metadata_path = None
                self._transport_metadata = metadata
                # multiprocessing 的正常退出会执行 Finalize，普通 atexit 不一定执行。
                # 只在进程退出时读取最终计数，不向逐层路径加采样或文件写入。
                from multiprocessing.util import Finalize

                self._finalizer = Finalize(self, self._finish, exitpriority=20)
                logging.getLogger(__name__).info("通信初始化元数据：%s", text)
                self._initialized = True
            except BaseException:
                owner.close()
                raise

    def close(self):
        """调用方须先停止调度和图重放；保留失败资源，不吞掉关闭错误。"""
        if getattr(self, "_owner", None):
            self._finish()
            self._finalizer.cancel()
        self._slot_handle = 0
        self._layers.clear()
        self._weight_retain.clear()
        self._initialized = False

    def read_daemon_stats(self):
        """释放后的退出统计读快照，绝不再解引用原生 slot。"""
        if not self._slot_handle:
            return dict(getattr(self, "_final_daemon_stats", {}))
        return super().read_daemon_stats()

    def _finish(self):
        if self._finished:
            return
        final = {"complete": False}
        try:
            if self._owner.handle:
                self._owner.synchronize()
                self._owner.check_health()
                slot = dict(self._engine_mod.cpu_client_slot_info(self._slot_handle))
                slot["error_code"] = self._owner.error_code()
                if slot["error_code"]:
                    raise RuntimeError(f"最终 CPU 请求失败：{slot['error_code']}")
                self._saved_final_slot = slot
                # 复用原统计字段转换；只在关闭时保存一次，不改变逐请求路径。
                self._final_daemon_stats = super().read_daemon_stats()
            final["slot"] = dict(self._saved_final_slot)
            try:
                self._owner.close()
            finally:
                # 原生视图一旦释放，所有后续诊断入口必须看不到旧地址。
                # 内存注销失败也执行；重试关闭时仍保留前面的最终快照。
                if not self._owner.handle:
                    self._slot_handle = 0
                    self._initialized = False
            final["complete"] = True
            self._finished = True
        except BaseException as exc:
            final["error"] = str(exc)
            raise
        finally:
            self._transport_metadata["final"] = final
            if self._metadata_path:
                self._metadata_path.write_text(
                    json.dumps(self._transport_metadata, ensure_ascii=False, indent=2)
                    + "\n"
                )
