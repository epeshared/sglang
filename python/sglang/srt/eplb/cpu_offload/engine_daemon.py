"""Daemon backend for ``CpuOffloadEngine``.

Connects the SGLang-side engine to standalone ``cpu_expertd`` daemons via
the shared-memory slot protocol validated in Phase 3.1-3.3:

- ``init_runtime``: optionally spawn the daemons via
  ``launch_cpu_expertd.sh``, then attach + open one slot per local NUMA.
- ``register_layer``: instantiate a ``RemoteMOE`` per layer and upload
  weights into the per-layer weight-SHM file the daemon mmaps.
- ``launch_layer``: enqueue ``RemoteMOE.forward_experts_kernel`` on the
  caller's CUDA stream (D2D memcpy + submit + wait + memcpy-out, all on
  stream; no host sync).
- ``commit_layer``: no-op for the kernel path (output memcpy is already
  enqueued by ``launch_layer``); kept on the interface for parity with the
  legacy backend.

The daemon backend is the default for fresh deployments and is the only
path that talks to the standalone ``cpu_expertd`` worker process. The
legacy in-process ``numa_initialize`` path stays in ``engine.py`` behind
``cpu_offload_backend="legacy"`` for regression debugging.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from sglang.srt.eplb.cpu_offload.engine import LayerSpec, LayerWeights

from sglang.srt.eplb.cpu_offload import trace as _trace
from sglang.srt.eplb.cpu_offload.config import CpuOffloadConfig
from sglang.srt.eplb.cpu_offload.metrics import CpuOffloadMetrics

logger = logging.getLogger(__name__)

# Resolved once, at import -- see trace.py's "COST WHEN DISABLED".
_MOE_TRACE = _trace.ENABLED
_STEP_TRACE_ENABLED = os.environ.get("SGLANG_MOE_STEP_TRACE", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


# Default location inside the repo. Used when ``cfg.cpu_offload_daemon_launcher``
# is empty AND the caller asked us to spawn the daemon.
_DEFAULT_LAUNCHER = "/workspace/moe-cpu-engine/launch_cpu_expertd.sh"


def _import_remote_moe():
    """Lazy import of ``RemoteMOE`` (lives in moe-cpu-engine/ext)."""
    # Try the installed location first (preferred), then fall back to the
    # in-tree source path so dev environments work without a reinstall.
    try:
        from cpu_offload_remote_moe import RemoteMOE  # type: ignore

        return RemoteMOE
    except ImportError:
        pass
    src_dir = "/workspace/moe-cpu-engine/ext"
    if src_dir not in sys.path and os.path.isdir(src_dir):
        sys.path.insert(0, src_dir)
    from cpu_offload_remote_moe import RemoteMOE  # type: ignore

    return RemoteMOE


def _import_engine_module():
    """Lazy import of the ``moe_cpu_engine`` C extension."""
    import moe_cpu_engine  # type: ignore

    return moe_cpu_engine


class DaemonBackend:
    """Attaches a single GPU rank to one cpu_expertd slot per local NUMA.

    A single instance is owned by one ``CpuOffloadEngine`` (one per GPU
    rank / process). Slot lifecycle:

        backend.init_runtime(num_numa_nodes=N)
            -> optionally spawn cpu_expertd; attach to local NUMA's daemon;
               open one slot (slot_id = position of this rank within
               ``config.active_ranks_on_numa(local_numa)``).
        backend.register_layer(spec, weights)
            -> build RemoteMOE on the local slot; SHM weight upload.
        backend.launch_layer(layer_id, dev_*, n_tokens, cuda_stream)
            -> RemoteMOE.forward_experts_kernel on the caller's stream.
        backend.commit_layer(...)  -> no-op (kernel path is self-contained).

    The per-rank slot is fixed for the process lifetime. The fast-loop
    selection of which logical expert is currently "on CPU" is handled
    GPU-side in ``dispatch.split_topk_for_cpu``; the backend just needs
    the full-layer weight set loaded so any expert id can route through.
    """

    # Process-level dedup so multi-rank-in-one-process tests don't launch
    # the daemon mesh repeatedly. Maps shm_prefix -> True once launched.
    _launched_prefixes: dict[str, bool] = {}
    _launch_lock = threading.Lock()

    def __init__(
        self,
        config: CpuOffloadConfig,
        *,
        host_rank: int,
        world_size: int,
        num_numa_nodes: int,
        metrics: Optional[CpuOffloadMetrics] = None,
    ) -> None:
        self._config = config
        self._host_rank = int(host_rank)
        self._world_size = int(world_size)
        self._num_numa_nodes = max(1, int(num_numa_nodes))
        self._metrics = metrics or CpuOffloadMetrics()

        self._engine_mod = None
        self._remote_moe_cls = None
        self._slot_handle: int = 0
        self._socket_id: int = -1
        self._slot_id: int = -1
        # layer_id -> RemoteMOE instance. Each RemoteMOE owns its weight
        # SHM file mmap on the daemon side.
        self._layers: dict[int, object] = {}
        self._layer_specs: dict[int, LayerSpec] = {}
        # Keep tensors alive while daemon holds their data pointers via
        # the weight SHM mapping. RemoteMOE.set_weights copies into SHM,
        # so this isn't strictly required for correctness but matches the
        # legacy backend's behavior and prevents accidental UAF if the
        # caller passes views.
        self._weight_retain: dict[int, LayerWeights] = {}
        self._initialized = False
        self._init_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_runtime(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            cfg = self._config
            self._engine_mod = _import_engine_module()
            self._remote_moe_cls = _import_remote_moe()

            # Decide which NUMA this rank belongs to.
            local_numa = cfg.numa_for_rank(
                self._host_rank,
                world_size=self._world_size,
                num_numa_nodes=self._num_numa_nodes,
            )
            # local_numa is an index in [0, num_numa_nodes).
            # Map it to the actual Linux NUMA node ID used by the daemon
            # (from CPU_NUMA_NODE_IDS), because the daemon's ready-file path
            # is .socket{actual_id}.ready, not .socket{index}.ready.
            # Example: CPU_NUMA_NODE_IDS="0,3" → index 1 → socket_id 3.
            _numa_ids_env = os.environ.get("CPU_NUMA_NODE_IDS", "").strip()
            if _numa_ids_env:
                _numa_ids = [
                    int(x.strip()) for x in _numa_ids_env.split(",") if x.strip()
                ]
                if 0 <= local_numa < len(_numa_ids):
                    self._socket_id = _numa_ids[int(local_numa)]
                else:
                    self._socket_id = int(local_numa)
            else:
                self._socket_id = int(local_numa)

            # Slot id within this NUMA = index of this rank among the
            # active ranks pinned to this NUMA.
            # With sub-NUMA reservation, the number of worker slots may be
            # less than the number of active ranks. We use the configured
            # assignment strategy. If no worker is available for this rank
            # (returns None), the rank will skip CPU offload.
            num_workers = cfg.cpu_offload_num_workers_per_numa
            assigned_slot = cfg.compute_worker_slot(
                self._host_rank,
                world_size=self._world_size,
                num_numa_nodes=self._num_numa_nodes,
                num_workers_per_numa=num_workers,
            )

            if assigned_slot is None:
                # No worker available for this rank - it will skip CPU offload
                logger.warning(
                    "[cpu_offload][daemon] rank %d has no available worker "
                    "on NUMA %d (workers=%d); rank will skip CPU offload.",
                    self._host_rank,
                    local_numa,
                    num_workers,
                )
                self._slot_id = -1  # Mark as no slot assigned
                self._initialized = True  # Allow register_layer to handle gracefully
                return

            self._slot_id = assigned_slot
            logger.info(
                "[cpu_offload][daemon] rank %d on NUMA %d -> slot %d "
                "(strategy=%s, workers=%d)",
                self._host_rank,
                local_numa,
                self._slot_id,
                cfg.cpu_offload_worker_assign_strategy,
                num_workers,
            )

            # If no worker slot was assigned (_slot_id == -1), skip daemon attach
            # This rank will operate in GPU-only mode
            if self._slot_id < 0:
                logger.warning(
                    "[cpu_offload][daemon] rank %d has no worker slot; "
                    "skipping CPU offload (GPU-only mode).",
                    self._host_rank,
                )
                self._initialized = True
                return

            shm_prefix = cfg.cpu_offload_shm_prefix

            # Optionally launch cpu_expertd. We only launch if a launcher
            # path is configured AND no daemon is already up (best-effort
            # detect via ready file existence).
            self._maybe_launch_daemon(shm_prefix)

            # Attach to the daemon and open our slot.
            timeout_ms = max(
                10_000,
                int(cfg.cpu_offload_timeout_us // 1000) or 30_000,
            )
            pid = self._engine_mod.cpu_client_attach(
                shm_prefix, self._socket_id, timeout_ms
            )
            if pid < 0:
                raise RuntimeError(
                    f"DaemonBackend: cpu_client_attach failed "
                    f"prefix={shm_prefix} socket={self._socket_id}"
                )
            handle = self._engine_mod.cpu_client_open_slot(
                shm_prefix, self._socket_id, self._slot_id, timeout_ms
            )
            if handle == 0:
                # attach_slot() refuses on a magic/version mismatch as well as
                # on a plain timeout, and returns the same 0 for both. The
                # SlotHeader is mmapped across two processes, so continuing
                # past a version skew would read one layout through the other's
                # field offsets -- garbage, silently. Name the mismatch instead.
                raise RuntimeError(self._explain_open_slot_failure(shm_prefix))
            self._slot_handle = int(handle)
            self._maybe_arm_trace()
            logger.info(
                "[cpu_offload][daemon] attached rank=%d numa=%d slot=%d pid=%d",
                self._host_rank,
                self._socket_id,
                self._slot_id,
                pid,
            )
            self._initialized = True

    def _explain_open_slot_failure(self, shm_prefix: str) -> str:
        """Turn a bare open_slot failure into a message that names the cause.

        The one cause that MUST NOT be reported vaguely is an ABI skew: the
        SlotHeader is a shared mmap, so a client built against one
        ``SLOT_VERSION`` reading a file stamped with another does not fail
        loudly on its own -- it reads whatever bytes happen to sit at the old
        offsets. attach_slot() already refuses in that case; this turns the
        refusal into an actionable sentence naming both versions.
        """
        base = (
            f"DaemonBackend: cpu_client_open_slot failed "
            f"prefix={shm_prefix} socket={self._socket_id} slot={self._slot_id}"
        )
        try:
            probe = self._engine_mod.cpu_client_probe_slot_version(
                shm_prefix, self._socket_id, self._slot_id
            )
        except Exception:
            return base + " (slot version could not be probed)"
        if not probe.get("exists"):
            return base + f" -- slot file {probe.get('path')} does not exist yet"
        client_ver = None
        try:
            client_ver = int(self._engine_mod.cpu_client_slot_abi_version())
        except Exception:
            pass
        file_ver = int(probe.get("version", 0))
        magic = int(probe.get("magic", 0))
        if client_ver is not None and file_ver != client_ver:
            msg = (
                f"{base}\n"
                f"  *** SLOT ABI VERSION MISMATCH ***\n"
                f"  slot file {probe.get('path')} was created by a daemon "
                f"built for SLOT_VERSION={file_ver},\n"
                f"  but this client (moe_cpu_engine) was built for "
                f"SLOT_VERSION={client_ver}.\n"
                f"  The SlotHeader is mmapped by both processes; reading one "
                f"layout through the other's\n"
                f"  field offsets returns garbage, so the attach was refused. "
                f"Rebuild BOTH sides from the\n"
                f"  same moe-cpu-engine checkout (ext/setup.py install) and "
                f"restart cpu_expertd."
            )
            # Also go straight to stderr: TP worker logging is not reliably
            # visible in this deployment, and this must not be missed.
            print(msg, file=sys.stderr, flush=True)
            return msg
        return base + (
            f" (slot file magic=0x{magic:08x} version={file_ver}, "
            f"client version={client_ver}; not a version mismatch -- "
            f"most likely the daemon never marked the slot READY)"
        )

    def _maybe_arm_trace(self) -> None:
        """分别打开旧区间汇总和新的逐任务 ring，两个开关互不串线。"""
        if self._slot_handle == 0:
            return
        if _MOE_TRACE:
            fn = getattr(self._engine_mod, "cpu_client_slot_set_trace", None)
            if fn is None:
                logger.warning(
                    "[cpu_offload][daemon] SGLANG_MOE_TRACE=1 but this "
                    "moe_cpu_engine build has no cpu_client_slot_set_trace; "
                    "per-layer daemon stats will be unavailable"
                )
            else:
                try:
                    fn(self._slot_handle, True)
                    logger.info(
                        "[cpu_offload][daemon] per-layer trace armed on socket=%d slot=%d",
                        self._socket_id,
                        self._slot_id,
                    )
                except Exception as exc:
                    _trace.exc("engine_daemon.init_runtime/arm-trace", exc)
        step_fn = getattr(self._engine_mod, "cpu_client_slot_set_step_trace", None)
        if step_fn is None:
            if _STEP_TRACE_ENABLED:
                logger.warning(
                    "[moe_step_trace] 当前 moe_cpu_engine 没有逐任务 ring 开关；"
                    "请同时重编客户端和 daemon"
                )
        else:
            try:
                step_fn(self._slot_handle, _STEP_TRACE_ENABLED)
                if _STEP_TRACE_ENABLED:
                    logger.info(
                        "[moe_step_trace] task ring armed on socket=%d slot=%d",
                        self._socket_id,
                        self._slot_id,
                    )
            except Exception as exc:
                action = "开启" if _STEP_TRACE_ENABLED else "关闭"
                logger.warning("[moe_step_trace] task ring %s失败：%s", action, exc)

    def read_task_trace_snapshot(self, after_task_id: int = 0) -> dict:
        """读取 v14 有界 ring 中 ``after_task_id`` 之后完成的真实任务。

        读取不清空 ring。调用方保存 ``write_task_id`` 作为下次游标；若消费
        太慢，native 层通过 ``lost_records`` 明确报告缺口。
        """
        if not self._initialized or self._slot_handle == 0:
            return {
                "valid": False,
                "supported": False,
                "enabled": False,
                "write_task_id": int(after_task_id),
                "records": [],
                "reason": "daemon_slot_unavailable",
                "socket_id": self._socket_id,
                "slot_id": self._slot_id,
            }
        fn = getattr(self._engine_mod, "cpu_client_slot_trace_snapshot", None)
        if fn is None:
            return {
                "valid": False,
                "supported": False,
                "enabled": False,
                "write_task_id": int(after_task_id),
                "records": [],
                "reason": "moe_cpu_engine_build_predates_slot_v14",
                "socket_id": self._socket_id,
                "slot_id": self._slot_id,
            }
        try:
            snapshot = dict(fn(self._slot_handle, int(after_task_id)) or {})
        except Exception as exc:
            return {
                "valid": False,
                "supported": False,
                "enabled": False,
                "write_task_id": int(after_task_id),
                "records": [],
                "reason": f"snapshot_error:{type(exc).__name__}",
                "socket_id": self._socket_id,
                "slot_id": self._slot_id,
            }
        snapshot.setdefault("records", [])
        snapshot["socket_id"] = self._socket_id
        snapshot["slot_id"] = self._slot_id
        for record in snapshot["records"]:
            record.setdefault("socket_id", self._socket_id)
            record.setdefault("slot_id", self._slot_id)
        return snapshot

    def read_layer_stats(self) -> dict:
        """Per-layer daemon counters (v13+). ``{}`` when unavailable.

        Unlike ``read_daemon_stats`` these are split by ``layer_id``, which is
        the granularity Python cannot reconstruct on its own: under CUDA graph
        both the submit and the wait live inside recorded kernels, so no
        individual request is ever observed from Python.
        """
        if not self._initialized or self._slot_handle == 0:
            return {}
        fn = getattr(self._engine_mod, "cpu_client_slot_layer_stats", None)
        if fn is None:
            return {
                "valid": False,
                "supported": False,
                "reason": "moe_cpu_engine build predates slot v13",
            }
        try:
            info = fn(self._slot_handle)
        except Exception:
            return {}
        if not info.get("valid", False):
            return {}
        return {
            "version": int(info.get("version", 0)),
            "supported": bool(info.get("supported", False)),
            "trace_enabled": int(info.get("trace_enabled", 0)),
            "overflow": int(info.get("overflow", 0)),
            "max_layers": int(info.get("max_layers", 0)),
            "layers": {
                int(k): {kk: int(vv) for kk, vv in v.items()}
                for k, v in (info.get("layers", {}) or {}).items()
            },
        }

    def _maybe_launch_daemon(self, shm_prefix: str) -> None:
        cfg = self._config
        launcher = cfg.cpu_offload_daemon_launcher or ""
        if not launcher:
            # User opted into attach-only; assume the operator started
            # cpu_expertd separately. We'll just block in attach.
            return
        with DaemonBackend._launch_lock:
            if DaemonBackend._launched_prefixes.get(shm_prefix):
                return
            ready_file = f"{shm_prefix}.socket{self._socket_id}.ready"
            if os.path.exists(ready_file):
                # Someone else already launched it.
                DaemonBackend._launched_prefixes[shm_prefix] = True
                return
            launcher_path = (
                launcher if launcher and launcher != "auto" else _DEFAULT_LAUNCHER
            )
            if not os.path.isfile(launcher_path):
                raise RuntimeError(
                    f"DaemonBackend: launcher script not found at {launcher_path}"
                )
            num_sockets = max(1, self._num_numa_nodes)
            env = dict(os.environ)
            env.setdefault("CPU_OFFLOAD_SHM_PREFIX", shm_prefix)
            if cfg.cpu_offload_num_workers_per_numa >= 0:
                env.setdefault(
                    "CPU_OFFLOAD_NUM_WORKERS_PER_NUMA",
                    str(cfg.cpu_offload_num_workers_per_numa),
                )
            env.setdefault(
                "CPU_OFFLOAD_RESERVED_CORES_PER_NUMA",
                str(cfg.cpu_offload_reserved_cores_per_numa),
            )
            logger.info(
                "[cpu_offload][daemon] launching %d cpu_expertd socket(s) via %s",
                num_sockets,
                launcher_path,
            )
            t0 = time.time()
            result = subprocess.run(
                [launcher_path, str(num_sockets)],
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode != 0:
                logger.error(
                    "[cpu_offload][daemon] launcher stdout:\n%s\nstderr:\n%s",
                    result.stdout,
                    result.stderr,
                )
                raise RuntimeError(
                    f"DaemonBackend: launcher exited {result.returncode}"
                )
            logger.info(
                "[cpu_offload][daemon] launcher ready in %.1fs", time.time() - t0
            )
            DaemonBackend._launched_prefixes[shm_prefix] = True

    # ------------------------------------------------------------------
    # Layer registration
    # ------------------------------------------------------------------

    def register_layer(self, spec: LayerSpec, weights: LayerWeights) -> None:
        if not self._initialized:
            raise RuntimeError(
                "DaemonBackend: init_runtime() must be called before register_layer()"
            )
        if self._slot_id < 0:
            # No worker slot assigned for this rank (e.g. custom strategy with
            # rank absent from cpu_offload_rank_to_worker_map); skip silently.
            return
        if spec.layer_id in self._layers:
            raise RuntimeError(
                f"DaemonBackend: layer {spec.layer_id} already registered"
            )

        from sglang.srt.eplb.cpu_offload.engine import (
            daemon_loads_weights_from_disk,
        )

        daemon_loads = daemon_loads_weights_from_disk()

        if not daemon_loads:
            # All weight tensors must be CPU + contiguous (RemoteMOE memcpy's
            # raw bytes into the weight-SHM file).
            for name, t in (
                ("gate_up_weights", weights.gate_up_weights),
                ("down_weights", weights.down_weights),
                ("gate_up_scales", weights.gate_up_scales),
                ("down_scales", weights.down_scales),
            ):
                if t is None or t.device.type != "cpu" or not t.is_contiguous():
                    raise ValueError(
                        f"DaemonBackend: layer {spec.layer_id} weight {name!r} "
                        f"must be CPU + contiguous"
                    )

        rm = self._remote_moe_cls(
            slot_handle=self._slot_handle,
            layer_id=spec.layer_id,
            expert_num=spec.num_experts,
            num_experts_per_tok=spec.num_experts_per_tok,
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            norm_topk_prob=spec.norm_topk_prob,
            n_group=spec.n_group,
            topk_group=spec.topk_group,
            shm_prefix=self._config.cpu_offload_shm_prefix,
        )

        if not daemon_loads:
            bias_ptr = (
                weights.e_score_correction_bias.data_ptr()
                if weights.e_score_correction_bias is not None
                else 0
            )
            rm.set_weights(
                weights.gate_up_weights.data_ptr(),
                weights.down_weights.data_ptr(),
                weights.gate_up_scales.data_ptr(),
                weights.down_scales.data_ptr(),
                bias_ptr,
                1,  # tp_size: per-rank slot is single-tp from daemon's view
                0,  # tp_rank
            )
        else:
            # Phase-2 mode: daemon owns the weight bytes via its own
            # safetensors mmap. Client never writes the .weights SHM file,
            # but we STILL must call set_weights so that RemoteMOE:
            #   (a) sends the LOAD_WEIGHTS opcode to inform the daemon of
            #       this layer's dims (E/K/H/I/n_group/topk_group/tp),
            #       which it uses to build the per-layer MOE instance
            #       against its preloaded per-socket weight file;
            #   (b) flips its internal ``_weights_loaded`` flag so the
            #       hot-path ``forward_experts_kernel`` does not bail
            #       with "weights not loaded".
            # In Phase-2 the function skips the .weights SHM write itself
            # (see env-gate inside RemoteMOE.set_weights). Pointer args
            # are unused on that path; pass zeros.
            rm.set_weights(0, 0, 0, 0, 0, 1, 0)

        # Enable kernel path once per layer (idempotent inside RemoteMOE).
        rm.enable_kernel_path()

        self._layers[spec.layer_id] = rm
        self._layer_specs[spec.layer_id] = spec
        self._weight_retain[spec.layer_id] = weights
        logger.debug(
            "[cpu_offload][daemon] registered layer=%d E=%d K=%d H=%d I=%d",
            spec.layer_id,
            spec.num_experts,
            spec.num_experts_per_tok,
            spec.hidden_size,
            spec.intermediate_size,
        )

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def launch_layer(
        self,
        layer_id: int,
        *,
        dev_input: torch.Tensor,
        dev_topk_ids: torch.Tensor,
        dev_topk_weights: torch.Tensor,
        dev_output: torch.Tensor,
        n_tokens: int,
        cuda_stream: int,
    ) -> None:
        """Enqueue the full kernel-path forward on ``cuda_stream``.

        Unlike the legacy backend, the daemon backend bundles input-copy,
        submit, wait, and output-copy into a single ``forward_experts_kernel``
        call — there is no separate ``commit_layer`` step.
        """
        if not self._initialized:
            raise RuntimeError("DaemonBackend: init_runtime() not called")
        if n_tokens <= 0:
            return
        rm = self._layers.get(layer_id)
        if rm is None:
            raise KeyError(f"DaemonBackend: layer {layer_id} not registered")
        rm.forward_experts_kernel(
            int(cuda_stream),
            int(dev_input.data_ptr()),
            int(dev_topk_ids.data_ptr()),
            int(dev_topk_weights.data_ptr()),
            int(dev_output.data_ptr()),
            int(n_tokens),
            max_spins=int(self._config.cpu_offload_max_spin),
        )
        # PCIe accounting: 4 transfers (in/ids/weights/out). Sizes
        # mirror the kernel-path ``forward_experts_kernel`` calc.
        spec = self._layer_specs[layer_id]
        in_bytes = n_tokens * spec.hidden_size * 2
        ids_bytes = n_tokens * spec.num_experts_per_tok * 4
        w_bytes = n_tokens * spec.num_experts_per_tok * 4
        out_bytes = n_tokens * spec.hidden_size * 2
        self._metrics.cpu_offload_pcie_bytes_per_step += (
            in_bytes + ids_bytes + w_bytes + out_bytes
        )

    def commit_layer(
        self,
        layer_id: int,
        *,
        dev_output: torch.Tensor,
        n_tokens: int,
        cuda_stream: int,
    ) -> None:
        # Kernel path is self-contained: forward_experts_kernel already
        # bundled submit + wait + output memcpy on the caller's stream,
        # so there is nothing to enqueue here.
        #
        # However, when CPU_OFFLOAD_VERIFY_COMMIT=1 we treat this as the
        # natural place to confirm the daemon really serviced the last
        # request. RemoteMOE.forward_experts_kernel piggy-backs a D2H
        # copy of the wait-kernel status word and the slot-header
        # error_code; here we synchronously read them and raise on any
        # non-zero, which causes commit_cpu_offload's existing handler
        # to record a combine_fail (so the metric stops being a
        # tautology). Default off for zero perf overhead in production.
        if n_tokens <= 0:
            return
        flag = os.environ.get("CPU_OFFLOAD_VERIFY_COMMIT", "").strip().lower()
        if flag not in ("1", "true", "yes", "on"):
            return
        rm = self._layers.get(layer_id)
        if rm is None:
            return
        try:
            timeout_flag, error_code = rm.read_last_status_blocking()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"DaemonBackend.commit_layer: status probe failed "
                f"layer={layer_id}: {exc}"
            ) from exc
        if timeout_flag != 0 or error_code != 0:
            raise RuntimeError(
                f"DaemonBackend.commit_layer: daemon did not produce a "
                f"valid result layer={layer_id} "
                f"wait_timeout={timeout_flag} error_code={error_code} "
                f"n_tokens={n_tokens}"
            )

    # ------------------------------------------------------------------
    # Split launch/commit for CPU-GPU parallelism (Phase 3.3b)
    # ------------------------------------------------------------------

    def launch_submit(
        self,
        layer_id: int,
        *,
        dev_input: torch.Tensor,
        dev_topk_ids: torch.Tensor,
        dev_topk_weights: torch.Tensor,
        n_tokens: int,
        cuda_stream: int,
        trace_events: Optional[dict] = None,
    ) -> None:
        """Phase 1: D2H + DMA fence + submit.  No host sync.

        Enqueues input copy + DMA fence + submit on ``cuda_stream``.
        After this returns the CPU daemon begins computing.  The caller
        should enqueue GPU MOE work on the same stream, then call
        :meth:`launch_wait_and_copyout`.
        """
        if not self._initialized:
            raise RuntimeError("DaemonBackend: init_runtime() not called")
        if n_tokens <= 0:
            return
        rm = self._layers.get(layer_id)
        if rm is None:
            raise KeyError(f"DaemonBackend: layer {layer_id} not registered")
        args = (
            int(cuda_stream),
            int(dev_input.data_ptr()),
            int(dev_topk_ids.data_ptr()),
            int(dev_topk_weights.data_ptr()),
            int(n_tokens),
        )
        if trace_events is None:
            rm.launch_submit(*args)
        else:
            rm.launch_submit(*args, trace_events=trace_events)
        # PCIe accounting: D2H transfers only (input, ids, weights).
        spec = self._layer_specs[layer_id]
        in_bytes = n_tokens * spec.hidden_size * 2
        ids_bytes = n_tokens * spec.num_experts_per_tok * 4
        w_bytes = n_tokens * spec.num_experts_per_tok * 4
        self._metrics.cpu_offload_pcie_bytes_per_step += in_bytes + ids_bytes + w_bytes

    def launch_wait_and_copyout(
        self,
        layer_id: int,
        *,
        dev_output: torch.Tensor,
        n_tokens: int,
        cuda_stream: int,
        trace_events: Optional[dict] = None,
    ) -> None:
        """Phase 3: Wait + H2D copy-out.  No host sync.

        Must be called after :meth:`launch_submit` and after GPU MOE
        work has been enqueued on the same stream.
        """
        if not self._initialized:
            raise RuntimeError("DaemonBackend: init_runtime() not called")
        if n_tokens <= 0:
            return
        rm = self._layers.get(layer_id)
        if rm is None:
            raise KeyError(f"DaemonBackend: layer {layer_id} not registered")
        kwargs = {"max_spins": int(self._config.cpu_offload_max_spin)}
        if trace_events is not None:
            kwargs["trace_events"] = trace_events
        rm.launch_wait_and_copyout(
            int(cuda_stream),
            int(dev_output.data_ptr()),
            int(n_tokens),
            **kwargs,
        )
        # PCIe accounting: H2D transfer (output).
        spec = self._layer_specs[layer_id]
        out_bytes = n_tokens * spec.hidden_size * 2
        self._metrics.cpu_offload_pcie_bytes_per_step += out_bytes

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def has_layer(self, layer_id: int) -> bool:
        return layer_id in self._layers

    def registered_layers(self) -> list[int]:
        return sorted(self._layers)

    def read_daemon_stats(self) -> dict:
        """Read live daemon-side FORWARD_EXPERTS statistics from SHM.

        These reflect REAL activity (including all CUDA graph replays),
        unlike the Python-side ``CpuOffloadMetrics`` which only count
        during graph capture.  Returns a dict with keys:

          - ``fwd_count``: total FORWARD_EXPERTS requests processed
          - ``fwd_tokens``: total tokens across all forward requests
          - ``compute_ns_sum``: total per-request handling time in
            nanoseconds (param decode + forwardExperts() + CLWB output
            flush + stat bookkeeping -- NOT pure GEMM time)
          - ``gemm_ns_sum``: total time spent ONLY inside
            forwardExperts() itself (v9+ daemon only; absent on older
            daemons). Use this instead of compute_ns_sum when comparing
            against a dummy warm-up call's timing.
          - ``prefetch_exec_count``: total next-layer prefetches that
            actually touched weight bytes (v5+ daemon only; 0/absent on
            older daemons or if prefetch is disabled/always deduped so
            far). This is the ground-truth signal that the "prewarm next
            layer's expert" feature is really doing work, as opposed to
            being silently skipped (e.g. XFT backend, or no daemon
            version support).
          - ``prefetch_bytes_sum``: cumulative weight bytes pulled into
            CPU cache by prefetch.
          - ``prefetch_ns_sum``: cumulative wall time spent prefetching.

        Returns an empty dict if the slot is not initialized or the
        daemon version does not support stats (version < 3).
        """
        if not self._initialized or self._slot_handle == 0:
            return {}
        try:
            info = self._engine_mod.cpu_client_slot_info(self._slot_handle)
            if not info.get("valid", False):
                return {}
            fwd_count = info.get("stat_fwd_count")
            if fwd_count is None:
                # Daemon version < 3 — stats not available.
                return {}
            result = {
                "fwd_count": int(fwd_count),
                "fwd_tokens": int(info.get("stat_fwd_tokens", 0)),
                "fwd_active_toks": int(info.get("stat_fwd_active_toks", 0)),
                # v4+: actual (token, expert) pair count routed to CPU.
                # Falls back to fwd_active_toks for v3 daemons (single-cpu-slot
                # assumption: one valid id per active token).
                "fwd_cpu_pairs": int(
                    info.get("stat_fwd_cpu_pairs", info.get("stat_fwd_active_toks", 0))
                ),
                "compute_ns_sum": int(info.get("stat_compute_ns_sum", 0)),
            }
            # v9+: pure forwardExperts() time (excludes param decode / CLWB
            # flush / bookkeeping that compute_ns_sum bundles in). Absent on
            # older daemon builds -- only add when the native module reports it.
            if "stat_gemm_ns_sum" in info:
                result["gemm_ns_sum"] = int(info.get("stat_gemm_ns_sum", 0))
            # v10+: pre-GEMM dispatch time and post-GEMM CLWB flush time --
            # the two real components of the compute_ns_sum - gemm_ns_sum
            # gap. Same absent-means-unsupported convention as gemm_ns_sum.
            if "stat_dispatch_ns_sum" in info:
                result["dispatch_ns_sum"] = int(info.get("stat_dispatch_ns_sum", 0))
                result["flush_ns_sum"] = int(info.get("stat_flush_ns_sum", 0))
            # v11+: the two remaining measured slices of the v10 gap --
            # post-GEMM diagnostic blocks and the post-flush stats-atomics
            # block. Same absent-means-unsupported convention.
            if "stat_diag_ns_sum" in info:
                result["diag_ns_sum"] = int(info.get("stat_diag_ns_sum", 0))
                result["stats_ns_sum"] = int(info.get("stat_stats_ns_sum", 0))
            # v12+: steady-state GEMM time. gemm_ns_sum above is a lifetime sum
            # whose mean keeps the startup transient and is skewed by rare stalls.
            if "stat_steady_count" in info:
                result["steady_ns_sum"] = int(info.get("stat_steady_ns_sum", 0))
                result["steady_count"] = int(info.get("stat_steady_count", 0))
                result["steady_max_ns"] = int(info.get("stat_steady_max_ns", 0))
                result["stall_count"] = int(info.get("stat_stall_count", 0))
                result["stall_ns_sum"] = int(info.get("stat_stall_ns_sum", 0))
            # v5+: next-layer prefetch execution stats. Only add these keys
            # when the loaded native module ACTUALLY reports them — do not
            # default to 0 here, since that would make "genuinely zero
            # executions" indistinguishable from "this build/daemon doesn't
            # support prefetch stats at all" (e.g. a stale moe_cpu_engine.so
            # that predates the feature). The caller (runtime.py METRICS
            # printer) uses key-presence to decide whether to print the
            # [PREFETCH] line at all.
            if "stat_prefetch_exec_count" in info:
                result["prefetch_exec_count"] = int(
                    info.get("stat_prefetch_exec_count", 0)
                )
                result["prefetch_bytes_sum"] = int(
                    info.get("stat_prefetch_bytes_sum", 0)
                )
                result["prefetch_ns_sum"] = int(info.get("stat_prefetch_ns_sum", 0))
            # v6+: prefetch pipeline diagnostics. Same absent-means-unsupported
            # convention as above -- only added when the native module (and
            # thus the running daemon) actually reports them.
            if "stat_prefetch_seen_count" in info:
                result["prefetch_seen_count"] = int(
                    info.get("stat_prefetch_seen_count", 0)
                )
                result["prefetch_attempt_count"] = int(
                    info.get("stat_prefetch_attempt_count", 0)
                )
                result["prefetch_miss_count"] = int(
                    info.get("stat_prefetch_miss_count", 0)
                )
            if "attention_window_ns" in info:
                result["attention_window_ns"] = int(info.get("attention_window_ns", 0))
            return result
        except Exception:
            return {}

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def slot_handle(self) -> int:
        return self._slot_handle

    @property
    def socket_id(self) -> int:
        return self._socket_id

    @property
    def slot_id(self) -> int:
        return self._slot_id
