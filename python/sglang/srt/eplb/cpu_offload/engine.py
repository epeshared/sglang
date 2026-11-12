"""SGLang-side wrapper around ``moe_cpu_engine``.

Responsibilities (see CPU COMMON EXECUTION PATH in design doc):
- Initialize the per-NUMA daemon mesh exactly once per process via
  ``moe_cpu_engine.numa_initialize``.
- Register one ``moe.MOE`` instance per MoE layer with the engine.
- Upload weights once at startup.
- Expose a single hot-path entry point ``forward_layer`` that runs the
  full GPU-stream-async pipeline with no host sync.

This wrapper requires ``moe_cpu_engine`` to be installed when CPU offload
is enabled (user choice 2). It is imported lazily so SGLang can still
boot without the package when CPU offload is off.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.eplb.cpu_offload.config import CpuOffloadConfig
from sglang.srt.eplb.cpu_offload.metrics import CpuOffloadMetrics

logger = logging.getLogger(__name__)


def _lazy_import_moe_cpu_engine():
    try:
        import moe_cpu_engine  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "moe_cpu_engine is required when CPU offload is enabled. "
            "Install from /workspace/moe-cpu-engine: "
            "`cd moe-cpu-engine/ext && python setup.py install`"
        ) from exc
    return moe_cpu_engine


def daemon_loads_weights_from_disk() -> bool:
    """Phase-2 toggle: when set, GPU ranks do NOT upload expert weights.

    Long-term design (see ``dual-loop-eplb-design.md`` and the repo memory
    note ``cpu_offload_notes.md``): each per-NUMA ``cpu_expertd`` daemon
    holds the FULL E=128 expert set in RAM up-front by mmaping safetensors
    itself. The GPU client only attaches and dispatches forward requests;
    it never writes the ``.weights`` SHM file. This env-var gates the
    transition so the legacy per-rank shard upload path remains available
    until the daemon-side loader (Phase-2 step B) is wired up.
    """
    return os.getenv("CPU_OFFLOAD_DAEMON_LOADS_WEIGHTS", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@dataclass
class LayerSpec:
    """Per-layer configuration the engine needs to build an MOE object."""

    layer_id: int
    num_experts: int
    num_experts_per_tok: int
    hidden_size: int
    intermediate_size: int
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    expert_id_offset: int = 0

    def cpu_expert_id(self, global_id: int) -> int:
        """将全局路由编号换成 CPU 权重下标；全量权重的偏移为零。"""
        if global_id < 0:
            return -1
        local_id = int(global_id) - self.expert_id_offset
        if not 0 <= local_id < self.num_experts:
            raise ValueError(
                f"CPU 专家编号越界：layer={self.layer_id}, global_id={global_id}, "
                f"offset={self.expert_id_offset}, count={self.num_experts}"
            )
        return local_id


@dataclass
class LayerWeights:
    """Raw pointer / tensor weights for one layer.

    All tensors must be on CPU and contiguous. The engine reads them via
    raw data pointers; passing GPU tensors here is a bug.

    When ``daemon_loads_weights_from_disk()`` is true (Phase-2), all four
    weight tensors may be ``None`` — the daemon will source weights from
    its own safetensors mmap and the client-side payload is just metadata.
    """

    gate_up_weights: Optional[torch.Tensor] = None
    down_weights: Optional[torch.Tensor] = None
    gate_up_scales: Optional[torch.Tensor] = None
    down_scales: Optional[torch.Tensor] = None
    e_score_correction_bias: Optional[torch.Tensor] = None


@dataclass
class _LayerIOBuffers:
    """Per-layer pinned host buffers + mapped device pointers."""

    capacity_tokens: int
    hidden_size: int
    topk: int

    host_input: torch.Tensor
    host_topk_ids: torch.Tensor
    host_topk_weights: torch.Tensor
    host_output: torch.Tensor

    # Mapped device pointers (int) into the same pinned host pages.
    dev_input_ptr: int
    dev_topk_ids_ptr: int
    dev_topk_weights_ptr: int
    dev_output_ptr: int


# CUDA host-register flags: Portable | Mapped (see cuda_runtime.h).
_CUDA_HOST_REGISTER_PORTABLE = 0x01
_CUDA_HOST_REGISTER_MAPPED = 0x02


def _cuda_host_register_mapped(tensor: torch.Tensor) -> int:
    """Pin ``tensor``'s storage and return its mapped device pointer.

    The caller owns the pinning and must keep the tensor alive. We
    intentionally do NOT call ``cudaHostUnregister`` automatically;
    process exit cleans up.
    """
    import ctypes

    assert tensor.is_contiguous(), "pinned tensors must be contiguous"
    libcudart = ctypes.CDLL("libcudart.so")

    flags = _CUDA_HOST_REGISTER_PORTABLE | _CUDA_HOST_REGISTER_MAPPED
    n_bytes = tensor.numel() * tensor.element_size()
    status = libcudart.cudaHostRegister(
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_size_t(n_bytes),
        ctypes.c_uint(flags),
    )
    if status != 0:
        raise RuntimeError(f"cudaHostRegister failed: status={status}")

    cudaHostGetDevicePointer = libcudart.cudaHostGetDevicePointer
    cudaHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    cudaHostGetDevicePointer.restype = ctypes.c_int
    dev_ptr = ctypes.c_void_p()
    status = cudaHostGetDevicePointer(
        ctypes.byref(dev_ptr), ctypes.c_void_p(tensor.data_ptr()), ctypes.c_uint(0)
    )
    if status != 0:
        raise RuntimeError(f"cudaHostGetDevicePointer failed: status={status}")
    return int(dev_ptr.value or 0)


class CpuOffloadEngine:
    """Process-singleton wrapper around moe_cpu_engine for this GPU rank.

    Lifecycle:

        engine = CpuOffloadEngine(config, host_rank=tp_rank, num_hosts=...)
        engine.init_runtime(num_numa_nodes=...)
        for spec in layer_specs:
            engine.register_layer(spec, weights)
        engine.warmup(...)
        # hot path:
        engine.forward_layer(layer_id, dev_input, dev_topk_ids,
                              dev_topk_weights, dev_output, n_tokens, stream)

    The constructor only stores config; the heavy lifting happens in
    ``init_runtime`` and ``register_layer`` so callers can pre-check the
    feature flag.
    """

    _instance: Optional[CpuOffloadEngine] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        config: CpuOffloadConfig,
        *,
        host_rank: int,
        num_hosts: int,
        ep_size: int = 1,
        num_numa_nodes: int = 1,
        metrics: Optional[CpuOffloadMetrics] = None,
    ) -> None:
        self._config = config
        self._host_rank = int(host_rank)
        self._num_hosts = int(num_hosts)
        self._ep_size = int(ep_size)
        self._num_numa_nodes = max(1, int(num_numa_nodes))
        self._metrics = metrics or CpuOffloadMetrics()

        self._engine_mod = None
        self._cpu_infer = None
        self._layers: dict[int, object] = {}
        self._layer_specs: dict[int, LayerSpec] = {}
        self._io_buffers: dict[int, _LayerIOBuffers] = {}
        self._initialized = False

        # Optional daemon backend; built lazily on init_runtime() when the
        # config selects "daemon". When set, ALL lifecycle and hot-path
        # methods on this class delegate to it; the legacy in-process
        # numa_initialize path (rest of this file) stays dormant.
        self._daemon = None  # type: Optional[object]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _propagate_env(self) -> None:
        """Forward Python-side knobs into env vars read by the C++ engine."""
        cfg = self._config
        # The engine reads NUM_RESERVED_CORES at NumaLauncher construction.
        os.environ.setdefault(
            "NUM_RESERVED_CORES", str(cfg.cpu_offload_reserved_cores_per_numa)
        )
        # ENABLE_NUMA must be set so the engine actually forks per-NUMA daemons.
        os.environ.setdefault("ENABLE_NUMA", "1")

    def init_runtime(self) -> None:
        """Boot the per-NUMA daemon mesh exactly once."""
        if self._initialized:
            return
        from sglang.srt.eplb.cpu_offload.engine_transport import (
            make_backend,
            transport_name,
        )

        transport_name(self._config)
        # Daemon backend: delegate to standalone cpu_expertd attach path.
        if getattr(self._config, "cpu_offload_backend", "daemon") == "daemon":
            self._daemon = make_backend(
                self._config,
                host_rank=self._host_rank,
                world_size=self._num_hosts,
                num_numa_nodes=self._num_numa_nodes,
                metrics=self._metrics,
            )
            self._daemon.init_runtime()
            self._initialized = True
            CpuOffloadEngine._instance = self
            return
        # Legacy in-process backend (regression / debugging only).
        with self._instance_lock:
            if self._initialized:
                return
            self._propagate_env()
            self._engine_mod = _lazy_import_moe_cpu_engine()
            logger.info(
                "[cpu_offload] numa_initialize(num_hosts=%d, host_rank=%d, ep_size=%d)",
                self._num_hosts,
                self._host_rank,
                self._ep_size,
            )
            self._engine_mod.numa_initialize(
                self._num_hosts, self._host_rank, self._ep_size
            )
            # CPUInfer thread_num is unused by the engine (legacy), pass 0.
            self._cpu_infer = self._engine_mod.CPUInfer(0)
            self._initialized = True
            CpuOffloadEngine._instance = self

    # ------------------------------------------------------------------
    # Per-layer registration
    # ------------------------------------------------------------------

    def register_layer(self, spec: LayerSpec, weights: LayerWeights) -> None:
        """Build a ``moe.MOE`` for this layer and upload its weights once."""
        if self._daemon is not None:
            self._daemon.register_layer(spec, weights)
            self._layer_specs[spec.layer_id] = spec
            return
        if not self._initialized:
            raise RuntimeError("init_runtime() must be called before register_layer()")
        mod = self._engine_mod
        cfg = mod.moe.MOEConfig(
            spec.layer_id,
            spec.num_experts,
            spec.num_experts_per_tok,
            spec.hidden_size,
            spec.intermediate_size,
            spec.norm_topk_prob,
            spec.n_group,
            spec.topk_group,
        )
        moe = mod.moe.MOE(cfg)
        # All weight tensors must be CPU + contiguous.
        for name, t in (
            ("gate_up_weights", weights.gate_up_weights),
            ("down_weights", weights.down_weights),
            ("gate_up_scales", weights.gate_up_scales),
            ("down_scales", weights.down_scales),
        ):
            if not t.is_contiguous() or t.device.type != "cpu":
                raise ValueError(
                    f"layer {spec.layer_id} weight '{name}' must be CPU + contiguous"
                )
        bias_ptr = (
            weights.e_score_correction_bias.data_ptr()
            if weights.e_score_correction_bias is not None
            else 0
        )
        moe.set_weights(
            weights.gate_up_weights.data_ptr(),
            weights.down_weights.data_ptr(),
            weights.gate_up_scales.data_ptr(),
            weights.down_scales.data_ptr(),
            bias_ptr,
        )
        self._layers[spec.layer_id] = moe
        self._layer_specs[spec.layer_id] = spec

    # ------------------------------------------------------------------
    # Pinned IO buffers (one per layer; reused every step)
    # ------------------------------------------------------------------

    def allocate_io_buffers(
        self, layer_id: int, *, capacity_tokens: int
    ) -> _LayerIOBuffers:
        if layer_id in self._io_buffers:
            existing = self._io_buffers[layer_id]
            if existing.capacity_tokens >= capacity_tokens:
                return existing
            # Need a bigger buffer; drop the old one and rebuild.
            del self._io_buffers[layer_id]

        spec = self._layer_specs[layer_id]
        hsize = spec.hidden_size
        topk = spec.num_experts_per_tok

        host_input = torch.zeros(
            (capacity_tokens, hsize), dtype=torch.bfloat16, device="cpu"
        )
        host_topk_ids = torch.zeros(
            (capacity_tokens, topk), dtype=torch.int32, device="cpu"
        )
        host_topk_weights = torch.zeros(
            (capacity_tokens, topk), dtype=torch.float32, device="cpu"
        )
        host_output = torch.zeros(
            (capacity_tokens, hsize), dtype=torch.bfloat16, device="cpu"
        )

        dev_input_ptr = _cuda_host_register_mapped(host_input)
        dev_topk_ids_ptr = _cuda_host_register_mapped(host_topk_ids)
        dev_topk_weights_ptr = _cuda_host_register_mapped(host_topk_weights)
        dev_output_ptr = _cuda_host_register_mapped(host_output)

        buf = _LayerIOBuffers(
            capacity_tokens=capacity_tokens,
            hidden_size=hsize,
            topk=topk,
            host_input=host_input,
            host_topk_ids=host_topk_ids,
            host_topk_weights=host_topk_weights,
            host_output=host_output,
            dev_input_ptr=dev_input_ptr,
            dev_topk_ids_ptr=dev_topk_ids_ptr,
            dev_topk_weights_ptr=dev_topk_weights_ptr,
            dev_output_ptr=dev_output_ptr,
        )
        self._io_buffers[layer_id] = buf
        return buf

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def forward_layer(
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
        """Run one MoE expert call on the CPU engine, asynchronously on ``cuda_stream``.

        ALL operations are enqueued on ``cuda_stream``. No host sync. The
        order is:

            1. copy_into_host_async  : GPU -> mapped pinned host
            2. submit_with_cuda_stream: wake CPU worker (via mapped flag)
            3. sync_with_cuda_stream : GPU polling kernel waits for flag
            4. combine_with_device_async: pinned host -> GPU + add

        Steps 1-2 are concurrent with whatever else is on the stream
        before step 3. Step 3 only stalls if the CPU was slower than the
        intervening GPU work; the demote-loop in fast_loop disables
        chronically-slow slots so this stall stays bounded.
        """
        self.launch_layer(
            layer_id,
            dev_input=dev_input,
            dev_topk_ids=dev_topk_ids,
            dev_topk_weights=dev_topk_weights,
            n_tokens=n_tokens,
            cuda_stream=cuda_stream,
        )
        self.commit_layer(
            layer_id,
            dev_output=dev_output,
            n_tokens=n_tokens,
            cuda_stream=cuda_stream,
        )

    def launch_layer(
        self,
        layer_id: int,
        *,
        dev_input: torch.Tensor,
        dev_topk_ids: torch.Tensor,
        dev_topk_weights: torch.Tensor,
        n_tokens: int,
        cuda_stream: int,
        dev_output: Optional[torch.Tensor] = None,
    ) -> None:
        """First half of the pipeline: copy-in + submit.

        Returns immediately after enqueuing both ops on ``cuda_stream``;
        the GPU work that follows on the same stream runs concurrently
        with the CPU expert exec (this is the OVERLAP requirement).

        ``dev_output`` is required for the daemon backend (kernel path is
        single-shot) and optional for the legacy backend (which only
        needs it at ``commit_layer`` time).
        """
        if self._daemon is not None:
            if dev_output is None:
                raise ValueError(
                    "launch_layer: dev_output is required when using the daemon backend"
                )
            self._daemon.launch_layer(
                layer_id,
                dev_input=dev_input,
                dev_topk_ids=dev_topk_ids,
                dev_topk_weights=dev_topk_weights,
                dev_output=dev_output,
                n_tokens=n_tokens,
                cuda_stream=cuda_stream,
            )
            return
        if not self._initialized:
            raise RuntimeError("init_runtime() must be called before launch_layer()")
        if layer_id not in self._layers:
            raise KeyError(f"layer {layer_id} not registered with CpuOffloadEngine")
        if n_tokens <= 0:
            return
        buf = self.allocate_io_buffers(layer_id, capacity_tokens=n_tokens)
        moe = self._layers[layer_id]
        self._cpu_infer.copy_into_host_async(
            cuda_stream,
            dev_input.data_ptr(),
            dev_topk_ids.data_ptr(),
            dev_topk_weights.data_ptr(),
            buf.dev_input_ptr,
            buf.dev_topk_ids_ptr,
            buf.dev_topk_weights_ptr,
            n_tokens,
            buf.hidden_size,
            buf.topk,
        )
        self._cpu_infer.submit_with_cuda_stream(
            cuda_stream,
            moe.forward_experts(
                buf.host_input.data_ptr(),
                buf.host_topk_ids.data_ptr(),
                buf.host_topk_weights.data_ptr(),
                buf.host_output.data_ptr(),
                n_tokens,
            ),
        )

    def commit_layer(
        self,
        layer_id: int,
        *,
        dev_output: torch.Tensor,
        n_tokens: int,
        cuda_stream: int,
    ) -> None:
        """Second half: GPU polling kernel + async combine.

        ``dev_output`` receives the CPU result added in-place; the caller
        is responsible for index-scattering the contiguous result back
        into the full hidden_states tensor if the input was a subset.

        For the daemon backend the kernel-path ``launch_layer`` already
        enqueued the output memcpy on the caller's stream; this method is
        a no-op there.
        """
        if self._daemon is not None:
            return
        if not self._initialized:
            raise RuntimeError("init_runtime() must be called before commit_layer()")
        if n_tokens <= 0:
            return
        buf = self._io_buffers[layer_id]
        self._cpu_infer.sync_with_cuda_stream(cuda_stream)
        self._cpu_infer.combine_with_device_async(
            cuda_stream,
            buf.dev_output_ptr,
            dev_output.data_ptr(),
            n_tokens,
            buf.hidden_size,
        )
        self._metrics.cpu_offload_pcie_bytes_per_step += (
            buf.host_input.element_size() * n_tokens * buf.hidden_size
            + buf.host_output.element_size() * n_tokens * buf.hidden_size
        )

    def has_layer(self, layer_id: int) -> bool:
        if self._daemon is not None:
            return self._daemon.has_layer(layer_id)
        return layer_id in self._layers

    def read_daemon_stats(self) -> dict:
        """Forward to DaemonBackend.read_daemon_stats (empty dict if N/A)."""
        if self._daemon is not None:
            return self._daemon.read_daemon_stats()
        return {}

    def read_layer_stats(self) -> dict:
        """Forward to DaemonBackend.read_layer_stats (empty dict if N/A).

        Per-layer daemon counters, only populated while the trace lane is
        armed. See engine_daemon.DaemonBackend.read_layer_stats.
        """
        if self._daemon is not None:
            return self._daemon.read_layer_stats()
        return {}

    def read_task_trace_snapshot(self, after_task_id: int = 0) -> dict:
        """读取 daemon 的逐任务 ring 快照；legacy 后端明确标成不支持。"""
        if self._daemon is not None:
            return self._daemon.read_task_trace_snapshot(after_task_id)
        return {
            "valid": False,
            "supported": False,
            "enabled": False,
            "write_task_id": int(after_task_id),
            "records": [],
            "reason": "legacy_backend_has_no_task_trace",
        }

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
        """Phase 1: D2H + DMA fence + submit.  No host sync."""
        if self._daemon is not None:
            kwargs = dict(
                layer_id=layer_id,
                dev_input=dev_input,
                dev_topk_ids=dev_topk_ids,
                dev_topk_weights=dev_topk_weights,
                n_tokens=n_tokens,
                cuda_stream=cuda_stream,
            )
            if trace_events is not None:
                kwargs["trace_events"] = trace_events
            self._daemon.launch_submit(**kwargs)
            return
        # Legacy in-process backend fallback: not parallel, not CUDA-graph safe.
        self.launch_layer(
            layer_id,
            dev_input=dev_input,
            dev_topk_ids=dev_topk_ids,
            dev_topk_weights=dev_topk_weights,
            n_tokens=n_tokens,
            cuda_stream=cuda_stream,
        )

    def launch_wait_and_copyout(
        self,
        layer_id: int,
        *,
        dev_output: torch.Tensor,
        n_tokens: int,
        cuda_stream: int,
        trace_events: Optional[dict] = None,
    ) -> None:
        """Phase 3: Wait kernel + H2D copy-out.  No host sync."""
        if self._daemon is not None:
            kwargs = dict(
                layer_id=layer_id,
                dev_output=dev_output,
                n_tokens=n_tokens,
                cuda_stream=cuda_stream,
            )
            if trace_events is not None:
                kwargs["trace_events"] = trace_events
            self._daemon.launch_wait_and_copyout(**kwargs)
            return
        # Legacy in-process backend fallback.
        self.commit_layer(
            layer_id,
            dev_output=dev_output,
            n_tokens=n_tokens,
            cuda_stream=cuda_stream,
        )

    # ------------------------------------------------------------------

    @property
    def metrics(self) -> CpuOffloadMetrics:
        return self._metrics

    @property
    def initialized(self) -> bool:
        return self._initialized

    def registered_layers(self) -> list[int]:
        if self._daemon is not None:
            return self._daemon.registered_layers()
        return sorted(self._layers)

    def layer_spec(self, layer_id: int) -> LayerSpec:
        """返回登记时的 CPU 权重布局，供 eager 和图路径共同使用。"""
        return self._layer_specs[layer_id]
