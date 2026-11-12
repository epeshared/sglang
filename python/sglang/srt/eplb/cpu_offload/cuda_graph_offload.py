"""CUDA-Graph-compatible CPU offload using SHM kernel path (方案 B).

Uses the same gpu-offload kernel path (cpu_offload_kernels.cu) as the eager
daemon backend, but driven from the CUDA graph capture/replay path.

During graph capture, the Python code runs once and enqueues GPU operations
(PyTorch ops + cudaMemcpyAsync + dma_fence + submit + wait + memcpy-out)
that are ALL CUDA-Graph capturable.  On replay, the recorded kernel sequence
re-executes with updated tensor contents (the device addresses are stable
because the CUDA graph runner reuses the same memory pool).

Critical design constraint:
  - All PyTorch ops (torch.full, torch.where, masked_fill, etc.) launch
    GPU kernels that get recorded into the graph and replay correctly.
  - BUT the *destination* tensors must be PRE-ALLOCATED with fixed device
    addresses so they survive across capture → replay.  Temporary tensors
    created during capture may be GC'd before replay.
  - We use persistent "scratch" device buffers (_cpu_topk_ids_buf, etc.)
    that are allocated once in initialize() and reused for every capture.

Pipeline per MoE layer (all on one CUDA stream, no host sync):
  1. torch GPU ops: build filtered cpu_topk_ids / cpu_topk_weights
     from original topk_ids into persistent scratch buffers
  2. cudaMemcpyAsync: D2D copy input/ids/weights → SHM payload (mapped host)
  3. dma_fence_kernel: read-back + write to ensure DMA ordering
  4. submit_kernel: write opcode/params, bump req_seq (device-side seq buf)
  5. [GPU MoE work runs concurrently with CPU daemon]
  6. wait_from_dev_kernel: poll done_seq via ld.global.cv (L2 bypass)
  7. cudaMemcpyAsync: D2D copy SHM output → device output
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import nullcontext as _nullcontext
from typing import Optional

import torch

from sglang.srt.eplb.cpu_offload import trace as _trace

logger = logging.getLogger(__name__)

# Resolved once, at import -- see trace.py's "COST WHEN DISABLED".
_MOE_TRACE = _trace.ENABLED

# Max tokens per MoE layer that the fixed-size SHM payload can hold.
_DEFAULT_MAX_TOKENS = 256
_MAX_TOPK = 8

# Wire opcodes matching cpu_offload::SlotOpcode.
_OP_FORWARD_EXPERTS = 2


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "")


_STEP_TRACE_ENABLED = _env_bool("SGLANG_MOE_STEP_TRACE", False)

# 合并准备段的开关。1 时 begin_offload 用一个单块 kernel 写出主流准备段的全部
# 结果（见 _launch_fused_prepare）；默认开（11-10 起，6.10 上机对照 11 组全部更快、
# 输出逐字节相同），设 0 走原来那串小算子。绑定缺失时自动退回原路径。进程里只读
# 这一次，initialize() 打一行取值。
_FUSED_PREPARE = _env_bool("CPU_OFFLOAD_FUSED_PREPARE", True)


class CudaGraphOffloadContext:
    """Per-rank context for CUDA-graph-compatible CPU offload.

    Delegates to RemoteMOE's kernel-path (SHM protocol via
    cpu_offload_kernels.cu), which is fully CUDA-Graph capturable.
    Each layer holds a reference to its RemoteMOE instance (owned by
    DaemonBackend) and uses launch_submit / launch_wait_and_copyout.
    """

    def __init__(
        self,
        hidden_size: int,
        topk: int,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        max_spins: int = 50_000_000,
        max_hits: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.topk = topk
        self.max_tokens = max_tokens
        self.max_spins = max_spins
        # GPU-side activation quantization: the gather kernel emits int8 + a
        # per-row scale instead of bf16, halving the D2H payload and letting the
        # daemon skip its stage-0 quantize. Bit-exact with the CPU quantizer.
        # Defined here, not in initialize(), because launch_submit reads it before
        # initialize() may have run. CPU_OFFLOAD_GPU_QUANT=0 disables.
        self._gpu_quant = os.environ.get("CPU_OFFLOAD_GPU_QUANT", "1") not in (
            "0",
            "false",
            "no",
            "",
        )
        # max_hits: compact D2H/H2D size.  Only hit tokens are gathered into
        # [max_hits, H] before DMA, saving N/max_hits× bandwidth for large N.
        # Default: min(max_tokens, 128) — covers the 3σ range for DeepSeek V3
        # with EP (expected hits ≈ N×8/256; 128 safely covers N≤1024).
        self.max_hits: int = max_hits if max_hits is not None else min(max_tokens, 128)
        # _gpu_total_pairs_counter / _cpu_true_demand_counter below only feed
        # the printed [ROUTING]/[OVERFLOW] METRICS lines, but each costs a real
        # kernel launch (or sum()+add_) on EVERY layer of EVERY decode step
        # regardless of hit/miss -- part of the fixed per-layer combine tax.
        # Set CPU_OFFLOAD_METRICS_DETAIL=0 to skip both in perf-sensitive runs.
        self._metrics_detail = _env_bool("CPU_OFFLOAD_METRICS_DETAIL", False)

        # layer_id -> RemoteMOE instance (borrowed from DaemonBackend)
        self._layers: dict[int, object] = {}

        # Persistent device buffers shared across layers (tiny: 4 + 8 bytes).
        self._status_buf: Optional[torch.Tensor] = None
        self._status_dev: int = 0
        self._submitted_seq_buf: Optional[torch.Tensor] = None
        self._submitted_seq_dev: int = 0

        # Persistent scratch device buffers for building filtered
        # cpu_topk_ids / cpu_topk_weights during graph capture.
        # These MUST survive from capture through all replays (fixed addrs).
        self._cpu_topk_ids_buf: Optional[torch.Tensor] = None
        self._cpu_topk_w_buf: Optional[torch.Tensor] = None

        # Persistent output buffer for CPU daemon results.
        # The daemon writes its output into the SHM payload; we memcpy it
        # here first, then add into final_hidden_states (not replace).
        self._cpu_output_buf: Optional[torch.Tensor] = None

        # Persistent int64 counter accumulating (token, expert) pairs routed
        # to THIS rank's OWN static GPU expert block across ALL graph replays
        # (GPU-local pairs ONLY -- see begin_offload's local_expert_lo/hi for
        # why cpu_slot must NOT be used as the range's upper bound).
        # Allocated in initialize(); counted by cpu_offload_count_valid_pairs_kernel
        # in begin_offload BEFORE masking cpu_slot → -1.
        # At atexit: total_pairs = gpu_pairs.item() + daemon.fwd_cpu_pairs.
        self._gpu_total_pairs_counter: Optional[torch.Tensor] = None

        # Persistent int64 counters for combine success/failure across ALL
        # graph replays.  cpu_offload_accumulate_wait_status_kernel reads
        # status_dev (written by wait_from_dev_kernel: 0=OK, else=TIMEOUT)
        # and atomicAdd-increments the appropriate counter on every replay.
        self._combine_ok_counter: Optional[torch.Tensor] = None
        self._combine_fail_counter: Optional[torch.Tensor] = None

        # Device-side slot buffer: one persistent [1] int32 tensor PER LAYER,
        # keyed by layer_id, allocated lazily in register_layer_from_daemon().
        # Written by update_slot(layer_id, expert_id) so fast-loop updates
        # propagate to graph replays. Kept per-layer (NOT shared) so each
        # MoE layer can have an independently-chosen hot expert offloaded
        # to CPU instead of every layer on a rank sharing one global slot.
        self._cpu_slot_bufs: dict[int, torch.Tensor] = {}

        # Dedicated fast-loop expert traffic counter: one persistent
        # [_EXPERT_ACC_SIZE] int64 GPU tensor PER LAYER, keyed by layer_id.
        # Incremented by scatter_add_ inside begin_offload() on every call
        # for that layer — fully CUDA-graph-capturable. Accumulates
        # monotonically; the fast loop reads it and takes deltas so it sees
        # only traffic since the last tick, PER LAYER. Completely independent
        # from EPLB's ExpertDistributionRecorder: EPLB's dump/reset cycle
        # does not affect this counter.
        self._expert_count_accs: dict[int, torch.Tensor] = {}

        # Next-layer prefetch (fire-and-forget cache warm-up). Disable via
        # CPU_OFFLOAD_PREFETCH_NEXT_LAYER=0. See _prefetch_next_layer().
        #
        # HISTORY: an initial A/B on the 94-layer dual-loop benchmark found
        # this to be a NET REGRESSION (total latency +11.7%, throughput
        # -10.5%, COMBINE wait_timeout 6-60x worse) -- root cause was
        # MOE::prefetchExpert() (moe.cpp) doing an UNTHROTTLED, single
        # -thread, ~18MB-per-expert burst of forced cache-line reads at
        # _MM_HINT_T0, saturating DRAM bandwidth and starving the
        # concurrent FORWARD_EXPERTS GEMM (SCHED_IDLE only protects CPU
        # scheduling, not memory-bus contention; L3 CAPACITY was never the
        # issue -- one subnuma's L3 slice fits ~9 experts, not just the 2
        # needed). FIXED by switching the hint to _MM_HINT_T2 (L3-only, no
        # L1/L2 pollution) and pacing the touch loop in small chunks
        # (CPU_OFFLOAD_PREFETCH_CHUNK_BYTES / CPU_OFFLOAD_PREFETCH_PACE_US,
        # default 256KiB / 150us) so prefetch trades wall-clock time (it
        # has slack -- no deadline until the next layer's forward) for a
        # much smaller instantaneous bandwidth footprint. Re-enabled by
        # default after this fix -- re-run the same A/B (baseline: off
        # 19.578s/4602.67 tok/s, pre-throttle on 21.873s/4119.87 tok/s) to
        # confirm the regression is gone before trusting this in prod.
        self._prefetch_enabled: bool = _env_bool(
            "CPU_OFFLOAD_PREFETCH_NEXT_LAYER", True
        )
        # One-shot-per-layer diagnostic: _prefetch_next_layer() normally
        # only runs Python-level logic once per layer_id (at CUDA-graph
        # capture time), so a per-call log would either spam once (eager
        # mode) or never repeat (graph mode) -- track which layer_ids we've
        # already logged the outcome for so every layer gets exactly one
        # diagnostic line regardless of mode.
        self._prefetch_diag_logged_layers: set = set()

        # Persistent side CUDA stream so D2H (submit) and H2D (copy-out)
        # DMA + the CPU-wait poll kernel can execute CONCURRENTLY with this
        # layer's GPU-local expert compute on the main/capturing stream,
        # instead of being fully serialized on it (measured via nsys: 0%
        # kernel/memcpy overlap on the main stream before this change --
        # see cpu-offload-notes memory, "D2H/H2D CONFIRMED FULLY
        # SERIALIZED"). Created once in initialize() and reused across every
        # layer/replay (fixed identity, like the persistent scratch buffers
        # above) -- fork/join is done via wait_stream()/wait_event(), the
        # same pattern already used by the eager EPMoESparseCPUInfer path's
        # cpu_eng_stream in ep_moe/layer.py, which is CUDA-graph-capture-safe.
        self._offload_stream: Optional[torch.cuda.Stream] = None

        # CPU_OFFLOAD_FUSED_PREPARE=1 才用：合并 kernel 的绑定（initialize() 里找到
        # 才非空）和它写 slot_weight 的常驻缓冲。每种退回原因只打印一次。
        self._fused_prepare_fn = None
        self._slot_weight_buf: Optional[torch.Tensor] = None
        self._fused_prepare_notices: set = set()

        self._initialized = False

    def initialize(self) -> bool:
        """Allocate shared device buffers. Returns True on success."""
        if self._initialized:
            return True

        try:
            T = self.max_tokens
            K = self.topk

            # Allocate tiny device buffers for wait-kernel status and
            # device-side submitted_seq (written by submit_kernel, read by
            # wait_from_dev_kernel — eliminates host-side seq tracking).
            self._status_buf = torch.zeros(1, dtype=torch.int32, device="cuda")
            self._status_dev = int(self._status_buf.data_ptr())
            self._submitted_seq_buf = torch.zeros(1, dtype=torch.int64, device="cuda")
            self._submitted_seq_dev = int(self._submitted_seq_buf.data_ptr())

            if self._submitted_seq_dev == 0:
                raise RuntimeError("submitted_seq_buf data_ptr is 0")

            # Persistent scratch buffers: these hold the filtered
            # cpu_topk_ids / cpu_topk_weights that launch_submit memcpy's
            # into the SHM payload.  Allocated at max_tokens capacity so
            # any batch size ≤ max_tokens fits.
            self._cpu_topk_ids_buf = torch.full(
                (T, K), -1, dtype=torch.int32, device="cuda"
            )
            self._cpu_topk_w_buf = torch.zeros(
                (T, K), dtype=torch.float32, device="cuda"
            )

            # Persistent output buffer: daemon result is memcpy'd here,
            # then added into final_hidden_states via a GPU add kernel.
            H = self.hidden_size
            self._cpu_output_buf = torch.zeros(
                (T, H), dtype=torch.bfloat16, device="cuda"
            )

            # Persistent per-token mask buffer: 1.0 for tokens that route to
            # the CPU slot, 0.0 for non-matching/padding rows.
            # Shape [T, 1] for broadcast-mul with [T, H] cpu_output_buf.
            # Written each forward by begin_offload (GPU op, graph-capturable).
            self._token_mask_f = torch.zeros(
                (T, 1), dtype=torch.bfloat16, device="cuda"
            )

            # Persistent int64 counter: total valid (token, expert) pairs
            # routed to this rank (both GPU experts and cpu_slot).
            # cpu_offload_count_valid_pairs_kernel does atomicAdd during
            # every graph replay without any host involvement.
            self._gpu_total_pairs_counter = torch.zeros(
                1, dtype=torch.int64, device="cuda"
            )

            # Persistent int64 counter: TRUE (uncapped) count of tokens whose
            # any top-K position == cpu_slot, measured BEFORE compaction
            # truncates to max_hits rows (see begin_offload — accumulated
            # directly from `token_mask`, the same dynamically-updated mask
            # used for real routing, via a plain GPU tensor add_ each
            # replay — NOT via a count_valid_pairs kernel call with cpu_slot
            # as an immediate arg, because that scalar gets baked into the
            # CUDA graph at CAPTURE TIME and never changes on replay, while
            # the real routing target (cpu_slot_buf) is deliberately mutable
            # across replays via update_slot() without recapture).
            # Needed because the delivered/compacted count (fwd_cpu_pairs
            # from daemon stats) is SILENTLY CAPPED at max_hits per call by
            # cpu_offload_collect_hit_indices_kernel (moe-cpu-engine/ext/
            # cpu_offload_kernels.cu — the ascending-scan loop simply stops
            # once max_hits hits are found; any remaining hit rows are
            # NEVER gathered/forwarded to the CPU, with no error and no GPU
            # fallback — that token's cpu_slot expert contribution is
            # silently missing for that step). Comparing this counter
            # against fwd_cpu_pairs at exit reveals the real overflow/drop
            # rate, which the old to_gpu/to_cpu breakdown cannot show (it
            # only reflects what was actually delivered, not true demand).
            self._cpu_true_demand_counter = torch.zeros(
                1, dtype=torch.int64, device="cuda"
            )

            # Persistent combine ok/fail counters.
            self._combine_ok_counter = torch.zeros(1, dtype=torch.int64, device="cuda")
            self._combine_fail_counter = torch.zeros(
                1, dtype=torch.int64, device="cuda"
            )

            # NOTE: the per-layer cpu_slot buffer and expert-traffic counter
            # (_cpu_slot_bufs / _expert_count_accs) are NOT allocated here —
            # they are lazily created per layer_id in
            # register_layer_from_daemon(), once the layer's identity is
            # known. See those dicts' declarations in __init__ for details.

            # Persistent per-layer hit flag: int32 scalar, 1 if any token
            # routes to the cpu_slot this step, 0 otherwise.  Written each
            # forward by begin_offload via a GPU copy (capturable).  Passed
            # to launch_submit's has_hit_dev so the submit kernel can skip
            # the daemon request — and thus the wait_kernel stream stall —
            # when no tokens hit the cpu_slot.
            self._has_any_hit_buf = torch.zeros(1, dtype=torch.int32, device="cuda")

            # ---- Token compaction buffers ----
            # Instead of DMA'ing all N rows of hidden states, we gather only
            # the n_hit rows into [max_hits, H] before D2H transfer, and
            # scatter back after H2D.  This saves N/max_hits× PCIe bandwidth.
            MH = self.max_hits
            H = self.hidden_size
            # int32 mask viewed as int (needed by collect_hit_indices kernel)
            self._token_mask_i_buf = torch.zeros(T, dtype=torch.int32, device="cuda")
            # Ascending list of source row indices for hit tokens (-1 = empty)
            self._hit_indices_buf = torch.full(
                (MH,), -1, dtype=torch.int32, device="cuda"
            )
            # Compact [max_hits, H] BF16 buffers for input/output
            self._compact_input_buf = torch.zeros(
                (MH, H), dtype=torch.bfloat16, device="cuda"
            )
            self._compact_output_buf = torch.zeros(
                (MH, H), dtype=torch.bfloat16, device="cuda"
            )
            # Compact top-k buffers: column 0 written per-step by gather kernel;
            # columns 1..K-1 stay at their pre-initialised -1 / 0 values so the
            # daemon correctly treats only one expert per hit token as active.
            self._compact_ids_buf = torch.full(
                (MH, K), -1, dtype=torch.int32, device="cuda"
            )
            self._compact_weights_buf = torch.zeros(
                (MH, K), dtype=torch.float32, device="cuda"
            )
            if self._gpu_quant:
                self._compact_q_buf = torch.zeros(
                    (MH, H), dtype=torch.int8, device="cuda"
                )
                self._compact_scale_buf = torch.zeros(
                    (MH,), dtype=torch.float32, device="cuda"
                )

            # Side stream used to overlap D2H/H2D + the CPU-wait poll with
            # this layer's local GPU expert compute on the main stream (see
            # the field's declaration in __init__ for the full rationale).
            self._offload_stream = torch.cuda.Stream()

            # 合并准备段：开关开、绑定在才启用，否则照原路径走。
            binding = "not checked"
            if _FUSED_PREPARE:
                try:
                    import moe_cpu_engine as _engine

                    self._fused_prepare_fn = getattr(
                        _engine, "cpu_client_cuda_fused_prepare", None
                    )
                except Exception:
                    self._fused_prepare_fn = None
                binding = "present" if self._fused_prepare_fn is not None else "missing"
                if self._fused_prepare_fn is not None:
                    self._slot_weight_buf = torch.zeros(
                        T, dtype=torch.float32, device="cuda"
                    )
            print(
                "[cuda_graph_offload] fused_prepare=%s "
                "(CPU_OFFLOAD_FUSED_PREPARE=%r, binding=%s)"
                % (
                    "ON" if self._fused_prepare_fn is not None else "OFF",
                    os.environ.get("CPU_OFFLOAD_FUSED_PREPARE"),
                    binding,
                ),
                flush=True,
            )

            self._initialized = True
            if _MOE_TRACE:
                _trace.record_graph_facts(self)
            # Bytes crossing PCIe per layer per step, both ways.
            _bf16_b = self.max_hits * self.hidden_size * 2
            _int8_b = self.max_hits * self.hidden_size + self.max_hits * 4
            logger.info(
                "[cuda_graph_offload] gpu_quant=%s  D2H input/layer: "
                "%.0fKB -> %.0fKB (%.0f%% less); CPU stage-0 quantize %s",
                "ON" if self._gpu_quant else "OFF",
                _bf16_b / 1024.0,
                (_int8_b if self._gpu_quant else _bf16_b) / 1024.0,
                100.0 * (1 - _int8_b / _bf16_b) if self._gpu_quant else 0.0,
                "SKIPPED" if self._gpu_quant else "runs on CPU",
            )
            logger.info(
                "[cuda_graph_offload] initialized (SHM kernel path): "
                "max_tokens=%d hidden=%d topk=%d max_spins=%d max_hits=%d",
                T,
                self.hidden_size,
                K,
                self.max_spins,
                MH,
            )
            return True

        except Exception as exc:
            logger.warning("[cuda_graph_offload] initialization failed: %s", exc)
            return False

    def register_layer(self, layer_id: int, slot_handle: int) -> bool:
        """Register a layer for CUDA-graph-capturable offload.

        ``slot_handle`` is unused — we look up the RemoteMOE from the
        DaemonBackend by layer_id. Kept for API compatibility.
        """
        # Actual RemoteMOE registration happens in
        # register_layer_from_daemon (called by DaemonBackend after
        # it creates the RemoteMOE). This method is a no-op placeholder
        # for the old MappedHostCallback path.
        return True

    def register_layer_from_daemon(self, layer_id: int, remote_moe: object) -> bool:
        """Store a reference to the DaemonBackend's RemoteMOE for this layer.

        Shared device buffers:
          - submitted_seq: safe to share because stream ordering guarantees
            each layer's submit → wait pair completes before the next layer's
            submit starts (submit writes, wait reads, then next submit writes).
          - status: safe to share (single word, only read after stream sync).
        """
        if not self._initialized:
            return False
        if hasattr(remote_moe, "_submitted_seq_dev"):
            remote_moe._submitted_seq_dev = self._submitted_seq_dev
            remote_moe._submitted_seq_buf = self._submitted_seq_buf
        if hasattr(remote_moe, "_status_dev"):
            remote_moe._status_dev = self._status_dev
            remote_moe._status_buf = self._status_buf
        self._layers[layer_id] = remote_moe

        # Per-layer stateful buffers (deliberately NOT shared with other
        # layers): each layer gets its own cpu_slot assignment and traffic
        # counter so the fast loop can pick an independently hot expert per
        # layer instead of one global expert shared by every layer on this
        # rank. Allocated once per layer_id (idempotent on re-registration).
        if layer_id not in self._cpu_slot_bufs:
            # Initialised to 0x7FFF (a value that never appears in any valid
            # topk_ids tensor) so this layer's slot never false-positive
            # matches before the fast loop installs a real assignment.
            # NEVER initialise to -1: local EP topk_ids use -1 to mark "not
            # this rank's expert", so -1 would match almost every token.
            self._cpu_slot_bufs[layer_id] = torch.tensor(
                [0x7FFF], dtype=torch.int32, device="cuda"
            )
        if layer_id not in self._expert_count_accs:
            # Sized to 512 — safely larger than any realistic expert count.
            self._expert_count_accs[layer_id] = torch.zeros(
                512, dtype=torch.int64, device="cuda"
            )

        if _MOE_TRACE:
            # Re-record so the reported layer count is the final one.
            _trace.record_graph_facts(self)
        logger.info(
            "[cuda_graph_offload] registered layer=%d from daemon "
            "(kernel_path_ready=%s)",
            layer_id,
            getattr(remote_moe, "_kernel_path_ready", False),
        )
        return True

    def has_layer(self, layer_id: int) -> bool:
        return layer_id in self._layers

    def registered_layer_ids(self) -> list[int]:
        """All layer_ids currently registered for CUDA-graph offload."""
        return list(self._layers.keys())

    def get_layer_expert_counts(self, layer_id: int) -> Optional[torch.Tensor]:
        """Return this layer's raw (monotonically increasing) GPU expert
        traffic accumulator, or ``None`` if the layer isn't registered.

        Callers (the fast loop) are responsible for copying to host and
        taking the delta since their last observation.
        """
        return self._expert_count_accs.get(layer_id)

    def update_slot(self, layer_id: int, expert_id: int) -> None:
        """Update layer ``layer_id``'s device-side slot buffer so graph
        replays use the new expert.

        Called by the fast loop after it changes the registry assignment
        for this (rank, layer_id) pair.  The write is a plain Python H2D
        tensor assignment (4 bytes) — not captured into the CUDA graph.
        The next graph replay will read the new value from the device
        buffer via the dynamically-connected ``slot_dev`` tensor in
        begin_offload.

        Negative ``expert_id`` (including the sentinel -1 used by the fast
        loop when it has no valid assignment) is mapped to 0x7FFF, a value
        that never appears in a valid topk_ids tensor.  This prevents the
        GPU from matching the EP topk_ids padding sentinel (-1) and
        accidentally flooding the daemon with spurious work.
        """
        buf = self._cpu_slot_bufs.get(layer_id)
        if buf is not None:
            safe_val = int(expert_id) if expert_id >= 0 else 0x7FFF
            buf[0] = safe_val
            if _STEP_TRACE_ENABLED:
                from sglang.srt.eplb.cpu_offload.step_trace import note_plan_update

                # 只有 device slot 真正写入后才发布版本，registry 中尚未落到
                # device 的候选方案不算生效。
                _plan_epoch = None
                try:
                    from sglang.srt.eplb.cpu_offload import get_cpu_offload_runtime

                    _runtime = get_cpu_offload_runtime()
                    if _runtime is not None:
                        _plan_epoch = _runtime.registry.plan_epoch
                except Exception:
                    pass
                note_plan_update(layer_id, expert_id, _plan_epoch)

    def _prefetch_next_layer(self, layer_id: int, rm: object, stream: int) -> None:
        """Fire-and-forget: ask the daemon to warm THIS rank's NEXT MoE
        layer's currently-assigned CPU expert weights into cache, while
        the current layer's FORWARD_EXPERTS request (just submitted) and
        the GPU-local experts compute — see the module docstring and
        RemoteMOE.launch_prefetch_next_layer / cpu_expertd.cpp's
        run_prefetch_loop for the daemon side.

        The prefetch target is read from ``_cpu_slot_bufs[next_layer_id]``
        (the SAME persistent per-layer device tensor update_slot() writes)
        dynamically INSIDE the submitted kernel on every replay, so it
        always reflects the fast loop's latest assignment without needing
        to recapture the graph. Never awaited; failures are swallowed so
        this can never break the real offload path.

        Submitted through the CURRENT layer's ``rm`` (not the next layer's)
        because every layer registered on this rank shares the same
        daemon process / socket, and the daemon's LayerSlot lookup
        (g_layers) is a single process-wide map — any slot belonging to
        that daemon can service a prefetch request for any layer_id it
        knows about. Using ``rm`` avoids an extra dict lookup + readiness
        check on the next layer's RemoteMOE.
        """
        # NOTE: use print(), not logger.*, for these one-shot diagnostics --
        # mirrors the METRICS dump in runtime.py, which deliberately uses
        # print() instead of logger.info() because logger output from TP
        # worker subprocesses is not reliably visible in this environment
        # (confirmed: these logger.warning() calls never appeared in a real
        # benchmark run even though the print()-based METRICS block did).
        first_log = layer_id not in self._prefetch_diag_logged_layers
        if first_log:
            self._prefetch_diag_logged_layers.add(layer_id)
        try:
            ids = self.registered_layer_ids()
            if len(ids) < 2 or layer_id not in ids:
                if first_log:
                    print(
                        f"[cuda_graph_offload] prefetch SKIPPED (layer={layer_id}): "
                        f"not enough registered layers yet (registered={sorted(ids)})",
                        flush=True,
                    )
                return
            ids.sort()
            idx = ids.index(layer_id)
            next_layer_id = ids[(idx + 1) % len(ids)]
            if next_layer_id == layer_id:
                if first_log:
                    print(
                        f"[cuda_graph_offload] prefetch SKIPPED (layer={layer_id}): "
                        f"next_layer_id resolved to itself (registered={sorted(ids)})",
                        flush=True,
                    )
                return
            next_slot_buf = self._cpu_slot_bufs.get(next_layer_id)
            if next_slot_buf is None:
                if first_log:
                    print(
                        f"[cuda_graph_offload] prefetch SKIPPED (layer={layer_id}): "
                        f"no _cpu_slot_bufs entry for next_layer_id={next_layer_id}",
                        flush=True,
                    )
                return
            kernel_ready = getattr(rm, "_kernel_path_ready", False)
            rm.launch_prefetch_next_layer(
                next_layer_id, int(next_slot_buf.data_ptr()), stream
            )
            if first_log:
                print(
                    f"[cuda_graph_offload] prefetch SUBMITTED (layer={layer_id} -> "
                    f"next_layer={next_layer_id}, kernel_path_ready={kernel_ready}, "
                    f"registered={sorted(ids)})",
                    flush=True,
                )
        except Exception as exc:
            if first_log:
                import traceback

                print(
                    f"[cuda_graph_offload] prefetch_next_layer EXCEPTION "
                    f"(layer={layer_id}): {exc!r}\n{traceback.format_exc()}",
                    flush=True,
                )
            else:
                logger.debug(
                    "[cuda_graph_offload] prefetch_next_layer skipped (layer=%d): %s",
                    layer_id,
                    exc,
                )

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _count_local_pairs(
        self,
        stream: int,
        topk_ids: torch.Tensor,
        local_expert_lo: int,
        local_expert_hi: int,
    ) -> None:
        """METRICS 用：把本卡静态 GPU 专家块收到的 (token, 专家) 对数原子加进计数器。

        原样搬自 begin_offload，原路径和合并路径共用；为什么用静态块、不用
        cpu_slot 作上界，见 begin_offload 里调用处的注释。
        """
        try:
            if local_expert_lo >= 0 and local_expert_hi >= 0:
                import moe_cpu_engine as _engine

                _engine.cpu_client_cuda_count_valid_pairs(
                    stream,
                    int(topk_ids.data_ptr()),
                    int(topk_ids.numel()),
                    int(self._gpu_total_pairs_counter.data_ptr()),
                    local_expert_lo,  # first physical GPU expert id on this rank
                    local_expert_hi,  # last  physical GPU expert id on this rank
                )
        except Exception as _ce:
            # Silently ignore if the binding is unavailable (old build).
            pass

    def _launch_fused_prepare(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cpu_slot_buf: torch.Tensor,
        expert_count_acc: Optional[torch.Tensor],
        stream: int,
    ) -> Optional[tuple]:
        """用一个单块 kernel 写出准备段的全部结果。

        写的东西和原路径相同：专家计数、_token_mask_f、_token_mask_i_buf、
        _has_any_hit_buf、_hit_indices_buf，外加返回的 slot_weight（常驻缓冲的
        前 N 行）和 masked_topk_ids。原路径那两块只写不读的
        _cpu_topk_ids_buf / _cpu_topk_w_buf 不再写。各项语义见
        cpu_offload_kernels.cu 的 cpu_offload_fused_prepare_kernel。

        条件不满足或启动失败返回 None，由调用方照原路径重算；每种原因只打印一次。
        """
        why = None
        if topk_ids.dtype != torch.int32:
            why = "topk_ids dtype=%s, need int32" % topk_ids.dtype
        elif topk_weights.dtype != torch.float32:
            why = "topk_weights dtype=%s, need float32" % topk_weights.dtype
        elif topk_ids.dim() != 2 or tuple(topk_weights.shape) != tuple(topk_ids.shape):
            why = "shape mismatch ids=%s weights=%s" % (
                tuple(topk_ids.shape),
                tuple(topk_weights.shape),
            )
        elif not (topk_ids.is_contiguous() and topk_weights.is_contiguous()):
            why = "topk tensors not contiguous"
        elif expert_count_acc is not None and (
            expert_count_acc.dtype != torch.int64
            or not expert_count_acc.is_contiguous()
        ):
            why = "expert_count_acc not contiguous int64"

        if why is None:
            num_tokens, topk_k = topk_ids.shape
            masked = torch.empty(
                (num_tokens, topk_k), dtype=torch.int32, device=topk_ids.device
            )
            slot_weight = self._slot_weight_buf[:num_tokens]
            try:
                rc = self._fused_prepare_fn(
                    int(stream),
                    int(topk_ids.data_ptr()),
                    int(topk_weights.data_ptr()),
                    int(cpu_slot_buf.data_ptr()),
                    int(expert_count_acc.data_ptr())
                    if expert_count_acc is not None
                    else 0,
                    int(expert_count_acc.shape[0])
                    if expert_count_acc is not None
                    else 0,
                    int(masked.data_ptr()),
                    int(slot_weight.data_ptr()),
                    int(self._token_mask_f.data_ptr()),
                    int(self._token_mask_i_buf.data_ptr()),
                    int(self._has_any_hit_buf.data_ptr()),
                    int(self._hit_indices_buf.data_ptr()),
                    int(num_tokens),
                    int(topk_k),
                    int(self.max_hits),
                )
            except Exception as exc:
                rc = None
                why = "binding raised %s: %s" % (type(exc).__name__, exc)
            if rc == 0:
                if _MOE_TRACE:
                    _trace.hit("gctx.begin.fused_prepare")
                return slot_weight, masked
            if why is None:
                why = "kernel launch returned cudaError %s" % rc

        if why not in self._fused_prepare_notices:
            self._fused_prepare_notices.add(why)
            print(
                "[cuda_graph_offload] fused_prepare fell back to the original "
                "path: %s" % why,
                flush=True,
            )
        if _MOE_TRACE:
            _trace.reason("gctx.begin.fused_prepare_fallback", why)
        return None

    def begin_offload(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cpu_slot: int,
        stream: int,
        local_expert_lo: int = -1,
        local_expert_hi: int = -1,
    ) -> Optional[dict]:
        """Enqueue capturable filtering + D2H + DMA fence + submit.

        All PyTorch ops below launch GPU kernels that are CUDA-Graph
        capturable.  The scratch buffers (_cpu_topk_ids_buf, _cpu_topk_w_buf)
        are persistent — their device addresses are stable across replay.

        During graph REPLAY, the same kernel sequence re-executes with
        updated topk_ids / topk_weights content (the router writes new
        values into the same intermediate tensor addresses that the graph
        recorded during capture).
        """
        if _MOE_TRACE:
            _trace.hit("gctx.begin")
        if not self._initialized:
            if _MOE_TRACE:
                _trace.reason("gctx.begin.reject", "context not initialized")
            print(
                f"[cuda_graph_offload] begin_offload: NOT INITIALIZED layer={layer_id}",
                flush=True,
            )
            return None
        if layer_id not in self._layers:
            if _MOE_TRACE:
                _trace.reason("gctx.begin.reject", "layer not registered")
            print(
                f"[cuda_graph_offload] begin_offload: layer {layer_id} not registered (have layers={sorted(self._layers.keys())[:5]}...)",
                flush=True,
            )
            return None

        rm = self._layers[layer_id]
        if not getattr(rm, "_kernel_path_ready", False):
            if _MOE_TRACE:
                _trace.reason("gctx.begin.reject", "RemoteMOE kernel path not ready")
            print(
                f"[cuda_graph_offload] begin_offload: layer {layer_id} kernel_path not ready",
                flush=True,
            )
            return None

        # Per-layer stateful buffers, allocated for every registered layer in
        # register_layer_from_daemon(). Guaranteed present here because
        # layer_id is already known to be in self._layers above.
        expert_count_acc = self._expert_count_accs.get(layer_id)
        cpu_slot_buf = self._cpu_slot_bufs.get(layer_id)
        if cpu_slot_buf is None:
            if _MOE_TRACE:
                _trace.reason("gctx.begin.reject", "missing per-layer cpu_slot buffer")
            print(
                f"[cuda_graph_offload] begin_offload: layer {layer_id} missing cpu_slot_buf",
                flush=True,
            )
            return None

        num_tokens = topk_ids.shape[0]
        topk_k = topk_ids.shape[1]

        # Skip if num_tokens exceeds scratch buffer capacity.
        # This happens during prefill/extend capture (e.g. bs=160).
        # CPU offload is only needed for decode (bs <= max_tokens).
        if num_tokens > self.max_tokens:
            if _MOE_TRACE:
                # Name the threshold: "too many tokens" is not actionable
                # without knowing what the scratch buffers were sized for.
                _trace.reason(
                    "gctx.begin.reject",
                    "num_tokens > max_tokens=%d (scratch capacity)" % self.max_tokens,
                )
            return None

        # CPU_OFFLOAD_FUSED_PREPARE=1 时，下面 else 里这串小算子和副流上的
        # collect_hit_indices 合成一个 kernel 一次做完；条件不满足或启动失败
        # 返回 None，照原路径走，结果相同。开关关时这里只多一次 is None 判断。
        fused = (
            self._launch_fused_prepare(
                topk_ids, topk_weights, cpu_slot_buf, expert_count_acc, stream
            )
            if self._fused_prepare_fn is not None
            else None
        )
        if fused is not None:
            slot_weight, masked_topk_ids = fused
            token_mask = None
            if self._metrics_detail:
                self._count_local_pairs(
                    stream, topk_ids, local_expert_lo, local_expert_hi
                )
                self._cpu_true_demand_counter += (
                    self._token_mask_i_buf[:num_tokens].sum().to(torch.int64)
                )
        else:
            # ---- GPU-side filtering into persistent scratch buffers ----
            # These ops launch CUDA kernels that get recorded in the graph.
            # On replay, topk_ids/topk_weights have new content but the same
            # device addresses → the filtering kernels produce correct results.

            # Accumulate per-expert routing counts into THIS LAYER's fast-loop
            # counter (kept separate per layer_id so hot-expert traffic is not
            # blended across layers).  Pure GPU op — CUDA-graph-capturable.
            # Uses the ORIGINAL topk_ids (all experts, before cpu-slot masking)
            # so every expert visit is counted.  -1 entries (EP padding for
            # non-local experts) are masked out.  The counter grows
            # monotonically; the fast loop takes deltas.
            if expert_count_acc is not None:
                _flat_ids = topk_ids.flatten().long()  # [N*K]
                _valid = (_flat_ids >= 0).long()  # 1 for real experts, 0 for -1
                _safe_idx = _flat_ids.clamp(min=0, max=expert_count_acc.shape[0] - 1)
                expert_count_acc.scatter_add_(0, _safe_idx, _valid)

            # Which rows have cpu_slot in any of their top-K positions?
            # Use this layer's persistent device tensor (cpu_slot_buf) so the
            # value can be updated between graph replays by update_slot()
            # without re-capture.
            # NOTE: do NOT write cpu_slot_buf here — tensor[i]=scalar issues a
            # synchronous H2D copy that is illegal when the stream is capturing.
            # The fast loop calls update_slot() between replays (outside capture),
            # and cpu_slot_buf is initialized to 0x7FFF at registration (safe
            # default during capture: all-false mask, no CPU tokens, no harm).
            slot_dev = cpu_slot_buf.to(topk_ids.dtype)  # [1], same dtype
            branch_mask = topk_ids == slot_dev  # [N, K] bool — dynamic
            token_mask = branch_mask.any(dim=-1)  # [N] bool

            # Count this rank's OWN GPU-local pairs BEFORE masking (the CPU-routed
            # portion is tracked separately -- see _cpu_true_demand_counter below
            # and daemon fwd_cpu_pairs -- and added back in at the metrics print
            # site). topk_ids here is the pre-EP-dispatch global expert id tensor
            # (0..num_experts-1 for all ranks). We filter to [local_expert_lo,
            # local_expert_hi], this rank's STATIC GPU expert block, NOT
            # [local_expert_lo, cpu_slot] -- cpu_slot is a dynamically rotating
            # expert id chosen specifically because it is OUTSIDE this rank's own
            # GPU block (see runtime.py's local_expert_set_fn), so it is never a
            # valid upper bound for this rank's local expert range, and being a
            # plain Python int it also gets frozen at CUDA-graph capture time and
            # never reflects later rotations. An earlier version used cpu_slot
            # here and produced meaningless/rank-dependent total/to_gpu figures.
            # The count kernel does atomicAdd on the persistent counter, making
            # it graph-capturable with no host involvement during replay.
            if self._metrics_detail:
                self._count_local_pairs(
                    stream, topk_ids, local_expert_lo, local_expert_hi
                )

            # TRUE (uncapped) cpu_slot demand, measured BEFORE compaction can
            # drop any excess hits. See _cpu_true_demand_counter's declaration
            # for why this is needed (delivered fwd_cpu_pairs is capped at
            # max_hits). IMPORTANT: this must be derived from `token_mask`
            # (a GPU tensor recomputed fresh from the DYNAMIC `slot_dev` buffer
            # every replay), NOT from the `cpu_slot`/`local_expert_lo` Python
            # ints used in the count_valid_pairs call above -- those ints are
            # baked into the graph as fixed kernel-launch args AT CAPTURE TIME
            # and never change on replay, whereas cpu_slot_buf (and hence
            # slot_dev/token_mask) is deliberately designed to be updated
            # between replays by update_slot() WITHOUT recapture (see the
            # "Which rows have cpu_slot..." comment above). An earlier version
            # of this counter used the same frozen-int approach as
            # _gpu_total_pairs_counter and silently under-counted true demand
            # (only counting traffic to whichever expert happened to be
            # cpu_slot at the moment the graph was captured, not the real,
            # rotating target) -- pure GPU tensor ops (sum + add_), no host
            # sync, so this stays CUDA-graph-capturable.
            if self._metrics_detail:
                self._cpu_true_demand_counter += token_mask.sum().to(torch.int64)

            # Write float mask to persistent [T, 1] buffer for commit_offload.
            # GPU op recorded in graph: on replay, token_mask is recomputed from
            # new topk_ids and the same SHM address gets updated values.
            self._token_mask_f[:num_tokens, 0] = token_mask.to(dtype=torch.bfloat16)

            # Write int32 hit flag to persistent [1] buffer: 1 if any token
            # routes to cpu_slot, 0 otherwise.  Used by launch_submit's
            # has_hit_dev so the submit kernel skips the daemon request (and
            # the wait_kernel stream stall) when no tokens hit.
            # All ops are GPU-to-GPU — fully CUDA-graph capturable.
            self._has_any_hit_buf.copy_(token_mask.any().reshape(1).to(torch.int32))

            # Build filtered topk_ids: position 0 = cpu_slot for matching rows,
            # -1 for non-matching.  Write into persistent scratch buffer.
            buf_ids = self._cpu_topk_ids_buf[:num_tokens]  # view into persistent
            buf_ids.fill_(-1)
            # Use the device slot tensor (not a Python constant) so replay picks
            # up whatever update_slot() last wrote without re-capture.
            slot_1d = slot_dev.expand(num_tokens)  # [N], view into cpu_slot_buf
            buf_ids[:, 0] = torch.where(
                token_mask,
                slot_1d,
                torch.full(
                    (num_tokens,), -1, dtype=topk_ids.dtype, device=topk_ids.device
                ),
            )

            # Build filtered topk_weights: weight of the cpu_slot branch.
            # Write into persistent scratch buffer.
            buf_w = self._cpu_topk_w_buf[:num_tokens]  # view into persistent
            buf_w.zero_()
            slot_pos = branch_mask.long().argmax(dim=-1)  # [N]
            slot_weight = topk_weights.gather(1, slot_pos.unsqueeze(-1)).squeeze(-1)
            slot_weight = slot_weight * token_mask.float()
            buf_w[:, 0] = slot_weight

            # Write int32 token mask for collect_hit_indices kernel.
            self._token_mask_i_buf[:num_tokens].copy_(token_mask.to(torch.int32))

        # ---- FORK onto the side offload stream ----
        # Everything below (compaction, D2H, submit) only depends on buffers
        # just written above on the main stream, and nothing on the main
        # stream depends on its OUTPUT until commit_offload's combine step
        # (which explicitly joins back — see there). Running it on a
        # separate stream lets the main stream immediately move on to this
        # layer's local GPU expert compute (masked_topk_ids) concurrently
        # with the D2H submit, instead of serializing them on one stream
        # (previously measured via nsys: 0% overlap — see cpu-offload-notes
        # memory). wait_stream() records an event on the main stream and
        # makes the offload stream wait on it — a cheap GPU-side dependency
        # edge, capturable into the CUDA graph exactly like the existing
        # cpu_eng_stream fork/join pattern in ep_moe/layer.py.
        off_stream = self._offload_stream
        use_side_stream = off_stream is not None
        if use_side_stream:
            off_stream.wait_stream(torch.cuda.current_stream())
            off_stream_ptr = int(off_stream.cuda_stream)
        else:
            off_stream_ptr = int(stream)
        stream_ctx = (
            torch.cuda.stream(off_stream) if use_side_stream else _nullcontext()
        )
        step_cpu_events = None
        if _STEP_TRACE_ENABLED:
            from sglang.srt.eplb.cpu_offload.step_trace import (
                begin_cpu_gpu_task,
                record_cpu_business_branches,
            )

            step_cpu_events = begin_cpu_gpu_task(layer_id)
            # 只为 trace 另算真实请求的分支数。实际 token_mask 保持原样，
            # 因而开关不会改变 padding 行是否进入 daemon，也不会污染性能对照。
            # 合并路径没有布尔 token_mask，只在 trace 开着时从 int32 那份现转一次。
            record_cpu_business_branches(
                step_cpu_events,
                token_mask
                if token_mask is not None
                else self._token_mask_i_buf[:num_tokens].bool(),
                max_branches=self.max_hits,
            )

        with stream_ctx:
            # ---- Compaction: collect hit indices, then gather hit rows ----
            # Reduces D2H/H2D DMA from N×H to max_hits×H rows.
            try:
                import moe_cpu_engine as _engine

                # 1. Build hit_indices[max_hits] (single-thread kernel, ~1µs).
                #    合并路径已在主流上写好 hit_indices，这里跳过。
                if fused is None:
                    _engine.cpu_client_cuda_collect_hit_indices(
                        off_stream_ptr,
                        int(self._token_mask_i_buf.data_ptr()),
                        int(self._hit_indices_buf.data_ptr()),
                        int(num_tokens),
                        self.max_hits,
                    )
                # 2. Gather: copy hidden rows at hit positions → compact_input_buf;
                #    write ids[:,0] = cpu_slot and weights[:,0] = slot_weight.
                #    Columns 1..K-1 stay at pre-init -1 / 0 (daemon ignores them).
                if self._gpu_quant:
                    _engine.cpu_client_cuda_gather_compact_int8(
                        off_stream_ptr,
                        int(hidden_states.data_ptr()),
                        int(self._compact_q_buf.data_ptr()),
                        int(self._compact_scale_buf.data_ptr()),
                        int(slot_weight.data_ptr()),
                        int(self._compact_weights_buf.data_ptr()),
                        int(self._compact_ids_buf.data_ptr()),
                        int(self._hit_indices_buf.data_ptr()),
                        int(cpu_slot_buf.data_ptr()),
                        self.max_hits,
                        self.hidden_size,
                        self.topk,
                    )
                else:
                    _engine.cpu_client_cuda_gather_compact(
                        off_stream_ptr,
                        int(hidden_states.data_ptr()),
                        int(self._compact_input_buf.data_ptr()),
                        int(slot_weight.data_ptr()),
                        int(self._compact_weights_buf.data_ptr()),
                        int(self._compact_ids_buf.data_ptr()),
                        int(self._hit_indices_buf.data_ptr()),
                        int(cpu_slot_buf.data_ptr()),
                        self.max_hits,
                        self.hidden_size,
                        self.topk,
                    )
            except Exception as _ce:
                if _MOE_TRACE:
                    _trace.exc("cuda_graph_offload.begin_offload/compaction", _ce)
                logger.warning(
                    "[cuda_graph_offload] compaction kernels failed: %s", _ce
                )
                # Fall back: compact_ids/weights already have safe defaults
                # (-1 / 0) from initialisation; daemon will treat all rows as
                # inactive and zero the output.
                self._compact_ids_buf.fill_(-1)
                self._compact_weights_buf.zero_()

            # ---- SHM kernel path: D2H + fence + submit ----
            # Submit max_hits rows from compact buffers.  has_hit_dev lets the
            # submit kernel emit a NOOP (instead of FORWARD_EXPERTS) when no token
            # hits, keeping daemon spin-alive without PCIe work.
            try:
                rm.launch_submit(
                    stream_ptr=off_stream_ptr,
                    input_ptr=int(
                        (
                            self._compact_q_buf
                            if self._gpu_quant
                            else self._compact_input_buf
                        ).data_ptr()
                    ),
                    topk_ids_ptr=int(self._compact_ids_buf.data_ptr()),
                    topk_weights_ptr=int(self._compact_weights_buf.data_ptr()),
                    ntokens=self.max_hits,
                    has_hit_dev=int(self._has_any_hit_buf.data_ptr()),
                    input_scale_ptr=int(self._compact_scale_buf.data_ptr())
                    if self._gpu_quant
                    else 0,
                    input_is_int8=self._gpu_quant,
                    **(
                        {"trace_events": step_cpu_events.submit_events()}
                        if step_cpu_events is not None
                        else {}
                    ),
                )
            except Exception as exc:
                if _MOE_TRACE:
                    _trace.exc("cuda_graph_offload.begin_offload/launch_submit", exc)
                    _trace.reason(
                        "gctx.begin.reject", "launch_submit raised (see sec.5)"
                    )
                import traceback

                print(
                    f"[cuda_graph_offload] launch_submit FAILED layer=%d: %s\n%s",
                    layer_id,
                    exc,
                    traceback.format_exc(),
                    flush=True,
                )
                return None

            # Fire-and-forget: warm the NEXT layer's currently-assigned CPU
            # expert into cache while this layer's FORWARD_EXPERTS request
            # (just submitted above) and the GPU-local experts compute. Never
            # awaited — see _prefetch_next_layer().
            if self._prefetch_enabled:
                if _MOE_TRACE:
                    _trace.hit("gctx.begin.prefetch")
                self._prefetch_next_layer(layer_id, rm, off_stream_ptr)
            elif layer_id not in self._prefetch_diag_logged_layers:
                self._prefetch_diag_logged_layers.add(layer_id)
                print(
                    f"[cuda_graph_offload] prefetch DISABLED (layer={layer_id}): "
                    f"CPU_OFFLOAD_PREFETCH_NEXT_LAYER=0",
                    flush=True,
                )

        # Diagnostic: log first few successful captures
        # Build masked topk_ids for GPU MoE (mask cpu_slot → -1)
        # 合并路径的 masked_topk_ids 已由合并 kernel 写好。
        if fused is None:
            masked_topk_ids = topk_ids.masked_fill(branch_mask, -1)

        if _MOE_TRACE:
            _trace.hit("gctx.begin.ok")

        return {
            "layer_id": layer_id,
            "stream": stream,
            "offload_stream_ptr": off_stream_ptr,
            "use_side_stream": use_side_stream,
            "masked_topk_ids": masked_topk_ids,
            "num_tokens": num_tokens,
            "step_trace_cpu_events": step_cpu_events,
        }

    def commit_offload(
        self,
        ctx: dict,
        final_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Enqueue capturable wait + H2D copy-out + add on ``stream``.

        Uses RemoteMOE.launch_wait_and_copyout which issues:
          1. wait_from_dev_kernel (poll done_seq via ld.global.cv)
          2. cudaMemcpyAsync (D2D: SHM output → persistent cpu_output_buf)

        Then adds the cpu_output_buf into final_hidden_states via a
        CUDA-Graph-capturable torch add operation.
        """
        if _MOE_TRACE:
            _trace.hit("gctx.commit")
        layer_id = ctx["layer_id"]
        rm = self._layers[layer_id]
        num_tokens = ctx["num_tokens"]
        off_stream_ptr = ctx.get("offload_stream_ptr", ctx["stream"])
        use_side_stream = (
            ctx.get("use_side_stream", False) and self._offload_stream is not None
        )
        step_cpu_events = ctx.get("step_trace_cpu_events")

        # Wait (poll done_seq) + H2D copy-out run on the SAME side stream
        # begin_offload's D2H/submit used, so they continue concurrently
        # with this layer's local GPU expert compute happening on the main
        # stream in between begin_offload's return and this call (no cross
        # -stream wait needed here — it's the same stream continuing in
        # program order from the earlier submit).
        try:
            # Copy daemon output into compact_output_buf [max_hits, H].
            # We DMA only max_hits rows (the same compact size sent in
            # begin_offload), saving (N - max_hits)/N × PCIe bandwidth.
            rm.launch_wait_and_copyout(
                stream_ptr=off_stream_ptr,
                output_ptr=int(self._compact_output_buf.data_ptr()),
                ntokens=self.max_hits,
                max_spins=self.max_spins,
                **(
                    {"trace_events": step_cpu_events.wait_events()}
                    if step_cpu_events is not None
                    else {}
                ),
            )
        except Exception as exc:
            if _MOE_TRACE:
                _trace.exc("cuda_graph_offload.commit_offload/wait_and_copyout", exc)
            import traceback

            print(
                f"[cuda_graph_offload] launch_wait_and_copyout FAILED "
                f"layer={layer_id}: {exc}\n{traceback.format_exc()}",
                flush=True,
            )
            return final_hidden_states

        # Accumulate wait-kernel status into persistent counters.
        # status_dev is written by wait_from_dev_kernel: 0=OK, else=TIMEOUT.
        # This kernel is graph-capturable and runs on every replay. Kept on
        # the offload stream (same stream that just wrote status_dev) so no
        # extra cross-stream dependency is needed for this step.
        try:
            import moe_cpu_engine as _engine

            if (
                self._combine_ok_counter is not None
                and self._combine_fail_counter is not None
                and self._status_buf is not None
            ):
                _engine.cpu_client_cuda_accumulate_wait_status(
                    off_stream_ptr,
                    self._status_dev,
                    int(self._combine_ok_counter.data_ptr()),
                    int(self._combine_fail_counter.data_ptr()),
                )
        except Exception:
            pass

        # ---- JOIN back onto the main stream ----
        # The combine step below writes into final_hidden_states, which the
        # main stream's local GPU expert compute (issued between
        # begin_offload's return and this call) also writes into — so the
        # combine must happen only after BOTH (a) the offload stream's H2D
        # copy-out above and (b) the main stream's local-expert writes are
        # visible. (b) is automatic (same stream, program order); (a) needs
        # this explicit wait_stream(), a cheap GPU-side event wait (no host
        # sync) that is captured into the CUDA graph as a dependency edge,
        # not a real blocking call. If the offload stream isn't in use
        # (fallback), this is a no-op self-wait.
        if _STEP_TRACE_ENABLED and step_cpu_events is not None:
            from sglang.srt.eplb.cpu_offload.step_trace import (
                record_visible_wait_start,
            )

            record_visible_wait_start(step_cpu_events)
        if use_side_stream:
            # 仍使用原有内部依赖完成 fork/join；external 计时 event 只测时，
            # 绝不拿来替换同步边。
            torch.cuda.current_stream().wait_stream(self._offload_stream)
        if _STEP_TRACE_ENABLED and step_cpu_events is not None:
            from sglang.srt.eplb.cpu_offload.step_trace import (
                record_visible_wait_end,
                record_wait_status,
            )

            record_visible_wait_end(step_cpu_events)
            if self._status_buf is not None:
                record_wait_status(step_cpu_events, self._status_buf)

        # Combine path A (preferred): scatter-ADD the compact daemon output
        # directly into final_hidden_states at the max_hits hit rows in one
        # kernel, skipping the _cpu_output_buf intermediate and the two
        # full-[N,H] elementwise passes (token_mask mul_ + add_) that combine
        # path B below needs. TIMEOUT / padding (-1) rows are skipped inside
        # the kernel (no add), so this is exactly equivalent to path B's
        # zero-then-add end state. Only valid when final_hidden_states is a
        # contiguous bf16 tensor (raw pointer arithmetic must match
        # _compact_output_buf's row layout).
        fused_add_ok = (
            final_hidden_states.dtype == torch.bfloat16
            and final_hidden_states.is_contiguous()
        )
        if fused_add_ok:
            try:
                import moe_cpu_engine as _engine

                if hasattr(_engine, "cpu_client_cuda_scatter_add_compact_status"):
                    _engine.cpu_client_cuda_scatter_add_compact_status(
                        int(ctx["stream"]),
                        int(self._compact_output_buf.data_ptr()),
                        int(final_hidden_states.data_ptr()),
                        int(self._hit_indices_buf.data_ptr()),
                        self._status_dev,
                        self.max_hits,
                        self.hidden_size,
                    )
                    if _MOE_TRACE:
                        _trace.hit("gctx.commit.fused_add")
                    return final_hidden_states
            except Exception as _se:
                if _MOE_TRACE:
                    _trace.exc(
                        "cuda_graph_offload.commit_offload/scatter_add_compact", _se
                    )
                logger.warning(
                    "[cuda_graph_offload] scatter_add_compact_status failed, "
                    "falling back to scatter_compact+mask+add: %s",
                    _se,
                )

        # Combine path B (fallback: older moe_cpu_engine build, or
        # final_hidden_states isn't a contiguous bf16 tensor). Scatter compact
        # daemon output back to full cpu_output_buf, fused with the TIMEOUT
        # zero-check (writes zero directly to the destination row on TIMEOUT
        # instead of a separate full-buffer zero pass beforehand -- one
        # kernel launch instead of two). Slots whose hit_indices entry is -1
        # are skipped (the token_mask_f mul_ below zeroes those rows safely).
        # Falls back further to the older two-kernel sequence if running
        # against an even older moe_cpu_engine build lacking that binding too.
        try:
            import moe_cpu_engine as _engine

            if hasattr(_engine, "cpu_client_cuda_scatter_compact_status"):
                _engine.cpu_client_cuda_scatter_compact_status(
                    int(ctx["stream"]),
                    int(self._compact_output_buf.data_ptr()),
                    int(self._cpu_output_buf.data_ptr()),
                    int(self._hit_indices_buf.data_ptr()),
                    self._status_dev,
                    self.max_hits,
                    self.hidden_size,
                )
            else:
                if self._status_buf is not None:
                    n_elements = self.max_hits * self.hidden_size
                    _engine.cpu_client_cuda_conditional_zero(
                        int(ctx["stream"]),
                        self._status_dev,
                        int(self._compact_output_buf.data_ptr()),
                        n_elements,
                    )
                _engine.cpu_client_cuda_scatter_compact(
                    int(ctx["stream"]),
                    int(self._compact_output_buf.data_ptr()),
                    int(self._cpu_output_buf.data_ptr()),
                    int(self._hit_indices_buf.data_ptr()),
                    self.max_hits,
                    self.hidden_size,
                )
        except Exception as _se:
            if _MOE_TRACE:
                _trace.exc("cuda_graph_offload.commit_offload/scatter_compact", _se)
            logger.warning("[cuda_graph_offload] scatter_compact failed: %s", _se)

        # Add CPU contribution into final_hidden_states.
        # Apply the token mask first: zero rows where the token did NOT route
        # to the CPU slot so stale daemon output doesn't corrupt those positions.
        # _token_mask_f[:N] is [N, 1] with 1.0 for real CPU tokens (broadcasts
        # over H). Both ops are CUDA-Graph capturable (pure device ops on
        # persistent buffers). The mul_ is in-place on _cpu_output_buf, which
        # is always overwritten by launch_wait_and_copyout at the next step.
        if _MOE_TRACE:
            _trace.hit("gctx.commit.fallback")
        cpu_out = self._cpu_output_buf[:num_tokens]
        cpu_out.mul_(self._token_mask_f[:num_tokens])  # zero non-matching rows
        final_hidden_states[:num_tokens].add_(cpu_out)

        return final_hidden_states


# Process-global singleton
_ctx_lock = threading.Lock()
_ctx: Optional[CudaGraphOffloadContext] = None


def get_cuda_graph_offload_ctx() -> Optional[CudaGraphOffloadContext]:
    return _ctx


def init_cuda_graph_offload(
    hidden_size: int,
    topk: int,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_spins: int = 50_000_000,
    max_hits: Optional[int] = None,
) -> Optional[CudaGraphOffloadContext]:
    """Create or return the process-global CudaGraphOffloadContext."""
    global _ctx
    if _ctx is not None:
        return _ctx

    with _ctx_lock:
        if _ctx is not None:
            return _ctx

        ctx = CudaGraphOffloadContext(
            hidden_size, topk, max_tokens, max_spins, max_hits=max_hits
        )
        if not ctx.initialize():
            return None
        _ctx = ctx
        return _ctx
