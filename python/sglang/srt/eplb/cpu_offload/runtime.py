"""Process-singleton runtime for CPU offload.

Holds the shared ``CpuOffloadConfig``, ``CpuShadowRegistry``,
``CpuOffloadEngine``, ``FastLoopScheduler``, and ``CpuOffloadMetrics``.

The runtime is created exactly once by ``ModelRunner`` early in startup.
Hot-path callers (FusedMoE.forward, EPLBManager.on_forward_pass_end)
fetch it via ``get_cpu_offload_runtime()`` and bail out cheaply when the
feature is off (returns ``None``).
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from sglang.srt.eplb.cpu_offload import trace as _trace
from sglang.srt.eplb.cpu_offload.config import CpuOffloadConfig
from sglang.srt.eplb.cpu_offload.engine import CpuOffloadEngine
from sglang.srt.eplb.cpu_offload.fast_loop import FastLoopScheduler
from sglang.srt.eplb.cpu_offload.metrics import CpuOffloadMetrics
from sglang.srt.eplb.cpu_offload.shadow_state import CpuShadowRegistry

logger = logging.getLogger(__name__)

# Resolved once, at import -- see trace.py's "COST WHEN DISABLED".
_MOE_TRACE = _trace.ENABLED


class _LocalExpertLoadCounter:
    """Per-process, per-layer, per-expert token counter.

    Independent from ``ExpertDistributionRecorder`` (which is global +
    destructive on read). This counter is local to the process and the
    fast loop drains it each tick.

    The counter only tracks experts the LOCAL rank routed to, so the
    "rank dimension" is implicit (always the local rank).
    """

    __slots__ = ("_layers", "_lock", "_local_rank")

    def __init__(self, local_rank: int) -> None:
        self._local_rank = int(local_rank)
        self._layers: dict[int, dict[int, int]] = defaultdict(dict)
        self._lock = threading.Lock()

    def record(self, layer_id: int, expert_id: int, n_tokens: int) -> None:
        if n_tokens <= 0 or expert_id < 0:
            return
        # No lock on the hot path; updates are additive and tear-tolerant.
        d = self._layers[layer_id]
        d[expert_id] = d.get(expert_id, 0) + int(n_tokens)

    def drain(self) -> dict:
        """Return ``{layer_id: {(local_rank, expert_id): tokens}}`` and clear."""
        with self._lock:
            out: dict = {}
            for layer_id, per_expert in self._layers.items():
                if not per_expert:
                    continue
                out[layer_id] = {
                    (self._local_rank, expert_id): tokens
                    for expert_id, tokens in per_expert.items()
                }
            self._layers.clear()
            return out


@dataclass
class CpuOffloadRuntime:
    """Bundle of runtime objects. Constructed once per process."""

    config: CpuOffloadConfig
    registry: CpuShadowRegistry
    engine: CpuOffloadEngine
    fast_loop: FastLoopScheduler
    metrics: CpuOffloadMetrics
    counter: _LocalExpertLoadCounter
    local_rank: int

    def tick_fast_loop(self) -> bool:
        """Convenience accessor; safe to call from any thread."""
        return self.fast_loop.tick()

    def is_active_locally(self) -> bool:
        """True iff CPU offload is enabled AND this rank participates."""
        return self.config.rank_is_active(self.local_rank)

    def local_cpu_slot(self, layer_id: int) -> int:
        """Currently-assigned CPU slot expert id for the local rank on
        ``layer_id``, or -1.

        Each MoE layer can have an independently-assigned hot expert (see
        fast_loop.py), so the caller MUST pass its own layer_id rather than
        assuming one rank-wide slot.
        """
        if not self.is_active_locally():
            return -1
        state = self.registry.local_state()
        if state is None or state.demoted:
            return -1
        return state.get_slot(layer_id)

    def record_routed(self, layer_id: int, expert_id: int, n_tokens: int) -> None:
        self.counter.record(layer_id, expert_id, n_tokens)


_runtime_lock = threading.Lock()
_runtime: Optional[CpuOffloadRuntime] = None


def get_cpu_offload_runtime() -> Optional[CpuOffloadRuntime]:
    """Hot-path accessor. Returns ``None`` if not initialized or disabled."""
    return _runtime


def _build_local_expert_set_fn(model_runner):
    """Build a ``(rank_id, layer_id) -> set[int]`` lookup over expert location metadata.

    Imported lazily so this module can be loaded by tests that don't have
    the full SGLang server stack.

    The EPLB ``physical_to_logical_map`` maps physical slots to logical expert
    IDs.  With the default identity map, rank r owns logical experts
    ``[r*num_local, (r+1)*num_local)`` (block layout), which matches how
    SGLang routes experts to GPU ranks in EP mode.  The fast loop uses this
    set to RESTRICT cpu_slot candidates to the rank's own GPU-local experts:
    offloading an expert this rank does not own would leave the owning rank
    computing it on GPU while this rank adds a CPU result on top, double
    counting it after the TP all-reduce.

    NOTE: this derives the block from ``num_local_physical_experts`` and the
    physical->logical map, whereas the weight loader and StandardDispatcher
    derive it from ``moe_ep_rank * num_local_experts``. Those agree only
    while the map is identity and the two counts match;
    ``model_load_hook.register_model_moe_layers`` asserts that at startup so a
    divergence fails loudly instead of silently mis-filtering here.
    """
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

    def fn(rank_id: int, layer_id: int) -> set[int]:
        meta = get_global_expert_location_metadata()
        if meta is None:
            return set()
        try:
            num_local = int(meta.num_local_physical_experts)
            phys_to_logical = meta.physical_to_logical_map  # (num_layers, num_phys)
        except Exception:
            return set()
        start = rank_id * num_local
        end = start + num_local
        try:
            slice_ = phys_to_logical[layer_id, start:end]
        except (IndexError, RuntimeError):
            return set()
        return {int(x) for x in slice_.tolist() if int(x) >= 0}

    return fn


def init_cpu_offload_runtime(
    server_args,
    *,
    local_rank: int,
    world_size: int,
    num_numa_nodes: int = 2,
    model_runner=None,
) -> Optional[CpuOffloadRuntime]:
    """Build the process-singleton runtime from ``server_args``.

    Returns ``None`` if the feature is disabled. Safe to call multiple
    times — subsequent calls are no-ops.
    """
    global _runtime
    if _runtime is not None:
        return _runtime
    if not getattr(server_args, "enable_cpu_offload_fast_loop", False):
        return None

    with _runtime_lock:
        if _runtime is not None:
            return _runtime

        config = _build_config_from_server_args(server_args)
        if not config.is_active():
            logger.info(
                "[cpu_offload] feature flag set but no active ranks; runtime stays None"
            )
            return None

        metrics = CpuOffloadMetrics()
        registry = CpuShadowRegistry(
            config,
            world_size=world_size,
            num_numa_nodes=num_numa_nodes,
            local_rank=local_rank,
        )
        engine = CpuOffloadEngine(
            config,
            host_rank=local_rank,
            num_hosts=world_size,
            ep_size=getattr(server_args, "ep_size", 1),
            num_numa_nodes=num_numa_nodes,
            metrics=metrics,
        )
        counter = _LocalExpertLoadCounter(local_rank=local_rank)

        local_set_fn = (
            _build_local_expert_set_fn(model_runner)
            if model_runner is not None
            else (lambda rid, lid: set())
        )

        fast_loop = FastLoopScheduler(
            config,
            registry,
            metrics,
            load_snapshot_fn=counter.drain,
            local_expert_set_fn=local_set_fn,
        )

        rt = CpuOffloadRuntime(
            config=config,
            registry=registry,
            engine=engine,
            fast_loop=fast_loop,
            metrics=metrics,
            counter=counter,
            local_rank=local_rank,
        )

        # NOTE: we do NOT call engine.init_runtime() here. The actual
        # NUMA daemon spawn happens lazily on the first
        # ``register_layer`` from a model adapter, so processes that
        # configure the flag but never integrate stay cheap and crash-free.
        _runtime = rt
        logger.info(
            "[cpu_offload] runtime initialized: local_rank=%d world_size=%d "
            "active_ranks=%s num_numa=%d",
            local_rank,
            world_size,
            config.cpu_active_ranks,
            num_numa_nodes,
        )
        if _MOE_TRACE:
            # Registered BEFORE the METRICS dump so that atexit's LIFO order
            # runs METRICS first and the trace report last.
            _trace.record_server_args(server_args)
            _trace.fact(
                "topology",
                "local_rank=%d world_size=%d num_numa=%d"
                % (local_rank, world_size, num_numa_nodes),
                "runtime",
            )
            _trace.install_atexit(rt)
        _maybe_install_metrics_atexit(rt)
        return _runtime


def _maybe_install_metrics_atexit(rt: CpuOffloadRuntime) -> None:
    """Print a one-shot metrics summary on process exit.

    Gated by ``CPU_OFFLOAD_LOG_METRICS_AT_EXIT`` (default ``1``). Set to
    ``0`` to silence. Each rank prints its own line so you can confirm
    every active rank actually fed traffic to the CPU.
    """
    flag = os.environ.get("CPU_OFFLOAD_LOG_METRICS_AT_EXIT", "1").strip().lower()
    if flag in ("0", "false", "no", "off", ""):
        return

    def _dump() -> None:
        try:
            snap = rt.metrics.snapshot()
        except Exception:
            return

        # Detect CUDA-graph path: if the graph offload context was
        # initialized, most work happens in graph replay where Python
        # metrics are not updated.  In that case, pull real stats from
        # the daemon (which counts every replay).
        graph_mode = False
        n_graph_layers = 0
        daemon_stats: dict = {}
        try:
            from sglang.srt.eplb.cpu_offload.cuda_graph_offload import (
                get_cuda_graph_offload_ctx,
            )

            gctx = get_cuda_graph_offload_ctx()
            if gctx is not None and gctx.initialized:
                graph_mode = True
                n_graph_layers = len(gctx._layers)
        except Exception:
            pass

        if graph_mode:
            try:
                daemon_stats = rt.engine.read_daemon_stats()
            except Exception:
                pass

        if daemon_stats:
            # Daemon has real statistics (includes all replays).
            fwd_count = daemon_stats.get("fwd_count", 0)
            fwd_cpu_pairs = daemon_stats.get(
                "fwd_cpu_pairs", daemon_stats.get("fwd_active_toks", 0)
            )
            compute_ns = daemon_stats.get("compute_ns_sum", 0)
            avg_us = (compute_ns / 1000.0 / fwd_count) if fwd_count else 0.0
            # v9+: pure forwardExperts() time -- compute_ns/avg_us above also
            # includes param decode + CLWB output flush + bookkeeping, so it
            # overstates real GEMM cost; use this when comparing against a
            # dummy warm-up call's timing. Absent -> omit from the printout.
            gemm_ns = daemon_stats.get("gemm_ns_sum")
            avg_gemm_us = (
                (gemm_ns / 1000.0 / fwd_count) if gemm_ns and fwd_count else None
            )
            # v10+: further split the compute_ns_sum - gemm_ns_sum gap into
            # pre-GEMM dispatch time and post-GEMM CLWB flush time, so the
            # remaining "everything else" (stats bookkeeping + sfence +
            # done_seq store) can be inferred by subtraction instead of
            # staying one opaque number. Absent -> omit the line entirely.
            breakdown_line = ""
            if "dispatch_ns_sum" in daemon_stats and fwd_count:
                dispatch_ns = daemon_stats.get("dispatch_ns_sum", 0)
                flush_ns = daemon_stats.get("flush_ns_sum", 0)
                avg_dispatch_us = dispatch_ns / 1000.0 / fwd_count
                avg_flush_us = flush_ns / 1000.0 / fwd_count
                # v11+: directly-measured post-GEMM diagnostic-block time and
                # post-flush stats-atomics time -- the two components v10's
                # avg_tail_us used to lump into one unexplained residual.
                # Absent on older daemon builds -> both default to 0 and
                # avg_tail_us falls back to the v10 (fully-derived) residual.
                diag_ns = daemon_stats.get("diag_ns_sum", 0)
                stats_ns = daemon_stats.get("stats_ns_sum", 0)
                avg_diag_us = diag_ns / 1000.0 / fwd_count
                avg_stats_us = stats_ns / 1000.0 / fwd_count
                avg_tail_us = max(
                    0.0,
                    avg_us
                    - avg_dispatch_us
                    - (avg_gemm_us or 0.0)
                    - avg_flush_us
                    - avg_diag_us
                    - avg_stats_us,
                )
                extra_line = (
                    f"  avg_diag_us={avg_diag_us:.1f}  avg_stats_us={avg_stats_us:.1f}"
                    if "diag_ns_sum" in daemon_stats
                    else ""
                )
                breakdown_line = (
                    f"\n  [BREAKDOWN] avg_dispatch_us={avg_dispatch_us:.1f}  "
                    f"avg_gemm_us={avg_gemm_us or 0.0:.1f}  "
                    f"avg_flush_us={avg_flush_us:.1f}{extra_line}  "
                    f"avg_tail_us={avg_tail_us:.1f}"
                )

            # v10+: steady-state mean, the figure to tune against. avg_gemm_us
            # divides a lifetime sum, so it never drops the startup transient and is
            # skewed by rare multi-ms stalls. Stalls are printed, not hidden.
            steady_ns = daemon_stats.get("steady_ns_sum", 0)
            steady_count = daemon_stats.get("steady_count", 0)
            steady_us = (steady_ns / 1000.0 / steady_count) if steady_count else None
            steady_max_us = daemon_stats.get("steady_max_ns", 0) / 1000.0
            stall_count = daemon_stats.get("stall_count", 0)
            stall_ns = daemon_stats.get("stall_ns_sum", 0)

            # gpu_pairs_this_rank measured GPU-side by count_valid_pairs kernel
            # against THIS rank's static GPU expert block (accumulated across
            # ALL replays without host sync). total = gpu_pairs + cpu_pairs
            # (previously derived as total_counter - cpu_pairs, but that
            # counter used cpu_slot -- a dynamically rotating, capture-time-
            # frozen id -- as its range's upper bound and silently miscounted;
            # it now counts ONLY this rank's own static GPU block, so we sum
            # instead of subtract).
            gpu_pairs = 0
            try:
                if (
                    gctx is not None
                    and getattr(gctx, "_metrics_detail", True)
                    and getattr(gctx, "_gpu_total_pairs_counter", None) is not None
                ):
                    gpu_pairs = int(gctx._gpu_total_pairs_counter.item())
            except Exception:
                pass
            total_pairs = gpu_pairs + fwd_cpu_pairs

            avg_toks_per_req = (fwd_cpu_pairs / fwd_count) if fwd_count else 0.0

            pairs_line = (
                f"total={total_pairs}  to_gpu={gpu_pairs}  "
                f"to_cpu={fwd_cpu_pairs} ({100.0 * fwd_cpu_pairs / total_pairs:.2f}%)"
                if total_pairs > 0
                else "total=N/A"
            )

            # TRUE (uncapped) cpu_slot demand vs what was actually delivered
            # (fwd_cpu_pairs, capped at max_hits per call by the compaction
            # kernel). If true_cpu_demand > fwd_cpu_pairs, the excess was
            # SILENTLY DROPPED -- never computed by CPU nor GPU (a real
            # correctness gap, not just a perf inefficiency). See
            # CudaGraphOffloadContext._cpu_true_demand_counter for details.
            overflow_line = ""
            try:
                if (
                    gctx is not None
                    and getattr(gctx, "_metrics_detail", True)
                    and getattr(gctx, "_cpu_true_demand_counter", None) is not None
                ):
                    true_cpu_demand = int(gctx._cpu_true_demand_counter.item())
                    dropped = max(0, true_cpu_demand - fwd_cpu_pairs)
                    drop_rate = (
                        (100.0 * dropped / true_cpu_demand)
                        if true_cpu_demand > 0
                        else 0.0
                    )
                    overflow_line = (
                        f"\n  [OVERFLOW] true_cpu_demand={true_cpu_demand}  "
                        f"delivered={fwd_cpu_pairs}  dropped={dropped} ({drop_rate:.2f}%)"
                    )
            except Exception:
                pass

            # Combine: pairs = cpu_slot_hits (daemon stat, pairs granularity).
            # ok/fail = wait-kernel request-level success (one per layer call).
            # avg_cpu_compute_us = daemon-side compute time per request
            #   (time from forwardExperts start to clwb/sfence done, NOT GPU wait time).
            c_ok, c_fail = 0, 0
            try:
                if getattr(gctx, "_combine_ok_counter", None) is not None:
                    c_ok = int(gctx._combine_ok_counter.item())
                    c_fail = int(gctx._combine_fail_counter.item())
            except Exception:
                pass
            c_total = c_ok + c_fail
            # ok+fail = total wait-kernel calls (= n_layers × n_decode_steps),
            # same for all ranks regardless of CPU load — not a useful per-rank
            # metric.  Only c_fail (timeouts) is interesting; show it prominently
            # when non-zero, otherwise just confirm no timeouts.
            if c_total > 0:
                if c_fail > 0:
                    c_rate = 100.0 * c_fail / c_total
                    combine_str = (
                        f"pairs={fwd_cpu_pairs}  "
                        f"wait_timeout={c_fail} ({c_rate:.2f}% of {c_total} calls)"
                    )
                else:
                    combine_str = f"pairs={fwd_cpu_pairs}  wait_timeout=0"
            else:
                combine_str = f"pairs={fwd_cpu_pairs}  n/a (old build)"

            # Next-layer prefetch (cache warm-up) verification. exec_count
            # only increments when the daemon's background prefetch thread
            # actually touched weight bytes (not on dedup/no-op) — so
            # exec_count > 0 here is direct proof the feature is running.
            # Absent (old daemon build / prefetch disabled) -> omit the line
            # entirely rather than print a misleading all-zero row.
            pf_exec = daemon_stats.get("prefetch_exec_count", 0)
            pf_line = ""
            if "prefetch_exec_count" in daemon_stats:
                pf_bytes = daemon_stats.get("prefetch_bytes_sum", 0)
                pf_ns = daemon_stats.get("prefetch_ns_sum", 0)
                pf_avg_us = (pf_ns / 1000.0 / pf_exec) if pf_exec else 0.0
                pf_mb = pf_bytes / (1024.0 * 1024.0)
                # attempt_count increments before the warm, exec_count only on
                # completion, so attempt-exec is the count preempted mid-stream.
                _pf_att = daemon_stats.get("prefetch_attempt_count", 0)
                _pf_abort = max(0, _pf_att - pf_exec)
                pf_line = (
                    f"\n  [PREFETCH] exec_count={pf_exec}  "
                    f"total_mb={pf_mb:.1f}  avg_us={pf_avg_us:.1f}"
                    f"  attempt={_pf_att} aborted={_pf_abort}"
                )
                # v6+: if nothing ever executed, show WHERE the pipeline
                # stalled instead of leaving a bare exec_count=0 to puzzle
                # over -- seen=0 means the GPU client never submitted a
                # request at all; attempt=0 (seen>0) means every request was
                # deduped (target never changed); miss>0 means the daemon
                # saw requests for a layer_id it has no registered MOE for;
                # otherwise attempt-miss-exec>0 calls ran but returned 0
                # bytes (XFT backend / weights not loaded / bad expert_id).
                if pf_exec == 0 and "prefetch_seen_count" in daemon_stats:
                    pf_seen = daemon_stats.get("prefetch_seen_count", 0)
                    pf_attempt = daemon_stats.get("prefetch_attempt_count", 0)
                    pf_miss = daemon_stats.get("prefetch_miss_count", 0)
                    pf_zero = max(0, pf_attempt - pf_miss - pf_exec)
                    pf_line += (
                        f"\n  [PREFETCH-DIAG] seen={pf_seen}  attempt={pf_attempt}  "
                        f"miss={pf_miss}  zero_bytes={pf_zero}"
                    )

            aw_line = ""
            if "attention_window_ns" in daemon_stats:
                aw_us = daemon_stats.get("attention_window_ns", 0) / 1000.0
                aw_line = f"\n  [IDLE_WINDOW] attention_window_us={aw_us:.1f} (EMA gap between GEMMs)"

            gemm_str = (
                f"  avg_gemm_us={avg_gemm_us:.1f}" if avg_gemm_us is not None else ""
            )

            gemm_line = ""
            if steady_us is not None:
                gemm_line = (
                    f"\n  [GEMM]     steady_us={steady_us:.1f} (n={steady_count}, "
                    f"max={steady_max_us:.1f}us, startup+stalls excluded)"
                )
                if stall_count:
                    gemm_line += (
                        f"  STALLS={stall_count} "
                        f"(avg {stall_ns / 1000.0 / stall_count:.0f}us)"
                    )

            print(
                f"[cpu_offload][rank={rt.local_rank}] METRICS\n"
                f"  [ROUTING]  {pairs_line}  "
                f"avg_cpu_compute_us={avg_us:.1f}{gemm_str} (per req, avg {avg_toks_per_req:.1f} toks/req)"
                f"{overflow_line}"
                f"{breakdown_line}"
                f"{gemm_line}\n"
                f"  [COMBINE]  {combine_str}"
                f"{pf_line}"
                f"{aw_line}",
                flush=True,
            )
            return

        # Non-graph path: use Python-side metrics (eager mode).
        # If graph mode is active but daemon_stats is empty AND Python-side
        # counters are also zero, there's nothing useful to print — skip.
        pairs_total = snap.get("cpu_offload_pairs_total", 0)
        pairs_cpu = snap.get("cpu_offload_pairs_to_cpu", 0)
        if graph_mode and pairs_total == 0 and pairs_cpu == 0:
            return
        pairs_gpu = snap.get("cpu_offload_pairs_to_gpu", 0)
        combine_ok = snap.get("cpu_offload_combine_success_pairs", 0)
        combine_bad = snap.get("cpu_offload_combine_fail_pairs", 0)
        commit_ok = snap.get("cpu_offload_commit_success_count", 0)
        commit_bad = snap.get("cpu_offload_commit_fail_count", 0)
        hit = snap.get("cpu_offload_hit_count", 0)
        miss = snap.get("cpu_offload_miss_count", 0)
        # Daemon request tracking: sent = all dispatched calls; exception = hard
        # failures (timeout / SHM error) in commit path; the rest succeeded.
        d_sent = snap.get("cpu_daemon_req_sent", hit)  # fallback to hit_count
        d_exc = snap.get("cpu_daemon_exception_count", 0)
        d_ok = d_sent - d_exc
        d_succ_rate = (100.0 * d_ok / d_sent) if d_sent else 0.0
        # Combine success rate at pair (token-expert) granularity.
        c_total = combine_ok + combine_bad
        c_rate = (100.0 * combine_ok / c_total) if c_total else 0.0
        pct_cpu = (100.0 * pairs_cpu / pairs_total) if pairs_total else 0.0

        # Use plain print so this lands in benchmark stdout regardless of
        # logging config / handler flushing during interpreter teardown.
        graph_suffix = ""
        if graph_mode:
            graph_suffix = (
                f"  [CUDA GRAPH MODE: {n_graph_layers} layers captured; "
                f"metrics only reflect capture-time counts, not replay]"
            )
        print(
            f"[cpu_offload][rank={rt.local_rank}] METRICS{graph_suffix}\n"
            f"  [DAEMON REQUESTS]  sent={d_sent}  "
            f"ok={d_ok}  exception={d_exc}  "
            f"success_rate={d_succ_rate:.1f}%\n"
            f"  [COMBINE]         "
            f"calls_ok={commit_ok}  calls_fail={commit_bad}  "
            f"pairs_ok={combine_ok}  pairs_nan={combine_bad}  "
            f"success_rate={c_rate:.1f}%\n"
            f"  [ROUTING]         "
            f"total_pairs={pairs_total}  to_gpu={pairs_gpu}  to_cpu={pairs_cpu}  "
            f"cpu_hit_rate={pct_cpu:.2f}%  "
            f"hit_layers={hit}  miss_layers={miss}",
            flush=True,
        )

    try:
        atexit.register(_dump)
    except Exception:
        pass


def _build_config_from_server_args(server_args) -> CpuOffloadConfig:
    from sglang.srt.eplb.cpu_offload.config import load_cpu_offload_config

    return load_cpu_offload_config(
        enable_cpu_offload_fast_loop=bool(
            getattr(server_args, "enable_cpu_offload_fast_loop", False)
        ),
        cpu_active_ranks=getattr(server_args, "cpu_active_ranks", None),
        cpu_offload_rank_to_numa_map=getattr(
            server_args, "cpu_offload_rank_to_numa_map", None
        ),
        cpu_offload_worker_assign_strategy=getattr(
            server_args, "cpu_offload_worker_assign_strategy", None
        ),
        cpu_offload_rank_to_worker_map=getattr(
            server_args, "cpu_offload_rank_to_worker_map", None
        ),
        cpu_offload_use_sub_numa_allocation=getattr(
            server_args, "cpu_offload_use_sub_numa_allocation", None
        ),
        cpu_offload_reserved_sub_numa_per_socket=getattr(
            server_args, "cpu_offload_reserved_sub_numa_per_socket", None
        ),
        cpu_offload_reserved_sub_numa_ids=getattr(
            server_args, "cpu_offload_reserved_sub_numa_ids", None
        ),
        cpu_offload_top_m=getattr(server_args, "cpu_offload_top_m", None),
        cpu_offload_slot_update_interval_ms=getattr(
            server_args, "cpu_offload_slot_update_interval_ms", None
        ),
        cpu_offload_enable_mode_a=getattr(
            server_args, "cpu_offload_enable_mode_a", None
        ),
        cpu_offload_ratio=getattr(server_args, "cpu_offload_ratio", None),
        cpu_offload_overflow_policy=getattr(
            server_args, "cpu_offload_overflow_policy", None
        ),
        cpu_offload_max_tokens_per_step=getattr(
            server_args, "cpu_offload_max_tokens_per_step", None
        ),
        cpu_offload_max_tokens_per_expert=getattr(
            server_args, "cpu_offload_max_tokens_per_expert", None
        ),
        cpu_offload_num_workers_per_numa=getattr(
            server_args, "cpu_offload_num_workers_per_numa", None
        ),
        cpu_offload_reserved_cores_per_numa=getattr(
            server_args, "cpu_offload_reserved_cores_per_numa", None
        ),
        cpu_offload_shm_prefix=getattr(server_args, "cpu_offload_shm_prefix", None),
        cpu_offload_amx_enabled=getattr(server_args, "cpu_offload_amx_enabled", None),
        cpu_offload_enable_cuda_graph_polling=getattr(
            server_args, "cpu_offload_enable_cuda_graph_polling", None
        ),
        cpu_offload_timeout_us=getattr(server_args, "cpu_offload_timeout_us", None),
        cpu_offload_max_spin=getattr(server_args, "cpu_offload_max_spin", None),
        cpu_offload_backend=getattr(server_args, "cpu_offload_backend", None),
        cpu_offload_daemon_launcher=getattr(
            server_args, "cpu_offload_daemon_launcher", None
        ),
    )


def reset_cpu_offload_runtime_for_tests() -> None:
    """Drop the process singleton. Tests only."""
    global _runtime
    with _runtime_lock:
        _runtime = None
