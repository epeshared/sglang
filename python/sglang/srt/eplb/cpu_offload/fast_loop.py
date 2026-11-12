"""Fast loop: per-rank CPU slot selection.

Runs inline in the scheduler thread (no extra OS thread) and is ticked
by ``EPLBManager.on_forward_pass_end``. Each tick:

1. Folds this forward pass's per-layer expert traffic into a running
   total that is summed ACROSS ALL RANKS via ``torch.distributed.all_reduce``
   (see ``_sync_layer_counts_cross_rank``) — every rank ranks experts by
   the SAME cluster-wide hotness for a given layer, not just its own local
   slice of the traffic.
2. For each active rank, walks that layer's cross-rank hotness ranking
   from hottest to coldest, keeping ONLY experts that are local to that
   rank's own GPU, and picks the first (top-M, M defaults to 1 per design
   CONFIGURATION KNOBS) remaining candidate. Local is required, not
   avoided: the CPU result is combined before the TP all-reduce, so an
   expert this rank does not own would still be computed on its owner's
   GPU and end up counted twice. See Phase 2 in _run_once.
3. Applies demotion: if a rank's EWMA latency exceeds
   ``cpu_offload_demote_latency_us`` for ``cpu_offload_demote_window``
   consecutive samples, force its slot to -1.
4. Writes the result through CpuShadowRegistry.set_layer_slots.

The fast loop NEVER assigns experts to inactive ranks and NEVER touches
GPU expert slots — it only manipulates the CPU shadow slot per rank.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)

from sglang.srt.eplb.cpu_offload import trace as _trace
from sglang.srt.eplb.cpu_offload.config import CpuOffloadConfig
from sglang.srt.eplb.cpu_offload.metrics import CpuOffloadMetrics
from sglang.srt.eplb.cpu_offload.shadow_state import (
    CpuShadowRegistry,
)

# Resolved once, at import -- see trace.py's "COST WHEN DISABLED".
_MOE_TRACE = _trace.ENABLED


def _trace_basis(
    layer_expert_counts: dict, rank_expert_tokens: dict, with_counts: bool = False
) -> dict:
    """Fingerprint the traffic a re-selection was decided from. Trace only.

    The journal without this records what was chosen but not what it was
    chosen from, and those two failures need opposite responses: identical
    traffic with a different outcome means the scheduler is nondeterministic,
    while different traffic means the divergence is already upstream, in
    batching or routing, and the scheduler is faithfully reflecting it.

    The digest covers the counts as integers, per layer, in layer order. It
    is deliberately NOT the argsort the selector uses: ties in the traffic
    resolve to an arbitrary order, so digesting the order would report a
    difference where the input was in fact identical.

    ``with_counts`` additionally keeps the numbers themselves under
    ``per_layer``. The digest can only say whether two runs saw the same
    traffic; it cannot say how close the top two experts were, and that
    margin is what separates real hot-expert movement from a selector
    flipping between two experts of near-equal load. Keeping the values adds
    no device synchronisation -- ``.tolist()`` below has already brought them
    across -- only journal size, which is why the caller gates it.

    Cost: one extra device-to-host copy per layer, on a path that already
    performs several (``argsort(...).tolist()`` and a ``.item()`` per
    candidate), at re-selection cadence -- not per forward, and never during
    CUDA-graph capture. It runs only behind ``if _MOE_TRACE``, so a default
    run does not pay it. It must not, and does not, feed anything the
    selector reads.
    """
    h = hashlib.sha1()
    pairs = 0
    n_layers = 0
    per_layer = {} if with_counts else None
    if layer_expert_counts:
        for layer_id in sorted(layer_expert_counts):
            vals = [int(x) for x in layer_expert_counts[layer_id].tolist()]
            if per_layer is not None:
                per_layer[layer_id] = vals
            pairs += sum(vals)
            n_layers += 1
            h.update(b"%d:" % layer_id)
            h.update(",".join(str(v) for v in vals).encode())
            h.update(b";")
        src = "counts"
    elif rank_expert_tokens:
        for layer_id in sorted(rank_expert_tokens):
            per_rank = rank_expert_tokens[layer_id]
            n_layers += 1
            h.update(b"%d:" % layer_id)
            if per_layer is not None:
                per_layer[layer_id] = {}
            for rank_id in sorted(per_rank):
                for eid in sorted(per_rank[rank_id]):
                    tok = int(per_rank[rank_id][eid])
                    pairs += tok
                    if per_layer is not None:
                        # Flat "rank/expert" keys: the fallback path is
                        # per-rank, unlike the cross-rank sum above, and
                        # collapsing that away here would silently change
                        # what the charts are plotting depending on which
                        # path the run happened to take.
                        per_layer[layer_id]["%d/%d" % (rank_id, eid)] = tok
                    h.update(b"%d/%d=%d," % (rank_id, eid, tok))
            h.update(b";")
        src = "snapshot"
    else:
        return {"src": "none", "layers": 0, "pairs": 0, "digest": ""}
    out = {
        "src": src,
        "layers": n_layers,
        "pairs": pairs,
        "digest": h.hexdigest()[:16],
    }
    if per_layer:
        out["per_layer"] = per_layer
    return out


# Type alias: snapshot callable returning a dict
#   { layer_id: { (rank_id, expert_id): token_count, ... }, ... }
# Provided by the caller — typically a thin adapter around
# ExpertDistributionRecorder.
LoadSnapshotFn = Callable[[], dict]


@dataclass
class _RankPlan:
    rank_id: int
    chosen_expert_id: int  # -1 means "no offload this round"
    score: float


class FastLoopScheduler:
    """Selects CPU slot experts per rank on a fixed-time tick."""

    def __init__(
        self,
        config: CpuOffloadConfig,
        registry: CpuShadowRegistry,
        metrics: CpuOffloadMetrics,
        *,
        load_snapshot_fn: LoadSnapshotFn,
        local_expert_set_fn: Callable[[int, int], set[int]],
    ) -> None:
        """
        Parameters
        ----------
        load_snapshot_fn:
            Returns the latest per-layer, per-(rank,expert) token counts.
        local_expert_set_fn:
            ``(rank_id, layer_id) -> set of logical expert ids local on GPU``.
            The fast loop uses this to ensure CPU never duplicates an
            already-local expert (which would waste PCIe with no relief).
        """
        self._config = config
        self._registry = registry
        self._metrics = metrics
        self._load_snapshot_fn = load_snapshot_fn
        self._local_expert_set_fn = local_expert_set_fn

        self._lock = threading.Lock()
        # ---- Re-selection cadence: FORWARD PASSES, not wall-clock ----
        # Gating on time made the whole run non-reproducible. Two runs never
        # spend the same milliseconds on the same decode step, so the step at
        # which the offloaded expert changed differed between runs; from that
        # step on the hidden state differed, and greedy decoding eventually
        # flipped a token. Counting tick() calls fixes both properties at
        # once: tick() is called exactly once per forward pass on EVERY rank
        # (from EPLBManager.on_forward_pass_end), so the counter is identical
        # across ranks at any given step and identical across runs. This is
        # the same argument _sync_layer_counts_cross_rank already relies on --
        # see its docstring.
        #
        # cpu_offload_slot_update_interval_ms keeps exactly one meaning:
        # 0 == disabled. A non-zero value only enables re-selection; it no
        # longer sets the cadence. That change of meaning is announced below
        # rather than applied silently.
        self._select_every_n = (
            0
            if int(config.cpu_offload_slot_update_interval_ms) == 0
            else max(1, int(config.cpu_offload_slot_update_every_n_forwards))
        )
        # Value of _tick_count when the selector last ran. Starting at 0
        # (with the bootstrap re-arming it) is what keeps the selector from
        # firing on the very first call and clobbering the init-time slot
        # assignment, e.g. SGLANG_CPU_OFFLOAD_INIT_SLOT.
        self._last_select_tick = 0
        if int(config.cpu_offload_slot_update_interval_ms) != 0:
            _notice = (
                "[cpu_offload] NOTICE: cpu_offload_slot_update_interval_ms=%d "
                "is no longer a wall-clock interval. Re-selection now fires "
                "every %d FORWARD PASSES (deterministic and identical on every "
                "rank); the old time-based gate made runs non-reproducible. "
                "Only 0 still has its original meaning (never re-select). Set "
                "CPU_OFFLOAD_SLOT_UPDATE_EVERY_N_FORWARDS to change the "
                "cadence."
                % (
                    int(config.cpu_offload_slot_update_interval_ms),
                    self._select_every_n,
                )
            )
            logger.warning("%s", _notice)
            # TP worker logging is not reliably visible in this deployment
            # (see the METRICS dump in runtime.py for the same reasoning), and
            # a silent semantic change is exactly what this must not be.
            print(_notice, flush=True)

        # Per-rank consecutive "slow" sample count for demotion.
        self._slow_count: dict[int, int] = {}

        # One-shot debug knob: apply SGLANG_CPU_OFFLOAD_INIT_SLOT(S) on the
        # first tick instead of at registry construction time. That way MoE
        # layers see cpu_slot=-1 during their own __init__.
        self._init_slot_pending = True

        # Snapshot of CudaGraphOffloadContext.get_layer_expert_counts(layer_id)
        # at the last tick — used to compute the LOCAL delta (this rank's own
        # traffic since the previous tick), keyed per layer_id. This is
        # purely local bookkeeping; it is never itself used for ranking (see
        # `_global_count_accum` below).
        self._expert_count_snap: dict[int, torch.Tensor] = {}

        # Cross-rank-summed delta accumulated since the last time
        # ``_run_once()`` consumed it for a ranking decision. Populated by
        # ``_sync_layer_counts_cross_rank()``, which runs every
        # ``cpu_offload_xrank_sync_every_n_ticks`` ``tick()`` calls, gated
        # by ``_tick_count`` (a per-process counter, NOT wall-clock time)
        # so the underlying ``all_reduce`` is always issued in lockstep
        # across ranks — see the comment on that method for why this
        # matters.
        self._global_count_accum: dict[int, torch.Tensor] = {}

        # Per-process count of tick() calls made while the fast loop is
        # active. Used ONLY to gate how often the cross-rank all_reduce
        # runs (see above) — never compared against wall-clock time, so it
        # cannot drift between ranks the way a wall-clock read can.
        self._tick_count = 0

        # Trace-only: a re-selection counter that, unlike ``_dbg_tick_n``,
        # advances even when CPU_OFFLOAD_LOG_FAST_LOOP_TICK is silenced.
        self._trace_reselect_n = 0

    def _lookup_num_experts(self) -> Optional[int]:
        """Derive the global logical expert count from EPLB metadata.

        Returns ``None`` if the metadata is not yet available; the caller
        will retry on a subsequent tick.
        """
        try:
            from sglang.srt.eplb.expert_location import (
                get_global_expert_location_metadata,
            )

            meta = get_global_expert_location_metadata()
            if meta is None:
                return None
            n = int(getattr(meta, "num_logical_experts", 0) or 0)
            if n > 0:
                return n
            n_local = int(getattr(meta, "num_local_physical_experts", 0) or 0)
            if n_local > 0 and self._registry._world_size > 0:
                return n_local * self._registry._world_size
        except Exception:
            return None
        return None

    def _lookup_num_layers(self) -> Optional[int]:
        """Derive the global MoE layer count from EPLB metadata.

        Identical on every rank (the metadata is fully replicated), unlike
        ``gctx.registered_layer_ids()`` which only reflects the layers THIS
        rank happens to have a CPU-offload daemon registered for. Used to
        size the fixed-shape cross-rank all_reduce tensor so every rank in
        the process group always participates with the same shape,
        regardless of its own local daemon/registration state.
        """
        try:
            from sglang.srt.eplb.expert_location import (
                get_global_expert_location_metadata,
            )

            meta = get_global_expert_location_metadata()
            if meta is None:
                return None
            n = int(meta.physical_to_logical_map.shape[0])
            return n if n > 0 else None
        except Exception:
            return None

    def _dump_slot_layout(self, num_experts: int) -> None:
        """One-shot diagnostic: dump per-rank cpu_slot ids + GPU phys_to_logical
        for layer 0, and verify every active rank's cpu_slot logical id
        also exists on some other rank's GPU (i.e. no expert is "lost").
        """
        try:
            from sglang.srt.eplb.expert_location import (
                get_global_expert_location_metadata,
            )

            meta = get_global_expert_location_metadata()
        except Exception:
            meta = None
        if meta is None:
            logger.info("[cpu_offload][slot_dump] no EPLB metadata available")
            return
        try:
            num_local = int(meta.num_local_physical_experts)
            phys = meta.physical_to_logical_map  # (layers, num_phys)
            layer0 = phys[0].tolist()
        except Exception as exc:
            logger.info("[cpu_offload][slot_dump] meta read failed: %s", exc)
            return
        ws = self._registry._world_size
        gpu_per_rank = {
            r: layer0[r * num_local : (r + 1) * num_local] for r in range(ws)
        }
        cpu_slots = {}
        for r in range(ws):
            st = self._registry.get(r)
            cpu_slots[r] = st.get_slot(0) if (st and st.active) else -1
        logger.info(
            "[cpu_offload][slot_dump] world=%d num_experts=%d num_local=%d",
            ws,
            num_experts,
            num_local,
        )
        for r in range(ws):
            cs = cpu_slots[r]
            if cs < 0:
                where = "n/a"
                lost = False
            else:
                holders = [r2 for r2, lst in gpu_per_rank.items() if cs in lst]
                where = ",".join(map(str, holders)) if holders else "NONE"
                lost = len(holders) == 0
            logger.info(
                "[cpu_offload][slot_dump] rank=%d gpu_logical=%s cpu_slot=%d "
                "cpu_slot_also_on_gpu_rank=[%s]%s",
                r,
                gpu_per_rank[r],
                cs,
                where,
                "  <<< LOST!" if lost else "",
            )
        all_gpu = set()
        for lst in gpu_per_rank.values():
            for e in lst:
                if e >= 0:
                    all_gpu.add(int(e))
        missing_global = sorted(set(range(num_experts)) - all_gpu)
        logger.info(
            "[cpu_offload][slot_dump] experts_on_no_GPU=%s",
            missing_global if missing_global else "(none)",
        )

    # ------------------------------------------------------------------
    # Cross-rank statistics sync (called from tick(), gated by a
    # per-process tick COUNTER — never by wall-clock time)
    # ------------------------------------------------------------------

    def _sync_layer_counts_cross_rank(self) -> None:
        """Fold accumulated per-layer expert traffic into a cross-rank
        (all-reduced) running total.

        Design note — WHY this is gated by a per-process CALL COUNTER
        (``self._tick_count`` in ``tick()``):

        ``torch.distributed.all_reduce`` is a blocking collective: every
        rank in the process group must call it, in the same order, or the
        call hangs forever. A wall-clock gate cannot provide that: two ranks
        cross a time boundary on different forward passes (clock skew,
        scheduling jitter), so one rank could enter the collective while
        another is still mid-forward-pass — a deadlock.

        The selection decision in ``tick()`` is now gated the SAME way, on
        ``_tick_count`` (it used to use a wall-clock interval, which is what
        made the offloaded expert change at a different decode step on every
        run and hence made whole runs non-reproducible). Both gates therefore
        rest on the identical argument below.

        ``tick()`` itself, however, IS called exactly once per forward pass
        on every rank (from ``EPLBManager.on_forward_pass_end``), and
        forward passes are already a synchronization point (TP/EP
        collectives inside the model). So a plain per-process call counter
        (incremented unconditionally on every ``tick()``, never reset by
        anything rank-local/racy) is IDENTICAL across ranks at any given
        step — gating on ``self._tick_count % N == 0`` is therefore just as
        deadlock-safe as running every tick, but lets ``N``
        (``cpu_offload_xrank_sync_every_n_ticks``) trade off communication
        overhead against data freshness. Raising ``N`` loses no traffic
        data: the per-layer counter this reads from is cumulative, so a
        skipped tick's traffic is simply included in the next sync's delta.
        The (wall-clock gated) decision logic in ``_run_once()`` then just
        drains whatever has accumulated here so far; it never triggers
        communication itself.

        CRITICAL: the tensor SHAPE and the decision to call ``all_reduce``
        at all must NEVER depend on any fact that can differ between ranks
        at runtime (e.g. whether THIS rank happens to have a CPU-offload
        daemon attached, or whether ``gctx`` has any layers registered on
        THIS rank). Not every rank in the TP process group necessarily has
        a local daemon/registered layers — e.g. with
        ``CPU_ACTIVE_RANKS_PER_SOCKET`` set, only a subset of ranks get
        real ``cuda_graph_offload`` registrations while the rest stay
        ``daemon=absent``. If those "inactive" ranks skip the collective
        while the "active" ones still call it, the active ranks hang
        forever — this previously caused a real deadlock in production.
        So we always determine ``(num_layers, num_slots)`` from EPLB
        metadata (identical on every rank, independent of local daemon
        state) and always issue the same all_reduce, contributing zeros
        for any layer this rank has no local data for.
        """
        num_layers = self._lookup_num_layers()
        num_slots = self._lookup_num_experts()
        if not num_layers or not num_slots:
            return

        try:
            from sglang.srt.eplb.cpu_offload.cuda_graph_offload import (
                get_cuda_graph_offload_ctx,
            )

            gctx = get_cuda_graph_offload_ctx()
        except Exception:
            gctx = None

        # Build this rank's LOCAL delta-since-last-tick, stacked into a
        # single fixed-shape [num_layers, num_slots] CPU tensor (zero rows
        # for any layer with no local registration) so the cross-rank
        # reduction is ONE collective call instead of one per layer.
        #
        # NOTE: this loop is wrapped in its own try/except that can only
        # ever leave `stacked` partially/not filled — it must NEVER be
        # allowed to raise past this point and skip the all_reduce call
        # below. If it did, and the failure is data-dependent (e.g. only
        # ranks with local registrations hit it), some ranks would call
        # all_reduce while others silently skipped it — the same class of
        # deadlock as the daemon-presence bug this method was already
        # hardened against, just triggered by an exception instead of an
        # early return. Every rank must reach the all_reduce call below,
        # no matter what happens while gathering local data.
        stacked = torch.zeros((num_layers, num_slots), dtype=torch.float32)
        try:
            if gctx is not None and gctx.initialized:
                for layer_id in gctx.registered_layer_ids():
                    if not (0 <= layer_id < num_layers):
                        continue
                    acc = gctx.get_layer_expert_counts(layer_id)
                    if acc is None:
                        continue
                    current = acc.cpu()  # D2H — safe, outside CUDA graph capture.
                    snap = self._expert_count_snap.get(layer_id)
                    delta = (
                        current.float()
                        if snap is None
                        else (current - snap).float().clamp(min=0)
                    )
                    self._expert_count_snap[layer_id] = current
                    # `acc` is an oversized fixed-size buffer (see
                    # cuda_graph_offload.py: allocated at a constant
                    # capacity "safely larger than any realistic expert
                    # count", NOT sized to `num_slots`/num_experts). Valid
                    # logical expert ids are always < num_slots, so the
                    # tail is always zero — truncate (never pad) to match
                    # the fixed all_reduce shape.
                    n = min(delta.numel(), num_slots)
                    stacked[layer_id, :n] = delta[:n]
        except Exception as exc:
            logger.warning(
                "[cpu_offload] building local delta for cross-rank sync "
                "failed, contributing zeros this tick: %s",
                exc,
            )
            stacked.zero_()

        # Cross-rank sum. Skipped (falls back to local-only counts) only
        # when there is no process group to reduce over at all (unit
        # tests, single-rank run) — a condition that is identical on every
        # rank, unlike local daemon/registration state above.
        try:
            import torch.distributed as dist

            if (
                dist.is_available()
                and dist.is_initialized()
                and self._registry._world_size > 1
            ):
                from sglang.srt.distributed.parallel_state import get_tp_group

                dist.all_reduce(
                    stacked, op=dist.ReduceOp.SUM, group=get_tp_group().cpu_group
                )
        except Exception as exc:
            logger.warning(
                "[cpu_offload] cross-rank count all_reduce failed, "
                "falling back to local-only counts this tick: %s",
                exc,
            )

        for layer_id in range(num_layers):
            prev = self._global_count_accum.get(layer_id)
            self._global_count_accum[layer_id] = (
                stacked[layer_id].clone() if prev is None else prev + stacked[layer_id]
            )

    # ------------------------------------------------------------------
    # Tick entry (called from EPLBManager.on_forward_pass_end)
    # ------------------------------------------------------------------

    def tick(self, *, now_ns: Optional[int] = None, force: bool = False) -> bool:
        """Run one fast-loop iteration if the interval has elapsed.

        Returns True if the loop actually ran.
        """
        if not self._config.is_active():
            return False

        if _MOE_TRACE:
            _trace.hit("loop.tick")

        # Gated by a per-process CALL COUNTER (never wall-clock time — see
        # the docstring on _sync_layer_counts_cross_rank for why that
        # distinction is what keeps this deadlock-safe). Every rank calls
        # tick() exactly once per forward pass, so this counter is always
        # identical across ranks at any given step; skipping most ticks
        # loses no traffic data because the underlying per-layer counter is
        # cumulative (the delta just covers more forward passes when synced
        # less often) — it only trades data freshness for fewer all_reduce
        # round trips.
        self._tick_count += 1
        if (
            self._tick_count % max(1, self._config.cpu_offload_xrank_sync_every_n_ticks)
            == 0
        ):
            if _MOE_TRACE:
                _trace.hit("loop.xrank_sync")
            try:
                self._sync_layer_counts_cross_rank()
            except Exception as exc:
                if _MOE_TRACE:
                    _trace.exc("fast_loop.tick/xrank_sync", exc)
                logger.warning("[cpu_offload] cross-rank count sync failed: %s", exc)

        # Always run the one-shot bootstrap on the very first tick attempt,
        # regardless of interval. Subsequent ticks fall through to the
        # normal interval check.
        if self._init_slot_pending:
            with self._lock:
                if self._init_slot_pending:
                    num_experts = self._lookup_num_experts()
                    try:
                        applied = self._registry.apply_initial_slots(num_experts)
                    except Exception as exc:
                        logger.warning(
                            "[cpu_offload] apply_initial_slots failed: %s", exc
                        )
                        applied = False
                    # Only clear the pending flag once we've actually pinned
                    # something. If num_experts isn't known yet we will
                    # retry on the next tick.
                    if applied:
                        self._init_slot_pending = False
                        self._registry.bump_plan_epoch()
                        # Re-arm the gate so the normal selector does not
                        # immediately overwrite the just-applied init slot:
                        # the first real selection happens _select_every_n
                        # forward passes from here.
                        self._last_select_tick = self._tick_count
                        # Push the initial per-layer slots for the local rank
                        # into the CUDA graph device buffers so that replays
                        # start routing to the correct expert immediately.
                        # Without this, each layer's cpu_slot buffer stays at
                        # the 0x7FFF sentinel (no-match) until the first
                        # _run_once() assigns one.
                        try:
                            local_rank = self._registry._local_rank
                            if local_rank is not None:
                                local_st = self._registry._ranks.get(local_rank)
                                if local_st is not None and local_st.cpu_slot_by_layer:
                                    from sglang.srt.eplb.cpu_offload.cuda_graph_offload import (
                                        get_cuda_graph_offload_ctx,
                                    )

                                    gctx = get_cuda_graph_offload_ctx()
                                    if gctx is not None and gctx.initialized:
                                        for (
                                            layer_id,
                                            expert_id,
                                        ) in local_st.cpu_slot_by_layer.items():
                                            if expert_id >= 0:
                                                gctx.update_slot(layer_id, expert_id)
                        except Exception:
                            pass
                        if _MOE_TRACE:
                            _trace.hit("loop.bootstrap")
                            try:
                                _lr = self._registry._local_rank
                                _xr = {}
                                for _r, _st in self._registry._ranks.items():
                                    _sl = dict(_st.cpu_slot_by_layer)
                                    if _sl:
                                        _xr[_r] = [
                                            (lid, -1, eid) for lid, eid in _sl.items()
                                        ]
                                _trace.reselect(
                                    event="bootstrap",
                                    tick=0,
                                    forward_no=self._tick_count,
                                    source="apply_initial_slots",
                                    changes=_xr.get(_lr, []),
                                    xrank=_xr,
                                    rank=_lr,
                                )
                            except Exception as _bexc:
                                _trace.exc("fast_loop.tick/bootstrap-trace", _bexc)
                        logger.info(
                            "[cpu_offload] initial CPU slots installed "
                            "(num_experts=%s) on first tick",
                            num_experts,
                        )
                        try:
                            self._dump_slot_layout(num_experts)
                        except Exception as exc:
                            logger.warning(
                                "[cpu_offload] _dump_slot_layout failed: %s", exc
                            )
                        return True

        if self._select_every_n <= 0 and not force:
            return False

        # Forward-pass gate. ``now_ns`` is accepted for API compatibility but
        # deliberately no longer consulted -- see __init__ for why time must
        # not decide this.
        if (
            not force
            and (self._tick_count - self._last_select_tick) < self._select_every_n
        ):
            return False

        with self._lock:
            if (
                not force
                and (self._tick_count - self._last_select_tick) < self._select_every_n
            ):
                return False
            self._last_select_tick = self._tick_count
            try:
                self._run_once()
            except Exception as exc:  # never let the fast loop crash the scheduler
                if _MOE_TRACE:
                    _trace.exc("fast_loop.tick/_run_once", exc)
                logger.warning("[cpu_offload] fast loop tick failed: %s", exc)
                return False
        return True

    # ------------------------------------------------------------------

    def _run_once(self) -> None:
        cfg = self._config
        if _MOE_TRACE:
            _trace.hit("loop.run_once")
        # Debug short-circuit: keep cpu_slot=-1 on every rank, every layer.
        import os as _os

        if _os.environ.get("CPU_OFFLOAD_FORCE_NO_SLOT", "").strip() in (
            "1",
            "true",
            "True",
        ):
            for st in self._registry._ranks.values():
                if st.active:
                    st.clear_slot()
            return
        # When the user requested the "last GPU slot of this rank" pinning
        # strategy, the bootstrap already installed the correct per-layer
        # cpu_slot for each rank from EPLB metadata. Subsequent ticks must NOT
        # overwrite it via the load-based selector — the whole point of
        # the strategy is a static, reproducible layout for correctness
        # validation. Bail out so the registry stays as-bootstrapped.
        if _os.environ.get("CPU_OFFLOAD_USE_LAST_GPU_SLOT", "").strip() in (
            "1",
            "true",
            "True",
            "yes",
            "on",
        ):
            if _MOE_TRACE:
                _trace.hit("loop.run_once.pinned")
            return

        # ----------------------------------------------------------------
        # Phase 1: Obtain per-LAYER, per-expert token counts, SUMMED ACROSS
        # ALL RANKS.
        #
        # The actual GPU-side counting (scatter_add_ inside begin_offload())
        # and the cross-rank all_reduce both happen unconditionally on every
        # ``tick()`` call, in ``_sync_layer_counts_cross_rank()`` — see that
        # method's docstring for why the collective must NOT live here in
        # the interval-gated path (deadlock risk). By the time we get here,
        # ``self._global_count_accum[layer_id]`` already holds this layer's
        # traffic summed over every rank in the TP group, accumulated since
        # the LAST time this method drained it. We drain (read + reset to
        # zero) it now so the ranking below reflects genuinely cluster-wide
        # "hottest expert for this layer", not just this rank's own slice —
        # this is what lets every rank agree on the same hotness order for
        # a given layer and only diverge based on which experts are already
        # local to THEIR OWN GPU (see Phase 2).
        #
        # Fallback: drain the Python-side _load_snapshot_fn counter, which
        # is already naturally bucketed per layer_id (this legacy path
        # remains per-rank-local; it's only used when the CUDA-graph
        # counters aren't available at all, e.g. during warm-up).
        # ----------------------------------------------------------------
        layer_expert_counts: dict[
            int, torch.Tensor
        ] = {}  # layer_id -> [N] float, cross-rank sum
        gctx = None
        try:
            from sglang.srt.eplb.cpu_offload.cuda_graph_offload import (
                get_cuda_graph_offload_ctx,
            )

            gctx = get_cuda_graph_offload_ctx()
        except Exception:
            gctx = None

        if gctx is not None and gctx.initialized:
            for layer_id, accum in list(self._global_count_accum.items()):
                if accum.sum().item() > 0:
                    layer_expert_counts[layer_id] = accum.clone()
                # Reset now that it has been drained for this decision.
                self._global_count_accum[layer_id] = torch.zeros_like(accum)

        # Fallback: legacy Python-side counter (only populated during warm-up).
        snapshot: dict = {}
        per_layer_rank_expert_tokens: dict[int, dict[int, dict[int, int]]] = {}
        if not layer_expert_counts:
            snapshot = self._load_snapshot_fn() or {}
            for layer_id, per_pair in snapshot.items():
                bucket = per_layer_rank_expert_tokens.setdefault(layer_id, {})
                for (rank_id, expert_id), tokens in per_pair.items():
                    if expert_id < 0:
                        continue
                    d = bucket.setdefault(rank_id, {})
                    d[expert_id] = d.get(expert_id, 0) + int(tokens)

        # Guard: bail out only if BOTH data sources are empty. When the
        # primary cuda_graph_acc path succeeds, `layer_expert_counts` is
        # populated and `snapshot`/`per_layer_rank_expert_tokens` are
        # intentionally left empty (the legacy branch is skipped) — that
        # must NOT be treated as "no evidence". Checking only the legacy
        # snapshot here would cause every tick to bail out early whenever
        # the primary path had valid data, permanently freezing cpu_slot at
        # the bootstrap assignment and disabling hot-expert switching. When
        # SGLANG_CPU_OFFLOAD_RECORD_LOAD=0 (default) and the cuda graph
        # counters have no traffic yet, both are empty and we still bail
        # out to avoid clobbering the bootstrap slots with -1.
        if not layer_expert_counts and not per_layer_rank_expert_tokens:
            if _MOE_TRACE:
                _trace.hit("loop.run_once.no_data")
            return

        # 1. Apply demotion based on latest latency EWMA.
        self._apply_demotion()

        # ----------------------------------------------------------------
        # Phase 2: For each LAYER independently, for each active rank pick
        # the hottest expert (for THAT layer) that IS one of that rank's own
        # GPU experts. This is what lets every layer offload the expert best
        # suited to ITS OWN traffic instead of all layers sharing one
        # globally-hottest expert.
        #
        # The candidate set used to be the COMPLEMENT of local_set ("never
        # offload an expert I already hold"), which looks reasonable but is
        # backwards for this pipeline. The CPU result is added into
        # final_hidden_states BEFORE the TP all-reduce, and masking cpu_slot
        # to -1 only suppresses the GPU copy on THIS rank. So:
        #   cpu_slot IS mine  -> mask removes the only GPU copy; the CPU
        #                        result replaces it. Counted once. Correct.
        #   cpu_slot NOT mine -> the mask is a no-op (this rank was never
        #                        going to compute it), the OWNING rank still
        #                        computes it on GPU, and this rank's CPU
        #                        result is added on top. Counted TWICE.
        # shadow_state.apply_initial_slots strategy 2.5 documents the same
        # invariant ("cpu_slot ∈ rank's GPU experts ... no double count").
        # ----------------------------------------------------------------
        all_layer_ids = sorted(
            set(layer_expert_counts.keys()) | set(per_layer_rank_expert_tokens.keys())
        )
        layer_assignments: dict[int, dict[int, int]] = {}

        for layer_id in all_layer_ids:
            counts = layer_expert_counts.get(layer_id)
            sorted_experts = (
                counts.argsort(descending=True).tolist() if counts is not None else None
            )
            rank_tokens = per_layer_rank_expert_tokens.get(layer_id, {})

            assignments: dict[int, int] = {}
            for state in self._registry.all_ranks():
                if not state.active or state.demoted:
                    assignments[state.rank_id] = -1
                    continue

                local_set = self._safe_local_set(state.rank_id, layer_id)
                top_m = max(1, cfg.cpu_offload_top_m)

                if counts is not None and sorted_experts is not None:
                    # Recorder path: iterate this layer's experts sorted by
                    # traffic, pick the hottest one this rank actually owns.
                    chosen = -1
                    found = 0
                    for eid in sorted_experts:
                        if eid not in local_set:
                            continue
                        if counts[eid].item() <= 0:
                            break  # remaining experts have zero traffic
                        chosen = int(eid)
                        found += 1
                        if found >= top_m:
                            break
                    assignments[state.rank_id] = chosen
                else:
                    # Legacy fallback path.
                    candidate_counts = rank_tokens.get(state.rank_id, {})
                    ranked = [
                        (eid, cnt)
                        for eid, cnt in candidate_counts.items()
                        if eid in local_set
                    ]
                    ranked.sort(key=lambda x: -x[1])
                    chosen = -1
                    for eid, cnt in ranked[:top_m]:
                        if cnt > 0:
                            chosen = eid
                            break
                    assignments[state.rank_id] = chosen

            layer_assignments[layer_id] = assignments

        # ----------------------------------------------------------------
        # 4. Publish assignments and print results.
        # ----------------------------------------------------------------

        # Capture old per-layer slots BEFORE publishing for the
        # change-indicator.
        old_slots: dict[int, dict[int, int]] = {
            r: dict(st.cpu_slot_by_layer) for r, st in self._registry._ranks.items()
        }

        self._registry.set_layer_slots(layer_assignments)
        self._metrics.cpu_offload_expert_slots = sum(
            1
            for assignments in layer_assignments.values()
            for v in assignments.values()
            if v >= 0
        )
        self._metrics.cpu_active_ranks_count = len(self._registry.active_ranks())

        # ----------------------------------------------------------------
        # 5. Push the local rank's new per-layer slots into the CUDA graph
        #    device buffers.
        # ----------------------------------------------------------------
        local_rank = self._registry._local_rank

        # ---- trace: the full assignment change, as a diffable event ------
        # Records EVERY rank, not just the local one. The defect this exists
        # to expose is double-compute, which is a property of the assignment
        # MATRIX (two ranks targeting one expert), not of any single column
        # of it -- and each rank's report only ever sees its own column.
        # Also fingerprints the traffic the choice was made from, so that a
        # mismatch between two runs can be attributed: same basis + different
        # outcome is the scheduler, different basis is upstream of it.
        if _MOE_TRACE and local_rank is not None:
            try:
                _xrank = {}
                for _r, _prev_r in old_slots.items():
                    _ch_r = [
                        (lid, _prev_r.get(lid, -1), a[_r])
                        for lid, a in layer_assignments.items()
                        if _r in a and a[_r] != _prev_r.get(lid, -1)
                    ]
                    if _ch_r:
                        _xrank[_r] = _ch_r
                _changes = _xrank.get(local_rank, [])
                if _xrank:
                    self._trace_reselect_n += 1
                    _trace.reselect(
                        event="reselect",
                        tick=self._trace_reselect_n,
                        forward_no=self._tick_count,
                        source=(
                            "cuda_graph_acc_xrank"
                            if layer_expert_counts
                            else "counter_fallback"
                        ),
                        changes=_changes,
                        xrank=_xrank,
                        basis=_trace_basis(
                            layer_expert_counts,
                            per_layer_rank_expert_tokens,
                            with_counts=_trace.want_counts(self._trace_reselect_n),
                        ),
                        rank=local_rank,
                    )
            except Exception as _rexc:
                _trace.exc("fast_loop._run_once/reselect-trace", _rexc)

        if gctx is not None and gctx.initialized and local_rank is not None:
            for layer_id, assignments in layer_assignments.items():
                if local_rank in assignments:
                    new_slot = assignments[local_rank]
                    if new_slot >= 0:
                        try:
                            gctx.update_slot(layer_id, new_slot)
                        except Exception:
                            pass

        # ----------------------------------------------------------------
        # Result print — always emitted so slot decisions are observable.
        # Kept to one compact line per tick regardless of layer count (a
        # full per-rank/per-layer dump would flood logs once there are
        # dozens of MoE layers each with their own assignment).
        # CPU_OFFLOAD_LOG_FAST_LOOP_TICK: 1 (default) = print; 0 = silence.
        # ----------------------------------------------------------------
        try:
            import os as _os

            if _os.environ.get("CPU_OFFLOAD_LOG_FAST_LOOP_TICK", "1").strip() in (
                "0",
                "false",
                "False",
                "no",
                "off",
            ):
                return

            tick_n = getattr(self, "_dbg_tick_n", 0) + 1
            self._dbg_tick_n = tick_n

            data_src = (
                "cuda_graph_acc_xrank" if layer_expert_counts else "counter_fallback"
            )

            n_changed = 0
            local_slots_by_layer: dict[int, int] = {}
            for layer_id in all_layer_ids:
                assignments = layer_assignments[layer_id]
                for rank_id, new_e in assignments.items():
                    old_e = old_slots.get(rank_id, {}).get(layer_id, -1)
                    if new_e != old_e:
                        n_changed += 1
                if local_rank is not None and local_rank in assignments:
                    local_slots_by_layer[layer_id] = assignments[local_rank]

            print(
                f"[fast_loop tick#{tick_n}] src={data_src} "
                f"layers_active_this_tick={len(all_layer_ids)} "
                f"slot_changes={n_changed} local_rank={local_rank} "
                f"local_slots_by_layer={local_slots_by_layer}",
                flush=True,
            )
        except Exception as _e:
            print(f"[fast_loop tick] result print error: {_e}", flush=True)

    # ------------------------------------------------------------------

    def _safe_local_set(self, rank_id: int, layer_id: int) -> set[int]:
        try:
            return self._local_expert_set_fn(rank_id, layer_id) or set()
        except Exception:
            return set()

    # ------------------------------------------------------------------

    def _apply_demotion(self) -> None:
        cfg = self._config
        threshold_us = float(cfg.cpu_offload_demote_latency_us)
        window = max(1, int(cfg.cpu_offload_demote_window))
        for state in self._registry.active_ranks():
            if state.latency_ewma_us > threshold_us:
                cnt = self._slow_count.get(state.rank_id, 0) + 1
                self._slow_count[state.rank_id] = cnt
                if cnt >= window and not state.demoted:
                    state.demote()
                    self._metrics.record_demotion()
                    logger.info(
                        "[cpu_offload] demoting rank %d: ewma=%.1fus > %.1fus for %d ticks",
                        state.rank_id,
                        state.latency_ewma_us,
                        threshold_us,
                        cnt,
                    )
            else:
                self._slow_count[state.rank_id] = 0
                if state.demoted:
                    # Cooldown elapsed and latency recovered; allow retry.
                    state.undemote()
