"""Per-rank CPU shadow state.

The shadow state is the GPU-side bookkeeping for an active rank's CPU
offload eligibility. The actual CPU workers live inside the per-NUMA
daemons spawned by ``moe_cpu_engine``; this module only tracks what the
fast loop has decided this rank may offload, and exposes a tiny
device-resident tensor the dispatcher reads on the hot path.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from sglang.srt.eplb.cpu_offload.config import CpuOffloadConfig


@dataclass
class CpuShadowRankState:
    """All bookkeeping for one GPU rank's CPU offload eligibility.

    Inactive ranks must still appear in the registry but with
    ``active=False`` and every layer's slot forever at -1.
    """

    rank_id: int
    numa_id: int
    daemon_id: int
    worker_id: int = -1
    active: bool = False
    # Per-layer CPU slot assignment: {layer_id: expert_id}. A missing
    # layer_id means "no CPU slot assigned yet" (equivalent to -1). This is
    # a dict (NOT a single scalar) so the fast loop can pick an
    # independently hot expert for each MoE layer instead of forcing every
    # layer on this rank to share one global slot.
    cpu_slot_by_layer: dict[int, int] = field(default_factory=dict)
    offload_ratio: float = 0.2

    # SHM / flag names are owned by the engine and stored here for traceability.
    shm_name: str = ""
    request_flag_name: str = ""
    completion_flag_name: str = ""

    # Health: rolling EWMA of CPU step latency in microseconds; used by the
    # fast loop to demote slow slots (see PERFORMANCE REQUIREMENTS).
    _latency_window: deque = field(default_factory=lambda: deque(maxlen=16), repr=False)
    latency_ewma_us: float = 0.0
    last_step_time_us: float = 0.0

    # Demotion flag the fast loop sets; the dispatcher MUST honor it
    # (skip CPU offload) on subsequent steps until the fast loop clears it.
    demoted: bool = False
    demoted_at_us: float = 0.0

    # ------------------------------------------------------------------

    def get_slot(self, layer_id: int) -> int:
        """Currently-assigned CPU slot expert id for ``layer_id``, or -1."""
        return self.cpu_slot_by_layer.get(layer_id, -1)

    def assign_slot(self, layer_id: int, expert_id: int) -> None:
        """Mark a new CPU slot expert for this rank's ``layer_id``.

        Idempotent. ``expert_id == -1`` means "no offload this round".
        """
        if not self.active and expert_id != -1:
            raise RuntimeError(
                f"rank {self.rank_id} is not in CPU_ACTIVE_RANKS; cannot assign slot"
            )
        if expert_id == -1:
            self.cpu_slot_by_layer.pop(layer_id, None)
        else:
            self.cpu_slot_by_layer[layer_id] = expert_id

    def clear_slot(self, layer_id: Optional[int] = None) -> None:
        """Clear one layer's slot, or every layer's slot if ``layer_id`` is None."""
        if layer_id is None:
            self.cpu_slot_by_layer.clear()
        else:
            self.cpu_slot_by_layer.pop(layer_id, None)

    def record_latency_us(self, us: float) -> None:
        self._latency_window.append(us)
        # Cheap EWMA with alpha = 1 / window_size_observed.
        n = len(self._latency_window)
        alpha = 1.0 / max(1, n)
        self.latency_ewma_us = (1.0 - alpha) * self.latency_ewma_us + alpha * us
        self.last_step_time_us = us

    def demote(self) -> None:
        self.demoted = True
        self.demoted_at_us = time.monotonic() * 1e6
        # Demoted slots must not dispatch CPU work; design rule. Clear every
        # layer's slot, not just one.
        self.cpu_slot_by_layer.clear()

    def undemote(self) -> None:
        self.demoted = False
        self.demoted_at_us = 0.0


class CpuShadowRegistry:
    """Process-local registry of per-rank shadow state.

    A single SGLang process typically hosts one GPU rank; the registry
    nevertheless tracks all ranks in the world so the dispatcher and the
    fast loop share a consistent view of which ranks are active.
    """

    def __init__(
        self,
        config: CpuOffloadConfig,
        *,
        world_size: int,
        num_numa_nodes: int,
        local_rank: Optional[int] = None,
    ) -> None:
        self._config = config
        self._world_size = int(world_size)
        self._num_numa_nodes = int(num_numa_nodes)
        self._local_rank = local_rank
        self._lock = threading.Lock()
        self._ranks: dict[int, CpuShadowRankState] = {}
        # 每次 fast loop 发布一整版 per-layer 方案时递增；同一版里的所有层
        # 共用一个 epoch，不能按层各自猜版本。
        self._plan_epoch = 0
        self._build()

    # ------------------------------------------------------------------

    def _build(self) -> None:
        cfg = self._config
        for rank_id in range(self._world_size):
            numa_id = cfg.numa_for_rank(
                rank_id,
                world_size=self._world_size,
                num_numa_nodes=self._num_numa_nodes,
            )
            active = cfg.rank_is_active(rank_id)
            self._ranks[rank_id] = CpuShadowRankState(
                rank_id=rank_id,
                numa_id=numa_id,
                daemon_id=numa_id,
                active=active,
                offload_ratio=cfg.cpu_offload_ratio,
            )

        if cfg.is_active():
            self._assign_workers_per_numa()

    def apply_initial_slots(self, num_experts: Optional[int] = None) -> bool:
        """Install the bootstrap CPU slot assignment on the first fast-loop tick.

        Selection order:

        1. ``SGLANG_CPU_OFFLOAD_INIT_SLOTS="0:16,1:17,..."``
           explicit per-rank override (debug aid).
        2. ``SGLANG_CPU_OFFLOAD_INIT_SLOT="16"``
           same expert id for every active rank (debug aid).
        3. Formula bootstrap (see design.md "INITIAL CPU SLOT
           ASSIGNMENT"): treat the CPU slot as a structural peer of the
           rank's GPU slots, contiguous and wrap-around::

               per_rank_slots = num_experts // world_size + 1
               base[r]        = (r * per_rank_slots) % num_experts
               cpu_slot[r]    = (base[r] + S_gpu)    % num_experts

           where ``S_gpu = num_experts // world_size``.

        Called by the fast loop on its first tick (NOT at construction
        time) so MoE layers see ``cpu_slot=-1`` during their own ``__init__``
        and build ``num_local_experts``/``local_expert_mapping`` from the
        unmodified physical->logical map.

        All strategies below install the SAME bootstrap decision on every
        registered layer (bootstrap runs before any per-layer traffic has
        been observed, so there's no basis to differentiate layers yet).
        The fast loop's regular per-tick selector is the one that later lets
        each layer diverge based on its own hot-expert traffic.

        Returns True if anything was assigned.
        """

        def _num_layers_hint() -> int:
            """Best-effort number of MoE layers, from EPLB metadata."""
            try:
                from sglang.srt.eplb.expert_location import (
                    get_global_expert_location_metadata,
                )

                meta = get_global_expert_location_metadata()
                if meta is not None:
                    n = int(meta.physical_to_logical_map.shape[0])
                    if n > 0:
                        return n
            except Exception:
                pass
            return 1

        # (0) Debug short-circuit: keep cpu_slot=-1 on every rank so the
        # offload path is wired up but never actually steals tokens.
        # Used for correctness validation: with this set, output should
        # be byte-identical to a GPU-only run.
        if os.environ.get("CPU_OFFLOAD_FORCE_NO_SLOT", "").strip() in (
            "1",
            "true",
            "True",
        ):
            for st in self._ranks.values():
                if st.active:
                    st.clear_slot()
            return True

        # (1) Explicit per-rank env override.
        per_rank = os.environ.get("SGLANG_CPU_OFFLOAD_INIT_SLOTS", "").strip()
        if per_rank:
            changed = False
            n_layers = _num_layers_hint()
            for token in per_rank.split(","):
                token = token.strip()
                if not token or ":" not in token:
                    continue
                try:
                    rid_s, eid_s = token.split(":", 1)
                    rid = int(rid_s)
                    eid = int(eid_s)
                except ValueError:
                    continue
                st = self._ranks.get(rid)
                if st is not None and st.active and eid >= 0:
                    for layer_id in range(n_layers):
                        st.assign_slot(layer_id, eid)
                    changed = True
            return changed

        # (2) Flat env override.
        flat = os.environ.get("SGLANG_CPU_OFFLOAD_INIT_SLOT", "").strip()
        if flat:
            try:
                eid = int(flat)
            except ValueError:
                return False
            if eid < 0:
                return False
            n_layers = _num_layers_hint()
            for st in self._ranks.values():
                if st.active:
                    for layer_id in range(n_layers):
                        st.assign_slot(layer_id, eid)
            return True

        # (2.5) Use the last GPU-owned logical expert id of each rank as
        # its cpu_slot. This guarantees ``cpu_slot ∈ rank's GPU experts``,
        # which is the invariant that makes the dispatch hook correct under
        # BOTH EP all-to-all (only owner rank holds the expert) and TP
        # all-reduce (only owner rank's _global_to_local_map maps the id;
        # other ranks already skip it). After mask=-1, no GPU on any rank
        # contributes to the cpu_slot expert, so ``index_add`` of the CPU
        # contribution is non-aliased — no double count, no missing tokens.
        # Pair this with ``CPU_OFFLOAD_SLOT_UPDATE_INTERVAL_MS=0`` to pin
        # the layout for a clean correctness comparison.
        flag_last = os.environ.get("CPU_OFFLOAD_USE_LAST_GPU_SLOT", "").strip()
        if flag_last in ("1", "true", "True", "yes", "on"):
            try:
                from sglang.srt.eplb.expert_location import (
                    get_global_expert_location_metadata,
                )

                meta = get_global_expert_location_metadata()
            except Exception:
                meta = None
            if meta is None:
                # Metadata not yet ready; let the bootstrap retry on the
                # next tick instead of falling through to formula (we want
                # this strategy to be sticky once requested).
                return False
            try:
                num_local = int(meta.num_local_physical_experts)
                phys = meta.physical_to_logical_map  # (layers, num_phys)
            except Exception:
                return False
            if num_local <= 0 or phys.shape[0] == 0:
                return False
            changed = False
            # Compute independently per layer: each layer's physical->logical
            # map can differ (e.g. after EPLB rebalancing), so "last GPU
            # expert of this rank" is not necessarily the same expert id on
            # every layer.
            for layer_id in range(phys.shape[0]):
                layer_row = phys[layer_id].tolist()
                for rid, st in self._ranks.items():
                    if not st.active:
                        continue
                    phys_idx = rid * num_local + (num_local - 1)
                    if 0 <= phys_idx < len(layer_row):
                        eid = int(layer_row[phys_idx])
                        if eid >= 0:
                            st.assign_slot(layer_id, eid)
                            changed = True
            return changed

        # (3) Formula bootstrap. Skip if we don't know num_experts yet —
        # the fast loop will retry on a later tick once metadata is ready.
        if num_experts is None or num_experts <= 0:
            return False
        if self._world_size <= 0:
            return False
        s_gpu = num_experts // self._world_size
        if s_gpu <= 0:
            return False
        # Use the last expert in each rank's own range so that
        # cpu_slot ∈ rank's GPU experts (same invariant as strategy 2.5).
        #
        # The layout is CONTIGUOUS BLOCK: rank r owns global expert ids
        # [r*s_gpu, (r+1)*s_gpu), so its last one is (r+1)*s_gpu - 1.
        #
        # HISTORY — this formula was previously changed to the INTERLEAVED
        # ``rank_id + (s_gpu - 1) * world_size`` on the belief that "the
        # standard EP layout is interleaved: expert e is on rank
        # e % world_size". That belief is wrong for this code path, and the
        # interleaved formula put EVERY rank's cpu_slot in the LAST rank's
        # block (r + (s_gpu-1)*ws lands in [ws*(s_gpu-1), ...] for all r) --
        # precisely the failure the comment claimed to be fixing. Three
        # independent places agree the layout is block:
        #   * FusedMoE._map_global_expert_id_to_local_expert_id -- decides
        #     which expert's WEIGHTS are loaded into which local slot:
        #     start = moe_ep_rank * num_local_experts;
        #   * StandardDispatcher.dispatch -- builds local_expert_mapping over
        #     [moe_ep_rank*n, (moe_ep_rank+1)*n) and sends everything else to
        #     -1, so a routed pair outside the block is never computed here;
        #   * this module's own _dump_slot_layout diagnostic, which printed
        #     ``rank=0 gpu_logical=[0..15] cpu_slot=120
        #     cpu_slot_also_on_gpu_rank=[7]`` on a real 8-rank run -- i.e. it
        #     was already reporting that the interleaved formula sent rank 0's
        #     slot to rank 7's GPU.
        # The interleaved map that motivated the change
        # (DeepEPMoEHeto._create_fast_loop_expert_map -> FusedMoE.
        # _global_to_local_map) is assigned and logged but has NO read site
        # anywhere in python/, so it decides nothing.
        #
        # Getting this wrong is not cosmetic: if cpu_slot is not one of this
        # rank's own experts, masking it to -1 is a no-op here, the rank that
        # DOES own it still computes it on GPU, and this rank adds a CPU
        # result on top -- every hit token is counted twice after the TP
        # all-reduce.
        n_layers = _num_layers_hint()
        changed = False
        for rank_id, st in self._ranks.items():
            if not st.active:
                continue
            cpu_expert = (rank_id + 1) * s_gpu - 1
            for layer_id in range(n_layers):
                st.assign_slot(layer_id, cpu_expert)
            changed = True
        return changed

    # Backward-compat alias for any external caller still using the old name.
    apply_init_slot_env = apply_initial_slots

    def _assign_workers_per_numa(self) -> None:
        """For MVP, NUM_WORKERS per NUMA must equal active ranks on that NUMA."""
        cfg = self._config
        for numa_id in range(self._num_numa_nodes):
            active_ranks = cfg.active_ranks_on_numa(
                numa_id,
                world_size=self._world_size,
                num_numa_nodes=self._num_numa_nodes,
            )
            requested = cfg.cpu_offload_num_workers_per_numa
            if requested == -1:
                num_workers = len(active_ranks)
            else:
                num_workers = requested
                if num_workers < len(active_ranks):
                    raise ValueError(
                        f"NUMA {numa_id}: cpu_offload_num_workers_per_numa="
                        f"{num_workers} < active ranks on this NUMA "
                        f"({len(active_ranks)}). Need workers >= active ranks."
                    )
            for worker_id, rank_id in enumerate(active_ranks):
                self._ranks[rank_id].worker_id = worker_id
                prefix = cfg.cpu_offload_shm_prefix
                self._ranks[rank_id].shm_name = f"{prefix}.rank{rank_id}.shm"
                self._ranks[rank_id].request_flag_name = f"{prefix}.rank{rank_id}.req"
                self._ranks[
                    rank_id
                ].completion_flag_name = f"{prefix}.rank{rank_id}.done"

    # ------------------------------------------------------------------

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def num_numa_nodes(self) -> int:
        return self._num_numa_nodes

    def get(self, rank_id: int) -> CpuShadowRankState:
        return self._ranks[rank_id]

    def all_ranks(self) -> list[CpuShadowRankState]:
        return [self._ranks[r] for r in sorted(self._ranks)]

    def active_ranks(self) -> list[CpuShadowRankState]:
        return [s for s in self.all_ranks() if s.active]

    def local_state(self) -> Optional[CpuShadowRankState]:
        if self._local_rank is None:
            return None
        return self._ranks.get(self._local_rank)

    @property
    def plan_epoch(self) -> int:
        return self._plan_epoch

    def bump_plan_epoch(self) -> int:
        """发布 bootstrap 方案时显式推进版本。"""
        with self._lock:
            self._plan_epoch += 1
            return self._plan_epoch

    # ------------------------------------------------------------------
    # Slot updates (called by the fast loop)
    # ------------------------------------------------------------------

    def set_layer_slots(self, layer_assignments: dict[int, dict[int, int]]) -> None:
        """Apply a fast-loop output ``{layer_id: {rank_id: expert_id_or_-1}}``.

        Strict design rule (see FAST LOOP HOT EXPERT SELECTION), now applied
        per layer:
        - inactive ranks MUST receive -1 on every layer.
        - active demoted ranks MUST receive -1 on every layer (the demotion
          is what tells the fast loop the slot is unsafe).
        - active healthy ranks may receive any expert id or -1, independently
          per layer.
        """
        with self._lock:
            for layer_id, assignments in layer_assignments.items():
                for rank_id, expert_id in assignments.items():
                    if rank_id not in self._ranks:
                        raise KeyError(f"rank {rank_id} not in shadow registry")
                    state = self._ranks[rank_id]
                    if not state.active:
                        if expert_id != -1:
                            raise ValueError(
                                f"rank {rank_id} is inactive; assignment must be -1"
                            )
                        state.clear_slot(layer_id)
                        continue
                    if state.demoted and expert_id != -1:
                        # Silently force -1; the fast loop should not even try.
                        state.clear_slot(layer_id)
                        continue
                    state.assign_slot(layer_id, expert_id)
            self._plan_epoch += 1
