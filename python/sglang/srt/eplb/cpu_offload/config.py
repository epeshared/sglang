"""CPU offload configuration.

Holds all knobs for the dual-loop EPLB CPU offload mechanism. Values are
sourced (in priority order) from:

1. Explicit constructor arguments (typically from ``ServerArgs``).
2. Environment variables (the ``CPU_*`` / ``CPU_OFFLOAD_*`` aliases listed
   in the design doc).
3. The hard-coded defaults below.

The class is intentionally a plain dataclass with explicit validation so
the configuration surface stays auditable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional


class OverflowPolicy(str, Enum):
    """What to do when Mode B handoff exceeds per-step caps."""

    DROP = "drop"
    ERROR = "error"
    BLOCK = "block"


_DEFAULT_RESERVED_CORES_PER_NUMA = 8
_DEFAULT_NUM_WORKERS_PER_NUMA = -1  # -1 = auto = number of active ranks on the NUMA
_DEFAULT_OFFLOAD_RATIO = 0.2
_DEFAULT_TOP_M = 1
_DEFAULT_SLOT_UPDATE_INTERVAL_MS = 100
# How many forward passes between CPU-slot re-selections.
#
# Re-selection used to be gated on wall-clock time, which made the whole run
# non-reproducible: two runs never spend the same number of milliseconds on
# the same decode step, so "which forward pass re-selected" differed, the
# offloaded expert changed at a different step, and greedy decoding eventually
# flipped a token. Counting forward passes removes that: tick() is called
# exactly once per forward pass on every rank, so the counter is identical
# across ranks and across runs.
#
# 100 is the measured equivalent of the previous 500 ms default: on the
# 94-layer Qwen3-235B TP8 benchmark, re-selection fired at forward passes
# ~2 / ~107 / ~200 / ~303, i.e. roughly every 100 passes.
_DEFAULT_SLOT_UPDATE_EVERY_N_FORWARDS = 100
_DEFAULT_XRANK_SYNC_EVERY_N_TICKS = 100
_DEFAULT_METRICS_INTERVAL_MS = 1000
_DEFAULT_TIMEOUT_US = 5_000_000
_DEFAULT_MAX_SPIN = 100_000
_DEFAULT_MAX_TOKENS_PER_STEP = 65536
_DEFAULT_MAX_TOKENS_PER_EXPERT = 4096
_DEFAULT_SHM_PREFIX = "/dev/shm/sglang_cpu_offload"


def _env_str(name: str, default: Optional[str]) -> Optional[str]:
    v = os.environ.get(name)
    return v if v is not None else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError as exc:
        raise ValueError(f"env {name}={v!r} is not an integer") from exc


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")


def _parse_int_list(s: Optional[str]) -> Optional[list[int]]:
    if s is None or s == "":
        return None
    out: list[int] = []
    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        out.append(int(piece))
    return out


def _parse_rank_to_numa(s: Optional[str]) -> Optional[dict[int, int]]:
    """Parse a ``rank:numa,rank:numa,...`` string into a dict.

    Returns ``None`` if ``s`` is empty/None.
    """
    if s is None or s == "":
        return None
    out: dict[int, int] = {}
    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        k, _, v = piece.partition(":")
        out[int(k)] = int(v)
    return out


def _parse_rank_to_worker(s: Optional[str]) -> Optional[dict[int, int]]:
    """Parse a ``rank:slot,rank:slot,...`` string into a dict.

    Used for custom worker assignment strategy.
    Returns ``None`` if ``s`` is empty/None.
    """
    if s is None or s == "":
        return None
    out: dict[int, int] = {}
    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        k, _, v = piece.partition(":")
        out[int(k)] = int(v)
    return out


@dataclass
class CpuOffloadConfig:
    """Configuration for the dual-loop EPLB + CPU offload mechanism."""

    # General
    enable_cpu_offload_fast_loop: bool = False
    cpu_offload_decode_only: bool = True
    cpu_active_ranks: Optional[list[int]] = None
    cpu_offload_rank_to_numa_map: Optional[dict[int, int]] = None

    # Fast loop (selector)
    cpu_offload_top_m: int = _DEFAULT_TOP_M
    # Kept for one purpose only: 0 still means "never re-select". Any
    # non-zero value only means "re-selection is enabled" -- the cadence now
    # comes from cpu_offload_slot_update_every_n_forwards. FastLoopScheduler
    # logs a loud one-shot notice when a non-zero value is supplied, so the
    # change of meaning can never be silent.
    cpu_offload_slot_update_interval_ms: int = _DEFAULT_SLOT_UPDATE_INTERVAL_MS
    cpu_offload_slot_update_every_n_forwards: int = (
        _DEFAULT_SLOT_UPDATE_EVERY_N_FORWARDS
    )
    cpu_offload_metrics_interval_ms: int = _DEFAULT_METRICS_INTERVAL_MS
    # How often (in tick() calls, i.e. forward passes — NOT wall-clock ms)
    # the cross-rank hot-expert all_reduce actually runs. 1 = every tick
    # (freshest data, most communication). The underlying per-layer counter
    # is cumulative, so raising this loses no traffic data — it only
    # batches more forward-passes' worth of delta into each collective
    # call, trading data freshness for fewer all_reduce round trips. Safe
    # to raise because it is gated by a per-process call counter (always
    # incremented in lockstep across ranks), never by wall-clock time.
    cpu_offload_xrank_sync_every_n_ticks: int = _DEFAULT_XRANK_SYNC_EVERY_N_TICKS

    # Token split (Mode A is off by default on L20)
    cpu_offload_enable_mode_a: bool = False
    cpu_offload_ratio: float = _DEFAULT_OFFLOAD_RATIO
    cpu_offload_overflow_policy: OverflowPolicy = OverflowPolicy.ERROR
    cpu_offload_max_tokens_per_step: int = _DEFAULT_MAX_TOKENS_PER_STEP
    cpu_offload_max_tokens_per_expert: int = _DEFAULT_MAX_TOKENS_PER_EXPERT

    # Daemon
    cpu_offload_num_workers_per_numa: int = _DEFAULT_NUM_WORKERS_PER_NUMA
    cpu_offload_reserved_cores_per_numa: int = _DEFAULT_RESERVED_CORES_PER_NUMA
    cpu_offload_shm_prefix: str = _DEFAULT_SHM_PREFIX
    cpu_offload_amx_enabled: bool = True
    # Sub-NUMA allocation mode
    cpu_offload_use_sub_numa_allocation: bool = False
    cpu_offload_reserved_sub_numa_per_socket: int = 0
    cpu_offload_reserved_sub_numa_ids: str = ""
    # Worker assignment strategy: "order" (default) or "custom"
    # order: rank i -> worker slot i % num_workers (round-robin when overflow)
    # custom: use cpu_offload_rank_to_worker_map for explicit mapping
    cpu_offload_worker_assign_strategy: str = "order"
    # Custom rank-to-worker mapping, format: "rank0:slot0,rank1:slot1,..."
    # Only used when cpu_offload_worker_assign_strategy="custom"
    cpu_offload_rank_to_worker_map: Optional[dict[int, int]] = None
    # Backend selector: "daemon" attaches to standalone cpu_expertd (default,
    # validated in Phase 3.1-3.3). "legacy" forks NUMA workers in-process via
    # moe_cpu_engine.numa_initialize (kept for regression / debugging only).
    cpu_offload_backend: str = "daemon"
    # Path to launch_cpu_expertd.sh; used by the daemon backend when it must
    # spawn cpu_expertd itself. Empty string means "assume already running".
    cpu_offload_daemon_launcher: str = ""

    # CUDA graph polling
    cpu_offload_enable_cuda_graph_polling: bool = True
    cpu_offload_timeout_us: int = _DEFAULT_TIMEOUT_US
    cpu_offload_max_spin: int = _DEFAULT_MAX_SPIN

    # Performance auto-disable thresholds (see PERFORMANCE REQUIREMENTS).
    cpu_offload_demote_latency_us: int = 2_000
    cpu_offload_demote_window: int = 16

    # Free-form extras for testing / forward compat.
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._normalize_and_validate()

    def _normalize_and_validate(self) -> None:
        if isinstance(self.cpu_offload_overflow_policy, str):
            self.cpu_offload_overflow_policy = OverflowPolicy(
                self.cpu_offload_overflow_policy
            )

        if not isinstance(self.cpu_offload_ratio, float):
            self.cpu_offload_ratio = float(self.cpu_offload_ratio)
        if not 0.0 < self.cpu_offload_ratio < 1.0:
            raise ValueError(
                f"cpu_offload_ratio must be in (0,1), got {self.cpu_offload_ratio}"
            )

        if self.cpu_offload_reserved_cores_per_numa < 0:
            raise ValueError(
                "cpu_offload_reserved_cores_per_numa must be >= 0, got "
                f"{self.cpu_offload_reserved_cores_per_numa}"
            )

        if self.cpu_offload_top_m < 1:
            raise ValueError(
                f"cpu_offload_top_m must be >= 1, got {self.cpu_offload_top_m}"
            )

        if self.cpu_offload_slot_update_interval_ms < 0:
            raise ValueError(
                "cpu_offload_slot_update_interval_ms must be >= 0, got "
                f"{self.cpu_offload_slot_update_interval_ms}"
            )

        if self.cpu_offload_slot_update_every_n_forwards < 0:
            raise ValueError(
                "cpu_offload_slot_update_every_n_forwards must be >= 0, got "
                f"{self.cpu_offload_slot_update_every_n_forwards}"
            )

        if self.cpu_offload_xrank_sync_every_n_ticks < 1:
            raise ValueError(
                "cpu_offload_xrank_sync_every_n_ticks must be >= 1, got "
                f"{self.cpu_offload_xrank_sync_every_n_ticks}"
            )

        if self.cpu_offload_max_tokens_per_step < 0:
            raise ValueError("cpu_offload_max_tokens_per_step must be >= 0")
        if self.cpu_offload_max_tokens_per_expert < 0:
            raise ValueError("cpu_offload_max_tokens_per_expert must be >= 0")

        if self.cpu_offload_timeout_us < 0:
            raise ValueError("cpu_offload_timeout_us must be >= 0")
        if self.cpu_offload_max_spin < 1:
            raise ValueError("cpu_offload_max_spin must be >= 1")

        if self.cpu_offload_backend not in ("daemon", "legacy"):
            raise ValueError(
                f"cpu_offload_backend must be 'daemon' or 'legacy', "
                f"got {self.cpu_offload_backend!r}"
            )

        if self.cpu_offload_worker_assign_strategy not in ("order", "custom"):
            raise ValueError(
                f"cpu_offload_worker_assign_strategy must be 'order' or 'custom', "
                f"got {self.cpu_offload_worker_assign_strategy!r}"
            )

        if self.cpu_offload_rank_to_worker_map is not None:
            self.cpu_offload_rank_to_worker_map = {
                int(k): int(v) for k, v in self.cpu_offload_rank_to_worker_map.items()
            }
            if any(v < 0 for v in self.cpu_offload_rank_to_worker_map.values()):
                raise ValueError("rank-to-worker slot values must be non-negative")

        if self.cpu_active_ranks is not None:
            cleaned = sorted(set(int(r) for r in self.cpu_active_ranks))
            if any(r < 0 for r in cleaned):
                raise ValueError(
                    f"cpu_active_ranks must be non-negative, got {self.cpu_active_ranks}"
                )
            self.cpu_active_ranks = cleaned

        if self.cpu_offload_rank_to_numa_map is not None:
            self.cpu_offload_rank_to_numa_map = {
                int(k): int(v) for k, v in self.cpu_offload_rank_to_numa_map.items()
            }
            if any(v < 0 for v in self.cpu_offload_rank_to_numa_map.values()):
                raise ValueError("rank-to-numa values must be non-negative")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        return self.enable_cpu_offload_fast_loop and bool(self.cpu_active_ranks)

    def rank_is_active(self, rank_id: int) -> bool:
        if not self.is_active():
            return False
        return rank_id in (self.cpu_active_ranks or ())

    def numa_for_rank(
        self, rank_id: int, *, world_size: int, num_numa_nodes: int
    ) -> int:
        """Return the NUMA node id for ``rank_id``.

        Priority: explicit map > even split of ranks across NUMA nodes.
        """
        if self.cpu_offload_rank_to_numa_map is not None:
            try:
                return self.cpu_offload_rank_to_numa_map[rank_id]
            except KeyError as exc:
                raise KeyError(
                    f"rank {rank_id} not present in cpu_offload_rank_to_numa_map"
                ) from exc
        if num_numa_nodes <= 0:
            raise ValueError("num_numa_nodes must be > 0")
        if world_size <= 0:
            raise ValueError("world_size must be > 0")
        ranks_per_numa = max(1, world_size // num_numa_nodes)
        return min(num_numa_nodes - 1, rank_id // ranks_per_numa)

    def active_ranks_on_numa(
        self, numa_id: int, *, world_size: int, num_numa_nodes: int
    ) -> list[int]:
        if not self.is_active():
            return []
        return [
            r
            for r in (self.cpu_active_ranks or ())
            if self.numa_for_rank(
                r, world_size=world_size, num_numa_nodes=num_numa_nodes
            )
            == numa_id
        ]

    def validate_slot_mapping(
        self,
        num_numa_nodes: int,
        world_size: int,
        num_workers_per_numa: int,
    ) -> None:
        """Validate that active ranks can be mapped to available worker slots.

        When using sub-NUMA allocation with reserved sub-NUMA nodes, the number
        of worker slots per NUMA may be less than the number of active ranks.
        This method checks that the mapping is valid and raises if not.

        Args:
            num_numa_nodes: Total number of NUMA nodes (sockets)
            world_size: Total number of GPU ranks (TP world size)
            num_workers_per_numa: Number of worker slots per NUMA (from daemon)

        Raises:
            RuntimeError: If active ranks exceed available slots on any NUMA
        """
        if not self.is_active():
            return

        for numa_id in range(num_numa_nodes):
            active_ranks = self.active_ranks_on_numa(
                numa_id, world_size=world_size, num_numa_nodes=num_numa_nodes
            )
            num_active = len(active_ranks)

            if num_active > num_workers_per_numa:
                raise RuntimeError(
                    f"[cpu_offload] CONFIG MISMATCH: NUMA {numa_id} has "
                    f"{num_active} active ranks but only {num_workers_per_numa} "
                    f"worker slots available. Reduce active ranks or increase "
                    f"workers per NUMA. Active ranks on this NUMA: {active_ranks}"
                )

    def compute_worker_slot(
        self,
        rank_id: int,
        *,
        world_size: int,
        num_numa_nodes: int,
        num_workers_per_numa: int,
    ) -> Optional[int]:
        """Compute the worker slot for a given rank.

        Strategy "order" (default):
            Direct 1:1 mapping: rank position -> worker slot.
            If rank position >= num_workers, returns None (no worker available).

        Strategy "custom":
            Uses the explicit cpu_offload_rank_to_worker_map if provided.
            Returns None if rank not in custom map (no worker assigned).

        Args:
            rank_id: The GPU rank id
            world_size: Total number of GPU ranks
            num_numa_nodes: Total number of NUMA nodes
            num_workers_per_numa: Number of worker slots per NUMA

        Returns:
            The worker slot id (0-indexed within the NUMA), or None if no
            worker is assigned to this rank (i.e., rank should skip CPU offload).
        """
        if not self.is_active():
            return 0

        if self.cpu_offload_worker_assign_strategy == "custom":
            if self.cpu_offload_rank_to_worker_map is not None:
                if rank_id in self.cpu_offload_rank_to_worker_map:
                    return self.cpu_offload_rank_to_worker_map[rank_id]
                return None

        # Default "order" strategy: direct 1:1 mapping
        local_numa = self.numa_for_rank(
            rank_id, world_size=world_size, num_numa_nodes=num_numa_nodes
        )
        active_on_numa = self.active_ranks_on_numa(
            local_numa, world_size=world_size, num_numa_nodes=num_numa_nodes
        )

        try:
            rank_position = active_on_numa.index(rank_id)
        except ValueError:
            rank_position = 0

        # Direct mapping: if position >= workers, no worker available
        if rank_position >= num_workers_per_numa:
            return None

        return rank_position


def load_cpu_offload_config(
    *,
    enable_cpu_offload_fast_loop: Optional[bool] = None,
    cpu_active_ranks: Optional[Iterable[int] | str] = None,
    cpu_offload_rank_to_numa_map: Optional[dict[int, int] | str] = None,
    cpu_offload_worker_assign_strategy: Optional[str] = None,
    cpu_offload_rank_to_worker_map: Optional[dict[int, int] | str] = None,
    cpu_offload_num_workers_per_numa: Optional[int] = None,
    cpu_offload_reserved_cores_per_numa: Optional[int] = None,
    cpu_offload_use_sub_numa_allocation: Optional[bool] = None,
    cpu_offload_reserved_sub_numa_per_socket: Optional[int] = None,
    cpu_offload_reserved_sub_numa_ids: Optional[str] = None,
    cpu_offload_ratio: Optional[float] = None,
    cpu_offload_enable_mode_a: Optional[bool] = None,
    cpu_offload_overflow_policy: Optional[str] = None,
    cpu_offload_top_m: Optional[int] = None,
    cpu_offload_slot_update_interval_ms: Optional[int] = None,
    cpu_offload_slot_update_every_n_forwards: Optional[int] = None,
    cpu_offload_xrank_sync_every_n_ticks: Optional[int] = None,
    cpu_offload_max_tokens_per_step: Optional[int] = None,
    cpu_offload_max_tokens_per_expert: Optional[int] = None,
    cpu_offload_timeout_us: Optional[int] = None,
    cpu_offload_max_spin: Optional[int] = None,
    cpu_offload_shm_prefix: Optional[str] = None,
    cpu_offload_amx_enabled: Optional[bool] = None,
    cpu_offload_enable_cuda_graph_polling: Optional[bool] = None,
    cpu_offload_backend: Optional[str] = None,
    cpu_offload_daemon_launcher: Optional[str] = None,
) -> CpuOffloadConfig:
    """Build a ``CpuOffloadConfig`` honoring CLI args first, then env vars.

    All ``None`` arguments fall back to environment variables (when defined)
    and then the dataclass default.
    """

    def _coerce_ranks(v):
        if v is None:
            return None
        if isinstance(v, str):
            return _parse_int_list(v)
        return list(v)

    def _coerce_map(v):
        if v is None:
            return None
        if isinstance(v, str):
            return _parse_rank_to_numa(v)
        return dict(v)

    return CpuOffloadConfig(
        enable_cpu_offload_fast_loop=(
            enable_cpu_offload_fast_loop
            if enable_cpu_offload_fast_loop is not None
            else _env_bool("CPU_OFFLOAD_ENABLE_FAST_LOOP", False)
        ),
        cpu_active_ranks=_coerce_ranks(
            cpu_active_ranks
            if cpu_active_ranks is not None
            else _env_str("CPU_ACTIVE_RANKS", None)
        ),
        cpu_offload_rank_to_numa_map=_coerce_map(
            cpu_offload_rank_to_numa_map
            if cpu_offload_rank_to_numa_map is not None
            else _env_str("CPU_OFFLOAD_RANK_TO_NUMA_MAP", None)
        ),
        cpu_offload_worker_assign_strategy=(
            cpu_offload_worker_assign_strategy
            if cpu_offload_worker_assign_strategy is not None
            else _env_str("CPU_OFFLOAD_WORKER_ASSIGN_STRATEGY", "order")
        ),
        cpu_offload_rank_to_worker_map=_coerce_map(
            cpu_offload_rank_to_worker_map
            if cpu_offload_rank_to_worker_map is not None
            else _env_str("CPU_OFFLOAD_RANK_TO_WORKER_MAP", None)
        ),
        cpu_offload_num_workers_per_numa=(
            cpu_offload_num_workers_per_numa
            if cpu_offload_num_workers_per_numa is not None
            else _env_int(
                "CPU_OFFLOAD_NUM_WORKERS_PER_NUMA", _DEFAULT_NUM_WORKERS_PER_NUMA
            )
        ),
        cpu_offload_reserved_cores_per_numa=(
            cpu_offload_reserved_cores_per_numa
            if cpu_offload_reserved_cores_per_numa is not None
            else _env_int(
                "CPU_OFFLOAD_RESERVED_CORES_PER_NUMA",
                _DEFAULT_RESERVED_CORES_PER_NUMA,
            )
        ),
        cpu_offload_ratio=(
            cpu_offload_ratio
            if cpu_offload_ratio is not None
            else float(_env_str("CPU_OFFLOAD_RATIO", str(_DEFAULT_OFFLOAD_RATIO)))
        ),
        cpu_offload_enable_mode_a=(
            cpu_offload_enable_mode_a
            if cpu_offload_enable_mode_a is not None
            else _env_bool("CPU_OFFLOAD_ENABLE_MODE_A", False)
        ),
        cpu_offload_overflow_policy=(
            cpu_offload_overflow_policy
            if cpu_offload_overflow_policy is not None
            else _env_str("CPU_OFFLOAD_OVERFLOW_POLICY", "error")
        ),
        cpu_offload_top_m=(
            cpu_offload_top_m
            if cpu_offload_top_m is not None
            else _env_int("CPU_OFFLOAD_TOP_M", _DEFAULT_TOP_M)
        ),
        cpu_offload_slot_update_interval_ms=(
            cpu_offload_slot_update_interval_ms
            if cpu_offload_slot_update_interval_ms is not None
            else _env_int(
                "CPU_OFFLOAD_SLOT_UPDATE_INTERVAL_MS",
                _DEFAULT_SLOT_UPDATE_INTERVAL_MS,
            )
        ),
        cpu_offload_slot_update_every_n_forwards=(
            cpu_offload_slot_update_every_n_forwards
            if cpu_offload_slot_update_every_n_forwards is not None
            else _env_int(
                "CPU_OFFLOAD_SLOT_UPDATE_EVERY_N_FORWARDS",
                _DEFAULT_SLOT_UPDATE_EVERY_N_FORWARDS,
            )
        ),
        cpu_offload_xrank_sync_every_n_ticks=(
            cpu_offload_xrank_sync_every_n_ticks
            if cpu_offload_xrank_sync_every_n_ticks is not None
            else _env_int(
                "CPU_OFFLOAD_XRANK_SYNC_EVERY_N_TICKS",
                _DEFAULT_XRANK_SYNC_EVERY_N_TICKS,
            )
        ),
        cpu_offload_max_tokens_per_step=(
            cpu_offload_max_tokens_per_step
            if cpu_offload_max_tokens_per_step is not None
            else _env_int(
                "CPU_OFFLOAD_MAX_TOKENS_PER_STEP", _DEFAULT_MAX_TOKENS_PER_STEP
            )
        ),
        cpu_offload_max_tokens_per_expert=(
            cpu_offload_max_tokens_per_expert
            if cpu_offload_max_tokens_per_expert is not None
            else _env_int(
                "CPU_OFFLOAD_MAX_TOKENS_PER_EXPERT",
                _DEFAULT_MAX_TOKENS_PER_EXPERT,
            )
        ),
        cpu_offload_timeout_us=(
            cpu_offload_timeout_us
            if cpu_offload_timeout_us is not None
            else _env_int("CPU_OFFLOAD_TIMEOUT_US", _DEFAULT_TIMEOUT_US)
        ),
        cpu_offload_max_spin=(
            cpu_offload_max_spin
            if cpu_offload_max_spin is not None
            else _env_int("CPU_OFFLOAD_MAX_SPIN", _DEFAULT_MAX_SPIN)
        ),
        cpu_offload_shm_prefix=(
            cpu_offload_shm_prefix
            if cpu_offload_shm_prefix is not None
            else _env_str("CPU_OFFLOAD_SHM_PREFIX", _DEFAULT_SHM_PREFIX)
            or _DEFAULT_SHM_PREFIX
        ),
        cpu_offload_amx_enabled=(
            cpu_offload_amx_enabled
            if cpu_offload_amx_enabled is not None
            else _env_bool("CPU_OFFLOAD_AMX_ENABLED", True)
        ),
        cpu_offload_enable_cuda_graph_polling=(
            cpu_offload_enable_cuda_graph_polling
            if cpu_offload_enable_cuda_graph_polling is not None
            else _env_bool("CPU_OFFLOAD_ENABLE_CUDA_GRAPH_POLLING", True)
        ),
        cpu_offload_backend=(
            cpu_offload_backend
            if cpu_offload_backend is not None
            else (_env_str("CPU_OFFLOAD_BACKEND", "daemon") or "daemon")
        ),
        cpu_offload_daemon_launcher=(
            cpu_offload_daemon_launcher
            if cpu_offload_daemon_launcher is not None
            else (_env_str("CPU_OFFLOAD_DAEMON_LAUNCHER", "") or "")
        ),
        cpu_offload_use_sub_numa_allocation=(
            cpu_offload_use_sub_numa_allocation
            if cpu_offload_use_sub_numa_allocation is not None
            else _env_bool("CPU_OFFLOAD_USE_SUBNUMA_ALLOCATION", False)
        ),
        cpu_offload_reserved_sub_numa_per_socket=(
            cpu_offload_reserved_sub_numa_per_socket
            if cpu_offload_reserved_sub_numa_per_socket is not None
            else _env_int("CPU_OFFLOAD_RESERVED_SUBNUMA_PER_SOCKET", 0)
        ),
        cpu_offload_reserved_sub_numa_ids=(
            cpu_offload_reserved_sub_numa_ids
            if cpu_offload_reserved_sub_numa_ids is not None
            else (_env_str("CPU_OFFLOAD_RESERVED_SUBNUMA_IDS", "") or "")
        ),
    )
