"""Lightweight metric counters for the CPU offload path.

The hot path must NEVER call logging or expensive observability code
(see PERFORMANCE REQUIREMENTS). This module only mutates plain
integers and floats. A periodic flusher (driven from the slow loop)
can drain them into Prometheus / scheduler stats out of band.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class CpuOffloadMetrics:
    """Mutable counters shared by the dispatcher, engine, and fast loop.

    All updates are O(1) and use a single lock only for batched reads
    (snapshot). Hot-path writes are unsynchronized integer adds; we
    accept the rare ordering tear because totals are advisory.
    """

    # --- token accounting ---
    cpu_offload_tokens_total: int = 0  # = pairs routed to CPU (token x cpu_slot)
    cpu_offload_tokens_by_expert: dict[int, int] = field(default_factory=dict)
    cpu_offload_tokens_by_rank: dict[int, int] = field(default_factory=dict)
    # Total (token, expert) pairs seen by begin_cpu_offload across all
    # MoE layer calls (= sum of topk_ids.numel()). GPU-side pairs are
    # derived as total - cpu_offload_tokens_total in the snapshot.
    cpu_offload_pairs_total: int = 0
    # Combine accounting: pairs successfully written back from CPU vs lost.
    cpu_offload_combine_success_pairs: int = 0
    cpu_offload_combine_fail_pairs: int = 0
    cpu_offload_commit_success_count: int = 0
    cpu_offload_commit_fail_count: int = 0
    # Daemon request accounting: requests sent vs exceptions during commit.
    # An exception (launch_wait_and_copyout raised) usually means timeout or
    # SHM communication failure — the daemon did not respond in time or the
    # transfer failed. NaN in the output is NOT an exception; it is counted
    # only in combine_fail_pairs. Success rate = (sent - exception) / sent.
    cpu_daemon_req_sent: int = 0  # incremented in begin_cpu_offload
    cpu_daemon_exception_count: int = 0  # incremented when commit raises

    # --- timing / quality ---
    cpu_offload_roundtrip_us_sum: float = 0.0
    cpu_offload_roundtrip_us_count: int = 0
    cpu_offload_roundtrip_us_p99: float = 0.0
    cpu_offload_gpu_wait_us_sum: float = 0.0
    cpu_offload_gpu_wait_us_count: int = 0
    cpu_offload_overlap_ratio_sum: float = 0.0
    cpu_offload_overlap_ratio_count: int = 0
    cpu_offload_net_speedup_pct: float = 0.0

    # --- health / control ---
    cpu_offload_timeout_count: int = 0
    cpu_offload_wait_cycles: int = 0
    cpu_offload_hit_count: int = 0
    cpu_offload_miss_count: int = 0
    cpu_offload_demoted_slots_count: int = 0
    cpu_offload_pcie_bytes_per_step: int = 0
    cpu_offload_expert_slots: int = 0

    # --- daemon / worker side ---
    cpu_daemon_queue_depth: int = 0
    cpu_daemon_latency_us_sum: float = 0.0
    cpu_daemon_latency_us_count: int = 0
    cpu_worker_busy_ratio: float = 0.0
    cpu_active_ranks_count: int = 0
    gpu_tokens_saved_by_cpu: int = 0

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------
    # Hot-path mutators (no locking — additive only)
    # ------------------------------------------------------------------

    def record_offload(self, rank_id: int, expert_id: int, n_tokens: int) -> None:
        self.cpu_offload_tokens_total += n_tokens
        self.cpu_offload_tokens_by_expert[expert_id] = (
            self.cpu_offload_tokens_by_expert.get(expert_id, 0) + n_tokens
        )
        self.cpu_offload_tokens_by_rank[rank_id] = (
            self.cpu_offload_tokens_by_rank.get(rank_id, 0) + n_tokens
        )
        self.gpu_tokens_saved_by_cpu += n_tokens
        self.cpu_offload_hit_count += 1

    def record_skip(self) -> None:
        self.cpu_offload_miss_count += 1

    def record_pairs_seen(self, n_pairs: int) -> None:
        """Count every (token, expert) pair entering begin_cpu_offload."""
        if n_pairs > 0:
            self.cpu_offload_pairs_total += int(n_pairs)

    def record_combine_success(self, n_pairs: int) -> None:
        self.cpu_offload_commit_success_count += 1
        if n_pairs > 0:
            self.cpu_offload_combine_success_pairs += int(n_pairs)

    def record_combine_fail(self, n_pairs: int) -> None:
        self.cpu_offload_commit_fail_count += 1
        if n_pairs > 0:
            self.cpu_offload_combine_fail_pairs += int(n_pairs)

    def record_roundtrip_us(self, us: float) -> None:
        self.cpu_offload_roundtrip_us_sum += us
        self.cpu_offload_roundtrip_us_count += 1
        if us > self.cpu_offload_roundtrip_us_p99:
            # Cheap running-max as a P99 proxy. A real implementation
            # would use a t-digest; intentionally avoided on hot path.
            self.cpu_offload_roundtrip_us_p99 = us

    def record_gpu_wait_us(self, us: float) -> None:
        self.cpu_offload_gpu_wait_us_sum += us
        self.cpu_offload_gpu_wait_us_count += 1

    def record_overlap_ratio(self, ratio: float) -> None:
        self.cpu_offload_overlap_ratio_sum += ratio
        self.cpu_offload_overlap_ratio_count += 1

    def record_timeout(self) -> None:
        self.cpu_offload_timeout_count += 1

    def record_daemon_req_sent(self) -> None:
        """Increment when begin_cpu_offload dispatches a request to daemon."""
        self.cpu_daemon_req_sent += 1

    def record_daemon_exception(self) -> None:
        """Increment when commit_cpu_offload's wait/copyout raises an exception."""
        self.cpu_daemon_exception_count += 1

    def record_demotion(self) -> None:
        self.cpu_offload_demoted_slots_count += 1

    # ------------------------------------------------------------------
    # Out-of-band snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a stable dict of all counters; safe to call from any thread."""
        with self._lock:
            return {
                "cpu_offload_tokens_total": self.cpu_offload_tokens_total,
                "cpu_offload_tokens_by_expert": dict(self.cpu_offload_tokens_by_expert),
                "cpu_offload_tokens_by_rank": dict(self.cpu_offload_tokens_by_rank),
                "cpu_offload_roundtrip_us_mean": (
                    self.cpu_offload_roundtrip_us_sum
                    / self.cpu_offload_roundtrip_us_count
                    if self.cpu_offload_roundtrip_us_count
                    else 0.0
                ),
                "cpu_offload_roundtrip_us_p99": self.cpu_offload_roundtrip_us_p99,
                "cpu_offload_gpu_wait_us_mean": (
                    self.cpu_offload_gpu_wait_us_sum
                    / self.cpu_offload_gpu_wait_us_count
                    if self.cpu_offload_gpu_wait_us_count
                    else 0.0
                ),
                "cpu_offload_overlap_ratio_mean": (
                    self.cpu_offload_overlap_ratio_sum
                    / self.cpu_offload_overlap_ratio_count
                    if self.cpu_offload_overlap_ratio_count
                    else 0.0
                ),
                "cpu_offload_net_speedup_pct": self.cpu_offload_net_speedup_pct,
                "cpu_offload_timeout_count": self.cpu_offload_timeout_count,
                "cpu_offload_wait_cycles": self.cpu_offload_wait_cycles,
                "cpu_offload_hit_count": self.cpu_offload_hit_count,
                "cpu_offload_miss_count": self.cpu_offload_miss_count,
                "cpu_offload_pairs_total": self.cpu_offload_pairs_total,
                "cpu_offload_pairs_to_cpu": self.cpu_offload_tokens_total,
                "cpu_offload_pairs_to_gpu": max(
                    0,
                    self.cpu_offload_pairs_total - self.cpu_offload_tokens_total,
                ),
                "cpu_offload_combine_success_pairs": self.cpu_offload_combine_success_pairs,
                "cpu_offload_combine_fail_pairs": self.cpu_offload_combine_fail_pairs,
                "cpu_offload_commit_success_count": self.cpu_offload_commit_success_count,
                "cpu_offload_commit_fail_count": self.cpu_offload_commit_fail_count,
                "cpu_daemon_req_sent": self.cpu_daemon_req_sent,
                "cpu_daemon_exception_count": self.cpu_daemon_exception_count,
                "cpu_offload_demoted_slots_count": self.cpu_offload_demoted_slots_count,
                "cpu_offload_pcie_bytes_per_step": self.cpu_offload_pcie_bytes_per_step,
                "cpu_offload_expert_slots": self.cpu_offload_expert_slots,
                "cpu_daemon_queue_depth": self.cpu_daemon_queue_depth,
                "cpu_daemon_latency_us_mean": (
                    self.cpu_daemon_latency_us_sum / self.cpu_daemon_latency_us_count
                    if self.cpu_daemon_latency_us_count
                    else 0.0
                ),
                "cpu_worker_busy_ratio": self.cpu_worker_busy_ratio,
                "cpu_active_ranks_count": self.cpu_active_ranks_count,
                "gpu_tokens_saved_by_cpu": self.gpu_tokens_saved_by_cpu,
            }

    def reset(self) -> None:
        """Reset all counters. Intended for tests."""
        with self._lock:
            self.cpu_offload_tokens_total = 0
            self.cpu_offload_tokens_by_expert.clear()
            self.cpu_offload_tokens_by_rank.clear()
            self.cpu_offload_roundtrip_us_sum = 0.0
            self.cpu_offload_roundtrip_us_count = 0
            self.cpu_offload_roundtrip_us_p99 = 0.0
            self.cpu_offload_gpu_wait_us_sum = 0.0
            self.cpu_offload_gpu_wait_us_count = 0
            self.cpu_offload_overlap_ratio_sum = 0.0
            self.cpu_offload_overlap_ratio_count = 0
            self.cpu_offload_net_speedup_pct = 0.0
            self.cpu_offload_timeout_count = 0
            self.cpu_offload_wait_cycles = 0
            self.cpu_offload_hit_count = 0
            self.cpu_offload_miss_count = 0
            self.cpu_offload_pairs_total = 0
            self.cpu_daemon_req_sent = 0
            self.cpu_daemon_exception_count = 0
            self.cpu_offload_combine_success_pairs = 0
            self.cpu_offload_combine_fail_pairs = 0
            self.cpu_offload_commit_success_count = 0
            self.cpu_offload_commit_fail_count = 0
            self.cpu_offload_demoted_slots_count = 0
            self.cpu_offload_pcie_bytes_per_step = 0
            self.cpu_offload_expert_slots = 0
            self.cpu_daemon_queue_depth = 0
            self.cpu_daemon_latency_us_sum = 0.0
            self.cpu_daemon_latency_us_count = 0
            self.cpu_worker_busy_ratio = 0.0
            self.cpu_active_ranks_count = 0
            self.gpu_tokens_saved_by_cpu = 0
