"""Basic tests for the cpu_offload module.

These tests do NOT require moe_cpu_engine to be installed; the engine
import is lazy and only triggered when CpuOffloadEngine.init_runtime is
called. Tests that need the engine stub it via sys.modules.
"""

from __future__ import annotations

import sys
import types
import unittest

import torch

from sglang.srt.eplb.cpu_offload import (
    CpuOffloadConfig,
    CpuShadowRegistry,
    FastLoopScheduler,
    check_overflow,
    split_topk_for_cpu,
)
from sglang.srt.eplb.cpu_offload.metrics import CpuOffloadMetrics


class TestCpuOffloadConfig(unittest.TestCase):
    def test_ratio_validation(self) -> None:
        with self.assertRaises(ValueError):
            CpuOffloadConfig(
                enable_cpu_offload_fast_loop=True,
                cpu_active_ranks=[0],
                cpu_offload_ratio=0.0,
            )
        with self.assertRaises(ValueError):
            CpuOffloadConfig(
                enable_cpu_offload_fast_loop=True,
                cpu_active_ranks=[0],
                cpu_offload_ratio=1.5,
            )

    def test_reserved_cores_nonneg(self) -> None:
        with self.assertRaises(ValueError):
            CpuOffloadConfig(
                enable_cpu_offload_fast_loop=True,
                cpu_active_ranks=[0],
                cpu_offload_reserved_cores_per_numa=-1,
            )

    def test_active_ranks_dedup_sort(self) -> None:
        cfg = CpuOffloadConfig(
            enable_cpu_offload_fast_loop=True,
            cpu_active_ranks=[3, 1, 1, 0],
        )
        self.assertEqual(cfg.cpu_active_ranks, [0, 1, 3])
        self.assertTrue(cfg.rank_is_active(0))
        self.assertFalse(cfg.rank_is_active(2))

    def test_numa_for_rank_explicit_map_wins(self) -> None:
        cfg = CpuOffloadConfig(
            enable_cpu_offload_fast_loop=True,
            cpu_active_ranks=[0, 1, 2, 3],
            cpu_offload_rank_to_numa_map={0: 1, 1: 0, 2: 1, 3: 0},
        )
        self.assertEqual(cfg.numa_for_rank(0, world_size=4, num_numa_nodes=2), 1)
        self.assertEqual(cfg.numa_for_rank(1, world_size=4, num_numa_nodes=2), 0)
        self.assertEqual(cfg.numa_for_rank(2, world_size=4, num_numa_nodes=2), 1)

    def test_numa_for_rank_partial_map_raises(self) -> None:
        cfg = CpuOffloadConfig(
            enable_cpu_offload_fast_loop=True,
            cpu_active_ranks=[0, 1],
            cpu_offload_rank_to_numa_map={0: 1, 1: 0},
        )
        with self.assertRaises(KeyError):
            cfg.numa_for_rank(2, world_size=4, num_numa_nodes=2)

    def test_numa_for_rank_even_split_no_map(self) -> None:
        cfg = CpuOffloadConfig(
            enable_cpu_offload_fast_loop=True,
            cpu_active_ranks=[0, 1, 2, 3],
        )
        # 4 ranks / 2 numa => 2 per node => ranks 0,1 -> numa 0; 2,3 -> numa 1
        self.assertEqual(cfg.numa_for_rank(0, world_size=4, num_numa_nodes=2), 0)
        self.assertEqual(cfg.numa_for_rank(2, world_size=4, num_numa_nodes=2), 1)

    def test_is_active(self) -> None:
        self.assertFalse(CpuOffloadConfig().is_active())
        self.assertTrue(
            CpuOffloadConfig(
                enable_cpu_offload_fast_loop=True, cpu_active_ranks=[0]
            ).is_active()
        )


class TestCpuShadowRegistry(unittest.TestCase):
    def _make_cfg(self) -> CpuOffloadConfig:
        return CpuOffloadConfig(
            enable_cpu_offload_fast_loop=True,
            cpu_active_ranks=[0, 4],
        )

    def test_inactive_ranks_have_active_false(self) -> None:
        reg = CpuShadowRegistry(
            self._make_cfg(), world_size=8, num_numa_nodes=2, local_rank=0
        )
        for rank_id in range(8):
            state = reg.get(rank_id)
            self.assertEqual(state.active, rank_id in (0, 4))

    def test_inactive_rank_cannot_get_real_slot(self) -> None:
        reg = CpuShadowRegistry(
            self._make_cfg(), world_size=8, num_numa_nodes=2, local_rank=0
        )
        with self.assertRaises(ValueError):
            reg.set_layer_slots({0: {1: 5}})  # rank 1 inactive

    def test_set_slots_active(self) -> None:
        reg = CpuShadowRegistry(
            self._make_cfg(), world_size=8, num_numa_nodes=2, local_rank=0
        )
        reg.set_layer_slots({0: {0: 12, 4: -1}})
        self.assertEqual(reg.get(0).get_slot(0), 12)
        self.assertEqual(reg.get(4).get_slot(0), -1)

    def test_demoted_rank_forced_minus1(self) -> None:
        reg = CpuShadowRegistry(
            self._make_cfg(), world_size=8, num_numa_nodes=2, local_rank=0
        )
        reg.get(0).demote()
        reg.set_layer_slots({0: {0: 7}})
        self.assertEqual(reg.get(0).get_slot(0), -1)


class TestSplitTopkForCpu(unittest.TestCase):
    def test_slot_minus1_is_noop(self) -> None:
        topk = torch.tensor([[1, 2], [3, 4]])
        d = split_topk_for_cpu(topk, cpu_slot_expert_id=-1)
        self.assertEqual(d.n_cpu_tokens, 0)
        self.assertIs(d.masked_topk_ids, topk)

    def test_no_match_returns_original(self) -> None:
        topk = torch.tensor([[1, 2], [3, 4]])
        d = split_topk_for_cpu(topk, cpu_slot_expert_id=99)
        self.assertEqual(d.n_cpu_tokens, 0)
        self.assertIs(d.masked_topk_ids, topk)

    def test_mode_b_masks_branches(self) -> None:
        topk = torch.tensor([[1, 5], [3, 5], [2, 4]])
        d = split_topk_for_cpu(topk, cpu_slot_expert_id=5)
        self.assertEqual(d.n_cpu_tokens, 2)
        self.assertTrue(torch.equal(d.token_indices, torch.tensor([0, 1])))
        self.assertTrue(
            torch.equal(d.masked_topk_ids, torch.tensor([[1, -1], [3, -1], [2, 4]]))
        )

    def test_max_tokens_cap(self) -> None:
        topk = torch.tensor([[5, 1], [5, 2], [5, 3]])
        d = split_topk_for_cpu(topk, cpu_slot_expert_id=5, max_tokens=2)
        self.assertEqual(d.n_cpu_tokens, 2)
        self.assertTrue(torch.equal(d.token_indices, torch.tensor([0, 1])))
        # Token 2's CPU branch must NOT be masked, so GPU computes it.
        self.assertEqual(d.masked_topk_ids[2, 0].item(), 5)

    def test_overflow_error(self) -> None:
        topk = torch.tensor([[5, 1], [5, 2], [5, 3]])
        d = split_topk_for_cpu(topk, cpu_slot_expert_id=5)
        with self.assertRaises(OverflowError):
            check_overflow(d, max_per_step=2, max_per_expert=-1, policy="error")

    def test_overflow_drop(self) -> None:
        topk = torch.tensor([[5, 1], [5, 2], [5, 3]])
        d = split_topk_for_cpu(topk, cpu_slot_expert_id=5)
        d2 = check_overflow(d, max_per_step=2, max_per_expert=-1, policy="drop")
        self.assertEqual(d2.n_cpu_tokens, 2)


class TestMetrics(unittest.TestCase):
    def test_snapshot_keys(self) -> None:
        m = CpuOffloadMetrics()
        m.record_offload(rank_id=0, expert_id=7, n_tokens=12)
        m.record_roundtrip_us(150.0)
        snap = m.snapshot()
        for key in (
            "cpu_offload_tokens_total",
            "cpu_offload_roundtrip_us_p99",
            "cpu_offload_hit_count",
            "gpu_tokens_saved_by_cpu",
        ):
            self.assertIn(key, snap)
        self.assertEqual(snap["cpu_offload_tokens_total"], 12)
        self.assertEqual(snap["gpu_tokens_saved_by_cpu"], 12)
        self.assertAlmostEqual(snap["cpu_offload_roundtrip_us_p99"], 150.0)


class TestFastLoop(unittest.TestCase):
    def _make(self) -> tuple[FastLoopScheduler, CpuShadowRegistry, dict]:
        cfg = CpuOffloadConfig(
            enable_cpu_offload_fast_loop=True,
            cpu_active_ranks=[0, 1],
            cpu_offload_slot_update_interval_ms=0,  # always tick when forced
        )
        reg = CpuShadowRegistry(cfg, world_size=2, num_numa_nodes=2, local_rank=0)
        metrics = CpuOffloadMetrics()
        snapshot = {
            0: {
                (0, 10): 100,
                (0, 11): 5,
                (1, 12): 50,
            }
        }
        # cpu_slot must be an expert the rank OWNS (offloading a foreign
        # expert double-counts it -- see fast_loop._run_once Phase 2), so the
        # rank's own set has to contain the experts that carry traffic.
        local_sets = {(0, 0): {10, 11}, (1, 0): {12}}

        loop = FastLoopScheduler(
            cfg,
            reg,
            metrics,
            load_snapshot_fn=lambda: snapshot,
            local_expert_set_fn=lambda rid, lid: local_sets.get((rid, lid), set()),
        )
        return loop, reg, snapshot

    def test_picks_top_expert(self) -> None:
        """Hottest expert AMONG THE RANK'S OWN wins (10 has 100, 11 has 5)."""
        loop, reg, _ = self._make()
        ran = loop.tick(force=True)
        self.assertTrue(ran)
        self.assertEqual(reg.get(0).get_slot(0), 10)
        self.assertEqual(reg.get(1).get_slot(0), 12)

    def test_picks_only_from_local_experts(self) -> None:
        """Only the rank's OWN experts are candidates.

        This test previously asserted the opposite ("skip local experts").
        That rule is what made every offloaded token count twice: the CPU
        result is combined before the TP all-reduce, and masking cpu_slot to
        -1 only suppresses the GPU copy on THIS rank, so a foreign expert was
        still computed on its owner's GPU with the CPU result added on top.
        Here 20 is hotter-looking but foreign; 10 is ours, so 10 must win.
        """
        cfg = CpuOffloadConfig(
            enable_cpu_offload_fast_loop=True,
            cpu_active_ranks=[0],
            cpu_offload_slot_update_interval_ms=0,
        )
        reg = CpuShadowRegistry(cfg, world_size=1, num_numa_nodes=1, local_rank=0)
        metrics = CpuOffloadMetrics()
        loop = FastLoopScheduler(
            cfg,
            reg,
            metrics,
            load_snapshot_fn=lambda: {0: {(0, 10): 100, (0, 20): 50}},
            local_expert_set_fn=lambda rid, lid: {10},  # 10 is local
        )
        loop.tick(force=True)
        self.assertEqual(reg.get(0).get_slot(0), 10)

    def test_demoted_rank_gets_minus1_via_latency(self) -> None:
        # Tight demote window so a single bad sample is enough.
        cfg = CpuOffloadConfig(
            enable_cpu_offload_fast_loop=True,
            cpu_active_ranks=[0],
            cpu_offload_slot_update_interval_ms=0,
            cpu_offload_demote_window=1,
            cpu_offload_demote_latency_us=1000,
        )
        reg = CpuShadowRegistry(cfg, world_size=1, num_numa_nodes=1, local_rank=0)
        metrics = CpuOffloadMetrics()
        loop = FastLoopScheduler(
            cfg,
            reg,
            metrics,
            load_snapshot_fn=lambda: {0: {(0, 10): 100}},
            local_expert_set_fn=lambda rid, lid: set(),
        )
        # Inject a slow sample so _apply_demotion trips on the next tick.
        reg.get(0).record_latency_us(5000.0)
        loop.tick(force=True)
        self.assertTrue(reg.get(0).demoted)
        self.assertEqual(reg.get(0).get_slot(0), -1)
        self.assertEqual(metrics.cpu_offload_demoted_slots_count, 1)


class TestRuntimeSingleton(unittest.TestCase):
    def setUp(self) -> None:
        from sglang.srt.eplb.cpu_offload import reset_cpu_offload_runtime_for_tests

        reset_cpu_offload_runtime_for_tests()

    def tearDown(self) -> None:
        from sglang.srt.eplb.cpu_offload import reset_cpu_offload_runtime_for_tests

        reset_cpu_offload_runtime_for_tests()

    def test_disabled_returns_none(self) -> None:
        from sglang.srt.eplb.cpu_offload import (
            get_cpu_offload_runtime,
            init_cpu_offload_runtime,
        )

        sa = types.SimpleNamespace(enable_cpu_offload_fast_loop=False)
        self.assertIsNone(
            init_cpu_offload_runtime(sa, local_rank=0, world_size=1, num_numa_nodes=1)
        )
        self.assertIsNone(get_cpu_offload_runtime())

    def test_enabled_builds_runtime(self) -> None:
        from sglang.srt.eplb.cpu_offload import (
            get_cpu_offload_runtime,
            init_cpu_offload_runtime,
        )

        sa = types.SimpleNamespace(
            enable_cpu_offload_fast_loop=True,
            cpu_active_ranks="0",
            cpu_offload_rank_to_numa_map=None,
            cpu_offload_top_m=1,
            cpu_offload_slot_update_interval_ms=0,
            cpu_offload_enable_mode_a=False,
            cpu_offload_ratio=0.2,
            cpu_offload_overflow_policy="error",
            cpu_offload_max_tokens_per_step=1024,
            cpu_offload_max_tokens_per_expert=512,
            cpu_offload_num_workers_per_numa=-1,
            cpu_offload_reserved_cores_per_numa=8,
            cpu_offload_shm_prefix="/dev/shm/sglang_cpu_offload_test",
            cpu_offload_amx_enabled=False,
            cpu_offload_enable_cuda_graph_polling=True,
            cpu_offload_timeout_us=1_000_000,
            cpu_offload_max_spin=1000,
            ep_size=1,
        )
        rt = init_cpu_offload_runtime(sa, local_rank=0, world_size=1, num_numa_nodes=1)
        self.assertIsNotNone(rt)
        self.assertIs(rt, get_cpu_offload_runtime())
        # Hot-path accessors stay cheap and never touch the engine.
        self.assertFalse(rt.engine.initialized)
        self.assertEqual(rt.local_cpu_slot(0), -1)  # no slot assigned yet
        # Inactive remote rank stays None for slot lookup.
        rt.record_routed(layer_id=0, expert_id=7, n_tokens=5)
        snap = rt.counter.drain()
        self.assertEqual(snap, {0: {(0, 7): 5}})

    def test_tick_runs_when_active(self) -> None:
        from sglang.srt.eplb.cpu_offload import (
            init_cpu_offload_runtime,
        )

        sa = types.SimpleNamespace(
            enable_cpu_offload_fast_loop=True,
            cpu_active_ranks="0",
            cpu_offload_rank_to_numa_map=None,
            cpu_offload_top_m=1,
            cpu_offload_slot_update_interval_ms=0,
            cpu_offload_enable_mode_a=False,
            cpu_offload_ratio=0.2,
            cpu_offload_overflow_policy="error",
            cpu_offload_max_tokens_per_step=1024,
            cpu_offload_max_tokens_per_expert=512,
            cpu_offload_num_workers_per_numa=-1,
            cpu_offload_reserved_cores_per_numa=8,
            cpu_offload_shm_prefix="/dev/shm/sglang_cpu_offload_test",
            cpu_offload_amx_enabled=False,
            cpu_offload_enable_cuda_graph_polling=True,
            cpu_offload_timeout_us=1_000_000,
            cpu_offload_max_spin=1000,
            ep_size=1,
        )
        rt = init_cpu_offload_runtime(sa, local_rank=0, world_size=1, num_numa_nodes=1)
        assert rt is not None
        rt.record_routed(layer_id=0, expert_id=42, n_tokens=99)
        # With no model_runner the default local_expert_set_fn returns an
        # EMPTY set, and an empty set now means "this rank owns nothing, so
        # offload nothing" (the safe direction -- offloading a foreign expert
        # would double-count it). Declare 42 as owned so the tick has a legal
        # candidate; what this test is really about is the runtime singleton
        # wiring and that tick() reaches the selector.
        rt.fast_loop._local_expert_set_fn = lambda rid, lid: {42}
        ran = rt.fast_loop.tick(force=True)
        self.assertTrue(ran)
        self.assertEqual(rt.local_cpu_slot(0), 42)


class TestEngineLazyImport(unittest.TestCase):
    def test_missing_engine_raises_at_init_runtime(self) -> None:
        from sglang.srt.eplb.cpu_offload.engine import CpuOffloadEngine

        cfg = CpuOffloadConfig(enable_cpu_offload_fast_loop=True, cpu_active_ranks=[0])
        eng = CpuOffloadEngine(cfg, host_rank=0, num_hosts=1, ep_size=1)
        real = sys.modules.pop("moe_cpu_engine", None)
        try:
            with self.assertRaises(ImportError):
                eng.init_runtime()
        finally:
            if real is not None:
                sys.modules["moe_cpu_engine"] = real


if __name__ == "__main__":
    unittest.main()
