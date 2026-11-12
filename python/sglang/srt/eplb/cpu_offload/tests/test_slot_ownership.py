"""Tests for the two correctness fixes: deterministic re-selection cadence,
and block-based CPU-slot ownership.

Both bugs shared one property: they produced no error, only wrong numbers.
The wall-clock gate made runs silently non-reproducible; the interleaved
ownership formula made every offloaded token silently counted twice. So the
tests below assert the invariants directly rather than any observable symptom.
"""

from __future__ import annotations

import ast
import os
import typing
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

_SRT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _extract_method(rel_path, class_name, method_name):
    """Pull one method out of a source file without importing the module.

    Importing the MoE layer needs libcudart, so a test that imports it can
    only run on a GPU box -- which would mean almost nobody runs it and the
    guard is decorative. Both methods compared here are self-contained (they
    touch only their arguments and two attributes of ``self``), so they can be
    compiled on their own.

    The point is to test the REAL code. Restating either formula in the test
    would let the two drift apart in exactly the way this test exists to
    prevent: the test would keep agreeing with its own copy while the shipped
    code moved.
    """
    path = os.path.join(_SRT, rel_path)
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == method_name:
                mod = ast.fix_missing_locations(
                    ast.Module(body=[item], type_ignores=[])
                )
                ns = {
                    "Dict": typing.Dict,
                    "List": typing.List,
                    "Optional": typing.Optional,
                    "Union": typing.Union,
                }
                exec(compile(mod, path, "exec"), ns)
                return ns[method_name]
    raise AssertionError(
        f"{class_name}.{method_name} not found in {rel_path} -- it was "
        "renamed or removed; this test needs updating, not deleting."
    )


from sglang.srt.eplb.cpu_offload.config import CpuOffloadConfig
from sglang.srt.eplb.cpu_offload.fast_loop import FastLoopScheduler
from sglang.srt.eplb.cpu_offload.metrics import CpuOffloadMetrics
from sglang.srt.eplb.cpu_offload.shadow_state import CpuShadowRegistry


def _cfg(**kw):
    base = dict(enable_cpu_offload_fast_loop=True, cpu_active_ranks=[0])
    base.update(kw)
    return CpuOffloadConfig(**base)


def _loop(cfg, *, local_rank=0, world_size=1, local_set=None):
    reg = CpuShadowRegistry(
        cfg, world_size=world_size, num_numa_nodes=1, local_rank=local_rank
    )
    return FastLoopScheduler(
        cfg,
        reg,
        CpuOffloadMetrics(),
        load_snapshot_fn=lambda: {},
        local_expert_set_fn=lambda r, l: set(local_set or ()),
    )


class TestReselectCadenceIsDeterministic(unittest.TestCase):
    """The gate must count forward passes, never read a clock."""

    def test_module_does_not_read_the_clock_at_all(self):
        import inspect

        from sglang.srt.eplb.cpu_offload import fast_loop

        src = inspect.getsource(fast_loop)
        # Comments may mention it; executable code must not call it.
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        self.assertNotIn("time.monotonic_ns()", code)
        self.assertNotIn("time.time()", code)

    def test_fires_on_a_fixed_forward_number_regardless_of_timing(self):
        """Same input -> same firing steps, with no dependence on now_ns."""
        fired = []
        cfg = _cfg(
            cpu_offload_slot_update_interval_ms=500,
            cpu_offload_slot_update_every_n_forwards=10,
        )
        loop = _loop(cfg)
        loop._init_slot_pending = False  # skip bootstrap
        loop._run_once = lambda: fired.append(loop._tick_count)
        # Absurd, non-monotonic wall-clock values must change nothing.
        for i, fake_ns in enumerate([0, 10**18, 5, 10**12] * 10):
            loop.tick(now_ns=fake_ns)
        self.assertEqual(fired, [10, 20, 30, 40])

    def test_two_schedulers_agree_step_for_step(self):
        """Two ranks must select on the SAME forward numbers (no deadlock,
        no divergent slot history)."""
        runs = []
        for rank in (0, 1):
            fired = []
            cfg = _cfg(
                cpu_active_ranks=[0, 1],
                cpu_offload_slot_update_interval_ms=500,
                cpu_offload_slot_update_every_n_forwards=7,
            )
            loop = _loop(cfg, local_rank=rank, world_size=2)
            loop._init_slot_pending = False
            loop._run_once = lambda f=fired, l=None: None
            loop._run_once = lambda: fired.append(loop._tick_count)
            for i in range(30):
                # Deliberately different "time" on each rank.
                loop.tick(now_ns=i * (rank + 1) * 10**6)
            runs.append(fired)
        self.assertEqual(runs[0], runs[1])
        self.assertTrue(runs[0])

    def test_zero_ms_still_means_disabled(self):
        fired = []
        cfg = _cfg(cpu_offload_slot_update_interval_ms=0)
        loop = _loop(cfg)
        loop._init_slot_pending = False
        loop._run_once = lambda: fired.append(loop._tick_count)
        for _ in range(50):
            loop.tick()
        self.assertEqual(fired, [])

    def test_force_still_overrides_a_disabled_gate(self):
        fired = []
        cfg = _cfg(cpu_offload_slot_update_interval_ms=0)
        loop = _loop(cfg)
        loop._init_slot_pending = False
        loop._run_once = lambda: fired.append(loop._tick_count)
        self.assertTrue(loop.tick(force=True))
        self.assertEqual(len(fired), 1)

    def test_non_zero_ms_announces_the_changed_meaning(self):
        """A silent semantic change would be worse than an error."""
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            _loop(_cfg(cpu_offload_slot_update_interval_ms=500))
        out = buf.getvalue()
        self.assertIn("no longer a wall-clock interval", out)
        self.assertIn("FORWARD PASSES", out)

    def test_zero_ms_does_not_warn(self):
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            _loop(_cfg(cpu_offload_slot_update_interval_ms=0))
        self.assertEqual(buf.getvalue(), "")


class TestBootstrapPicksAnOwnedExpert(unittest.TestCase):
    """cpu_slot must land inside [r*n, (r+1)*n) -- this rank's own block."""

    def test_formula_bootstrap_is_inside_the_rank_block(self):
        num_experts, ws = 128, 8
        for rank in range(ws):
            cfg = _cfg(cpu_active_ranks=[rank])
            reg = CpuShadowRegistry(
                cfg, world_size=ws, num_numa_nodes=1, local_rank=rank
            )
            self.assertTrue(reg.apply_initial_slots(num_experts))
            slot = reg.get(rank).get_slot(0)
            n = num_experts // ws
            lo, hi = rank * n, (rank + 1) * n
            self.assertTrue(
                lo <= slot < hi,
                "rank %d got cpu_slot %d, outside its own block [%d, %d) -- "
                "that expert would be computed on its owner's GPU AND on this "
                "rank's CPU" % (rank, slot, lo, hi),
            )

    def test_the_old_interleaved_formula_would_fail_this(self):
        """Guard against re-introducing the regression this replaced."""
        num_experts, ws = 128, 8
        n = num_experts // ws
        for rank in range(ws - 1):  # every rank but the last
            interleaved = rank + (n - 1) * ws
            self.assertFalse(rank * n <= interleaved < (rank + 1) * n)


class TestSelectorPicksAnOwnedExpert(unittest.TestCase):
    def test_hottest_owned_expert_wins(self):
        import torch

        cfg = _cfg(
            cpu_offload_slot_update_interval_ms=500,
            cpu_offload_slot_update_every_n_forwards=1,
        )
        # rank 0 owns {0,1,2,3}; expert 9 is hotter but belongs to someone else
        loop = _loop(cfg, local_set={0, 1, 2, 3})
        loop._init_slot_pending = False
        counts = torch.zeros(16)
        counts[9] = 500.0  # hottest overall, NOT ours
        counts[2] = 100.0  # hottest of ours
        counts[0] = 10.0
        loop._global_count_accum = {0: counts}

        gctx = MagicMock()
        gctx.initialized = True
        import sglang.srt.eplb.cpu_offload.cuda_graph_offload as cg

        orig = cg.get_cuda_graph_offload_ctx
        cg.get_cuda_graph_offload_ctx = lambda: gctx
        try:
            loop._run_once()
        finally:
            cg.get_cuda_graph_offload_ctx = orig

        self.assertEqual(loop._registry.get(0).get_slot(0), 2)

    def test_no_owned_expert_with_traffic_means_no_offload(self):
        import torch

        cfg = _cfg(
            cpu_offload_slot_update_interval_ms=500,
            cpu_offload_slot_update_every_n_forwards=1,
        )
        loop = _loop(cfg, local_set={0, 1, 2, 3})
        loop._init_slot_pending = False
        counts = torch.zeros(16)
        counts[9] = 500.0  # traffic exists, but not on any expert we own
        loop._global_count_accum = {0: counts}

        gctx = MagicMock()
        gctx.initialized = True
        import sglang.srt.eplb.cpu_offload.cuda_graph_offload as cg

        orig = cg.get_cuda_graph_offload_ctx
        cg.get_cuda_graph_offload_ctx = lambda: gctx
        try:
            loop._run_once()
        finally:
            cg.get_cuda_graph_offload_ctx = orig

        # Better to offload nothing than to offload someone else's expert.
        self.assertEqual(loop._registry.get(0).get_slot(0), -1)


class TestOwnershipSourceAssertion(unittest.TestCase):
    def _module(self, ep_rank=0, n_local=4):
        m = MagicMock()
        m.moe_ep_rank = ep_rank
        m.num_local_experts = n_local
        return m

    def _meta(self, rows, n_phys):
        class _Row:
            def __init__(s, d):
                s._d = list(d)

            def tolist(s):
                return list(s._d)

        class _Map:
            def __init__(s, r):
                s._r = r
                s.shape = (len(r), len(r[0]))

            def __getitem__(s, k):
                row, sl = k
                return _Row(s._r[row][sl])

        class _Meta:
            num_local_physical_experts = n_phys
            physical_to_logical_map = _Map(rows)

        return _Meta()

    def _run(self, module, meta):
        import sys
        import types

        from sglang.srt.eplb.cpu_offload import model_load_hook as mlh

        mod = types.ModuleType("sglang.srt.eplb.expert_location")
        mod.get_global_expert_location_metadata = lambda: meta
        saved = sys.modules.get("sglang.srt.eplb.expert_location")
        sys.modules["sglang.srt.eplb.expert_location"] = mod
        try:
            mlh._assert_expert_block_sources_agree(module, 0)
        finally:
            if saved is not None:
                sys.modules["sglang.srt.eplb.expert_location"] = saved
            else:
                del sys.modules["sglang.srt.eplb.expert_location"]

    def test_agreement_passes(self):
        rows = [[0, 1, 2, 3, 4, 5, 6, 7]]
        self._run(self._module(0, 4), self._meta(rows, 4))  # both -> {0,1,2,3}

    def test_divergence_raises_loudly(self):
        # non-identity map: rank 0's physical slice holds logical {4,5,6,7}
        rows = [[4, 5, 6, 7, 0, 1, 2, 3]]
        with self.assertRaises(RuntimeError) as cm:
            self._run(self._module(0, 4), self._meta(rows, 4))
        msg = str(cm.exception)
        self.assertIn("DISAGREE", msg)
        self.assertIn("counted twice", msg)

    def test_missing_metadata_is_not_an_error(self):
        self._run(self._module(0, 4), None)


class TestPlacementMapAgreesWithWeightLoader(unittest.TestCase):
    """The placement map must use the same formula as the weight loader.

    These two forked once: the map was written interleaved
    (``{eid: eid % ep_size}``) while the weight loader stayed on the
    contiguous block. Nothing computed a wrong number, because the map's only
    consumer (``_global_to_local_map``) is never read -- but the map is also
    what gets logged as ``my_global_expert_ids``, and that log line is the most
    authoritative-looking claim about expert ownership in the run. Reading it
    yields the exact opposite of the truth, which is how an investigation
    concluded that 915 of 998 placements were out of range on a system that
    was fine.

    Both sides are invoked here rather than restated, so the test cannot drift
    into agreeing with a copy of the formula instead of the formula.
    """

    def _placement(self, num_experts, ep_size):
        fn = _extract_method(
            "eplb/cpu_offload/legacy/ep_moe_v055.py",
            "DeepEPMoEHeto",
            "_create_fast_loop_expert_map",
        )
        return fn(SimpleNamespace(ep_size=ep_size), num_experts)

    def _weight_loader_owns(self, expert_id, ep_rank, num_local):
        fn = _extract_method(
            "layers/moe/fused_moe_triton/layer.py",
            "FusedMoE",
            "_map_global_expert_id_to_local_expert_id",
        )
        # 新版按实际专家存储分片编号判断归属；本轮禁止共享专家融合。
        stub = SimpleNamespace(
            moe_ep_rank=ep_rank,
            num_local_experts=num_local,
            _expert_storage_rank=ep_rank,
            _num_local_routed=num_local,
            _has_fused_shared=False,
        )
        return fn(stub, expert_id) >= 0

    def test_every_expert_agrees_with_the_weight_loader(self):
        for num_experts, ep_size in ((128, 8), (128, 2), (64, 4), (8, 8)):
            with self.subTest(E=num_experts, EP=ep_size):
                placement = self._placement(num_experts, ep_size)
                num_local = num_experts // ep_size
                for eid in range(num_experts):
                    for rank in range(ep_size):
                        self.assertEqual(
                            placement[eid] == rank,
                            self._weight_loader_owns(eid, rank, num_local),
                            f"expert {eid}: map says rank {placement[eid]}, "
                            f"weight loader disagrees about rank {rank}",
                        )

    def test_each_rank_owns_a_contiguous_block(self):
        placement = self._placement(128, 8)
        for rank in range(8):
            mine = sorted(e for e, r in placement.items() if r == rank)
            self.assertEqual(mine, list(range(rank * 16, (rank + 1) * 16)))

    def test_interleaved_layout_would_be_rejected(self):
        # The shape of the old bug, asserted directly: under the interleaved
        # map rank 0 held {0, 8, 16, ...}, which the weight loader does not.
        num_local = 16
        interleaved_rank0 = {e for e in range(128) if e % 8 == 0}
        block_rank0 = {
            e for e in range(128) if self._weight_loader_owns(e, 0, num_local)
        }
        self.assertNotEqual(interleaved_rank0, block_rank0)
        self.assertEqual(len(interleaved_rank0 & block_rank0), 2)

    def test_cpu_slot_is_already_a_member_so_the_append_cannot_fire(self):
        # DeepEPMoEHeto appends cpu_slot to my_global_expert_ids when it is not
        # already there, and len(that list) becomes num_local_experts. Under
        # the old interleaved map a block-derived slot was never a member, so
        # the append would have pushed the count to E/EP+1 and shifted the
        # weight loader's window over every expert. Under the block map the
        # slot is always a member, so the branch is unreachable.
        placement = self._placement(128, 8)
        for rank in range(8):
            mine = [e for e, r in placement.items() if r == rank]
            for slot in (rank * 16, rank * 16 + 15):
                self.assertIn(
                    slot,
                    mine,
                    "a cpu_slot chosen from this rank's block must already be "
                    "in my_global_expert_ids, or num_local_experts changes",
                )

    def test_non_divisible_expert_count_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            self._placement(130, 8)
        self.assertIn("divisible", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
