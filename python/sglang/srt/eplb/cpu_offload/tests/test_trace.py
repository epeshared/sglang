"""Tests for the MoE call-path trace (``SGLANG_MOE_TRACE``).

Three things are being pinned down here:

1. The report shows paths that never ran, not only paths that did. A report
   listing only what executed would let a reader conclude "this is the path"
   without ever learning a second path existed and was skipped -- the exact
   mistake this feature exists to prevent.
2. The re-selection history is byte-identical for identical inputs, so two
   runs can be compared with ``diff``.
3. Tracing is off unless asked for, and every instrumented module says so
   through a module-level constant (which is what makes the disabled cost a
   single not-taken branch rather than a function call).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from sglang.srt.eplb.cpu_offload import journal, trace


def _fake_runtime(*, local_rank=0, slots=None, layer_stats=None):
    """A runtime stand-in with just the surface ``build_report`` touches."""
    slots = slots or {}
    engine = MagicMock()
    engine.read_layer_stats = MagicMock(return_value=layer_stats or {})
    engine.registered_layers = MagicMock(return_value=sorted(slots))
    rt = types.SimpleNamespace(
        local_rank=local_rank,
        config=types.SimpleNamespace(
            enable_cpu_offload_fast_loop=True,
            cpu_active_ranks=[0],
            cpu_offload_backend="daemon",
            cpu_offload_decode_only=True,
            cpu_offload_top_m=1,
            cpu_offload_slot_update_interval_ms=500,
            cpu_offload_xrank_sync_every_n_ticks=100,
            cpu_offload_max_spin=1000,
            cpu_offload_timeout_us=5_000_000,
            cpu_offload_enable_mode_a=False,
            cpu_offload_shm_prefix="/dev/shm/x",
            cpu_offload_num_workers_per_numa=2,
            cpu_offload_worker_assign_strategy="order",
            cpu_offload_enable_cuda_graph_polling=True,
        ),
        engine=engine,
        local_cpu_slot=lambda layer_id: slots.get(layer_id, -1),
    )
    return rt


class TestSwitchIsOffByDefault(unittest.TestCase):
    def test_module_constant_is_false(self):
        # The test suite never sets SGLANG_MOE_TRACE, so this documents the
        # default and fails loudly if the default ever flips.
        self.assertFalse(trace.ENABLED)

    def test_every_instrumented_module_exposes_the_constant(self):
        """Each call site must be gated by a module-level bool, not a lookup.

        A per-call ``os.environ`` read or a function call would defeat the
        whole point of the disabled path being free.
        """
        from sglang.srt.eplb.cpu_offload import (
            cuda_graph_offload,
            dispatch_hook,
            engine_daemon,
            fast_loop,
            runtime,
        )

        for mod in (
            dispatch_hook,
            cuda_graph_offload,
            fast_loop,
            runtime,
            engine_daemon,
        ):
            with self.subTest(module=mod.__name__):
                self.assertIn("_MOE_TRACE", vars(mod))
                self.assertIsInstance(mod._MOE_TRACE, bool)
                self.assertFalse(mod._MOE_TRACE)

    def test_moe_layer_modules_expose_the_constant(self):
        from sglang.srt.layers.moe.ep_moe import layer as ep_layer
        from sglang.srt.layers.moe.fused_moe_triton import layer as fused_layer

        for mod in (ep_layer, fused_layer):
            with self.subTest(module=mod.__name__):
                self.assertIsInstance(mod._MOE_TRACE, bool)
                self.assertFalse(mod._MOE_TRACE)
                # And the trace module must not have been imported by them.
                self.assertIsNone(mod._moe_trace_mod)


class TestPathTree(unittest.TestCase):
    def setUp(self):
        trace.reset_for_tests()

    def tearDown(self):
        trace.reset_for_tests()

    def test_unexecuted_paths_are_still_listed(self):
        trace.hit("moe.heto.forward", 10)
        lines = trace._tree_lines()
        text = "\n".join(lines)
        # Executed
        self.assertIn("DeepEPMoEHeto.forward", text)
        # Never executed -- must still appear, marked
        self.assertIn("in-class offload begin", text)
        dead = [l for l in lines if "in-class offload begin" in l][0]
        self.assertTrue(dead.lstrip().startswith("X"), dead)
        self.assertTrue(dead.rstrip().endswith("0"), dead)

    def test_heto_and_parent_forward_are_separate_rows(self):
        """DeepEPMoEHeto overrides forward and calls FusedMoE.forward
        directly, so DeepEPMoE.forward can be dead while the subclass runs.
        One shared row would let a zero on one be read as a zero on both."""
        trace.hit("moe.heto.forward", 12596)
        lines = trace._tree_lines()
        heto = [l for l in lines if "DeepEPMoEHeto.forward" in l][0]
        parent = [l for l in lines if "DeepEPMoE.forward (parent" in l][0]
        self.assertIn("12596", heto)
        self.assertTrue(parent.lstrip().startswith("X"), parent)

    def test_every_declared_key_has_a_row(self):
        rows = trace._tree_lines()
        self.assertGreaterEqual(len(rows), len(trace._PATH_ORDER))

    def test_reason_histogram_is_rendered_under_its_path(self):
        trace.reason("hook.begin.eager_skip", "cpu_slot=-1", 3)
        trace.reason("hook.begin.eager_skip", "layer not registered", 5)
        text = "\n".join(trace._tree_lines())
        self.assertIn("reason: layer not registered", text)
        self.assertIn("reason: cpu_slot=-1", text)
        # Ordered by count, largest first.
        self.assertLess(
            text.index("reason: layer not registered"),
            text.index("reason: cpu_slot=-1"),
        )


class TestExceptionAggregation(unittest.TestCase):
    def setUp(self):
        trace.reset_for_tests()

    def tearDown(self):
        trace.reset_for_tests()

    def test_first_traceback_kept_and_repeats_counted(self):
        for i in range(5):
            try:
                raise ValueError("boom %d" % i)
            except ValueError as e:
                trace.exc("site.a", e)
        entry = trace._S.exceptions["site.a"]
        self.assertEqual(entry["count"], 5)
        self.assertEqual(entry["type"], "ValueError")
        # First occurrence wins, so the message is from i == 0.
        self.assertIn("boom 0", entry["msg"])
        self.assertIn("ValueError", entry["tb"])

    def test_sites_do_not_merge(self):
        try:
            raise KeyError("k")
        except KeyError as e:
            trace.exc("site.a", e)
        try:
            raise IndexError("i")
        except IndexError as e:
            trace.exc("site.b", e)
        self.assertEqual(set(trace._S.exceptions), {"site.a", "site.b"})


class TestReselectHistory(unittest.TestCase):
    def setUp(self):
        trace.reset_for_tests()

    def tearDown(self):
        trace.reset_for_tests()

    def _emit(self, changes):
        trace.reselect(
            event="reselect",
            tick=1,
            forward_no=200,
            changes=changes,
            source="cuda_graph_acc_xrank",
        )

    def test_identical_input_gives_identical_text(self):
        self._emit([(3, 1, 2), (1, 5, 6)])
        a = trace._reselect_report()
        trace.reset_for_tests()
        # Same changes, different insertion order -- must still render the same.
        self._emit([(1, 5, 6), (3, 1, 2)])
        b = trace._reselect_report()
        self.assertEqual(a, b)

    def test_different_input_gives_different_text(self):
        self._emit([(3, 1, 2)])
        a = trace._reselect_report()
        trace.reset_for_tests()
        self._emit([(3, 1, 9)])
        b = trace._reselect_report()
        self.assertNotEqual(a, b)
        self.assertIn("L3:1>9", b)

    def test_identical_consecutive_moves_collapse_into_a_range(self):
        """94 identical bootstrap tokens hide the one layer that differs."""
        self._emit([(i, -1, 120) for i in range(10)])
        text = trace._reselect_report()
        self.assertIn("L0-9:-1>120", text)
        self.assertNotIn("L5:-1>120", text)

    def test_a_layer_that_differs_survives_the_collapse(self):
        ch = [(i, -1, 120) for i in range(10)]
        ch[4] = (4, -1, 77)
        self._emit(ch)
        text = trace._reselect_report()
        self.assertIn("L4:-1>77", text)
        self.assertIn("L0-3:-1>120", text)
        self.assertIn("L5-9:-1>120", text)

    def test_empty_history_explains_itself(self):
        text = trace._reselect_report()
        self.assertIn("no re-selection event", text)


class TestConfigProvenance(unittest.TestCase):
    def setUp(self):
        trace.reset_for_tests()

    def tearDown(self):
        trace.reset_for_tests()

    def test_server_args_wins_over_env(self):
        rt = _fake_runtime()
        trace.record_server_args(types.SimpleNamespace(cpu_offload_max_spin=1000))
        with patch.dict("os.environ", {"CPU_OFFLOAD_MAX_SPIN": "7"}):
            rows = {r[1]: r for r in trace._provenance_rows(rt.config)}
        self.assertEqual(
            rows["cpu_offload_max_spin"][3], "ServerArgs.cpu_offload_max_spin"
        )

    def test_env_wins_when_server_args_is_none(self):
        rt = _fake_runtime()
        trace.record_server_args(types.SimpleNamespace(cpu_offload_max_spin=None))
        with patch.dict("os.environ", {"CPU_OFFLOAD_MAX_SPIN": "7"}):
            rows = {r[1]: r for r in trace._provenance_rows(rt.config)}
        self.assertEqual(rows["cpu_offload_max_spin"][3], "env:CPU_OFFLOAD_MAX_SPIN")

    def test_inert_server_args_flag_is_flagged_not_trusted(self):
        """A CLI flag that nothing forwards must not be reported as its source.

        ``--cpu-offload-decode-only`` parses into ServerArgs but is never
        passed into CpuOffloadConfig, so the effective value comes from the
        dataclass default. Claiming "came from ServerArgs" here would be a
        confident wrong answer, which is the failure this report exists to
        prevent.
        """
        rt = _fake_runtime()
        trace.record_server_args(types.SimpleNamespace(cpu_offload_decode_only=True))
        rows = {r[1]: r for r in trace._provenance_rows(rt.config)}
        src = rows["cpu_offload_decode_only"][3]
        self.assertIn("INERT", src)
        self.assertNotIn("ServerArgs.cpu_offload_decode_only", src.split("!!")[0])

    def test_wired_flag_is_not_marked_inert(self):
        rt = _fake_runtime()
        trace.record_server_args(types.SimpleNamespace(cpu_offload_max_spin=1234))
        rows = {r[1]: r for r in trace._provenance_rows(rt.config)}
        self.assertNotIn("INERT", rows["cpu_offload_max_spin"][3])

    def test_default_when_neither_channel_is_set(self):
        rt = _fake_runtime()
        trace.record_server_args(types.SimpleNamespace(cpu_offload_max_spin=None))
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("CPU_OFFLOAD_MAX_SPIN", None)
            rows = {r[1]: r for r in trace._provenance_rows(rt.config)}
        self.assertEqual(rows["cpu_offload_max_spin"][3], "default")


class _FakeMeta:
    """Minimal EPLB metadata: rank r owns physical [r*n, (r+1)*n)."""

    def __init__(self, num_local, rows):
        self.num_local_physical_experts = num_local
        self.physical_to_logical_map = _FakeMap(rows)


class _FakeMap:
    def __init__(self, rows):
        self._rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, sl = key
            return _FakeRow(self._rows[row][sl])
        return _FakeRow(self._rows[key])


class _FakeRow:
    def __init__(self, data):
        self._data = list(data)

    def tolist(self):
        return list(self._data)


class TestPerLayerTable(unittest.TestCase):
    def setUp(self):
        trace.reset_for_tests()

    def tearDown(self):
        trace.reset_for_tests()

    def _patched_meta(self, rows, num_local):
        mod = types.ModuleType("sglang.srt.eplb.expert_location")
        mod.get_global_expert_location_metadata = lambda: _FakeMeta(num_local, rows)
        return patch.dict("sys.modules", {"sglang.srt.eplb.expert_location": mod})

    def _patched_meta_missing(self):
        mod = types.ModuleType("sglang.srt.eplb.expert_location")
        mod.get_global_expert_location_metadata = lambda: None
        return patch.dict("sys.modules", {"sglang.srt.eplb.expert_location": mod})

    def test_own_expert_is_the_correct_state(self):
        """cpu_slot inside this rank's own GPU block == no double count."""
        rows = [[0, 1, 2, 3]]
        trace._S.block_rank, trace._S.block_num_local = 0, 2  # owns [0, 2)
        rt = _fake_runtime(local_rank=0, slots={0: 1})
        with self._patched_meta(rows, num_local=2):
            table = trace._per_layer_table(rt, 0)
        self.assertIn("on 1, NOT on 0", table)
        self.assertNotIn("counted twice", table)

    def test_foreign_expert_is_flagged_as_double_counted(self):
        """cpu_slot owned by another rank == that pair is computed twice.

        Masking a non-local expert is a no-op, the owning rank still runs it
        on GPU, and the CPU result is added on top before the TP all-reduce.
        """
        rows = [[0, 1, 2, 3]]
        trace._S.block_rank, trace._S.block_num_local = 0, 2  # owns [0, 2)
        rt = _fake_runtime(local_rank=0, slots={0: 3})
        with self._patched_meta(rows, num_local=2):
            table = trace._per_layer_table(rt, 0)
        self.assertIn("on 0, NOT on 1", table)
        self.assertIn("counted twice", table)

    def test_own_follows_the_block_not_the_dead_interleaved_map(self):
        """The interleaved _global_to_local_map must NOT decide ownership.

        It is assigned and logged but never read; both the weight loader and
        StandardDispatcher use the contiguous block. An earlier version of
        this report followed the interleaved map and inverted the verdict.
        """
        rows = [[0, 1, 2, 3]]
        trace._S.block_rank, trace._S.block_num_local = 0, 2  # owns [0, 2)
        trace.record_local_expert_ids(0, [0, 2])  # interleaved
        rt = _fake_runtime(local_rank=0, slots={0: 2})
        with self._patched_meta(rows, num_local=2):
            table = trace._per_layer_table(rt, 0)
        # expert 2 IS in the interleaved map but NOT in the block -> NO
        self.assertIn("on 0, NOT on 1", table)
        self.assertIn("NOTHING READS IT", table)

    def test_the_legend_names_the_source_the_column_actually_used(self):
        """With the MoE module observed, ownership comes from the block."""
        rows = [[0, 1, 2, 3]]
        trace._S.block_rank, trace._S.block_num_local = 0, 2
        rt = _fake_runtime(local_rank=0, slots={0: 1})
        with self._patched_meta(rows, num_local=2):
            table = trace._per_layer_table(rt, 0)
        self.assertIn("CONTIGUOUS BLOCK", table)
        self.assertIn("(block [0, 2))", table)
        self.assertNotIn("FALLBACK", table)

    def test_falling_back_to_eplb_says_so_instead_of_claiming_the_block(self):
        """The legend must not name a source the column did not use.

        Without the MoE module the authoritative range is unknown and 'own?'
        is computed from the EPLB slice instead. Printing "block" there would
        be the same class of error this report exists to catch: agreeing with
        the code under inspection because both read the same metadata. The
        fast-loop candidate filter reads that very slice, so in this state
        the column cannot contradict the selector and must not pretend to.
        """
        rows = [[0, 1, 2, 3]]
        # block_rank deliberately left unset -> the module was never observed
        self.assertIsNone(trace._S.block_rank)
        rt = _fake_runtime(local_rank=0, slots={0: 1})
        with self._patched_meta(rows, num_local=2):
            table = trace._per_layer_table(rt, 0)
        self.assertIn("FALLBACK", table)
        self.assertIn("NOT an independent check", table)
        self.assertNotIn("CONTIGUOUS BLOCK", table)
        self.assertNotIn("(block [", table)

    def test_no_ownership_source_at_all_is_admitted_not_guessed(self):
        """Neither module nor EPLB metadata: say so, do not invent a range."""
        rt = _fake_runtime(local_rank=0, slots={0: 1})
        with self._patched_meta_missing():
            table = trace._per_layer_table(rt, 0)
        self.assertIn("UNAVAILABLE", table)
        self.assertIn("not determinable", table)
        self.assertNotIn("(block [", table)

    def test_double_count_note_does_not_blame_the_pre_fix_selector(self):
        """The selector now picks from THIS rank's own experts.

        An earlier version of this note told the reader that the selector
        deliberately picks a non-local expert and that shadow_state wanted
        the opposite. That was true before the ownership fix and is actively
        misleading after it: a non-zero NOT count now means the two ownership
        rules have diverged, not that the design intends the disagreement.
        Sending a reader after a cause that no longer exists is worse than
        saying nothing.
        """
        rows = [[0, 1, 2, 3]]
        trace._S.block_rank, trace._S.block_num_local = 0, 2
        rt = _fake_runtime(local_rank=0, slots={0: 3})
        with self._patched_meta(rows, num_local=2):
            table = trace._per_layer_table(rt, 0)
        self.assertIn("counted twice", table)
        self.assertIn("diverged", table)
        self.assertNotIn("deliberately", table)

    def test_missing_daemon_support_is_explained_not_silently_zero(self):
        rows = [[0, 1, 2, 3]]
        rt = _fake_runtime(
            local_rank=0,
            slots={0: 3},
            layer_stats={"supported": False, "version": 12, "layers": {}},
        )
        with self._patched_meta(rows, num_local=2):
            table = trace._per_layer_table(rt, 0)
        self.assertIn("predates slot v13", table)

    def test_daemon_numbers_are_reported_per_layer(self):
        rows = [[0, 1, 2, 3], [0, 1, 2, 3]]
        rt = _fake_runtime(
            local_rank=0,
            slots={0: 3, 1: 2},
            layer_stats={
                "supported": True,
                "version": 13,
                "trace_enabled": 1,
                "overflow": 0,
                "layers": {
                    0: {
                        "calls": 100,
                        "hit_calls": 80,
                        "active_toks": 400,
                        "pairs": 400,
                        "gemm_ns": 8_000_000,
                        "gemm_max_ns": 300_000,
                        "expert": 3,
                    },
                    1: {
                        "calls": 100,
                        "hit_calls": 10,
                        "active_toks": 20,
                        "pairs": 20,
                        "gemm_ns": 1_000_000,
                        "gemm_max_ns": 250_000,
                        "expert": 2,
                    },
                },
            },
        )
        with self._patched_meta(rows, num_local=2):
            table = trace._per_layer_table(rt, 0)
        # 8_000_000 ns over 80 hit calls == 100.0 us mean.
        self.assertIn("100.0", table)
        self.assertIn(
            "total CPU-side hit tokens (all layers, all replays) = 420", table
        )

    def test_layer_id_overflow_is_surfaced(self):
        rows = [[0, 1, 2, 3]]
        rt = _fake_runtime(
            local_rank=0,
            slots={0: 3},
            layer_stats={
                "supported": True,
                "version": 13,
                "trace_enabled": 1,
                "overflow": 7,
                "layers": {0: {"calls": 1}},
            },
        )
        with self._patched_meta(rows, num_local=2):
            table = trace._per_layer_table(rt, 0)
        self.assertIn("outside the traced range", table)


class TestPlacementCrossCheck(unittest.TestCase):
    """The two "which experts are mine" answers must be compared, not assumed.

    ``_global_to_local_map`` (what the dispatcher actually uses) and the EPLB
    ``physical_to_logical_map`` block slice (what the fast loop and the
    cuda-graph offload read) can describe different placements. Silently
    picking one would make the "own?" column authoritative-looking and wrong.
    """

    def setUp(self):
        trace.reset_for_tests()

    def tearDown(self):
        trace.reset_for_tests()

    def test_eplb_slice_agreeing_with_the_block_is_stated(self):
        trace._S.block_rank, trace._S.block_num_local = 0, 2
        msg = trace._placement_disagreement(0, {0, 1}, {0: {0, 1}})
        self.assertIn("AGREES with [1]", msg)
        # Agreement must be reported as contingent, not structural.
        self.assertIn("by coincidence, not by construction", msg)

    def test_dead_interleaved_map_is_named_as_unread(self):
        trace._S.block_rank, trace._S.block_num_local = 0, 2
        msg = trace._placement_disagreement(0, {0, 2}, {0: {0, 1}})
        self.assertIn("NOTHING READS IT", msg)
        self.assertIn("OPPOSITE", msg)

    def test_eplb_slice_disagreeing_with_the_block_is_stated(self):
        trace._S.block_rank, trace._S.block_num_local = 0, 2
        msg = trace._placement_disagreement(0, {0, 1}, {0: {2, 3}})
        self.assertIn("DISAGREES with [1]", msg)
        self.assertIn("wrong range", msg)

    def test_unknown_block_is_stated_not_guessed(self):
        trace._S.block_rank, trace._S.block_num_local = None, None
        msg = trace._placement_disagreement(0, None, {0: {0, 1}})
        self.assertIn("never observed", msg)


class TestReportAssembly(unittest.TestCase):
    def setUp(self):
        trace.reset_for_tests()

    def tearDown(self):
        trace.reset_for_tests()

    def test_all_five_sections_present(self):
        rt = _fake_runtime(slots={0: 5})
        trace.record_server_args(types.SimpleNamespace())
        trace.hit("moe.fused.forward", 3)
        trace.reason("hook.begin.eager_skip", "cpu_slot=-1", 2)
        trace.reselect(
            event="bootstrap",
            tick=0,
            forward_no=1,
            changes=[(0, -1, 5)],
            source="apply_initial_slots",
        )
        try:
            raise RuntimeError("swallowed")
        except RuntimeError as e:
            trace.exc("some.site", e)

        text = trace.build_report(rt)
        for marker in (
            "1. CODE PATHS EXECUTED",
            "2. HOW THE CONFIGURATION ADJUDICATED THIS PATH",
            "3. PER-LAYER OFFLOAD DECISION",
            "4. DYNAMIC RE-SELECTION HISTORY",
            "5. SWALLOWED EXCEPTIONS",
        ):
            self.assertIn(marker, text)
        self.assertIn("swallowed", text)
        self.assertIn("some.site", text)
        # The dead in-class hook must be visible as never-executed.
        self.assertIn("in-class offload begin", text)

    def test_report_points_at_the_journal_when_it_is_off(self):
        """A clean one-run report must not read as "nothing left to check"."""
        with patch.object(trace, "JOURNAL_PATH", ""):
            text = trace.build_report(_fake_runtime())
        self.assertIn("SGLANG_MOE_TRACE_JOURNAL", text)
        self.assertIn("cannot show", text)

    def test_report_names_the_journal_file_when_it_is_on(self):
        with patch.object(trace, "JOURNAL_PATH", "/tmp/run-a.jsonl"):
            text = trace.build_report(_fake_runtime())
        self.assertIn("/tmp/run-a.jsonl", text)
        self.assertIn("journal diff", text)

    def test_emit_report_prints_once(self):
        rt = _fake_runtime()
        with patch.object(trace, "ENABLED", True):
            buf = io.StringIO()
            with redirect_stdout(buf):
                trace.emit_report(rt)
                trace.emit_report(rt)
            # (the phrase also appears in the footer, so count a unique marker)
            self.assertEqual(buf.getvalue().count("1. CODE PATHS EXECUTED"), 1)

    def test_emit_report_is_a_noop_when_disabled(self):
        rt = _fake_runtime()
        buf = io.StringIO()
        with redirect_stdout(buf):
            trace.emit_report(rt)
        self.assertEqual(buf.getvalue(), "")

    def test_broken_report_does_not_raise_at_exit(self):
        rt = _fake_runtime()
        with (
            patch.object(trace, "ENABLED", True),
            patch.object(trace, "build_report", side_effect=RuntimeError("nope")),
        ):
            buf = io.StringIO()
            with redirect_stdout(buf):
                trace.emit_report(rt)  # must not propagate
            self.assertIn("report generation failed", buf.getvalue())

    def test_rank_filter(self):
        with (
            patch.object(trace, "ENABLED", True),
            patch.object(trace, "_REPORT_RANKS", {4}),
        ):
            self.assertFalse(trace.should_report(0))
            self.assertTrue(trace.should_report(4))


class TestSlotAbiMismatchIsLoud(unittest.TestCase):
    """A version skew on the shared mmap must be named, never degraded to a
    generic 'attach failed'."""

    def _backend(self, file_version, client_version):
        from sglang.srt.eplb.cpu_offload.config import CpuOffloadConfig
        from sglang.srt.eplb.cpu_offload.engine_daemon import DaemonBackend

        cfg = CpuOffloadConfig(enable_cpu_offload_fast_loop=True, cpu_active_ranks=[0])
        be = DaemonBackend(cfg, host_rank=0, world_size=1, num_numa_nodes=1)
        be._socket_id = 0
        be._slot_id = 0
        be._engine_mod = types.SimpleNamespace(
            cpu_client_probe_slot_version=lambda p, s, l: {
                "path": "/dev/shm/x.socket0.slot0",
                "exists": True,
                "magic": 0xC1100AD0,
                "version": file_version,
            },
            cpu_client_slot_abi_version=lambda: client_version,
        )
        return be

    def test_mismatch_names_both_versions(self):
        be = self._backend(file_version=12, client_version=13)
        buf = io.StringIO()
        with redirect_stdout(buf):
            msg = be._explain_open_slot_failure("/dev/shm/x")
        self.assertIn("SLOT ABI VERSION MISMATCH", msg)
        self.assertIn("SLOT_VERSION=12", msg)
        self.assertIn("SLOT_VERSION=13", msg)
        self.assertIn("restart cpu_expertd", msg)

    def test_matching_versions_are_not_reported_as_a_mismatch(self):
        be = self._backend(file_version=13, client_version=13)
        msg = be._explain_open_slot_failure("/dev/shm/x")
        self.assertNotIn("MISMATCH", msg)
        self.assertIn("not a version mismatch", msg)


class TestDecisionJournal(unittest.TestCase):
    """The journal is the machine-readable half of the trace.

    The exit report is written for a person and mixes in timings, which differ
    every run by construction. The journal must not: it is compared field by
    field against another run, so anything in it that varies without the
    decision varying turns into a false alarm.
    """

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.path = os.path.join(self._dir, "j.jsonl")
        trace.reset_for_tests()

    def tearDown(self):
        trace.reset_for_tests()
        shutil.rmtree(self._dir, ignore_errors=True)

    def _run(self, events):
        with (
            patch.object(trace, "ENABLED", True),
            patch.object(trace, "JOURNAL_PATH", self.path),
        ):
            for e in events:
                trace.reselect(**e)
        trace.reset_for_tests()  # closes the handle so the file is readable
        recs = [json.loads(x) for x in open(self._written()) if x.strip()]
        return recs[0], recs[1:]

    def _written(self):
        """The sink suffixes by rank, so the file is never exactly `path`."""
        import glob as _g

        hits = _g.glob(os.path.join(self._dir, "j.rank*.jsonl"))
        self.assertEqual(len(hits), 1, "expected one journal, got %s" % hits)
        return hits[0]

    def test_schema_version_is_declared_before_any_event(self):
        schema, evs = self._run(
            [
                dict(event="reselect", tick=1, forward_no=100, changes=[[3, 1, 2]]),
            ]
        )
        self.assertEqual(schema["event"], "_schema")
        self.assertEqual(schema["v"], trace.JOURNAL_SCHEMA)
        self.assertEqual(len(evs), 1)

    def test_nothing_is_written_without_an_explicit_path(self):
        with (
            patch.object(trace, "ENABLED", True),
            patch.object(trace, "JOURNAL_PATH", ""),
        ):
            trace.reselect(event="reselect", tick=1, forward_no=1, changes=[])
        # Check the whole directory, not one filename: the sink now derives
        # the name from the rank, so asserting on `self.path` alone would
        # pass even if a file had been written.
        self.assertEqual(os.listdir(self._dir), [])

    def test_forward_number_is_recorded_as_the_alignment_axis(self):
        _, evs = self._run(
            [
                dict(event="reselect", tick=1, forward_no=100, changes=[[3, 1, 2]]),
                dict(event="reselect", tick=2, forward_no=200, changes=[[3, 2, 5]]),
            ]
        )
        self.assertEqual([e["forward_no"] for e in evs], [100, 200])

    def test_no_wall_clock_field_is_emitted(self):
        """A timestamp would differ every run and defeat a field-wise diff."""
        _, evs = self._run(
            [
                dict(
                    event="reselect",
                    tick=1,
                    forward_no=100,
                    changes=[[3, 1, 2]],
                    basis={
                        "src": "counts",
                        "layers": 94,
                        "pairs": 790,
                        "digest": "abc",
                    },
                ),
            ]
        )
        blob = json.dumps(evs)
        for banned in ("time", "ts", "elapsed", "us", "ms", "sec"):
            self.assertNotIn('"%s"' % banned, blob)

    def test_every_rank_is_recorded_not_only_the_local_one(self):
        """Double-compute is a cross-rank event: one rank's view cannot see it."""
        _, evs = self._run(
            [
                dict(
                    event="reselect",
                    tick=1,
                    forward_no=100,
                    rank=0,
                    changes=[[3, -1, 20]],
                    xrank={0: [[3, -1, 20]], 1: [[3, -1, 90]]},
                ),
            ]
        )
        self.assertEqual(sorted(evs[0]["xrank"]), ["0", "1"])

    def test_records_are_key_ordered_so_two_runs_diff_cleanly(self):
        _, evs = self._run(
            [
                dict(
                    event="reselect",
                    tick=1,
                    forward_no=1,
                    changes=[],
                    source="z",
                    basis={"digest": "d", "pairs": 1},
                ),
            ]
        )
        line = [x for x in open(self._written()) if x.strip()][1]
        self.assertEqual(line.strip(), json.dumps(evs[0], sort_keys=True))

    def test_each_rank_gets_its_own_file(self):
        """One process per rank, all sharing one env var.

        A single literal path would have every rank appending to one file:
        records interleave, a long one can tear mid-write, and the ordering
        the comparison depends on is gone.
        """
        with (
            patch.object(trace, "ENABLED", True),
            patch.object(trace, "JOURNAL_PATH", os.path.join(self._dir, "d.%r.jsonl")),
        ):
            for r in (0, 3):
                trace.reset_for_tests()
                with (
                    patch.object(trace, "ENABLED", True),
                    patch.object(
                        trace, "JOURNAL_PATH", os.path.join(self._dir, "d.%r.jsonl")
                    ),
                ):
                    trace.reselect(
                        event="reselect", tick=1, forward_no=1, changes=[], rank=r
                    )
                trace.reset_for_tests()
        for r in (0, 3):
            self.assertTrue(
                os.path.exists(os.path.join(self._dir, "d.%d.jsonl" % r)),
                "rank %d wrote nowhere" % r,
            )

    def test_a_path_without_a_placeholder_still_separates_ranks(self):
        """Sharing must not be the silent default when %r is forgotten."""
        with (
            patch.object(trace, "ENABLED", True),
            patch.object(trace, "JOURNAL_PATH", self.path),
        ):
            trace.reselect(event="reselect", tick=1, forward_no=1, changes=[], rank=5)
        trace.reset_for_tests()
        self.assertFalse(os.path.exists(self.path))
        self.assertTrue(os.path.exists(os.path.join(self._dir, "j.rank5.jsonl")))

    def test_a_second_run_replaces_the_file_instead_of_appending(self):
        """Appending would read back as one long run that diverged midway."""
        for tick in (1, 2):
            trace.reset_for_tests()
            with (
                patch.object(trace, "ENABLED", True),
                patch.object(trace, "JOURNAL_PATH", self.path),
            ):
                trace.reselect(
                    event="reselect", tick=tick, forward_no=tick, changes=[], rank=0
                )
            trace.reset_for_tests()
        recs = [
            json.loads(x)
            for x in open(os.path.join(self._dir, "j.rank0.jsonl"))
            if x.strip()
        ]
        self.assertEqual(len(recs), 2)  # schema + exactly one event
        self.assertEqual(recs[1]["tick"], 2)

    def test_an_unwritable_path_does_not_kill_the_run(self):
        with (
            patch.object(trace, "ENABLED", True),
            patch.object(trace, "JOURNAL_PATH", "/nonexistent-dir/x.jsonl"),
        ):
            trace.reselect(event="reselect", tick=1, forward_no=1, changes=[], rank=0)
        self.assertTrue(trace._S.journal_broken)


def _basis(layer_counts, snapshot, *, with_counts=False):
    """``fast_loop._trace_basis``, imported late.

    The selector's module pulls in torch; this file otherwise does not, and
    keeping the import inside the call means the rest of the suite still runs
    where torch is absent.
    """
    from sglang.srt.eplb.cpu_offload.fast_loop import _trace_basis

    return _trace_basis(layer_counts, snapshot, with_counts=with_counts)


class _Counts:
    """Stands in for a per-layer count tensor: only ``.tolist()`` is read."""

    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class TestTrafficCounts(unittest.TestCase):
    """The counts are the margin, and the margin is what the digest destroys.

    A digest answers "did two runs see the same traffic". It cannot answer
    "how far ahead was the winner", which is the question a layer that keeps
    flipping between two experts actually raises -- and the two causes of
    that flipping (traffic genuinely moving vs. the selector chasing noise)
    have nothing in common except that the digest hides both.

    Keeping the numbers is only safe if it changes nothing else, so that is
    what these pin: same digest, absent unless asked for, capped, and the two
    traffic sources still told apart.
    """

    # Two layers, four experts. Layer 7's top two are one token apart: once
    # hashed, that is indistinguishable from a decisive win.
    def traffic(self):
        return {7: _Counts([10, 41, 40, 0]), 3: _Counts([0, 0, 5, 1])}

    # The fallback shape: per-layer, per-rank, per-expert token counts.
    def snapshot(self):
        return {5: {0: {1: 7}, 2: {1: 3}}}

    def test_counts_are_off_by_default(self):
        # The suite never sets SGLANG_MOE_TRACE_COUNTS, so this documents the
        # default and fails loudly if it ever flips. Size is the reason: a
        # 94-layer capture is hundreds of KB per rank per run.
        self.assertFalse(trace.COUNTS)
        self.assertFalse(trace.want_counts(1))

    def test_the_field_is_absent_unless_asked_for(self):
        """Absent, not empty: a consumer must not have to special-case it."""
        b = _basis(self.traffic(), {})
        self.assertNotIn("per_layer", b)

    def test_keeping_the_numbers_does_not_change_the_digest(self):
        """Otherwise turning charts on could flip a determinism verdict.

        ``diff`` compares the digest. If it moved when counts were enabled,
        a capture taken for the charts could not be compared against one
        taken without them -- and the comparison is the reason the journal
        exists.
        """
        off = _basis(self.traffic(), {})
        on = _basis(self.traffic(), {}, with_counts=True)
        self.assertEqual(off["digest"], on["digest"])
        for field in ("src", "layers", "pairs"):
            with self.subTest(field=field):
                self.assertEqual(off[field], on[field])

    def test_the_numbers_kept_are_the_ones_that_were_digested(self):
        """A vector recorded from anywhere else would not explain the choice."""
        b = _basis(self.traffic(), {}, with_counts=True)
        self.assertEqual(b["per_layer"][7], [10, 41, 40, 0])
        self.assertEqual(b["per_layer"][3], [0, 0, 5, 1])

    def test_the_margin_between_first_and_second_is_recoverable(self):
        """The whole point: 41 vs 40 must not read like 41 vs 0."""
        b = _basis(self.traffic(), {}, with_counts=True)
        top2 = sorted(b["per_layer"][7], reverse=True)[:2]
        self.assertEqual(top2[0] - top2[1], 1)

    def test_the_fallback_path_keeps_its_rank_prefix(self):
        """Folding it into a flat vector would plot the same run two ways.

        The cross-rank path sums to global traffic; the fallback is one
        rank's own. Collapsing the difference here would make a chart depend
        on which path the run happened to take, with nothing in the record
        saying so.
        """
        b = _basis({}, self.snapshot(), with_counts=True)
        self.assertEqual(b["src"], "snapshot")
        self.assertEqual(b["per_layer"][5], {"0/1": 7, "2/1": 3})

    def test_the_two_sources_are_never_the_same_shape(self):
        """So a reader can tell which one they are holding without the label."""
        summed = _basis(self.traffic(), {}, with_counts=True)
        fallback = _basis({}, self.snapshot(), with_counts=True)
        self.assertIsInstance(summed["per_layer"][7], list)
        self.assertIsInstance(fallback["per_layer"][5], dict)
        self.assertNotEqual(summed["src"], fallback["src"])

    def test_counts_stop_at_the_cap(self):
        """A capture whose length is set by generation length must not be
        able to become a multi-gigabyte journal by accident."""
        with patch.object(trace, "COUNTS", True), patch.object(trace, "COUNTS_MAX", 2):
            self.assertTrue(trace.want_counts(1))
            self.assertTrue(trace.want_counts(2))
            self.assertFalse(trace.want_counts(3))

    def test_past_the_cap_the_digest_survives(self):
        """Only the charts lose their tail; the comparison keeps working."""
        with patch.object(trace, "COUNTS", True), patch.object(trace, "COUNTS_MAX", 2):
            late = _basis(self.traffic(), {}, with_counts=trace.want_counts(3))
        self.assertNotIn("per_layer", late)
        self.assertEqual(late["digest"], _basis(self.traffic(), {})["digest"])

    def test_the_cap_reads_a_bad_value_as_the_default(self):
        """A typo in an env var must not silently disable the charts."""
        for bad in ("", "   ", "sixty", "6.5"):
            with self.subTest(value=bad):
                with patch.dict(os.environ, {"SGLANG_MOE_TRACE_COUNTS_MAX": bad}):
                    self.assertEqual(
                        trace._env_int("SGLANG_MOE_TRACE_COUNTS_MAX", 60), 60
                    )

    def test_no_traffic_produces_no_counts_field(self):
        """The early return must not leave a half-formed record behind."""
        b = _basis({}, {}, with_counts=True)
        self.assertEqual(b["src"], "none")
        self.assertNotIn("per_layer", b)


class TestJournalVerdicts(unittest.TestCase):
    """The verifier must name WHICH of three unrelated causes to chase.

    "The two runs differ" was already known before any of this existed. The
    value is entirely in the classification, so each class is pinned against
    a journal pair that differs in exactly one dimension.
    """

    def setUp(self):
        self._dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._dir, ignore_errors=True)

    def _write(self, name, events, schema=1):
        p = os.path.join(self._dir, name)
        with open(p, "w") as fh:
            if schema is not None:
                fh.write(json.dumps({"event": "_schema", "v": schema}) + "\n")
            for e in events:
                fh.write(json.dumps(e, sort_keys=True) + "\n")
        return p

    @staticmethod
    def _ev(tick, fwd, digest="aaa", pairs=790, xrank=None):
        xrank = xrank or {"0": [[3, -1, 20]], "1": [[3, -1, 90]]}
        return {
            "event": "reselect",
            "tick": tick,
            "forward_no": fwd,
            "source": "cuda_graph_acc_xrank",
            "changes": xrank["0"],
            "xrank": xrank,
            "basis": {"src": "counts", "layers": 94, "pairs": pairs, "digest": digest},
        }

    def _baseline(self):
        return [self._ev(1, 1), self._ev(2, 101), self._ev(3, 201)]

    def _verdict(self, other):
        a = self._write("a.jsonl", self._baseline())
        b = self._write("b.jsonl", other)
        ok, lines = journal.diff(a, b)
        return ok, "\n".join(lines)

    def test_same_basis_different_choice_blames_the_scheduler(self):
        other = self._baseline()
        other[1] = self._ev(2, 101, xrank={"0": [[3, -1, 77]], "1": [[3, -1, 90]]})
        ok, text = self._verdict(other)
        self.assertFalse(ok)
        self.assertIn("SCHEDULER", text)
        self.assertNotIn("UPSTREAM", text)

    def test_different_basis_blames_upstream_not_the_scheduler(self):
        other = self._baseline()
        other[1] = self._ev(2, 101, digest="bbb", pairs=812)
        ok, text = self._verdict(other)
        self.assertFalse(ok)
        self.assertIn("UPSTREAM", text)
        self.assertNotIn("SCHEDULER", text)

    def test_same_decision_at_a_different_step_blames_the_gate(self):
        """The original defect's exact shape: right choice, wrong step."""
        other = self._baseline()
        other[1] = self._ev(2, 107)
        ok, text = self._verdict(other)
        self.assertFalse(ok)
        self.assertIn("GATE", text)

    def test_identical_runs_are_reported_as_reproducible(self):
        ok, text = self._verdict(self._baseline())
        self.assertTrue(ok)
        self.assertIn("IDENTICAL", text)

    def test_only_the_first_divergence_is_reported(self):
        other = self._baseline()
        other[1] = self._ev(2, 107)
        other[2] = self._ev(3, 999, digest="zzz")
        _, text = self._verdict(other)
        self.assertIn("event #1", text)
        self.assertNotIn("event #2", text)

    def test_a_longer_run_is_not_mistaken_for_a_wrong_decision(self):
        ok, text = self._verdict(self._baseline() + [self._ev(4, 301)])
        self.assertFalse(ok)
        self.assertIn("length", text)
        for klass in ("SCHEDULER", "UPSTREAM", "GATE"):
            self.assertNotIn(klass, text)

    def test_two_ranks_taking_one_expert_is_flagged_as_double_compute(self):
        both = {"0": [[3, -1, 90]], "1": [[3, -1, 90]]}
        p = self._write(
            "x.jsonl", [self._ev(1, 1), self._ev(2, 101, xrank=both), self._ev(3, 201)]
        )
        ok, lines = journal.check(p)
        text = "\n".join(lines)
        self.assertFalse(ok)
        self.assertIn("expert=90", text)
        self.assertIn("twice", text)

    def test_a_wobbling_stride_is_flagged(self):
        p = self._write("x.jsonl", [self._ev(1, 1), self._ev(2, 101), self._ev(3, 217)])
        ok, lines = journal.check(p)
        self.assertFalse(ok)
        self.assertIn("stride is not constant", "\n".join(lines))

    def test_a_healthy_journal_passes_but_says_it_proves_less(self):
        p = self._write("x.jsonl", self._baseline())
        ok, lines = journal.check(p)
        text = "\n".join(lines)
        self.assertTrue(ok)
        self.assertIn("NOT determinism", text)

    def test_an_empty_journal_is_not_called_healthy(self):
        p = self._write("x.jsonl", [])
        ok, lines = journal.check(p)
        text = "\n".join(lines)
        self.assertIn("EMPTY", text)
        self.assertIn("not evidence", text)

    def test_an_unknown_schema_refuses_to_compare(self):
        p = self._write("x.jsonl", self._baseline(), schema=99)
        with self.assertRaises(journal.JournalError) as cm:
            journal.load(p)
        self.assertIn("Refusing", str(cm.exception))

    def test_a_file_without_a_schema_line_is_rejected(self):
        p = self._write("x.jsonl", self._baseline(), schema=None)
        with self.assertRaises(journal.JournalError):
            journal.load(p)

    def test_a_killed_run_leaves_a_usable_prefix(self):
        """Per-event flush means a crash costs at most the last line."""
        p = self._write("x.jsonl", self._baseline())
        with open(p, "a") as fh:
            fh.write('{"event": "reselect", "tick": 4, "forwa')
        events, _ = journal.load(p)
        self.assertEqual(len(events), 3)


if __name__ == "__main__":
    unittest.main()
