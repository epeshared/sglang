"""已知步号与真实 bench 假引擎对照，核对窗口两端和逐步同步次数。"""

import unittest
from unittest import mock

from sglang.bench_profile_window import focus_spec, profile_steps, validate_focus
from sglang.test import test_bench_one_batch_sweep as base


class WindowDefinitionTests(unittest.TestCase):
    def test_internal_eight_phases_and_guards(self):
        spec = focus_spec((8, 1024, 1024, 8, 15))
        self.assertEqual(spec["capture_steps"], [505, 528])
        self.assertEqual(profile_steps(1024, spec["args"]), (504, 527))
        self.assertEqual(spec["capture_count"], 24)
        self.assertEqual(profile_steps(1024), (512, 512))

    def test_reject_missing_duplicate_or_invalid_window(self):
        for values in ((8, 1024, 4, 8, 15), (8, 1024, 1024, -1, 15), (8,)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                focus_spec(values)
        for batches, pairs in (
            ([1], ["1024:1024"]),
            ([8, 8], ["1024:1024"]),
            ([8], ["1024:128"]),
        ):
            with (
                self.subTest(batches=batches, pairs=pairs),
                self.assertRaises(ValueError),
            ):
                validate_focus(
                    (8, 1024, 1024, 8, 15),
                    batches,
                    pairs,
                    True,
                    ("CUDA_PROFILER",),
                    "decode",
                )


class WindowExecutionTests(base.ProfileTests):
    def test_only_target_segment_profiles_and_records_every_step(self):
        args = self._bench_args(
            profile_focus_window=(2, 8, 5, 1, 1), trace_target_steps=1
        )
        original = base.bo.latency_test_run_once
        calls = []

        def run(*a, **kw):
            import inspect

            bound = inspect.signature(original).bind(*a, **kw).arguments
            calls.append(
                (bound["batch_size"], bound["profile"], kw.get("profile_focus_window"))
            )
            return original(*a, **kw)

        with (
            mock.patch.object(base.bo, "latency_test_run_once", run),
            mock.patch.object(
                base.bo.step_trace,
                "start_segment",
                wraps=base.bo.step_trace.start_segment,
            ) as start,
        ):
            self._sweep(args, 0)
        self.assertEqual(calls[-2:], [(1, False, ()), (2, True, (2, 8, 5, 1, 1))])
        self.assertEqual(
            [c.kwargs["sample_every"] for c in start.call_args_list], [4, 1]
        )
        begin, end = self.events.index("开抓"), self.events.index("停抓")
        self.assertEqual(self.events[begin:end].count("解码"), 3)
        self.assertEqual(self.events.count("对齐"), 2)


if __name__ == "__main__":
    unittest.main()
