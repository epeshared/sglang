"""CPU checks for sink normalization and unsupported backend combinations."""

import math
import unittest

import torch

from sglang.kernels.ops.attention.dsa.sparse_mla_sinks import (
    apply_sparse_mla_sinks,
    skip_sparse_mla_sink_schedule,
    validate_sparse_mla_sink_backend,
    validate_sparse_mla_sinks,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSparseMLASinks(unittest.TestCase):
    def test_sm89_short_context_schedule_guard(self):
        self.assertTrue(skip_sparse_mla_sink_schedule(True, 2048, 2048, True))
        self.assertFalse(skip_sparse_mla_sink_schedule(False, 4096, 2048, True))
        self.assertFalse(skip_sparse_mla_sink_schedule(True, 4096, 2048, False))
        with self.assertRaisesRegex(NotImplementedError, "context <= index_topk"):
            skip_sparse_mla_sink_schedule(True, 2049, 2048, True)

    def test_closed_form_and_base2_conversion(self):
        out, lse = apply_sparse_mla_sinks(
            torch.tensor([[[4.0]]], dtype=torch.bfloat16),
            torch.tensor([[2.0]]),
            torch.tensor([math.log(3.0)]),
            torch.tensor([[0, 1]]),
        )
        torch.testing.assert_close(out, torch.tensor([[[16 / 7]]]).bfloat16())
        torch.testing.assert_close(lse, torch.log2(torch.tensor([[7.0]])))

    def test_empty_row_clears_nan_and_finite_lse_sentinel(self):
        out, lse = apply_sparse_mla_sinks(
            torch.full((1, 2, 3), torch.nan, dtype=torch.bfloat16),
            torch.full((1, 2), -1073741824.0),
            torch.tensor([0.0, 1000.0]),
            torch.tensor([[-1, -1]]),
        )
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        torch.testing.assert_close(lse, torch.tensor([[0.0, 1000.0]]) / math.log(2))

    def test_length_masks_unscanned_keys(self):
        out, _ = apply_sparse_mla_sinks(
            torch.ones(1, 1, 1, dtype=torch.bfloat16),
            torch.zeros(1, 1),
            torch.zeros(1),
            torch.tensor([[0, 1]]),
            torch.tensor([0]),
        )
        self.assertEqual(out.item(), 0.0)

    def test_disabled_and_extreme_sinks(self):
        out, _ = apply_sparse_mla_sinks(
            torch.full((1, 3, 1), 4.0, dtype=torch.bfloat16),
            torch.zeros(1, 3),
            torch.tensor([-torch.inf, -1000.0, 1000.0]),
            torch.tensor([[0]]),
        )
        torch.testing.assert_close(
            out, torch.tensor([[[4.0], [4.0], [0.0]]]).bfloat16()
        )

    def test_bf16_and_sink_layout_guards(self):
        q, rope, kv = (torch.zeros(1, 2, 8).bfloat16() for _ in range(3))
        validate_sparse_mla_sinks(q, rope, kv, torch.zeros(2))
        for bad in (torch.zeros(3), torch.zeros(2).bfloat16(), torch.zeros(4)[::2]):
            with self.assertRaises(ValueError):
                validate_sparse_mla_sinks(q, rope, kv, bad)
        for inputs in (
            (q.float(), rope, kv),
            (q, rope.float(), kv),
            (q, rope, kv.float()),
        ):
            with self.assertRaisesRegex(ValueError, "BF16"):
                validate_sparse_mla_sinks(*inputs, torch.zeros(2))

    def test_backend_and_dcp_guards(self):
        sinks = torch.zeros(2)
        validate_sparse_mla_sink_backend(sinks, "triton", 1, False)
        validate_sparse_mla_sink_backend(sinks, "flashmla_sparse", 2, True)
        validate_sparse_mla_sink_backend(None, "tilelang", 2, True)
        with self.assertRaises(RuntimeError):
            validate_sparse_mla_sink_backend(sinks, "tilelang", 1, False)
        for size, shards in ((2, False), (1, True)):
            with self.assertRaises(NotImplementedError):
                validate_sparse_mla_sink_backend(sinks, "triton", size, shards)


if __name__ == "__main__":
    unittest.main()
