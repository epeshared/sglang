"""Closed-form sink checks for the existing BF16 fused/split-K MLA kernels."""

import unittest

import torch

from sglang.kernels.ops.attention.dsa.triton_sparse_mla import triton_sparse_mla_fwd
from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
    triton_sparse_mla_decode_splitk,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTritonSparseMLASinks(unittest.TestCase):
    def setUp(self):
        self.q = torch.zeros(4, 8, 576, device="cuda", dtype=torch.bfloat16)
        self.kv = torch.full((2048, 1, 576), 4.0, device="cuda", dtype=torch.bfloat16)
        self.indices = torch.full((4, 1, 2048), -1, device="cuda", dtype=torch.int32)
        self.counts = torch.tensor([0, 1, 77, 2048], device="cuda", dtype=torch.int32)
        for row, count in enumerate((0, 1, 77, 2048)):
            self.indices[row, 0, :count] = torch.arange(count, device="cuda")
        self.sinks = torch.tensor(
            [-float("inf"), -1000, -2, 0, 1, 2, 8, 1000],
            device="cuda",
            dtype=torch.float32,
        )

    def _check(self, output):
        # Zero query makes every real score zero. All values equal 4, so the
        # result is 4 * count / (count + exp(sink)), independent of dot rounding.
        count = self.counts.float()[:, None]
        factor = torch.exp(
            torch.log(count) - torch.logaddexp(torch.log(count), self.sinks)
        )
        expected = torch.where(count > 0, 4 * factor, 0.0)
        expected = expected[:, :, None].expand(4, 8, 512).bfloat16()
        self.assertTrue(bool(torch.isfinite(output).all()))
        torch.testing.assert_close(output.squeeze(0), expected, rtol=0.005, atol=0)

    def test_fused_and_split_k(self):
        for splits in (1, 4):
            with self.subTest(splits=splits):
                output, lse = triton_sparse_mla_decode_splitk(
                    self.q[..., :512],
                    self.q[..., 512:],
                    self.kv,
                    self.indices,
                    0.0625,
                    kv_splits=splits,
                    lengths=self.counts,
                    return_lse=True,
                    attn_sink=self.sinks,
                )
                self._check(output)
                expected_lse = (
                    torch.logaddexp(torch.log(self.counts.float()[:, None]), self.sinks)
                    / 0.6931471805599453
                )
                torch.testing.assert_close(lse, expected_lse, rtol=1e-5, atol=1e-5)

    def test_prefill_entry(self):
        self._check(
            triton_sparse_mla_fwd(
                self.q[..., :512],
                self.q[..., 512:],
                self.kv,
                self.indices,
                0.0625,
                topk_length=self.counts,
                attn_sink=self.sinks,
            )
        )

    def test_disabled_sink_keeps_nonempty_output(self):
        args = (
            self.q[1:, :, :512],
            self.q[1:, :, 512:],
            self.kv,
            self.indices[1:],
            0.0625,
        )
        base = triton_sparse_mla_decode_splitk(*args, kv_splits=4)
        actual = triton_sparse_mla_decode_splitk(
            *args, kv_splits=4, attn_sink=torch.full_like(self.sinks, -torch.inf)
        )
        self.assertTrue(torch.equal(actual, base))

    def test_prefill_preserves_two_dimensional_single_key_indices(self):
        indices = self.indices[:, 0, :1]
        output = triton_sparse_mla_fwd(
            self.q[..., :512],
            self.q[..., 512:],
            self.kv,
            indices,
            0.0625,
            attn_sink=self.sinks,
        ).squeeze(0)
        count = (indices >= 0).sum(-1, keepdim=True).float()
        factor = torch.exp(
            torch.log(count) - torch.logaddexp(torch.log(count), self.sinks)
        )
        expected = torch.where(count > 0, 4 * factor, 0.0)
        expected = expected[..., None].expand_as(output).bfloat16()
        torch.testing.assert_close(output, expected, rtol=0.005, atol=0)


if __name__ == "__main__":
    unittest.main()
