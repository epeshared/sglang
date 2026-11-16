"""Exercise the production KT adapter methods without loading CUDA dependencies."""

from __future__ import annotations

import ast
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SOURCE = (
    Path(__file__).resolve().parents[5]
    / "python/sglang/srt/layers/moe/kt_ep_wrapper.py"
)


@dataclass
class RunnerConfig:
    num_local_experts: int = 8
    routed_scaling_factor: float = 2.827
    swiglu_limit: float | None = 10.0


class GpuMethod:
    def create_weights(self, **kwargs):
        self.weight_args = kwargs

    def create_moe_runner(self, layer, config):
        self.runner_config = config


def adapter(rank=0, hot=2, limit=10.0, last=False):
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "KTEPWrapperMethod"
    )
    calls = []

    def record(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    namespace = dict(
        torch=torch,
        replace=replace,
        FusedMoEMethodBase=object,
        KTRANSFORMERS_AVAILABLE=True,
        KTMoEWrapper=record,
        get_parallel=lambda: SimpleNamespace(tp_rank=rank),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    config = SimpleNamespace(
        num_gpu_experts=hot,
        max_deferred_experts_per_token=1,
        num_layers=2,
        layer_idx=1 if last else 0,
        cpuinfer_threads=4,
        threadpool_count=1,
        weight_path="unused",
        chunked_prefill_size=8,
        method="BF16",
    )
    gpu = GpuMethod()
    method = namespace["KTEPWrapperMethod"](gpu, config)
    layer = SimpleNamespace(
        top_k=4,
        intermediate_size_per_partition=32,
        moe_tp_size=2,
        moe_runner_config=RunnerConfig(swiglu_limit=limit),
    )
    method.create_weights(layer, 8, 64, 32, torch.bfloat16)
    return method, gpu, layer, calls


class TestKtBf16Contract(unittest.TestCase):
    def test_physical_gpu_mask_and_full_cpu_shape(self):
        _, gpu, _, calls = adapter()
        mask = calls[0]["gpu_experts_mask"]
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.device.type, "cpu")
        self.assertEqual(mask.tolist(), [True, True] + [False] * 6)
        self.assertEqual(calls[0]["moe_intermediate_size"], 64)
        self.assertEqual(gpu.weight_args["num_experts"], 2)

    def test_cpu_receives_limit(self):
        self.assertEqual(adapter()[3][0]["swiglu_limit"], 10.0)

    def test_missing_limit_is_disabled(self):
        self.assertEqual(adapter(limit=None)[3][0]["swiglu_limit"], 0.0)

    def test_nonzero_rank_does_not_construct_cpu_wrapper(self):
        self.assertEqual(adapter(rank=1)[3], [])

    def test_gpu_scaling_is_deferred_without_mutating_shared_config(self):
        method, gpu, layer, _ = adapter()
        original = layer.moe_runner_config
        method.create_moe_runner(layer, original)
        self.assertIsNone(gpu.runner_config.routed_scaling_factor)
        self.assertEqual(gpu.runner_config.num_local_experts, 2)
        self.assertEqual(gpu.runner_config.swiglu_limit, 10.0)
        self.assertEqual(original.routed_scaling_factor, 2.827)
        self.assertEqual(original.num_local_experts, 8)

    def test_zero_and_all_gpu_masks(self):
        for hot in (0, 8):
            with self.subTest(hot=hot):
                self.assertEqual(
                    adapter(hot=hot)[3][0]["gpu_experts_mask"].sum().item(), hot
                )

    def test_last_layer_does_not_defer_experts(self):
        self.assertEqual(adapter(last=True)[3][0]["max_deferred_experts_per_token"], 0)


if __name__ == "__main__":
    unittest.main()
