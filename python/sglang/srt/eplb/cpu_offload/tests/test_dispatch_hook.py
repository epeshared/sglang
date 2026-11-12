"""Tests for the shared CPU-offload dispatch hook used by both Triton
FusedMoE and EP-MoE paths.

These tests exercise ``begin_cpu_offload`` / ``commit_cpu_offload`` at
the (topk_ids, topk_weights) signature level — no TopKOutput plumbing,
no real CUDA stream — so the same logic the EP-MoE path now calls is
verified.
"""

from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.eplb.cpu_offload.config import CpuOffloadConfig
from sglang.srt.eplb.cpu_offload.dispatch_hook import (
    begin_cpu_offload,
    commit_cpu_offload,
)
from sglang.srt.eplb.cpu_offload.metrics import CpuOffloadMetrics
from sglang.srt.eplb.cpu_offload.shadow_state import CpuShadowRegistry


def _install_runtime(
    *,
    active: bool = True,
    cpu_slot: int = 7,
    layer_registered: bool = True,
    layer_id: int = 0,
):
    """Build a fully-mocked CpuOffloadRuntime and install it as the singleton.

    ``layer_id`` must match whatever ``layer_id`` the test subsequently
    passes to ``begin_cpu_offload``/``commit_cpu_offload`` — the CPU slot is
    now assigned per layer (not one rank-wide scalar), so a mismatch means
    the runtime will report no slot for that layer.
    """
    from sglang.srt.eplb.cpu_offload import runtime as runtime_mod

    cfg = CpuOffloadConfig(
        enable_cpu_offload_fast_loop=True,
        cpu_active_ranks=[0] if active else [],
        cpu_offload_max_tokens_per_step=10_000,
        cpu_offload_max_tokens_per_expert=10_000,
    )
    metrics = CpuOffloadMetrics()
    registry = CpuShadowRegistry(cfg, world_size=1, num_numa_nodes=1, local_rank=0)
    counter = runtime_mod._LocalExpertLoadCounter(local_rank=0)
    fake_engine = MagicMock()
    fake_engine.initialized = True
    fake_engine.has_layer = MagicMock(return_value=layer_registered)
    fake_engine.launch_layer = MagicMock()
    fake_engine.commit_layer = MagicMock()

    rt = runtime_mod.CpuOffloadRuntime(
        config=cfg,
        registry=registry,
        engine=fake_engine,
        fast_loop=MagicMock(),
        metrics=metrics,
        counter=counter,
        local_rank=0,
    )
    # Force the local slot to the requested expert id.
    state = registry.local_state()
    if state is not None and cpu_slot >= 0:
        state.assign_slot(layer_id, cpu_slot)
    runtime_mod._runtime = rt
    return rt, fake_engine


def _uninstall_runtime():
    from sglang.srt.eplb.cpu_offload import runtime as runtime_mod

    runtime_mod._runtime = None


class TestBeginCpuOffload(unittest.TestCase):
    def setUp(self):
        self._patched_stream = torch.cuda.current_stream
        torch.cuda.current_stream = lambda: types.SimpleNamespace(cuda_stream=0x1)

    def tearDown(self):
        torch.cuda.current_stream = self._patched_stream
        _uninstall_runtime()

    def test_returns_none_when_no_runtime(self):
        _uninstall_runtime()
        h = torch.zeros((2, 4))
        ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        w = torch.ones_like(ids, dtype=torch.float32)
        self.assertIsNone(
            begin_cpu_offload(layer_id=0, hidden_states=h, topk_ids=ids, topk_weights=w)
        )

    def test_returns_none_when_no_cpu_tokens_match_slot(self):
        rt, _eng = _install_runtime(cpu_slot=99)  # 99 doesn't appear in topk
        h = torch.zeros((2, 4))
        ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        w = torch.ones_like(ids, dtype=torch.float32)
        self.assertIsNone(
            begin_cpu_offload(layer_id=0, hidden_states=h, topk_ids=ids, topk_weights=w)
        )
        # Skip should have been counted.
        self.assertGreaterEqual(rt.metrics.cpu_offload_miss_count, 1)

    def test_returns_none_when_engine_does_not_have_layer(self):
        _install_runtime(layer_registered=False)
        h = torch.zeros((2, 4))
        ids = torch.tensor([[7, 1], [7, 3]], dtype=torch.int32)
        w = torch.ones_like(ids, dtype=torch.float32)
        self.assertIsNone(
            begin_cpu_offload(layer_id=0, hidden_states=h, topk_ids=ids, topk_weights=w)
        )

    def test_routes_matching_tokens_to_engine(self):
        rt, eng = _install_runtime(cpu_slot=7, layer_id=3)
        h = torch.randn((4, 8))
        ids = torch.tensor([[7, 1], [2, 3], [7, 4], [5, 6]], dtype=torch.int32)
        w = torch.ones_like(ids, dtype=torch.float32) * 0.5
        ctx = begin_cpu_offload(
            layer_id=3, hidden_states=h, topk_ids=ids, topk_weights=w
        )
        self.assertIsNotNone(ctx)
        # Tokens 0 and 2 hit expert 7.
        self.assertEqual(ctx["n_cpu_tokens"], 2)
        self.assertEqual(ctx["layer_id"], 3)
        self.assertEqual(ctx["token_indices"].tolist(), [0, 2])
        # masked_topk_ids should -1 the matching entry.
        self.assertEqual(ctx["masked_topk_ids"][0, 0].item(), -1)
        self.assertEqual(ctx["masked_topk_ids"][2, 0].item(), -1)
        # Engine.launch_layer was invoked with dev_output preallocated.
        eng.launch_layer.assert_called_once()
        kwargs = eng.launch_layer.call_args.kwargs
        self.assertIn("dev_output", kwargs)
        self.assertEqual(kwargs["n_tokens"], 2)


class TestCommitCpuOffload(unittest.TestCase):
    def setUp(self):
        self._patched_stream = torch.cuda.current_stream
        torch.cuda.current_stream = lambda: types.SimpleNamespace(cuda_stream=0x1)

    def tearDown(self):
        torch.cuda.current_stream = self._patched_stream
        _uninstall_runtime()

    def test_commit_scatter_adds_into_final(self):
        rt, eng = _install_runtime(cpu_slot=7)
        h = torch.zeros((4, 4))
        ids = torch.tensor([[7, 1], [2, 3], [7, 4], [5, 6]], dtype=torch.int32)
        w = torch.ones_like(ids, dtype=torch.float32)
        ctx = begin_cpu_offload(
            layer_id=0, hidden_states=h, topk_ids=ids, topk_weights=w
        )
        self.assertIsNotNone(ctx)
        # Pretend the engine wrote a known pattern into cpu_contrib.
        ctx["cpu_contrib"][:] = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        )
        final = torch.zeros((4, 4))
        out = commit_cpu_offload(ctx, final)
        # Indices 0 and 2 must contain the CPU contribution.
        self.assertTrue(torch.equal(out[0], ctx["cpu_contrib"][0]))
        self.assertTrue(torch.equal(out[2], ctx["cpu_contrib"][1]))
        # Indices 1 and 3 untouched.
        self.assertTrue(torch.equal(out[1], torch.zeros(4)))
        self.assertTrue(torch.equal(out[3], torch.zeros(4)))
        eng.commit_layer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
