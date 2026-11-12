"""合并准备段（CPU_OFFLOAD_FUSED_PREPARE）在 begin_offload 里的接线。

真 kernel 的逐位正确性由 src/moe-fused-prepare/kernel_check 在 GPU 上验；这里只验
Python 这一侧：参数按绑定的次序、指到对的缓冲；合并路径交给下游的东西
（slot_weight、hit_indices、masked_topk_ids、计数）和原路径一样；用不了时退回原路径。

本机 torch 可以是 CPU 版：上下文不调 initialize()，缓冲直接用 CPU 张量，不开副流，
moe_cpu_engine 用桩模块。桩里的合并绑定按 engine_bindings.cpp 的参数次序，通过
指针直接读写这些 CPU 张量，所以参数次序错了、指针给错了，结果都会对不上。
"""

from __future__ import annotations

import contextlib
import ctypes
import io
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.eplb.cpu_offload import cuda_graph_offload as cgo

T, K, MH, H, E, LAYER = 64, 8, 16, 32, 512, 3

_CTYPE = {
    torch.int32: ctypes.c_int32,
    torch.int64: ctypes.c_int64,
    torch.float32: ctypes.c_float,
    torch.int16: ctypes.c_int16,
}


def _view(ptr: int, n: int, dtype: torch.dtype) -> torch.Tensor:
    """按指针取 n 个元素的 CPU 内存视图（共享内存，写进去就是写原张量）。"""
    if n == 0:
        return torch.empty(0, dtype=dtype)
    return torch.frombuffer((_CTYPE[dtype] * n).from_address(ptr), dtype=dtype)


def _bits(t: torch.Tensor) -> torch.Tensor:
    """浮点按位比较用。"""
    return t.contiguous().view(torch.int32 if t.dtype == torch.float32 else torch.int16)


class _FakeEngine:
    """moe_cpu_engine 桩：记录调用；合并绑定和 collect 按约定语义真写内存。"""

    def __init__(self, fused_rc: int = 0):
        self.calls: list[str] = []
        self.gathered = None
        self.fused_rc = fused_rc

    def module(self, with_fused: bool = True) -> types.ModuleType:
        m = types.ModuleType("moe_cpu_engine")
        m.cpu_client_cuda_collect_hit_indices = self.collect
        m.cpu_client_cuda_gather_compact = self.gather
        m.cpu_client_cuda_count_valid_pairs = self.count_pairs
        if with_fused:
            m.cpu_client_cuda_fused_prepare = self.fused
        return m

    def collect(self, stream, mask_ptr, hits_ptr, n, max_hits):
        self.calls.append("collect")
        mask = _view(mask_ptr, n, torch.int32)
        rows = torch.nonzero(mask, as_tuple=False).flatten().to(torch.int32)[:max_hits]
        hits = _view(hits_ptr, max_hits, torch.int32)
        hits.fill_(-1)
        hits[: rows.numel()] = rows

    def gather(
        self,
        stream,
        src_h,
        dst_h,
        w_ptr,
        dst_w,
        dst_ids,
        hits_ptr,
        slot_ptr,
        max_hits,
        hidden,
        topk,
    ):
        self.calls.append("gather")
        hits = _view(hits_ptr, max_hits, torch.int32).clone()
        rows = hits[hits >= 0].long()
        # 只读命中行的 slot_weight：没命中行的值下游不用。
        n_rows = int(rows.max()) + 1 if rows.numel() else 0
        w = _view(w_ptr, n_rows, torch.float32).clone()
        self.gathered = (hits, _bits(w[rows]) if rows.numel() else w)

    def count_pairs(self, *args):
        self.calls.append("count_pairs")

    def fused(
        self,
        stream,
        ids_ptr,
        w_ptr,
        slot_ptr,
        count_ptr,
        e,
        masked_ptr,
        sw_ptr,
        mf_ptr,
        mi_ptr,
        has_ptr,
        hits_ptr,
        n,
        k,
        max_hits,
    ):
        self.calls.append("fused")
        if self.fused_rc != 0:
            return self.fused_rc
        ids = _view(ids_ptr, n * k, torch.int32).view(n, k).clone()
        w = _view(w_ptr, n * k, torch.float32).view(n, k).clone()
        slot = int(_view(slot_ptr, 1, torch.int32)[0])
        if count_ptr:
            cnt = _view(count_ptr, e, torch.int64)
            valid = ids[ids >= 0].long().clamp(max=e - 1)
            cnt += torch.bincount(valid, minlength=e)
        branch = ids == slot
        hit = branch.any(dim=1)
        pos = branch.long().argmax(dim=1)
        _view(masked_ptr, n * k, torch.int32).copy_(
            ids.masked_fill(branch, -1).flatten()
        )
        _view(sw_ptr, n, torch.float32).copy_(
            w.gather(1, pos.unsqueeze(1)).squeeze(1) * hit.float()
        )
        _view(mf_ptr, n, torch.int16).copy_(
            torch.where(
                hit,
                torch.tensor(0x3F80, dtype=torch.int16),
                torch.tensor(0, dtype=torch.int16),
            )
        )
        _view(mi_ptr, n, torch.int32).copy_(hit.to(torch.int32))
        _view(has_ptr, 1, torch.int32).fill_(int(bool(hit.any())))
        rows = torch.nonzero(hit).flatten().to(torch.int32)[:max_hits]
        hits = _view(hits_ptr, max_hits, torch.int32)
        hits.fill_(-1)
        hits[: rows.numel()] = rows
        return 0


class _FakeRemoteMOE:
    _kernel_path_ready = True

    def __init__(self, ctx):
        self.ctx = ctx
        self.submitted_has_hit = None

    def launch_submit(self, **kw):
        self.submitted_has_hit = int(self.ctx._has_any_hit_buf[0])


def _make_ctx(fused_fn, metrics: bool = True):
    ctx = cgo.CudaGraphOffloadContext(hidden_size=H, topk=K, max_tokens=T, max_hits=MH)
    ctx._gpu_quant = False
    ctx._metrics_detail = metrics
    ctx._prefetch_enabled = False
    ctx._prefetch_diag_logged_layers.add(LAYER)
    ctx._offload_stream = None
    ctx._token_mask_f = torch.full((T, 1), 7.0, dtype=torch.bfloat16)
    ctx._token_mask_i_buf = torch.full((T,), 5, dtype=torch.int32)
    ctx._hit_indices_buf = torch.full((MH,), 9, dtype=torch.int32)
    ctx._has_any_hit_buf = torch.full((1,), 9, dtype=torch.int32)
    ctx._cpu_topk_ids_buf = torch.full((T, K), -1, dtype=torch.int32)
    ctx._cpu_topk_w_buf = torch.zeros((T, K), dtype=torch.float32)
    ctx._compact_input_buf = torch.zeros((MH, H), dtype=torch.bfloat16)
    ctx._compact_ids_buf = torch.full((MH, K), -1, dtype=torch.int32)
    ctx._compact_weights_buf = torch.zeros((MH, K), dtype=torch.float32)
    ctx._gpu_total_pairs_counter = torch.zeros(1, dtype=torch.int64)
    ctx._cpu_true_demand_counter = torch.zeros(1, dtype=torch.int64)
    ctx._slot_weight_buf = torch.full((T,), 3.0, dtype=torch.float32)
    ctx._fused_prepare_fn = fused_fn
    ctx._cpu_slot_bufs[LAYER] = torch.tensor([0], dtype=torch.int32)
    ctx._expert_count_accs[LAYER] = torch.arange(E, dtype=torch.int64) * 1000
    ctx._layers[LAYER] = _FakeRemoteMOE(ctx)
    ctx._initialized = True
    return ctx


def _inputs(n: int, slot: int, seed: int, ids_dtype=torch.int32):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 128, (n, K), generator=g, dtype=torch.int64)
    ids[ids == slot] = (slot + 1) % 128
    ids[torch.rand(n, K, generator=g) < 0.2] = -1
    rows = torch.rand(n, generator=g) < 0.6  # 命中行数超过 MH，验截断
    first = torch.randint(0, K, (n,), generator=g)
    ids[rows, first[rows]] = slot
    dup = rows & (torch.rand(n, generator=g) < 0.4)  # 一行两处命中，取第一个
    second = torch.randint(0, K, (n,), generator=g)
    ids[dup, second[dup]] = slot
    w = torch.randn(n, K, generator=g, dtype=torch.float32)  # 有负数：乘 0 得 -0.0
    return ids.to(ids_dtype), w


class FusedPrepareWiringTests(unittest.TestCase):
    def _run(self, engine, with_fused_fn, ids, w, slot, metrics=True):
        mod = engine.module()
        fn = mod.cpu_client_cuda_fused_prepare if with_fused_fn else None
        ctx = _make_ctx(fn, metrics=metrics)
        ctx._cpu_slot_bufs[LAYER][0] = slot
        hidden = torch.zeros((ids.shape[0], H), dtype=torch.bfloat16)
        out = io.StringIO()
        with (
            patch.dict("sys.modules", {"moe_cpu_engine": mod}),
            contextlib.redirect_stdout(out),
        ):
            res = ctx.begin_offload(
                LAYER, hidden, ids, w, slot, 0, local_expert_lo=0, local_expert_hi=63
            )
        self.assertIsNotNone(res)
        return ctx, res, out.getvalue()

    def _assert_same(self, a, b):
        (ca, ra), (cb, rb) = a, b
        self.assertTrue(
            torch.equal(ca._expert_count_accs[LAYER], cb._expert_count_accs[LAYER])
        )
        self.assertTrue(torch.equal(_bits(ca._token_mask_f), _bits(cb._token_mask_f)))
        self.assertTrue(torch.equal(ca._token_mask_i_buf, cb._token_mask_i_buf))
        self.assertTrue(torch.equal(ca._has_any_hit_buf, cb._has_any_hit_buf))
        self.assertTrue(torch.equal(ca._hit_indices_buf, cb._hit_indices_buf))
        self.assertTrue(torch.equal(ra["masked_topk_ids"], rb["masked_topk_ids"]))
        self.assertTrue(
            torch.equal(ca._cpu_true_demand_counter, cb._cpu_true_demand_counter)
        )
        self.assertEqual(
            ca._layers[LAYER].submitted_has_hit, cb._layers[LAYER].submitted_has_hit
        )

    def test_default_is_on_and_zero_turns_off(self):
        import importlib
        import os

        saved = os.environ.pop("CPU_OFFLOAD_FUSED_PREPARE", None)
        try:
            self.assertTrue(importlib.reload(cgo)._FUSED_PREPARE)
            os.environ["CPU_OFFLOAD_FUSED_PREPARE"] = "0"
            self.assertFalse(importlib.reload(cgo)._FUSED_PREPARE)
        finally:
            os.environ.pop("CPU_OFFLOAD_FUSED_PREPARE", None)
            if saved is not None:
                os.environ["CPU_OFFLOAD_FUSED_PREPARE"] = saved
            importlib.reload(cgo)

    def test_off_runs_original_small_ops(self):
        ids, w = _inputs(40, 7, seed=1)
        eng = _FakeEngine()
        self._run(eng, False, ids, w, 7)
        self.assertEqual(eng.calls, ["count_pairs", "collect", "gather"])

    def test_on_matches_original_path(self):
        for n, slot, seed in [
            (40, 7, 2),
            (1, 7, 3),
            (64, 0, 4),
            (17, 0x7FFF, 5),
            (0, 7, 6),
        ]:
            with self.subTest(n=n, slot=slot):
                ids, w = _inputs(n, slot, seed)
                off_eng, on_eng = _FakeEngine(), _FakeEngine()
                off_ctx, off_res, _ = self._run(off_eng, False, ids, w, slot)
                on_ctx, on_res, _ = self._run(on_eng, True, ids, w, slot)
                self.assertEqual(on_eng.calls, ["fused", "count_pairs", "gather"])
                self._assert_same((off_ctx, off_res), (on_ctx, on_res))
                # gather 读到的命中行和它们的 slot_weight 逐位相同
                self.assertTrue(torch.equal(off_eng.gathered[0], on_eng.gathered[0]))
                self.assertTrue(torch.equal(off_eng.gathered[1], on_eng.gathered[1]))
                # N 之后的旧值不动
                self.assertTrue(
                    torch.equal(
                        on_ctx._token_mask_i_buf[n:],
                        torch.full((T - n,), 5, dtype=torch.int32),
                    )
                )

    def test_falls_back_when_inputs_unusable(self):
        ids, w = _inputs(24, 7, seed=7, ids_dtype=torch.int64)
        off_ctx, off_res, _ = self._run(_FakeEngine(), False, ids, w, 7)
        eng = _FakeEngine()
        mod = eng.module()
        ctx = _make_ctx(mod.cpu_client_cuda_fused_prepare)
        ctx._cpu_slot_bufs[LAYER][0] = 7
        hidden = torch.zeros((24, H), dtype=torch.bfloat16)
        out = io.StringIO()
        with (
            patch.dict("sys.modules", {"moe_cpu_engine": mod}),
            contextlib.redirect_stdout(out),
        ):
            for i in range(2):  # 第二次计数在第一次之上累加，只比第一次
                res = ctx.begin_offload(LAYER, hidden, ids, w, 7, 0, 0, 63)
                if i == 0:
                    self._assert_same((off_ctx, off_res), (ctx, res))
        self.assertNotIn("fused", eng.calls)
        self.assertEqual(eng.calls.count("collect"), 2)
        self.assertEqual(out.getvalue().count("fell back"), 1)  # 同一原因只打一次

    def test_falls_back_when_launch_fails(self):
        ids, w = _inputs(30, 7, seed=8)
        off_ctx, off_res, _ = self._run(_FakeEngine(), False, ids, w, 7)
        eng = _FakeEngine(fused_rc=9)
        ctx, res, out = self._run(eng, True, ids, w, 7)
        self.assertEqual(eng.calls, ["fused", "count_pairs", "collect", "gather"])
        self._assert_same((off_ctx, off_res), (ctx, res))
        self.assertIn("cudaError 9", out)


if __name__ == "__main__":
    unittest.main()
