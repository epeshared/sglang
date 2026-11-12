"""使用真实 CUDA 准备内核和图重放验证专家编号；不启动 CPU 守护进程。"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.eplb.cpu_offload.cuda_graph_offload import CudaGraphOffloadContext
from sglang.srt.eplb.cpu_offload.engine import LayerSpec


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 和配套 moe_cpu_engine")
class CudaGraphNumberingTests(unittest.TestCase):
    def test_real_gather_after_slot_reselection(self):
        import moe_cpu_engine

        for offset in (0, 2):
            for fused in (False, True):
                if fused and not hasattr(
                    moe_cpu_engine, "cpu_client_cuda_fused_prepare"
                ):
                    continue
                with self.subTest(offset=offset, fused=fused):
                    context = CudaGraphOffloadContext(
                        hidden_size=128, topk=2, max_tokens=4, max_hits=4
                    )
                    self.assertTrue(context.initialize())
                    context._fused_prepare_fn = (
                        moe_cpu_engine.cpu_client_cuda_fused_prepare if fused else None
                    )
                    # 保留真实 GPU 筛选和压紧，仅替代向守护进程提交请求的操作。
                    remote = SimpleNamespace(
                        _kernel_path_ready=True, launch_submit=lambda **kwargs: None
                    )
                    context.register_layer_from_daemon(
                        0,
                        remote,
                        spec=LayerSpec(
                            0, 2 if offset else 4, 2, 128, 128, expert_id_offset=offset
                        ),
                    )
                    ids = torch.tensor(
                        [[3, 0], [2, 1], [0, 1], [3, 2]],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    weights = torch.full((4, 2), 0.5, device="cuda")
                    hidden = (
                        torch.arange(512, device="cuda", dtype=torch.float32)
                        .reshape(4, 128)
                        .to(torch.bfloat16)
                    )
                    context.update_slot(0, 3)

                    def forward():
                        result = context.begin_offload(
                            0,
                            hidden,
                            ids,
                            weights,
                            3,
                            torch.cuda.current_stream().cuda_stream,
                        )
                        torch.cuda.current_stream().wait_stream(context._offload_stream)
                        return result[
                            "masked_topk_ids"
                        ], context._compact_ids_buf.clone()

                    # 暖机和捕获均使用同一个侧流，并在主流等待它完成。
                    warmup = torch.cuda.Stream()
                    warmup.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(warmup):
                        forward()
                    torch.cuda.current_stream().wait_stream(warmup)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        masked, compact_ids = forward()
                    for slot in (3, 2, 3):
                        context.update_slot(0, slot)
                        graph.replay()
                        torch.cuda.synchronize()
                        expected = ids.masked_fill(ids == slot, -1)
                        self.assertTrue(torch.equal(masked, expected))
                        hits = int((ids == slot).any(dim=1).sum())
                        self.assertEqual(
                            compact_ids[:hits, 0].tolist(), [slot - offset] * hits
                        )
                        self.assertTrue(bool((compact_ids[hits:, 0] == -1).all()))


if __name__ == "__main__":
    unittest.main()
