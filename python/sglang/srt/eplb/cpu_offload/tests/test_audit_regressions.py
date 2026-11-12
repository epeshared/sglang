"""复现移植审计发现的编号、初始化及暖机边界问题。"""

from __future__ import annotations

import ast
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.eplb.cpu_offload import (
    dispatch_hook,
    graph_hooks,
    model_load_hook,
    runner_hooks,
    step_trace,
)
from sglang.srt.eplb.cpu_offload.config import CpuOffloadConfig
from sglang.srt.eplb.cpu_offload.engine import CpuOffloadEngine, LayerSpec
from sglang.srt.eplb.cpu_offload.tests.test_fused_prepare import (
    LAYER,
    H,
    K,
    _FakeEngine,
    _make_ctx,
    _view,
)


class ExpertNumberingTests(unittest.TestCase):
    def test_rank_local_weights_receive_local_ids(self):
        for full_weights, expected in ((False, 1), (True, 3)):
            with self.subTest(full_weights=full_weights):
                registered = []
                submitted = []
                engine = CpuOffloadEngine.__new__(CpuOffloadEngine)
                engine._layer_specs = {}
                engine._initialized = True
                engine.init_runtime = lambda: None
                engine.has_layer = lambda layer: True

                def register(spec, weights):
                    engine._layer_specs[spec.layer_id] = spec
                    registered.append((spec, weights))

                engine.register_layer = register
                engine.launch_submit = lambda layer, **kw: submitted.append(kw)
                cfg = CpuOffloadConfig(
                    enable_cpu_offload_fast_loop=True,
                    cpu_offload_decode_only=False,
                    cpu_active_ranks=[1],
                )
                runtime = NS(
                    engine=engine,
                    config=cfg,
                    local_rank=1,
                    local_cpu_slot=lambda layer: 3,
                    is_active_locally=lambda: True,
                    metrics=MagicMock(),
                )
                experts = NS(
                    num_local_experts=2,
                    num_experts=4,
                    moe_ep_rank=1,
                    top_k=1,
                    hidden_size=4,
                    intermediate_size=2,
                    w13_weight=torch.zeros(2, 4, 4),
                    w2_weight=torch.zeros(2, 4, 2),
                    w13_weight_scale_inv=torch.ones(2, 1, 1),
                    w2_weight_scale_inv=torch.ones(2, 1, 1),
                )
                model = NS(modules=lambda: [NS(experts=experts, layer_id=0)])
                with (
                    patch(
                        "sglang.srt.eplb.cpu_offload.engine.daemon_loads_weights_from_disk",
                        return_value=full_weights,
                    ),
                    patch.object(model_load_hook, "_assert_expert_block_sources_agree"),
                ):
                    model_load_hook.register_model_moe_layers(model, runtime)
                with (
                    patch(
                        "sglang.srt.eplb.cpu_offload.get_cpu_offload_runtime",
                        return_value=runtime,
                    ),
                    patch(
                        "sglang.srt.eplb.cpu_offload.cuda_graph_offload.get_cuda_graph_offload_ctx",
                        return_value=None,
                    ),
                    patch.object(
                        torch.cuda, "is_current_stream_capturing", return_value=False
                    ),
                    patch.object(
                        torch.cuda, "current_stream", return_value=NS(cuda_stream=0)
                    ),
                ):
                    result = dispatch_hook.begin_cpu_offload(
                        layer_id=0,
                        hidden_states=torch.ones(1, 4),
                        topk_ids=torch.tensor([[3]], dtype=torch.int32),
                        topk_weights=torch.ones(1, 1),
                    )
                self.assertEqual(submitted[0]["dev_topk_ids"].tolist(), [[expected]])
                self.assertEqual(result["masked_topk_ids"].tolist(), [[-1]])
                self.assertLess(expected, registered[0][0].num_experts)


class GraphInitializationTests(unittest.TestCase):
    def test_capture_does_not_silently_skip_a_missing_layer(self):
        runtime = NS(is_active_locally=lambda: True)
        with (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
            patch(
                "sglang.srt.eplb.cpu_offload.get_cpu_offload_runtime",
                return_value=runtime,
            ),
            patch(
                "sglang.srt.eplb.cpu_offload.cuda_graph_offload.get_cuda_graph_offload_ctx",
                return_value=None,
            ),
            self.assertRaisesRegex(RuntimeError, "捕获失败"),
        ):
            dispatch_hook.begin_cpu_offload(
                layer_id=0,
                hidden_states=torch.ones(1, 4),
                topk_ids=torch.tensor([[3]], dtype=torch.int32),
                topk_weights=torch.ones(1, 1),
            )

    def test_partial_registration_is_fatal(self):
        runner = NS(
            server_args=NS(enable_cpu_offload_fast_loop=True),
            tp_rank=1,
            model_config=NS(hf_config=NS(hidden_size=4, num_experts_per_tok=1)),
        )
        engine = NS(
            initialized=True,
            _daemon=NS(_layers={0: object()}),
            registered_layers=lambda: [0],
            layer_spec=lambda layer: object(),
        )
        runtime = NS(engine=engine, is_active_locally=lambda: True)
        graph = NS(register_layer_from_daemon=lambda *a, **kw: False)
        with (
            patch.object(runner_hooks, "decode_graph_capacity", return_value=8),
            patch(
                "sglang.srt.eplb.cpu_offload.get_cpu_offload_runtime",
                return_value=runtime,
            ),
            patch(
                "sglang.srt.eplb.cpu_offload.cuda_graph_offload.init_cuda_graph_offload",
                return_value=graph,
            ),
            self.assertRaisesRegex(RuntimeError, "层登记失败"),
        ):
            runner_hooks.RunnerView._init_cuda_graph_offload(runner)

    def test_active_rank_initialization_failure_is_fatal(self):
        runner = NS(
            server_args=NS(enable_cpu_offload_fast_loop=True),
            tp_rank=0,
            model_config=NS(hf_config=NS(hidden_size=4, num_experts_per_tok=1)),
        )
        runtime = NS(
            engine=NS(initialized=True, _daemon=object()),
            is_active_locally=lambda: True,
        )
        for value in (None, RuntimeError("注入初始化故障")):
            with (
                self.subTest(value=value),
                patch.object(runner_hooks, "decode_graph_capacity", return_value=8),
                patch(
                    "sglang.srt.eplb.cpu_offload.get_cpu_offload_runtime",
                    return_value=runtime,
                ),
                patch(
                    "sglang.srt.eplb.cpu_offload.cuda_graph_offload.init_cuda_graph_offload",
                    **(
                        {"side_effect": value}
                        if isinstance(value, Exception)
                        else {"return_value": value}
                    ),
                ),
                self.assertRaises(RuntimeError),
            ):
                runner_hooks.RunnerView._init_cuda_graph_offload(runner)


class GraphExpertNumberingTests(unittest.TestCase):
    def test_reselection_updates_cpu_ids_but_keeps_global_gpu_mask(self):
        class Engine(_FakeEngine):
            def gather(self, *args):
                super().gather(*args)
                self.weight_slot = int(_view(args[7], 1, torch.int32)[0])

        for fused in (False, True):
            with self.subTest(fused=fused):
                engine = Engine()
                ctx = _make_ctx(engine.fused if fused else None)
                spec = LayerSpec(LAYER, 2, K, H, H, expert_id_offset=2)
                tensor, zeros = torch.tensor, torch.zeros

                def cpu_tensor(*a, **kw):
                    kw["device"] = "cpu"
                    return tensor(*a, **kw)

                def cpu_zeros(*a, **kw):
                    kw["device"] = "cpu"
                    return zeros(*a, **kw)

                with (
                    patch.object(torch, "tensor", side_effect=cpu_tensor),
                    patch.object(torch, "zeros", side_effect=cpu_zeros),
                ):
                    ctx.register_layer_from_daemon(LAYER, ctx._layers[LAYER], spec=spec)
                for slot in (3, 2):
                    ctx.update_slot(LAYER, slot)
                    ids = torch.full((1, K), -1, dtype=torch.int32)
                    ids[0, 0] = slot
                    ids[0, 1] = 0
                    with patch.dict("sys.modules", {"moe_cpu_engine": engine.module()}):
                        result = ctx.begin_offload(
                            LAYER,
                            torch.zeros(1, H, dtype=torch.bfloat16),
                            ids,
                            torch.ones(1, K),
                            slot,
                            0,
                            local_expert_lo=2,
                            local_expert_hi=3,
                        )
                    self.assertEqual(engine.weight_slot, slot - 2)
                    self.assertEqual(result["masked_topk_ids"][0, :2].tolist(), [-1, 0])
                with self.assertRaisesRegex(ValueError, "越界"):
                    ctx.update_slot(LAYER, 0)
                self.assertEqual(int(ctx._cpu_slot_bufs[LAYER][0]), 2)
                ctx.update_slot(LAYER, -1)
                self.assertEqual(int(ctx._cpu_slot_bufs[LAYER][0]), 0x7FFF)
                self.assertEqual(int(ctx._cpu_weight_slot_bufs[LAYER][0]), 0x7FFF)


class RealGraphBackendTests(unittest.TestCase):
    def test_only_actual_capture_records_layer_events(self):
        path = (
            Path(__file__).parents[3]
            / "model_executor/runner_backend/full_cuda_graph_backend.py"
        )
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "FullCudaGraphBackend"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "capture_one"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
        capturing = False

        @contextmanager
        def graph_context(**kwargs):
            nonlocal capturing
            capturing = True
            try:
                yield
            finally:
                capturing = False

        namespace = {
            "torch": NS(cuda=NS(CUDAGraph=lambda: object())),
            "graph_pool_capture_scope": nullcontext,
        }
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        backend = NS(
            _cuda_graph_runner=NS(enable_profile_cuda_graph=False),
            _device_module=NS(synchronize=lambda: None, graph=graph_context),
            _tp_group=NS(barrier=lambda: None),
            _precarve=NS(measure=nullcontext, mint=lambda: None),
            _reuse_output_buffer=False,
            _memory_saver_adapter=None,
            _pool=None,
            _capture_stream=None,
            _graphs={},
            _outputs={},
            _capture_inputs={},
        )
        backend.capture_one = MethodType(namespace["capture_one"], backend)
        recorder = step_trace.StepTraceRecorder.__new__(step_trace.StepTraceRecorder)
        recorder._capture_trace = recorder._active_step = None
        recorder._event_factory = lambda: NS(record=lambda: None)
        state = NS(layers=[])
        phases = []

        def forward():
            phases.append(recorder._capture_trace is not None)
            recorder.begin_layer(7)

        with (
            patch.object(
                torch.cuda, "is_current_stream_capturing", side_effect=lambda: capturing
            ),
            patch.object(
                graph_hooks.step_trace, "prepare_graph_capture", return_value=state
            ),
            patch.object(
                graph_hooks.step_trace,
                "enter_graph_capture",
                side_effect=recorder.enter_graph_capture,
            ),
            patch.object(
                graph_hooks.step_trace,
                "leave_graph_capture",
                side_effect=recorder.leave_graph_capture,
            ),
        ):
            graph_hooks.capture(
                backend, NS(device="cpu"), 4, forward, NS(batch_size=4), 4, None
            )
        self.assertEqual(phases, [False, False, True])
        self.assertEqual([x.layer_id for x in state.layers], [7])
        self.assertIsNone(recorder._capture_trace)


if __name__ == "__main__":
    unittest.main()
