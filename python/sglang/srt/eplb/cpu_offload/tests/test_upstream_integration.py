"""验证新版真实 MoE 入口的分流、合并次序及图捕获边界。"""

from __future__ import annotations

import argparse
import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import torch

from sglang.srt.eplb.cpu_offload import (
    dispatch_hook,
    graph_hooks,
    layer_hooks,
    runner_hooks,
)
from sglang.srt.eplb.cpu_offload.request_identity import attach, from_requests


class TopK(NamedTuple):
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


def native_forward(namespace):
    path = Path(__file__).parents[3] / "layers/moe/fused_moe_triton/layer.py"
    tree = ast.parse(path.read_text())
    klass = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FusedMoE"
    )
    method = next(
        node
        for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward_impl"
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace.update(
        torch=torch,
        TopKOutput=TopK,
        Optional=__import__("typing").Optional,
        Tuple=__import__("typing").Tuple,
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["forward_impl"]


class TestNativeMoeIntegration(unittest.TestCase):
    def run_forward(self, enabled):
        events = []
        source_ids = torch.tensor([[0, 1], [0, 2]])
        topk = TopK(source_ids, torch.full((2, 2), 0.5))
        inputs = torch.tensor([[1.0], [2.0]])
        factors = torch.tensor([10.0, 20.0, 30.0])

        class Dispatcher:
            def combine(self, *, combine_input):
                events.append("GPU合并")
                return combine_input

        class Layer:
            layer_id = 7
            quant_method = object()
            _dwdp_bound = False
            _cpu_offload_observe = enabled
            dispatcher = Dispatcher()
            reduce_results = True
            moe_tp_size = 2

            def _dispatch_with_pre_quant(self, hidden, selected, pre_quant):
                events.append("GPU分发")
                self.selected_ids = selected.topk_ids
                return SimpleNamespace(hidden=hidden, selected=selected)

            def run_moe_core(self, *, dispatch_output):
                events.append("GPU计算")
                ids = dispatch_output.selected.topk_ids
                weights = dispatch_output.selected.topk_weights
                values = factors[ids.clamp_min(0)] * (ids >= 0) * weights
                return dispatch_output.hidden * values.sum(dim=1, keepdim=True)

        def begin(**kwargs):
            events.append("CPU启动")
            self.assertTrue(torch.equal(kwargs["topk_ids"], source_ids))
            masked = source_ids.clone()
            masked[0, 1] = -1
            return {"masked_topk_ids": masked, "cpu": torch.tensor([[10.0], [0.0]])}

        def commit(context, output):
            events.append("CPU合并")
            return output + context["cpu"]

        def reduce(output):
            events.append("跨卡求和")
            self.assertTrue(torch.equal(output, torch.tensor([[15.0], [40.0]])))
            return output * 2

        topk_module = ModuleType("sglang.srt.layers.moe.topk")
        topk_module.TopKOutputChecker = SimpleNamespace(
            format_is_standard=lambda value: True
        )
        runtime = SimpleNamespace(is_active_locally=lambda: True)
        namespace = {
            "use_symmetric_memory": lambda *args, **kwargs: nullcontext(),
            "get_parallel": lambda: SimpleNamespace(tp_group=None),
            "is_allocation_symmetric": lambda: False,
            "tensor_model_parallel_all_reduce": reduce,
        }
        layer = Layer()
        with (
            patch.dict("sys.modules", {topk_module.__name__: topk_module}),
            patch.object(layer_hooks, "get_cpu_offload_runtime", return_value=runtime),
            patch.object(layer_hooks, "MOE_TRACE", True),
            patch.object(layer_hooks.trace, "hit") as traced,
            patch.object(layer_hooks.trace, "record_moe_layer_facts"),
            patch.object(
                dispatch_hook, "begin_cpu_offload", side_effect=begin
            ) as launch,
            patch.object(dispatch_hook, "commit_cpu_offload", side_effect=commit),
        ):
            result = native_forward(namespace)(layer, inputs, topk)
            self.assertEqual(launch.call_count, int(enabled))
            self.assertEqual(
                [call.args[0] for call in traced.call_args_list],
                [
                    "moe.fused.forward",
                    "moe.fused.begin",
                    "moe.fused.begin_armed",
                    "moe.fused.commit",
                ]
                if enabled
                else [],
            )
        self.assertTrue(torch.equal(source_ids, torch.tensor([[0, 1], [0, 2]])))
        self.assertTrue(torch.equal(result, torch.tensor([[30.0], [80.0]])))
        return events

    def test_cpu_result_is_added_before_cross_rank_sum(self):
        self.assertEqual(
            self.run_forward(True),
            ["CPU启动", "GPU分发", "GPU计算", "GPU合并", "CPU合并", "跨卡求和"],
        )

    def test_disabled_path_keeps_native_output(self):
        self.assertEqual(
            self.run_forward(False), ["GPU分发", "GPU计算", "GPU合并", "跨卡求和"]
        )


class TestGraphLifecycle(unittest.TestCase):
    def test_trace_starts_after_warmup_and_is_always_closed(self):
        events = []
        state = object()

        class Backend:
            def capture_one(self, key, run_once, **kwargs):
                events.append("暖机")
                kwargs["post_warmup_hook"]()
                events.append("正式捕获")
                raise RuntimeError("测试捕获失败")

        with (
            patch.object(
                graph_hooks.step_trace, "prepare_graph_capture", return_value=state
            ),
            patch.object(
                graph_hooks.step_trace,
                "enter_graph_capture",
                side_effect=lambda value: events.append("开始记录"),
            ),
            patch.object(
                graph_hooks.step_trace,
                "leave_graph_capture",
                side_effect=lambda value: events.append("结束记录"),
            ),
            self.assertRaisesRegex(RuntimeError, "测试捕获失败"),
        ):
            graph_hooks.capture(
                Backend(),
                SimpleNamespace(device="cpu"),
                "key",
                lambda: None,
                SimpleNamespace(batch_size=4),
                4,
                lambda: events.append("原有暖机钩子"),
            )
        self.assertEqual(
            events, ["暖机", "原有暖机钩子", "开始记录", "正式捕获", "结束记录"]
        )

    def test_padding_uses_the_actual_token_count(self):
        runner = SimpleNamespace(bs=8, raw_num_token=3)
        with patch.object(graph_hooks.step_trace, "mark_graph_replay") as mark:
            graph_hooks.before_replay(runner)
        mark.assert_called_once_with(id(runner), 8, 3)


class TestRequestIdentity(unittest.TestCase):
    def test_scheduler_ids_are_preserved(self):
        requests = [SimpleNamespace(rid="真实请求", output_ids=[1, 2, 3])]
        batch = SimpleNamespace()
        attach(batch, requests)
        self.assertEqual(vars(batch), from_requests(requests))
        self.assertEqual(
            from_requests(requests),
            {
                "request_ids": ["真实请求"],
                "request_decode_steps": [3],
                "request_id_source": "scheduler",
            },
        )

    def test_synthetic_ids_are_explicit(self):
        request = SimpleNamespace(
            rid="池位置",
            output_ids=[],
            _step_trace_request_id="基准请求",
            _step_trace_decode_step=9,
            _step_trace_identity="synthetic",
        )
        self.assertEqual(
            from_requests([request]),
            {
                "request_ids": ["基准请求"],
                "request_decode_steps": [9],
                "request_id_source": "synthetic",
            },
        )


class TestArguments(unittest.TestCase):
    def test_cli_preserves_switches_and_defaults(self):
        from sglang.srt.arg_groups.arg_utils import add_cli_args_from_dataclass
        from sglang.srt.arg_groups.fields.cpu_experts import CpuOffloadFields

        parser = argparse.ArgumentParser()
        add_cli_args_from_dataclass(parser, CpuOffloadFields)
        values = parser.parse_args(
            [
                "--enable-cpu-offload-fast-loop",
                "--cpu-active-ranks",
                "0,1",
                "--no-cpu-offload-decode-only",
            ]
        )
        self.assertTrue(values.enable_cpu_offload_fast_loop)
        self.assertEqual(values.cpu_active_ranks, "0,1")
        self.assertFalse(values.cpu_offload_decode_only)
        self.assertEqual(values.cpu_offload_max_hits, 0)


class TestRunnerGuards(unittest.TestCase):
    def setUp(self):
        self.runner = SimpleNamespace(
            server_args=SimpleNamespace(
                enable_cpu_offload_fast_loop=True, enable_ep_moe_heto=False
            ),
            is_draft_worker=False,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    quantization_config={
                        "quant_method": "fp8",
                        "weight_block_size": [128, 128],
                    }
                )
            ),
        )
        from sglang.srt import runtime_context

        self.graphs = SimpleNamespace(
            decode=SimpleNamespace(backend="full", max_bs=512, bs=[1, 8, 512]),
            prefill=SimpleNamespace(backend="disabled"),
        )
        self.context = patch.multiple(
            runtime_context,
            get_parallel=lambda: SimpleNamespace(pp_size=1, moe_ep_size=8, tp_size=8),
            get_exec=lambda: SimpleNamespace(
                moe=SimpleNamespace(moe_a2a_backend="none"),
                graph=SimpleNamespace(cuda_graph_config=self.graphs),
            ),
            get_spec=lambda: SimpleNamespace(speculative_algorithm=None),
        )
        self.context.start()
        self.addCleanup(self.context.stop)

    def test_original_block_fp8_is_accepted(self):
        self.assertTrue(runner_hooks.validate_runner(self.runner))

    def test_native_bf16_is_unchanged_when_offload_is_disabled(self):
        self.runner.server_args.enable_cpu_offload_fast_loop = False
        self.runner.model_config.hf_config.quantization_config = None
        self.assertFalse(runner_hooks.validate_runner(self.runner))

    def test_bf16_offload_is_rejected_instead_of_silently_skipped(self):
        self.runner.model_config.hf_config.quantization_config = None
        with self.assertRaisesRegex(ValueError, "Hy4 BF16"):
            runner_hooks.validate_runner(self.runner)

    def test_mxfp8_is_not_misread_as_the_old_fp8_format(self):
        self.runner.model_config.hf_config.quantization_config = {
            "quant_method": "modelopt",
            "quantization": {"quant_algo": "MXFP8"},
        }
        with self.assertRaisesRegex(ValueError, "128×128"):
            runner_hooks.validate_runner(self.runner)

    def test_incomplete_startup_weights_are_not_registered(self):
        self.runner.startup_weight_load = object()
        with patch(
            "sglang.srt.eplb.cpu_offload.get_cpu_offload_runtime", return_value=object()
        ):
            with self.assertRaisesRegex(RuntimeError, "完整权重"):
                runner_hooks.register_runner_model(self.runner)

    def test_no_registered_experts_is_an_error(self):
        self.runner.model = object()
        runtime = SimpleNamespace(is_active_locally=lambda: True)
        with (
            patch(
                "sglang.srt.eplb.cpu_offload.get_cpu_offload_runtime",
                return_value=runtime,
            ),
            patch(
                "sglang.srt.eplb.cpu_offload.model_load_hook.register_model_moe_layers",
                return_value=0,
            ),
            self.assertRaisesRegex(RuntimeError, "没有成功登记"),
        ):
            runner_hooks.register_runner_model(self.runner)

    def test_graph_buffers_follow_the_resolved_capture_buckets(self):
        self.graphs.decode.bs = [1, 256, 1024]
        self.assertEqual(runner_hooks.decode_graph_capacity(), 1024)
        self.graphs.decode.bs = None
        self.assertEqual(runner_hooks.decode_graph_capacity(), 512)

    def test_disabled_decode_does_not_initialize_graph_offload(self):
        self.graphs.decode.backend = "disabled"
        self.assertEqual(runner_hooks.decode_graph_capacity(), 0)
        view = runner_hooks.RunnerView(self.runner)
        with patch(
            "sglang.srt.eplb.cpu_offload.cuda_graph_offload.init_cuda_graph_offload"
        ) as initialize:
            view._init_cuda_graph_offload()
        initialize.assert_not_called()

    def test_unresolved_graph_size_is_rejected(self):
        self.graphs.decode.bs = None
        self.graphs.decode.max_bs = None
        with self.assertRaisesRegex(ValueError, "已解析"):
            runner_hooks.decode_graph_capacity()

    def test_prefill_and_breakable_graphs_are_rejected(self):
        self.graphs.prefill.backend = "breakable"
        with self.assertRaisesRegex(ValueError, "prefill"):
            runner_hooks.validate_runner(self.runner)
        self.graphs.prefill.backend = "disabled"
        self.graphs.decode.backend = "breakable"
        with self.assertRaisesRegex(ValueError, "full"):
            runner_hooks.validate_runner(self.runner)

    def test_runtime_receives_resolved_parallel_settings(self):
        self.runner.server_args.ep_size = 1
        self.runner.server_args._resolved_overrides = [
            ("解析后的专家并行数", {"ep_size": 8})
        ]
        view = runner_hooks.RunnerView(self.runner)
        self.assertEqual(view.server_args.ep_size, 8)
        self.assertEqual(self.runner.server_args.ep_size, 1)

    def test_pinned_startup_keeps_the_original_thread_check(self):
        self.runner._cpu_offload_view = SimpleNamespace(tp_rank=3)
        with (
            patch.dict("os.environ", {"CPU_OFFLOAD_TRANSPORT": "pinned"}),
            patch(
                "sglang.srt.eplb.cpu_offload.transport_threads.record_rank_startup"
            ) as record,
        ):
            runner_hooks.finish_initialization(self.runner)
        record.assert_called_once_with(3)
