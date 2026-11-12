"""bench_one_batch 多组参数扫描的单测：不需要 GPU、模型和守护进程。

bench_one_batch 顶层会导入整套运行时（zmq、triton、sgl-kernel 等），开发机上
装不全。这里先正常导入真正要用的 step_trace 和 bench_real_text，再把其余运行时
模块临时换成占位模块导入 bench_one_batch，导入完立刻还原 sys.modules。extend
和 decode 换成按固定规则出 token 的假实现，所以测的只是这个脚本自己的逻辑：
扫描网格、分组预填充、输出收集、分段文件、剖析时在哪一步开抓停抓，以及几处防呆检查。
"""

from __future__ import annotations

import collections
import hashlib
import importlib
import inspect
import json
import multiprocessing
import os
import statistics
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sglang import bench_real_text
from sglang.srt.eplb.cpu_offload import step_trace  # noqa: F401  先导入真身


class _StubReq:
    def __init__(self, rid, origin_input_text, origin_input_ids, sampling_params):
        self.rid = rid
        self.origin_input_ids = origin_input_ids
        self.prefix_indices = []


def _noop(*args, **kwargs):
    return None


def _env_flag(name, default="false"):
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _import_bench_one_batch():
    stubs = {
        name: _stub(name, **attrs)
        for name, attrs in {
            "sglang.srt.configs.model_config": {"ModelConfig": object},
            "sglang.srt.distributed.parallel_state": {
                "destroy_distributed_environment": _noop
            },
            "sglang.srt.entrypoints.engine": {"_set_envs_and_config": _noop},
            "sglang.srt.layers.dp_attention": {"set_is_extend_in_batch": _noop},
            "sglang.srt.layers.moe": {"initialize_moe_config": _noop},
            "sglang.srt.managers.schedule_batch": {
                "Req": _StubReq,
                "ScheduleBatch": object,
            },
            "sglang.srt.managers.scheduler": {"Scheduler": object},
            "sglang.srt.model_executor.forward_batch_info": {"ForwardBatch": object},
            "sglang.srt.model_executor.model_runner": {"ModelRunner": object},
            "sglang.srt.sampling.sampling_params": {
                "SamplingParams": lambda **kw: SimpleNamespace(**kw)
            },
            "sglang.srt.server_args": {"PortArgs": object, "ServerArgs": object},
            "sglang.srt.speculative.spec_info": {"SpeculativeAlgorithm": object},
            "sglang.srt.utils": {
                "configure_logger": _noop,
                "get_bool_env_var": _env_flag,
                "is_cuda_alike": lambda: False,
                "is_xpu": lambda: False,
                "kill_process_tree": _noop,
                "maybe_reindex_device_id": _noop,
                "require_mlp_sync": lambda *a: False,
                "require_mlp_tp_gather": lambda *a: False,
                "set_gpu_proc_affinity": _noop,
                "suppress_other_loggers": _noop,
            },
            "sglang.srt.utils.hf_transformers_utils": {"get_tokenizer": _noop},
        }.items()
    }
    target = "sglang.srt.eplb.cpu_offload.legacy.bench_one_batch_v055"
    saved = {name: sys.modules.get(name) for name in [*stubs, target]}
    sys.modules.pop(target, None)
    sys.modules.update(stubs)
    try:
        return importlib.import_module(target)
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


bo = _import_bench_one_batch()


class _Tokenizer:
    """一个字符一个 token，带最简单的对话模板。"""

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        return f"<u>{messages[0]['content']}</u><a>"

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = list(reqs)

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)


_VOCAB = 5


def _fake_logits_row(rid):
    return [1000 + 100 * rid + j / 4 for j in range(_VOCAB)]


class _FakeEngine:
    """假的 extend/decode：第 r 个请求预填充出 1000+100r，之后每步加一。

    预填充的 logits 第 r 行是 :func:`_fake_logits_row`。故意每次都写进同一块
    缓冲区再返回，模拟输出缓冲区被下一次 forward 复用：不先复制就会被覆盖。
    """

    def __init__(self):
        self.extend_calls = []
        self._logits_buffer = torch.zeros((64, _VOCAB), dtype=torch.float32)

    def extend(self, reqs, model_runner):
        self.extend_calls.append([list(r.origin_input_ids) for r in reqs])
        ids = torch.tensor([1000 + 100 * r.rid for r in reqs], dtype=torch.int64)
        logits = self._logits_buffer[: len(reqs)]
        logits.copy_(torch.tensor([_fake_logits_row(r.rid) for r in reqs]))
        return ids, logits, _FakeBatch(reqs)

    def decode(self, next_token_ids, batch, model_runner):
        assert next_token_ids.shape[0] == len(batch.reqs)
        return next_token_ids + 1, None


class _TraceRecorder:
    def __init__(self):
        self.calls = []

    def pause(self):
        self.calls.append(("pause",))

    def resume(self):
        self.calls.append(("resume",))

    def start_segment(self, name, fields=None, *, sample_every=None):
        self.calls.append(("start", name, sample_every, dict(fields or {})))

    def end_segment(self):
        self.calls.append(("end",))


def _runner(max_total=10**6, slots=64, context_len=10**6, tp_size=1, tp_group=None):
    return SimpleNamespace(
        max_total_num_tokens=max_total,
        req_to_token_pool=SimpleNamespace(size=slots, clear=_noop),
        token_to_kv_pool_allocator=SimpleNamespace(clear=_noop),
        model_config=SimpleNamespace(context_len=context_len),
        tp_size=tp_size,
        tp_group=tp_group,
    )


def _server_args(**overrides):
    fields = dict(
        device="cpu",
        tp_size=1,
        enable_cpu_offload_fast_loop=False,
        disable_cuda_graph=False,
        cpu_offload_max_hits=0,
        cuda_graph_max_bs=8,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _write_dataset(path: Path, count=60):
    with path.open("w", encoding="utf-8") as f:
        for i in range(count):
            f.write(
                json.dumps(
                    {"text": f"第{i}条记录：" + "真实内容" * 8}, ensure_ascii=False
                )
            )
            f.write("\n")


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "data.jsonl"
        _write_dataset(self.dataset)
        self.engine = _FakeEngine()
        self.trace = _TraceRecorder()
        for name, value in [
            ("extend", self.engine.extend),
            ("decode", self.engine.decode),
            ("step_trace", self.trace),
        ]:
            patcher = mock.patch.object(bo, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def _bench_args(self, **overrides):
        fields = dict(
            batch_size=(2, 1, 2),
            len_pairs=("64:4", "128:3"),
            dataset_path=str(self.dataset),
            prefill_group_tokens=100,
            segment_output_dir=str(self.root / "out"),
        )
        fields.update(overrides)
        return bo.BenchArgs(**fields)

    def _sweep(self, bench_args, runner=None, server_args=None):
        return bo.sweep_latency_test(
            server_args or _server_args(),
            bench_args,
            runner or _runner(),
            _Tokenizer(),
            0,
            _noop,
        )

    def test_real_text_sweep_writes_one_directory_per_segment(self):
        results = self._sweep(self._bench_args())
        names = ["bs2-in64-out4", "bs1-in64-out4", "bs2-in128-out3", "bs1-in128-out3"]
        self.assertEqual([r["batch_size"] for r in results], [2, 1, 2, 1])

        out = self.root / "out"
        summary = json.loads((out / "segments.json").read_text(encoding="utf-8"))
        self.assertEqual([s["segment"] for s in summary["segments"]], names)

        for name in names:
            seg = json.loads((out / name / "segment.json").read_text(encoding="utf-8"))
            bs, il, ol = seg["batch_size"], seg["input_len"], seg["output_len"]
            prompts = json.loads((out / name / "output_ids.json").read_text())[
                "prompts"
            ]
            self.assertEqual(len(prompts), bs)
            inputs = []
            for r, prompt in enumerate(prompts):
                ids = prompt["output_ids"]
                self.assertEqual(len(ids), il + ol)
                # 输出接在输入后面，顺序和请求顺序一致（分组预填充也一样）。
                self.assertEqual(ids[il:], [1000 + 100 * r + k for k in range(ol)])
                inputs.append(ids[:il])
            self.assertEqual(seg["workload"], "real_text")
            self.assertEqual(seg["input_sha256"], bench_real_text.input_digest(inputs))
            self.assertEqual(seg["decode_steps"], ol - 1)
            self.assertTrue(seg["ignore_eos"])
            self.assertEqual(seg["template_prefix_tokens"], len("<u>"))
            self.assertEqual(seg["template_suffix_tokens"], len("</u><a>"))
            self.assertEqual(len(seg["dataset_records"]), bs)
            # 每个请求 64 或 128 个 token，按 100 分组时 bs 2 一定拆成两组。
            self.assertEqual(
                seg["timing"].get("prefill_groups"), 2 if bs == 2 else None
            )

    def test_segments_and_pause_wrap_the_right_calls(self):
        self._sweep(self._bench_args(trace_target_steps=2))
        kinds = [c[0] for c in self.trace.calls]
        self.assertEqual(kinds[:2], ["pause", "resume"])
        self.assertEqual(kinds[2:], ["start", "end"] * 4)
        starts = [c for c in self.trace.calls if c[0] == "start"]
        self.assertEqual([c[1] for c in starts][:2], ["bs2-in64-out4", "bs1-in64-out4"])
        # 3 个 decode 步、目标 2 步 → 每 2 步记一次；2 个 decode 步 → 每步都记。
        self.assertEqual([c[2] for c in starts], [2, 2, 1, 1])
        self.assertEqual(starts[0][3]["workload"], "real_text")

    def test_warmup_uses_text_the_segments_never_see(self):
        self._sweep(self._bench_args())
        # 预热是第一组参数（bs 2、输入 64），按 100 分组拆成两次 extend。
        warmup = self.engine.extend_calls[0] + self.engine.extend_calls[1]
        measured = [ids for call in self.engine.extend_calls[2:] for ids in call]
        for ids in warmup:
            self.assertEqual(len(ids), 64)
            self.assertNotIn(ids, measured)

    def test_same_length_segments_share_leading_prompts(self):
        self._sweep(self._bench_args())
        out = self.root / "out"

        def prompts(name):
            data = json.loads((out / name / "output_ids.json").read_text())
            return [p["output_ids"] for p in data["prompts"]]

        self.assertEqual(
            prompts("bs1-in64-out4")[0][:64], prompts("bs2-in64-out4")[0][:64]
        )

    def test_prefill_logits_and_decode_latencies_are_saved(self):
        self._sweep(self._bench_args())
        out = self.root / "out"
        names = ["bs2-in64-out4", "bs1-in64-out4", "bs2-in128-out3", "bs1-in128-out3"]
        for name in names:
            seg = json.loads((out / name / "segment.json").read_text(encoding="utf-8"))
            bs, ol = seg["batch_size"], seg["output_len"]
            info = seg["prefill_logits"]
            path = out / name / info["file"]
            array = np.load(path, allow_pickle=False)
            # bs 2 按 100 分组拆成两次 extend，两次写的是同一块缓冲区：
            # 不先复制的话，第 0 行会被第二组覆盖。
            self.assertEqual(array.dtype, np.float32)
            self.assertEqual(array.tolist(), [_fake_logits_row(r) for r in range(bs)])
            self.assertEqual(info["shape"], [bs, _VOCAB])
            self.assertEqual(
                (info["dtype"], info["source_dtype"]), ("float32", "float32")
            )
            self.assertEqual(
                info["sha256"], hashlib.sha256(array.tobytes()).hexdigest()
            )
            self.assertEqual(
                info["row_sha256"],
                [hashlib.sha256(row.tobytes()).hexdigest() for row in array],
            )
            # 摘要算的就是 .npy 文件头后面的数据段。
            raw = path.read_bytes()
            self.assertEqual(
                hashlib.sha256(raw[-bs * _VOCAB * 4 :]).hexdigest(), info["sha256"]
            )

            data = json.loads((out / name / seg["decode_latencies_file"]).read_text())
            self.assertEqual(data["unit"], "ns")
            values = data["latencies_ns"]
            self.assertEqual(len(values), ol - 1)
            self.assertTrue(all(isinstance(v, int) and v >= 0 for v in values))
            self.assertAlmostEqual(
                statistics.median(values),
                seg["timing"]["median_decode_latency"] * 1e9,
                delta=1,
            )

    def test_prefill_logits_shape_must_match_batch(self):
        with self.assertRaisesRegex(RuntimeError, "和 batch 3 对不上"):
            bo._write_prefill_logits(
                str(self.root), [torch.zeros((2, _VOCAB), dtype=torch.float32)], 3
            )

    def test_real_text_refuses_segments_that_do_not_fit(self):
        runner = _runner(max_total=2 * (128 + 3) - 1)
        with self.assertRaisesRegex(ValueError, "bs2-in128-out3"):
            self._sweep(self._bench_args(), runner=runner)
        self.assertEqual(self.engine.extend_calls, [])

    def test_short_dataset_fails_before_running_anything(self):
        _write_dataset(self.dataset, count=2)
        with self.assertRaisesRegex(ValueError, "不够"):
            self._sweep(self._bench_args())
        self.assertEqual(self.engine.extend_calls, [])

    def test_random_mode_skips_what_does_not_fit_and_is_seeded(self):
        args = self._bench_args(dataset_path="", seed=7)
        runner = _runner(max_total=2 * (128 + 3) - 1)
        results = self._sweep(args, runner=runner)
        self.assertEqual(
            [(r["batch_size"], r["input_len"]) for r in results],
            [(2, 64), (1, 64), (1, 128)],
        )
        warmup = self.engine.extend_calls[0] + self.engine.extend_calls[1]
        first_measured = self.engine.extend_calls[2] + self.engine.extend_calls[3]
        self.assertNotEqual(warmup, first_measured)

        engine = _FakeEngine()
        with mock.patch.object(bo, "extend", engine.extend):
            self._sweep(args, runner=runner)
        self.assertEqual(engine.extend_calls, self.engine.extend_calls)

    def test_inputs_must_match_across_ranks(self):
        digests = []
        group = SimpleNamespace(all_gather_object=lambda d: [d, "另一个 rank 的输入"])
        with self.assertRaisesRegex(RuntimeError, "各 rank 拼出的输入不一样"):
            bo._check_same_input_on_all_ranks(
                _runner(tp_size=2, tp_group=group), "x", "abc"
            )
        group = SimpleNamespace(all_gather_object=lambda d: digests.append(d) or [d, d])
        bo._check_same_input_on_all_ranks(
            _runner(tp_size=2, tp_group=group), "x", "abc"
        )
        self.assertEqual(digests, ["abc"])


class SyntheticInputTests(unittest.TestCase):
    def test_same_seed_gives_same_tokens(self):
        a = bo.prepare_synthetic_inputs_for_latency_test(2, 16, seed=3)
        b = bo.prepare_synthetic_inputs_for_latency_test(2, 16, seed=3)
        c = bo.prepare_synthetic_inputs_for_latency_test(2, 16, seed=4)
        ids = lambda reqs: [list(r.origin_input_ids) for r in reqs]  # noqa: E731
        self.assertEqual(ids(a), ids(b))
        self.assertNotEqual(ids(a), ids(c))


class _FakeCudart:
    """torch.cuda.cudart() 的替身：开抓、停抓各记一笔。"""

    def __init__(self, events):
        self._events = events

    def cudaProfilerStart(self):
        self._events.append("开抓")

    def cudaProfilerStop(self):
        self._events.append("停抓")


class _FakeTorchProfiler:
    """torch.profiler.profile 的替身：开始、结束各记一笔。"""

    def __init__(self, events, **kwargs):
        self._events = events

    def start(self):
        self._events.append("剖析开始")

    def stop(self):
        self._events.append("剖析结束")


class _Clock:
    """bench 里的 time 模块只换掉 perf_counter：每取一次时刻记一笔，别的照旧。"""

    def __init__(self, events):
        self._events = events

    def perf_counter(self):
        self._events.append("取时刻")
        return time.perf_counter()

    def __getattr__(self, name):
        return getattr(time, name)


# 不剖析时，预填充（不分组）和每一步解码各自的调用顺序，和 77be5ae90 的写法一一对应。
_PREFILL = ["同步", "取时刻", "预填充", "同步", "取时刻"]
_STEP = ["同步", "取时刻", "解码", "同步", "取时刻"]
# CUDA_PROFILER 时要抓的那一步：开抓 → 对齐 → 解码，synchronize → 对齐 → 停抓。
# 两次对齐都在两次取时刻的外面，计时的起止点不变。
_CAPTURED_STEP = [
    "同步",
    "开抓",
    "对齐",
    "取时刻",
    "解码",
    "同步",
    "取时刻",
    "对齐",
    "停抓",
]


class ProfileTests(unittest.TestCase):
    """剖析开关：CUDA_PROFILER 时每个 rank 都在要抓的那一步前后各开抓、停抓一次，前后对齐；
    torch profiler 照旧只在 0 号上剖析；不开剖析时调用顺序和以前一样。"""

    def setUp(self):
        self.events = []
        events = self.events
        engine = _FakeEngine()

        def extend(reqs, model_runner):
            events.append("预填充")
            return engine.extend(reqs, model_runner)

        def decode(next_token_ids, batch, model_runner):
            events.append("解码")
            return engine.decode(next_token_ids, batch, model_runner)

        for target, name, value in [
            (bo, "extend", extend),
            (bo, "decode", decode),
            (bo, "synchronize", lambda device: events.append("同步")),
            (bo, "time", _Clock(events)),
            (bo, "step_trace", _TraceRecorder()),
            (bo, "_save_profile_trace_results", lambda *a: events.append("存结果")),
            (torch.cuda, "cudart", lambda: _FakeCudart(events)),
            (torch.profiler, "profile", lambda **kw: _FakeTorchProfiler(events, **kw)),
        ]:
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # TP 组只用到两样：对齐，以及各 rank 核对输入时的汇总。
        self.group = SimpleNamespace(
            barrier=lambda: events.append("对齐"),
            all_gather_object=lambda obj: [obj, obj],
        )

    def _run_once(
        self, profile, activities, *, stage="decode", tp_size=2, output_len=5
    ):
        reqs = bo.prepare_synthetic_inputs_for_latency_test(2, 8, seed=1)
        bo.latency_test_run_once(
            "profile-test",
            _runner(tp_size=tp_size, tp_group=self.group),
            _noop,
            reqs,
            2,
            8,
            output_len,
            "cpu",
            0,
            profile,
            False,
            activities,
            "profile-test",
            stage,
            0,
        )
        return self.events

    def _bench_args(self, **overrides):
        fields = dict(
            batch_size=(1, 2),
            len_pairs=("8:5",),
            dataset_path="",
            seed=3,
            result_filename="",
            profile=True,
            profile_activities=("CUDA_PROFILER",),
            profile_stage="decode",
        )
        fields.update(overrides)
        return bo.BenchArgs(**fields)

    def _sweep(self, bench_args, tp_rank):
        self.events.clear()
        bo.sweep_latency_test(
            _server_args(tp_size=2),
            bench_args,
            _runner(tp_size=2, tp_group=self.group),
            _Tokenizer(),
            tp_rank,
            _noop,
        )
        return collections.Counter(self.events)

    def test_without_profile_call_order_is_unchanged(self):
        for profile, activities in [
            (False, ("CPU", "GPU")),
            (None, ("CPU", "GPU")),
            (False, ("CUDA_PROFILER",)),
        ]:
            with self.subTest(profile=profile, activities=activities):
                self.events.clear()
                self.assertEqual(
                    self._run_once(profile, activities), _PREFILL + _STEP * 4
                )

    def test_cuda_profiler_captures_the_middle_step_and_aligns_around_it(self):
        # 输出 5 个 token：解码 4 步（i=0..3），抓 i = 5 // 2 = 2 那一步。
        self.assertEqual(
            self._run_once(True, ("CUDA_PROFILER",)),
            _PREFILL + _STEP * 2 + _CAPTURED_STEP + _STEP,
        )
        # 只有一张卡时不对齐，其余照旧。
        self.events.clear()
        self.assertEqual(
            self._run_once(True, ("CUDA_PROFILER",), tp_size=1),
            _PREFILL + _STEP * 2 + [e for e in _CAPTURED_STEP if e != "对齐"] + _STEP,
        )

    def test_cuda_profiler_aligns_prefill_too_when_it_is_captured(self):
        self.assertEqual(
            self._run_once(True, ("CUDA_PROFILER",), stage="all"),
            ["开抓", "对齐"]
            + _PREFILL
            + ["对齐", "停抓"]
            + _STEP * 2
            + _CAPTURED_STEP
            + _STEP,
        )

    def test_cuda_profiler_runs_on_every_rank(self):
        # 两组参数、各抓一步：每个 rank 每组开抓、停抓各一次，对齐两次；预热不抓。
        for tp_rank in (0, 1):
            with self.subTest(tp_rank=tp_rank):
                counts = self._sweep(self._bench_args(), tp_rank)
                self.assertEqual(
                    (counts["开抓"], counts["停抓"], counts["对齐"]), (2, 2, 4)
                )
                self.assertEqual(counts["剖析开始"], 0)

    def test_torch_profiler_stays_on_rank_zero(self):
        args = self._bench_args(profile_activities=("CPU", "GPU"))
        counts = self._sweep(args, 0)
        self.assertEqual(
            (counts["剖析开始"], counts["剖析结束"], counts["存结果"]), (2, 2, 2)
        )
        counts = self._sweep(args, 1)
        self.assertEqual((counts["剖析开始"], counts["剖析结束"]), (0, 0))
        for tp_rank in (0, 1):
            counts = self._sweep(args, tp_rank)
            self.assertEqual(
                (counts["开抓"], counts["停抓"], counts["对齐"]), (0, 0, 0)
            )

    def test_without_profile_no_rank_profiles_or_aligns(self):
        for activities in (("CPU", "GPU"), ("CUDA_PROFILER",)):
            for tp_rank in (0, 1):
                with self.subTest(activities=activities, tp_rank=tp_rank):
                    counts = self._sweep(
                        self._bench_args(profile=False, profile_activities=activities),
                        tp_rank,
                    )
                    for event in ("开抓", "停抓", "对齐", "剖析开始", "剖析结束"):
                        self.assertEqual(counts[event], 0, event)

    def test_legacy_path_passes_profile_the_same_way(self):
        # 不走扫描的老流程（latency_test）也是同一个规则：只记交给每组的 profile 和 record_shapes。
        seen = []
        signature = inspect.signature(bo.latency_test_run_once)

        def run_once(*args, **kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            seen.append((bound["profile"], bound["profile_record_shapes"]))

        runner = _runner(tp_size=2, tp_group=self.group)
        with (
            mock.patch.object(bo, "load_model", lambda *a: (runner, _Tokenizer())),
            mock.patch.object(bo, "latency_test_run_once", run_once),
        ):
            for activities, rank1 in [
                (("CUDA_PROFILER",), (True, None)),
                (("CPU", "GPU"), (None, None)),
            ]:
                for tp_rank, want in [(0, (True, True)), (1, rank1)]:
                    with self.subTest(activities=activities, tp_rank=tp_rank):
                        seen.clear()
                        args = bo.BenchArgs(
                            batch_size=(1,),
                            input_len=(8,),
                            output_len=(5,),
                            result_filename="",
                            profile=True,
                            profile_record_shapes=True,
                            profile_activities=activities,
                        )
                        bo.latency_test(_server_args(tp_size=2), None, args, 0, tp_rank)
                        # 第一次是预热，不剖析；第二次才是正式那组。
                        self.assertEqual(seen, [(False, False), want])


class GuardTests(unittest.TestCase):
    def test_grid_without_len_pairs_crosses_input_and_output_lengths(self):
        args = bo.BenchArgs(batch_size=(4, 1, 4), input_len=(8, 16), output_len=(3,))
        self.assertEqual(
            bo._sweep_grid(args), [(4, 8, 3), (1, 8, 3), (4, 16, 3), (1, 16, 3)]
        )

    def test_max_hits(self):
        problem = bo._max_hits_problem
        on = dict(enable_cpu_offload_fast_loop=True)
        self.assertIsNone(problem(_server_args(**on, cuda_graph_max_bs=8), [1, 8]))
        self.assertIn(
            "max_hits=4",
            problem(_server_args(**on, cpu_offload_max_hits=4), [1, 8]),
        )
        # 自动取值是 min(cuda_graph_max_bs, 128)。
        self.assertIn(
            "max_hits=128", problem(_server_args(**on, cuda_graph_max_bs=200), [200])
        )
        self.assertIsNone(problem(_server_args(cpu_offload_max_hits=4), [8]))
        self.assertIsNone(
            problem(
                _server_args(**on, cpu_offload_max_hits=4, disable_cuda_graph=True), [8]
            )
        )

    def _main_server_args(self, **overrides):
        args = _server_args(
            model_path="m", cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4]
        )
        args.__dict__.update(overrides)
        args._generate_cuda_graph_batch_sizes = lambda: [
            b for b in (1, 2, 4, 8, 12, 16) if b <= args.cuda_graph_max_bs
        ]
        return args

    def test_main_recomputes_cuda_graph_batch_sizes(self):
        seen = []
        server_args = self._main_server_args()
        with (
            mock.patch.object(
                bo,
                "latency_test",
                lambda sa, *rest: seen.append(list(sa.cuda_graph_bs)),
            ),
            mock.patch.object(bo, "PortArgs", SimpleNamespace(init_new=_noop)),
        ):
            bo.main(server_args, bo.BenchArgs(batch_size=(1, 4, 8)))
        self.assertEqual(seen, [[1, 2, 4, 8]])

    def test_main_rejects_batch_above_max_hits_before_loading(self):
        server_args = self._main_server_args(
            enable_cpu_offload_fast_loop=True, cpu_offload_max_hits=4
        )
        worker = mock.Mock()
        with (
            mock.patch.object(bo, "latency_test", worker),
            mock.patch.object(bo, "PortArgs", SimpleNamespace(init_new=_noop)),
        ):
            with self.assertRaisesRegex(ValueError, "max_hits=4"):
                bo.main(server_args, bo.BenchArgs(batch_size=(1, 8)))
        worker.assert_not_called()


class WaitForWorkersTests(unittest.TestCase):
    def test_failure_is_reported_without_waiting_for_a_stuck_rank(self):
        ctx = multiprocessing.get_context("spawn")
        stuck = ctx.Process(target=time.sleep, args=(60,))
        failed = ctx.Process(target=os._exit, args=(3,))
        for proc in (stuck, failed):
            proc.start()
        self.addCleanup(stuck.join)
        self.addCleanup(stuck.kill)
        tic = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, r"\(1, 3\)"):
            bo._wait_for_workers([stuck, failed])
        self.assertLess(time.monotonic() - tic, 30)

    def test_all_ranks_succeed(self):
        ctx = multiprocessing.get_context("spawn")
        workers = [ctx.Process(target=time.sleep, args=(0.1,)) for _ in range(2)]
        for proc in workers:
            proc.start()
        bo._wait_for_workers(workers)
        self.assertEqual([p.exitcode for p in workers], [0, 0])


if __name__ == "__main__":
    unittest.main()
