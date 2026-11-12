"""
Benchmark the latency of running a single static batch without a server.

This script does not launch a server and uses the low-level APIs.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (e.g., batch size, input lengths).

# Usage (latency test)
## with dummy weights:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --load-format dummy
## sweep through multiple data points and store (append) the results in a jsonl file:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --output-len 32 256 --run-name test_run
## run with profiling:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --profile
## run with profiling to custom directory:
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 --input-len 256 --profile
## run with CUDA profiler (nsys):
nsys profile --force-overwrite=true -o bench_one_batch python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 --input-len 256 --profile --profile-activities CUDA_PROFILER
## 真实文本、多组 batch 与长度交叉扫描（每组参数的输出写进各自的子目录）：
python -m sglang.bench_one_batch --model-path Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 --batch-size 1 4 8 --len-pairs 1024:1024 1024:8192 8192:1024 --dataset-path /data/ShareGPT_V3_unfiltered_cleaned_split.json --segment-output-dir /tmp/bench-out --result-filename ""
# Usage (correctness test):
python -m sglang.bench_one_batch --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --correct

## Reference output (of the correctness test above, can be gpu dependent):
input_ids=[[1, 450, 7483, 310, 3444, 338], [1, 450, 7483, 310, 278, 3303, 13187, 290, 338], [1, 20628, 338, 263, 6575, 1460, 2462, 322, 306, 763]]

prefill logits (first half): tensor([[-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [ -9.1875, -10.2500,   2.7129,  ...,  -4.3359,  -4.0664,  -4.1328]],
       device='cuda:0')

prefill logits (final): tensor([[-8.3125, -7.1172,  3.3457,  ..., -4.9570, -4.1328, -3.4141],
        [-8.9141, -9.0156,  4.1445,  ..., -4.9922, -4.4961, -4.0781],
        [-9.6328, -9.0547,  4.0195,  ..., -5.3047, -4.7148, -4.4570]],
       device='cuda:0')

========== Prompt 0 ==========
<s> The capital of France is Paris.
The capital of the United States is Washington, D.C.


========== Prompt 1 ==========
<s> The capital of the United Kindom is London.
The capital of the United Kingdom is London.
The capital of the

========== Prompt 2 ==========
<s> Today is a sunny day and I like to go for a walk in the park.
I'm going to the park
"""

import argparse
import copy
import dataclasses
import hashlib
import itertools
import json
import logging
import multiprocessing
import multiprocessing.connection
import os
import time
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from sglang.bench_profile_window import focus_matches, profile_steps, validate_focus
from sglang.bench_real_text import (
    RealTextPromptBuilder,
    group_by_token_budget,
    input_digest,
    iter_dataset_texts,
    parse_len_pairs,
    segment_name,
    trace_sample_every,
)
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.eplb.cpu_offload import step_trace
from sglang.srt.layers.dp_attention import set_is_extend_in_batch
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    is_cuda_alike,
    is_xpu,
    kill_process_tree,
    maybe_reindex_device_id,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

profile_activities = [torch.profiler.ProfilerActivity.CPU] + [
    profiler_activity
    for available, profiler_activity in [
        (is_cuda_alike(), torch.profiler.ProfilerActivity.CUDA),
        (is_xpu(), torch.profiler.ProfilerActivity.XPU),
    ]
    if available
]

_STEP_TRACE_ENABLED = get_bool_env_var("SGLANG_MOE_STEP_TRACE")


def start_profile(profile_activities, profile_record_shapes=False, rank_print=print):
    """
    Abstracted function to start profiling based on profile_activities.
    Returns profiler object (or None).
    """
    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStart()
            rank_print("CUDA Profiler started (nsys will begin capturing)")
        except Exception as e:
            rank_print(f"Failed to start CUDA profiler: {e}")
        return None
    else:
        activities = []
        if "CPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if "GPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if activities:
            profiler = torch.profiler.profile(
                activities=activities,
                with_stack=True,
                record_shapes=profile_record_shapes,
            )
            profiler.start()
            return profiler
        return None


def stop_profile(
    profiler,
    profile_activities,
    rank_print=print,
    save_trace=False,
    trace_filename=None,
    stage=None,
):
    """
    Abstracted function to stop profiling based on profile_activities.
    Optionally saves trace results and prints completion messages.
    """
    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStop()
            rank_print("CUDA Profiler stopped (nsys should dump traces)")
        except Exception as e:
            rank_print(f"Failed to stop CUDA profiler: {e}")
    elif profiler is not None:
        profiler.stop()

    if save_trace:
        if profiler is not None:
            if trace_filename:
                _save_profile_trace_results(profiler, trace_filename)
                stage_desc = f"for {stage}" if stage else ""
                rank_print(
                    f"torch profiler chrome trace {stage_desc} saved to {trace_filename}"
                )
        if "CUDA_PROFILER" in profile_activities:
            rank_print(f"CUDA profiler trace for {stage} completed")


def _profile_on_every_rank(profile_activities):
    """CUDA_PROFILER 只是通知 nsys 开抓、停抓，每个 rank 都自己通知，前后再对齐；
    torch profiler（CPU、GPU）照旧只在 0 号上剖析。

    10-16 试抓时只有 0 号通知，别的卡要抓的那一步一个 kernel 都没记上。
    """
    return "CUDA_PROFILER" in profile_activities


def _profile_for_rank(bench_args, tp_rank):
    """交给 latency_test_run_once 的 profile：0 号照旧；别的 rank 只在每个 rank 都要通知时才给。"""
    if tp_rank == 0 or _profile_on_every_rank(bench_args.profile_activities):
        return bench_args.profile
    return None


def _align_profiled_ranks(model_runner, profile_activities):
    """每个 rank 都开抓、停抓时，让 TP 组里的各 rank 在这里对齐一次。

    开抓之后、发要抓的那一步之前对齐一次；这一步 synchronize 之后、停抓之前再对齐一次。
    这样任何一个 rank 发这一步之前，所有 rank 都已经开抓；任何一个 rank 停抓之前，所有 rank
    这一步都已经跑完。多个进程各自开抓、停抓时 nsys 怎么合并，手册没写清，这样都切不掉哪张卡的一截。
    GroupCoordinator.barrier() 走的是 TP 组的 CPU 组（gloo），不发 GPU kernel，也不走 NCCL。
    两次对齐都放在计时区外面。torch profiler 只有 0 号进来，这里不能对齐，不然会卡住。
    """
    if not _profile_on_every_rank(profile_activities):
        return
    if int(getattr(model_runner, "tp_size", 1)) <= 1:
        return
    model_runner.tp_group.barrier()


def _decode_profile_steps(output_len):
    """解码时在哪一步开抓、哪一步停抓，都是 latency_test_run_once 循环里的 i；这两步都在抓取范围里。

    现在只抓正中间那一步：i = output_len // 2，输出 1024 时是 i=512。要换成"只让 0 号开抓，
    提前一步开、推后一步停"，改这里返回 (i - 1, i + 1)，再让 _profile_on_every_rank 返回 False。
    """
    step = output_len // 2
    return step, step


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    prompt_filename: str = ""
    result_filename: str = "result.jsonl"
    correctness_test: bool = False
    # This is only used for correctness test
    cut_len: int = 4
    apply_chat_template: bool = False
    log_decode_step: int = 0
    profile: bool = False
    profile_record_shapes: bool = False
    profile_activities: Tuple[str] = ("CPU", "GPU")
    profile_stage: str = "all"
    profile_filename_prefix: str = "profile"
    # 以下用于真实文本负载和多组参数扫描。from_cli_args 按默认值的类型做转换，
    # 所以这里的默认值不能写 None。
    dataset_path: str = ""
    dataset_offset: int = 0
    len_pairs: Tuple[str] = ()
    dataset_raw_text: bool = False
    seed: int = 0
    prefill_group_tokens: int = 8192
    trace_target_steps: int = 0
    segment_output_dir: str = ""
    profile_focus_window: Tuple[int] = ()

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--prompt-filename", type=str, default=BenchArgs.prompt_filename
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        parser.add_argument(
            "--apply-chat-template",
            action="store_true",
            help="Apply the tokenizer's chat template to correctness-test prompts.",
        )
        parser.add_argument(
            "--log-decode-step",
            type=int,
            default=BenchArgs.log_decode_step,
            help="Log decode latency by step, default is set to zero to disable.",
        )
        parser.add_argument("--profile", action="store_true", help="Enable profiling.")
        parser.add_argument(
            "--profile-focus-window", type=int, nargs=5, default=(),
            metavar=("BATCH", "INPUT", "OUTPUT", "BEFORE", "AFTER"),
            help="只抓指定组中点前后的一段；目标组 trace 每步记录，其余组保持原抽样。",
        )
        parser.add_argument(
            "--profile-record-shapes",
            action="store_true",
            help="Record tensor shapes in profiling results.",
        )
        parser.add_argument(
            "--profile-activities",
            type=str,
            nargs="+",
            default=["CPU", "GPU"],
            choices=["CPU", "GPU", "CUDA_PROFILER"],
            help="Profiler activities: CPU, GPU, CUDA_PROFILER. If CPU/GPU, use torch profiler. If CUDA_PROFILER, use CUDA profiler.",
        )
        parser.add_argument(
            "--profile-stage",
            type=str,
            default=BenchArgs.profile_stage,
            choices=["all", "prefill", "decode"],
            help="Which stage to profile: all, prefill, or decode only.",
        )
        parser.add_argument(
            "--profile-filename-prefix",
            type=str,
            default=BenchArgs.profile_filename_prefix,
            help="Prefix of the profiling file names. The full profiling result file(s) be "
            '"[profile_filename_prefix]_batch[batch_size]_input[input_len]_output[output_len].trace.json.gz"',
        )
        parser.add_argument(
            "--dataset-path",
            type=str,
            default=BenchArgs.dataset_path,
            help="用这个数据集里的真实文本拼输入（ShareGPT/ChatMed 的 JSON、JSONL 或纯文本）。"
            "MoE 的专家冷热取决于输入内容，评估挑热专家时不要用随机 token。",
        )
        parser.add_argument(
            "--dataset-offset",
            type=int,
            default=BenchArgs.dataset_offset,
            help="从第几条非空正文开始取，从 0 数。",
        )
        parser.add_argument(
            "--len-pairs",
            type=str,
            nargs="+",
            default=BenchArgs.len_pairs,
            help="输入:输出 长度组合，例如 1024:1024 1024:8192 8192:1024。给了就和 "
            "--batch-size 交叉扫描，不再用 --input-len/--output-len。",
        )
        parser.add_argument(
            "--dataset-raw-text",
            action="store_true",
            help="不套对话模板，直接拿正文当输入。",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=BenchArgs.seed,
            help="随机 token 输入的种子。每个 rank 各自生成输入，必须用同一个种子。",
        )
        parser.add_argument(
            "--prefill-group-tokens",
            type=int,
            default=BenchArgs.prefill_group_tokens,
            help="按整个请求分组预填充，每组 token 总数的上限；0 表示整批一次预填充。",
        )
        parser.add_argument(
            "--trace-target-steps",
            type=int,
            default=BenchArgs.trace_target_steps,
            help="开了逐步追踪（SGLANG_MOE_STEP_TRACE=1）时，每组参数大约记多少个 "
            "decode 步；0 表示每步都记。",
        )
        parser.add_argument(
            "--segment-output-dir",
            type=str,
            default=BenchArgs.segment_output_dir,
            help="每组参数一个子目录，写输出 token 和本组参数；开了逐步追踪时，"
            "追踪记录也按组分开写。",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def load_model(server_args, port_args, gpu_id, tp_rank):
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer


def prepare_inputs_for_correctness_test(bench_args, tokenizer, custom_prompts):
    prompts = (
        custom_prompts
        if custom_prompts
        else [
            "The capital of France is",
            "The capital of the United Kindom is",
            "Today is a sunny day and I like",
        ]
    )
    if bench_args.apply_chat_template:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(prompts)):
        assert len(input_ids[i]) > bench_args.cut_len

        tmp_input_ids = input_ids[i][: bench_args.cut_len]
        req = Req(
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=tmp_input_ids,
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        if _STEP_TRACE_ENABLED:
            req._step_trace_identity = "synthetic"
            req._step_trace_request_id = f"synthetic:correctness:{i}"
            req._step_trace_decode_step = 0
        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args, input_ids, reqs, model_runner
):
    for i in range(len(reqs)):
        req = reqs[i]
        req.fill_ids += input_ids[i][bench_args.cut_len :]
        req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
            i, : bench_args.cut_len
        ]
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
    return reqs


def prepare_synthetic_inputs_for_latency_test(
    batch_size,
    input_len,
    custom_inputs=None,
    *,
    seed=0,
    request_id_prefix="synthetic:latency",
):
    # 多 rank 时每个进程各自生成输入。不固定种子的话各 rank 拿到的 token 不一样，
    # 张量并行把不同输入的部分结果加在一起，算出来的东西没有意义。
    input_ids = (
        custom_inputs
        if custom_inputs
        else np.random.default_rng(seed).integers(
            0, 10000, (batch_size, input_len), dtype=np.int32
        )
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        if _STEP_TRACE_ENABLED:
            req._step_trace_identity = "synthetic"
            req._step_trace_request_id = f"{request_id_prefix}:{i}"
            req._step_trace_decode_step = 0
        reqs.append(req)

    return reqs


@torch.no_grad
def extend(reqs, model_runner):
    # Create dummy tree_cache for benchmarks (no prefix caching, just allocation)
    dummy_tree_cache = SimpleNamespace(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )

    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=dummy_tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    # bench_one_batch does not run the full scheduler's DP all-gather that
    # normally calls set_is_extend_in_batch(). Set the global flag explicitly
    # so that get_is_extend_in_batch() returns True during prefill even when
    # prepare_mlp_sync_batch is not called (e.g. when require_mlp_sync=False).
    batch.is_extend_in_batch = True
    set_is_extend_in_batch(True)
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad
def extend_in_groups(reqs, model_runner, group_tokens, device, logits_sink=None):
    """按整个请求分组预填充，再合成一个 batch 交给 decode。

    8 个 8192 token 的请求一次塞进一个 forward，规模远超在线服务（在线服务按块
    预填充），显存也未必扛得住。这里按 ``group_tokens`` 分组，每组单独 extend，
    最后用 merge_batch 合并，之后的 decode 和一次预填充完全一样。

    ``logits_sink`` 给了列表时，把每组预填充算出的 logits 复制一份，按请求顺序
    追加进去，用来核对开不开 offload 时预填充的结果是否逐位相同。复制的只有
    batch × 词表个数，几微秒，和几百毫秒的预填充比可以忽略。

    返回 ``(next_token_ids, batch, groups, group_latencies)``；只有一组时直接
    走 :func:`extend`，``group_latencies`` 为空。
    """
    groups = group_by_token_budget([len(r.fill_ids) for r in reqs], group_tokens)
    if len(groups) <= 1:
        next_token_ids, logits, batch = extend(reqs, model_runner)
        if logits_sink is not None:
            logits_sink.append(logits.clone())
        return next_token_ids, batch, groups, []

    batch = None
    next_token_parts = []
    group_latencies = []
    for indices in groups:
        synchronize(device)
        tic = time.perf_counter()
        group_next_ids, group_logits, group_batch = extend(
            [reqs[i] for i in indices], model_runner
        )
        synchronize(device)
        group_latencies.append(time.perf_counter() - tic)
        if logits_sink is not None:
            # 下一组的 forward 可能复用同一块输出缓冲区，先复制出来。
            logits_sink.append(group_logits.clone())
        next_token_parts.append(group_next_ids)
        if batch is None:
            batch = group_batch
        else:
            # 分组是连续的下标区间，按顺序合并后请求次序和 next_token_ids 一致。
            batch.merge_batch(group_batch)
    return torch.cat(next_token_parts), batch, groups, group_latencies


@torch.no_grad
def decode(input_token_ids, batch, model_runner):
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    if _STEP_TRACE_ENABLED:
        # bench_one_batch 不走 scheduler 的 output_ids 更新逻辑，因此单独维护
        # synthetic 请求的 decode 步号；第一轮 decode 前已经有 prefill 产出的
        # 第一个 token，所以从 1 开始。
        for req in batch.reqs:
            if getattr(req, "_step_trace_identity", "") == "synthetic":
                req._step_trace_decode_step = (
                    int(getattr(req, "_step_trace_decode_step", 0)) + 1
                )
    set_is_extend_in_batch(False)
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits


def _maybe_prepare_mlp_sync_batch(batch: ScheduleBatch, model_runner):
    if require_mlp_sync(model_runner.server_args):
        Scheduler.prepare_mlp_sync_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=1,
            tp_group=model_runner.tp_group,
            get_idle_batch=None,
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            speculative_num_draft_tokens=None,
            require_mlp_tp_gather=require_mlp_tp_gather(model_runner.server_args),
            disable_overlap_schedule=model_runner.server_args.disable_overlap_schedule,
            offload_tags=set(),
        )


def _read_prompts_from_file(prompt_file, rank_print):
    """Read custom prompts from the file specified by `--prompt-filename`."""
    if not prompt_file:
        return []
    if not os.path.exists(prompt_file):
        rank_print(
            f"Custom prompt file {prompt_file} not found. Using default inputs..."
        )
        return []
    with open(prompt_file, "r") as pf:
        return pf.readlines()


def _get_torch_profiler_output_dir():
    return os.environ.get("SGLANG_TORCH_PROFILER_DIR", "/tmp")


def _create_torch_profiler_filename(
    profile_filename_prefix, batch_size, input_len, output_len, stage
):
    output_dir = _get_torch_profiler_output_dir()
    filename = f"{profile_filename_prefix}_batch{batch_size}_input{input_len}_output{output_len}_{stage}.trace.json.gz"
    return os.path.join(output_dir, filename)


def _save_profile_trace_results(profiler, filename):
    parent_dir = os.path.dirname(os.path.abspath(filename))
    os.makedirs(parent_dir, exist_ok=True)
    profiler.export_chrome_trace(filename)
    print(
        profiler.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total"
        )
    )


def correctness_test(
    server_args,
    port_args,
    bench_args,
    gpu_id,
    tp_rank,
):
    # Initialize MoE config to ensure correctness test uses the same configuration
    # as latency test. This is critical for --enable-ep-moe-heto and other MoE
    # parameters to take effect properly.
    initialize_moe_config(server_args)

    # Set CPU affinity
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, tp_rank
        )
        import psutil
        _cpus = psutil.Process().cpu_affinity()
        print(f"[TP{tp_rank}] NUMA_BIND: gpu_id={gpu_id} cpus={_cpus[:4]}... (total={len(_cpus)})", flush=True)

    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)

    # Prepare inputs
    custom_prompts = _read_prompts_from_file(bench_args.prompt_filename, rank_print)
    input_ids, reqs = prepare_inputs_for_correctness_test(
        bench_args, tokenizer, custom_prompts
    )
    rank_print(f"\n{input_ids=}\n")

    if bench_args.cut_len > 0:
        # Prefill
        next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
        rank_print(f"prefill logits (first half): {next_token_logits} \n")

    # Prepare extend inputs
    reqs = prepare_extend_inputs_for_correctness_test(
        bench_args, input_ids, reqs, model_runner
    )

    # Extend (prefill w/ KV cache)
    next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
    rank_print(f"prefill logits (final): {next_token_logits} \n")

    # Decode
    output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    for _ in range(bench_args.output_len[0] - 1):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        next_token_ids_list = next_token_ids.tolist()
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids_list[i])

    # Print output texts
    for i in range(len(reqs)):
        rank_print(f"========== Prompt {i} ==========")
        rank_print(tokenizer.decode(output_ids[i]), "\n")

    check_output_path = os.environ.get(
        "SGLANG_MOE_STEP_TRACE_CHECK_OUTPUT", ""
    ).strip()
    if tp_rank == 0 and check_output_path:
        parent = os.path.dirname(os.path.abspath(check_output_path))
        os.makedirs(parent, exist_ok=True)
        with open(check_output_path, "w", encoding="utf-8") as check_file:
            json.dump(
                {
                    "schema_version": 1,
                    "prompts": [
                        {
                            "prompt_index": i,
                            "output_ids": [int(token_id) for token_id in ids],
                        }
                        for i, ids in enumerate(output_ids)
                    ],
                },
                check_file,
                ensure_ascii=False,
                indent=2,
            )


def synchronize(device):
    torch.get_device_module(device).synchronize()


def latency_test_run_once(
    run_name,
    model_runner,
    rank_print,
    reqs,
    batch_size,
    input_len,
    output_len,
    device,
    log_decode_step,
    profile,
    profile_record_shapes,
    profile_activities,
    profile_filename_prefix,
    profile_stage,
    tp_rank,
    *,
    prefill_group_tokens=0,
    output_sink: Optional[List[List[int]]] = None,
    prefill_logits_sink: Optional[List[torch.Tensor]] = None,
    decode_latency_sink: Optional[List[float]] = None,
    profile_focus_window=(),
):
    """跑一组参数。

    ``output_sink`` 给了列表时，把每个请求生成的 token（含预填充产出的第一个）
    按请求顺序追加进去，用来比对 offload 前后输出是否一致。
    ``prefill_logits_sink`` 收预填充算出的 logits，见 :func:`extend_in_groups`。
    ``decode_latency_sink`` 收每一步解码的耗时（秒），按步号顺序。
    """
    max_batch_size = model_runner.max_total_num_tokens // (input_len + output_len)
    if batch_size > max_batch_size:
        rank_print(
            f"skipping ({batch_size}, {input_len}, {output_len}) due to max batch size limit"
        )
        return

    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
    }

    tot_latency = 0

    profiler = None
    enable_profile_prefill = profile and profile_stage in ["all", "prefill"]
    if enable_profile_prefill:
        profiler = start_profile(
            profile_activities,
            profile_record_shapes=profile_record_shapes,
            rank_print=rank_print,
        )
        _align_profiled_ranks(model_runner, profile_activities)

    synchronize(device)
    tic = time.perf_counter()
    next_token_ids, batch, prefill_groups, group_latencies = extend_in_groups(
        reqs, model_runner, prefill_group_tokens, device, prefill_logits_sink
    )
    synchronize(device)
    prefill_latency = time.perf_counter() - tic
    first_token_ids = next_token_ids.clone() if output_sink is not None else None

    if enable_profile_prefill:
        _align_profiled_ranks(model_runner, profile_activities)
        trace_filename = _create_torch_profiler_filename(
            profile_filename_prefix, batch_size, input_len, output_len, "prefill"
        )
        stop_profile(
            profiler,
            profile_activities,
            rank_print=rank_print,
            save_trace=True,
            trace_filename=trace_filename,
            stage="prefill",
        )

    tot_latency += prefill_latency
    throughput = input_len * batch_size / prefill_latency
    rank_print(
        f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput
    if len(prefill_groups) > 1:
        measurement_results["prefill_groups"] = len(prefill_groups)
        measurement_results["prefill_group_latencies"] = group_latencies

    decode_latencies = []
    decoded_ids = None
    profile_start_step, profile_stop_step = (
        profile_steps(output_len, profile_focus_window)
        if profile_focus_window else _decode_profile_steps(output_len)
    )
    enable_profile_decode = profile and profile_stage in ["all", "decode"]
    profiler = None
    for i in range(output_len - 1):
        synchronize(device)
        if enable_profile_decode and i == profile_start_step:
            profiler = start_profile(
                profile_activities,
                profile_record_shapes=profile_record_shapes,
                rank_print=rank_print,
            )
            # 所有 rank 都开抓了，才发要抓的这一步。
            _align_profiled_ranks(model_runner, profile_activities)

        tic = time.perf_counter()
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        synchronize(device)
        latency = time.perf_counter() - tic
        if output_sink is not None:
            # 计时已经结束；拷贝留在 GPU 上，下一步开始计时前的同步会等它做完。
            if decoded_ids is None:
                decoded_ids = torch.empty(
                    (output_len - 1, batch_size),
                    dtype=next_token_ids.dtype,
                    device=next_token_ids.device,
                )
            decoded_ids[i].copy_(next_token_ids)

        if enable_profile_decode and i == profile_stop_step:
            # 所有 rank 这一步都跑完了，才停抓。
            _align_profiled_ranks(model_runner, profile_activities)
            trace_filename = _create_torch_profiler_filename(
                profile_filename_prefix, batch_size, input_len, output_len, "decode"
            )
            stop_profile(
                profiler,
                profile_activities,
                rank_print=rank_print,
                save_trace=True,
                trace_filename=trace_filename,
                stage="decode",
            )
            profiler = None

        tot_latency += latency
        throughput = batch_size / latency
        decode_latencies.append(latency)
        if i < 5 or (log_decode_step > 0 and i % log_decode_step == 0):
            rank_print(
                f"Decode {i}. Batch size: {batch_size}, latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s"
            )

    if decode_latency_sink is not None:
        decode_latency_sink.extend(decode_latencies)

    # Record decode timing from 2nd output
    if output_len > 1:
        med_decode_latency = np.median(decode_latencies)
        med_decode_throughput = batch_size / med_decode_latency
        rank_print(
            f"Decode.  median latency: {med_decode_latency:6.5f} s, median throughput: {med_decode_throughput:9.2f} token/s"
        )
        measurement_results["median_decode_latency"] = med_decode_latency
        measurement_results["median_decode_throughput"] = med_decode_throughput

    throughput = (input_len + output_len) * batch_size / tot_latency
    rank_print(
        f"Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["total_latency"] = tot_latency
    measurement_results["overall_throughput"] = throughput
    if output_sink is not None:
        steps = [first_token_ids.tolist()]
        if decoded_ids is not None:
            steps.extend(decoded_ids.tolist())
        output_sink.extend([int(step[r]) for step in steps] for r in range(batch_size))
    return measurement_results


def latency_test(
    server_args,
    port_args,
    bench_args,
    gpu_id,
    tp_rank,
):
    initialize_moe_config(server_args)

    # Set CPU affinity
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, tp_rank
        )

    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)

    if _sweep_requested(bench_args):
        result_list = sweep_latency_test(
            server_args, bench_args, model_runner, tokenizer, tp_rank, rank_print
        )
        _write_latency_results(bench_args, tp_rank, result_list)
        if server_args.tp_size > 1:
            destroy_distributed_environment()
        return

    # Prepare inputs for warm up
    reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args.batch_size[0], bench_args.input_len[0], seed=bench_args.seed
    )

    # Warm up
    rank_print("Warmup ...")
    latency_test_run_once(
        bench_args.run_name,
        model_runner,
        rank_print,
        reqs,
        bench_args.batch_size[0],
        bench_args.input_len[0],
        min(32, bench_args.output_len[0]),  # shorter decoding to speed up the warmup
        server_args.device,
        log_decode_step=0,
        profile=False,
        profile_record_shapes=False,
        profile_activities=("CPU", "GPU"),
        profile_filename_prefix="",
        profile_stage="all",
        tp_rank=tp_rank,
    )

    rank_print("Benchmark ...")

    custom_inputs = _read_prompts_from_file(bench_args.prompt_filename, rank_print)
    custom_inputs = [tokenizer.encode(p.strip()) for p in custom_inputs]
    custom_input_len = len(custom_inputs)

    # Run the sweep
    result_list = []
    for bs, il, ol in itertools.product(
        bench_args.batch_size, bench_args.input_len, bench_args.output_len
    ):
        bs_aligned_inputs = []
        if custom_inputs:
            if custom_input_len == bs:
                bs_aligned_inputs = custom_inputs
            elif custom_input_len > bs:
                rank_print(
                    f"Custom input size ({custom_input_len}) is larger than batch_size ({bs}). "
                    f"Using the first {bs} prompts."
                )
                bs_aligned_inputs = copy.deepcopy(custom_inputs[:bs])
            else:
                rank_print(
                    f"Custom input size ({custom_input_len}) is smaller than batch_size ({bs}). "
                    f"Pad to the desired batch_size with the last prompt."
                )
                bs_aligned_inputs = copy.deepcopy(custom_inputs)
                bs_aligned_inputs.extend(
                    [bs_aligned_inputs[-1]] * (bs - custom_input_len)
                )

        reqs = prepare_synthetic_inputs_for_latency_test(
            bs, il, bs_aligned_inputs, seed=bench_args.seed
        )
        ret = latency_test_run_once(
            bench_args.run_name,
            model_runner,
            rank_print,
            reqs,
            bs,
            il,
            ol,
            server_args.device,
            bench_args.log_decode_step,
            _profile_for_rank(bench_args, tp_rank),
            bench_args.profile_record_shapes if tp_rank == 0 else None,
            bench_args.profile_activities,
            bench_args.profile_filename_prefix,
            bench_args.profile_stage,
            tp_rank,
        )
        if ret is not None:
            result_list.append(ret)

    _write_latency_results(bench_args, tp_rank, result_list)

    if server_args.tp_size > 1:
        destroy_distributed_environment()


def _write_latency_results(bench_args, tp_rank, result_list):
    # Write results in jsonlines format on rank 0.
    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")


def _sweep_requested(bench_args) -> bool:
    return bool(
        bench_args.dataset_path or bench_args.len_pairs or bench_args.segment_output_dir
    )


def _sweep_grid(bench_args):
    """外层是长度组合、内层是 batch：短组合排在前面，出问题能早发现。"""
    if bench_args.len_pairs:
        pairs = parse_len_pairs(bench_args.len_pairs)
    else:
        pairs = list(itertools.product(bench_args.input_len, bench_args.output_len))
    batch_sizes = list(dict.fromkeys(int(bs) for bs in bench_args.batch_size))
    return [(bs, il, ol) for il, ol in pairs for bs in batch_sizes]


def _max_hits_problem(server_args, batch_sizes) -> Optional[str]:
    """走 CUDA Graph 的 offload 每层每步最多把 max_hits 个 token 送去 CPU。

    batch 里每个 token 最多选同一个专家一次，所以 batch 不超过 max_hits 就不会
    有 token 被悄悄丢掉。自动取值规则和 ModelRunner 初始化 offload 时一致。
    """
    if not getattr(server_args, "enable_cpu_offload_fast_loop", False):
        return None
    if getattr(server_args, "disable_cuda_graph", False):
        return None
    max_hits = int(getattr(server_args, "cpu_offload_max_hits", 0) or 0)
    if max_hits <= 0:
        max_hits = min(int(server_args.cuda_graph_max_bs or 256), 128)
    largest = max(batch_sizes)
    if largest <= max_hits:
        return None
    return (
        f"batch {largest} 超过 CPU 专家每步能接收的 token 数 max_hits={max_hits}，"
        f"多出的 token 会被悄悄丢掉；把 --cpu-offload-max-hits 调到不小于 {largest}"
    )


def _capacity_problems(model_runner, batch_size, input_len, output_len) -> List[str]:
    problems = []
    need = batch_size * (input_len + output_len)
    have = int(model_runner.max_total_num_tokens)
    if need > have:
        problems.append(
            f"KV 缓存要放 {need} 个 token，只有 {have} 个位置"
            "（调大 --max-total-tokens 或 --mem-fraction-static）"
        )
    slots = getattr(model_runner.req_to_token_pool, "size", None)
    if slots is not None and batch_size > slots:
        problems.append(f"请求槽只有 {slots} 个（调大 --max-running-requests）")
    context_len = getattr(model_runner.model_config, "context_len", None)
    if context_len and input_len + output_len > context_len:
        problems.append(
            f"输入加输出共 {input_len + output_len} 个 token，超过模型上下文长度 {context_len}"
        )
    return problems


def _check_same_input_on_all_ranks(model_runner, name, digest):
    if int(getattr(model_runner, "tp_size", 1)) <= 1:
        return
    # 每个 rank 拿到的是同一份汇总结果，所以要报错就一起报错，不会有 rank 卡在
    # 后面的集合通信里等。
    digests = model_runner.tp_group.all_gather_object(digest)
    if len(set(digests)) != 1:
        raise RuntimeError(
            f"{name}：各 rank 拼出的输入不一样（sha256 依次为 {digests}），结果没有意义"
        )


_TIMING_KEYS = (
    "prefill_latency",
    "prefill_throughput",
    "prefill_groups",
    "prefill_group_latencies",
    "median_decode_latency",
    "median_decode_throughput",
    "total_latency",
    "overall_throughput",
)


def _write_prefill_logits(segment_dir, parts, batch_size):
    """预填充 logits 存成 float32 的 .npy，一行一个请求，按请求顺序。

    返回写进 segment.json 的说明。sha256 按 float32 原始字节算（小端、按行连续），
    和 .npy 文件头后面的数据段一致；``row_sha256`` 逐个请求算，不读 .npy 也能
    比两遍是否逐位相同。
    """
    logits = torch.cat(parts) if len(parts) > 1 else parts[0]
    if logits.dim() != 2 or logits.shape[0] != batch_size:
        raise RuntimeError(
            f"预填充 logits 的形状是 {tuple(logits.shape)}，和 batch {batch_size} 对不上"
        )
    array = np.ascontiguousarray(
        logits.detach().to(torch.float32).cpu().numpy(), dtype="<f4"
    )
    np.save(os.path.join(segment_dir, "prefill_logits.npy"), array, allow_pickle=False)
    return {
        "file": "prefill_logits.npy",
        "dtype": "float32",
        "source_dtype": str(logits.dtype).removeprefix("torch."),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "row_sha256": [hashlib.sha256(row.tobytes()).hexdigest() for row in array],
    }


def _write_decode_latencies(segment_dir, latencies):
    """每一步解码的耗时，整数纳秒，按步号顺序。中位数就是 timing 里那个。"""
    name = "decode_latencies.json"
    with open(os.path.join(segment_dir, name), "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": 1,
                "unit": "ns",
                "latencies_ns": [round(x * 1e9) for x in latencies],
            },
            f,
            separators=(",", ":"),
        )
    return name


def _write_segment_outputs(
    out_root,
    name,
    workload,
    builder,
    records,
    input_ids,
    outputs,
    result,
    extra,
    *,
    prefill_logits=None,
    decode_latencies=None,
):
    segment_dir = os.path.join(out_root, name)
    os.makedirs(segment_dir, exist_ok=True)
    with open(os.path.join(segment_dir, "output_ids.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": 1,
                "prompts": [
                    {"prompt_index": i, "output_ids": list(ids) + list(outputs[i])}
                    for i, ids in enumerate(input_ids)
                ],
            },
            f,
            separators=(",", ":"),
        )
    timing = {key: result[key] for key in _TIMING_KEYS if key in result}
    info = {"schema_version": 1, "segment": name} | workload | extra
    info |= {
        "chat_template": builder.chat_template if builder is not None else None,
        "template_prefix_tokens": len(builder.prefix_ids) if builder is not None else 0,
        "template_suffix_tokens": len(builder.suffix_ids) if builder is not None else 0,
        "dataset_records": records,
        "timing": timing,
    }
    if prefill_logits:
        info["prefill_logits"] = _write_prefill_logits(
            segment_dir, prefill_logits, workload["batch_size"]
        )
    if decode_latencies is not None:
        info["decode_latencies_file"] = _write_decode_latencies(
            segment_dir, decode_latencies
        )
    with open(os.path.join(segment_dir, "segment.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return {
        "segment": name,
        "batch_size": workload["batch_size"],
        "input_len": workload["input_len"],
        "output_len": workload["output_len"],
    } | timing


def sweep_latency_test(
    server_args, bench_args, model_runner, tokenizer, tp_rank, rank_print
):
    """真实文本（或固定种子的随机 token）在多组 batch 和长度上的扫描。

    每组参数对应一个追踪分段和一个输出子目录。各组之间不重置 offload 状态，
    fast loop 挑出的热专家会带到下一组，和在线服务一直跑着是一样的。decode
    不看 EOS，每个请求都生成满 output_len 个 token。
    """
    grid = _sweep_grid(bench_args)
    real_text = bool(bench_args.dataset_path)
    runnable = []
    for bs, il, ol in grid:
        problems = _capacity_problems(model_runner, bs, il, ol)
        if not problems:
            runnable.append((bs, il, ol))
            continue
        message = f"{segment_name(bs, il, ol)} 跑不了：" + "；".join(problems)
        if real_text:
            # 真实文本扫描一般是成组对照，悄悄少跑一组比直接报错更难发现。
            raise ValueError(message)
        rank_print(f"跳过 {message}")
    if not runnable:
        rank_print("没有能跑的参数组合")
        return []

    builder = None
    if real_text:
        texts = itertools.islice(
            iter_dataset_texts(bench_args.dataset_path), bench_args.dataset_offset, None
        )
        builder = RealTextPromptBuilder(
            tokenizer,
            texts,
            chat_template=not bench_args.dataset_raw_text,
            first_record_index=bench_args.dataset_offset,
        )
        rank_print(
            f"真实文本：{bench_args.dataset_path}，从第 {bench_args.dataset_offset} 条正文起取，"
            f"对话模板占 {builder.template_len} 个 token"
        )

    def build_inputs(bs, il, *, warmup=False):
        if builder is not None:
            built = builder.build_batch(bs, il, warmup_start if warmup else 0)
            return built.input_ids, built.record_indices
        # 预热换一个随机数流，不和正式测量用同一批 token。
        rng = np.random.default_rng([bench_args.seed, 1] if warmup else bench_args.seed)
        rows = rng.integers(0, 10000, (bs, il), dtype=np.int32)
        return [[int(x) for x in row] for row in rows], None

    warmup_start = 0
    if builder is not None:
        # 先把每组要用的正文都分好词：数据集不够长的话现在就报错，不用等前面几组
        # 跑完才发现。预热接在这些正文后面取，fast loop 不会提前见到要测的内容。
        warmup_start = max(
            builder.build_batch(bs, il).next_record for bs, il, _ in runnable
        )

    # 预热用第一组的 batch 和长度，只跑 32 步 decode，期间不记逐步追踪。
    bs0, il0, ol0 = runnable[0]
    warmup_ids, _ = build_inputs(bs0, il0, warmup=True)
    rank_print("Warmup ...")
    step_trace.pause()
    try:
        latency_test_run_once(
            bench_args.run_name,
            model_runner,
            rank_print,
            prepare_synthetic_inputs_for_latency_test(
                bs0, il0, warmup_ids, request_id_prefix="synthetic:warmup"
            ),
            bs0,
            il0,
            min(32, ol0),
            server_args.device,
            log_decode_step=0,
            profile=False,
            profile_record_shapes=False,
            profile_activities=("CPU", "GPU"),
            profile_filename_prefix="",
            profile_stage="all",
            tp_rank=tp_rank,
            prefill_group_tokens=bench_args.prefill_group_tokens,
        )
    finally:
        step_trace.resume()

    out_root = bench_args.segment_output_dir
    result_list = []
    summaries = []
    for bs, il, ol in runnable:
        name = segment_name(bs, il, ol)
        input_ids, records = build_inputs(bs, il)
        digest = input_digest(input_ids)
        _check_same_input_on_all_ranks(model_runner, name, digest)
        sample_every = trace_sample_every(ol, bench_args.trace_target_steps)
        focus = bench_args.profile_focus_window
        targeted = focus_matches(focus, bs, il, ol)
        if focus and targeted:
            sample_every = 1
        workload = {
            "workload": "real_text" if real_text else "random",
            "batch_size": bs,
            "input_len": il,
            "output_len": ol,
            "dataset": bench_args.dataset_path or None,
            "dataset_offset": bench_args.dataset_offset if real_text else None,
            "seed": None if real_text else bench_args.seed,
            "input_sha256": digest,
            "ignore_eos": True,
        }
        reqs = prepare_synthetic_inputs_for_latency_test(
            bs, il, input_ids, request_id_prefix=f"synthetic:latency:{name}"
        )
        outputs: List[List[int]] = []
        prefill_logits: List[torch.Tensor] = []
        decode_latencies: List[float] = []
        rank_print(f"Benchmark {name} ...")
        step_trace.start_segment(name, workload, sample_every=sample_every)
        try:
            ret = latency_test_run_once(
                bench_args.run_name,
                model_runner,
                rank_print,
                reqs,
                bs,
                il,
                ol,
                server_args.device,
                bench_args.log_decode_step,
                _profile_for_rank(bench_args, tp_rank) if targeted else False,
                bench_args.profile_record_shapes if tp_rank == 0 else None,
                bench_args.profile_activities,
                bench_args.profile_filename_prefix,
                bench_args.profile_stage,
                tp_rank,
                prefill_group_tokens=bench_args.prefill_group_tokens,
                output_sink=outputs,
                prefill_logits_sink=prefill_logits,
                decode_latency_sink=decode_latencies,
                profile_focus_window=focus if targeted else (),
            )
        finally:
            step_trace.end_segment()
        if ret is None:
            raise RuntimeError(f"{name} 没有跑：容量检查通过了，运行时却判定放不下")
        result_list.append(ret)
        if tp_rank == 0:
            preview = tokenizer.decode(outputs[0][:48])
            rank_print(f"{name} 第 0 个请求生成的开头：{preview!r}")
        if tp_rank == 0 and out_root:
            summaries.append(
                _write_segment_outputs(
                    out_root,
                    name,
                    workload,
                    builder,
                    records,
                    input_ids,
                    outputs,
                    ret,
                    {
                        "decode_steps": max(0, ol - 1),
                        "prefill_group_tokens": bench_args.prefill_group_tokens,
                        "trace": {
                            "enabled": _STEP_TRACE_ENABLED,
                            "sample_every": sample_every,
                            "target_steps": bench_args.trace_target_steps,
                            "expected_steps": (
                                len(range(1, ol, sample_every))
                                if _STEP_TRACE_ENABLED
                                else 0
                            ),
                        },
                    },
                    prefill_logits=prefill_logits,
                    decode_latencies=decode_latencies,
                )
            )

    if tp_rank == 0 and out_root:
        with open(os.path.join(out_root, "segments.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"schema_version": 1, "segments": summaries},
                f,
                ensure_ascii=False,
                indent=2,
            )
    return result_list


def _wait_for_workers(workers):
    """等所有 rank 退出；一旦有 rank 出错就不再等其余的。

    张量并行时一个 rank 出错退出，其余 rank 会卡在下一次集合通信上。按顺序
    join 的话，要是先等的正好是卡住的那个，就会一直等下去。这里谁先退出就先
    检查谁，发现非零退出码立刻报错，卡住的进程由入口处的 kill_process_tree 清理。
    """
    pending = {proc.sentinel: (rank, proc) for rank, proc in enumerate(workers)}
    while pending:
        failed = []
        for sentinel in multiprocessing.connection.wait(list(pending)):
            rank, proc = pending.pop(sentinel)
            proc.join()
            if proc.exitcode != 0:
                failed.append((rank, proc.exitcode))
        if failed:
            raise RuntimeError(
                f"rank 异常退出（rank, 退出码）：{failed}；其余 {len(pending)} 个 rank 不再等待"
            )


def main(server_args, bench_args):
    validate_focus(
        bench_args.profile_focus_window, bench_args.batch_size, bench_args.len_pairs,
        bench_args.profile, bench_args.profile_activities, bench_args.profile_stage,
        bench_args.correctness_test,
    )
    # Set cuda_graph_max_bs to the bench batch size for proper graph capture.
    # CPU offload now works WITH CUDA graph via MappedHostCallback kernels.
    if not getattr(server_args, "disable_cuda_graph", False):
        server_args.cuda_graph_max_bs = max(bench_args.batch_size)
        # 抓图档位在 ServerArgs 初始化时已经按当时的上限生成好了，抓图只读这份
        # 列表、不看上限。只改上限的话，比原上限大的 batch 找不到图，改走 eager；
        # 而 CUDA Graph offload 开着时 eager 路径不做 offload，那一组测的其实是纯 GPU。
        server_args.cuda_graph_bs = server_args._generate_cuda_graph_batch_sizes()

    # 只看启动参数就能判断，放在加载模型之前，省得等几分钟才报错。
    hits_problem = _max_hits_problem(server_args, bench_args.batch_size)
    if hits_problem:
        raise ValueError(hits_problem)

    _set_envs_and_config(server_args)

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        else:
            work_func = latency_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            with maybe_reindex_device_id(tp_rank) as gpu_id:
                proc = multiprocessing.Process(
                    target=work_func,
                    args=(
                        server_args,
                        port_args,
                        bench_args,
                        gpu_id,
                        tp_rank,
                    ),
                )
                proc.start()
                workers.append(proc)

        _wait_for_workers(workers)

        proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
