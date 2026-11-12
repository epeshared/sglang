"""沿用实验分支的 CPU 专家参数，接入新版参数命名空间。"""

from __future__ import annotations

import argparse
from typing import Literal, Optional

import msgspec

from sglang.srt.arg_groups.arg_utils import A, Arg


class CpuOffloadFields(msgspec.Struct):
    """动态 CPU 专家参数；默认关闭。"""

    _NS_PATH = "exec.moe"
    enable_ep_moe_heto: A[
        bool,
        Arg(
            help="CPU 专家实验参数：enable_ep_moe_heto",
            action=argparse.BooleanOptionalAction,
        ),
    ] = False
    ep_moe_heto_gpu_experts: A[int, "CPU 专家实验参数：ep_moe_heto_gpu_experts"] = -1
    enable_cpu_offload_fast_loop: A[
        bool,
        Arg(
            help="CPU 专家实验参数：enable_cpu_offload_fast_loop",
            action=argparse.BooleanOptionalAction,
        ),
    ] = False
    cpu_offload_decode_only: A[
        bool,
        Arg(
            help="CPU 专家实验参数：cpu_offload_decode_only",
            action=argparse.BooleanOptionalAction,
        ),
    ] = True
    cpu_active_ranks: A[Optional[str], "CPU 专家实验参数：cpu_active_ranks"] = None
    cpu_offload_rank_to_numa_map: A[
        Optional[str], "CPU 专家实验参数：cpu_offload_rank_to_numa_map"
    ] = None
    cpu_offload_top_m: A[int, "CPU 专家实验参数：cpu_offload_top_m"] = 1
    cpu_offload_slot_update_interval_ms: A[
        int, "CPU 专家实验参数：cpu_offload_slot_update_interval_ms"
    ] = 100
    cpu_offload_metrics_interval_ms: A[
        int, "CPU 专家实验参数：cpu_offload_metrics_interval_ms"
    ] = 1000
    cpu_offload_enable_mode_a: A[
        bool,
        Arg(
            help="CPU 专家实验参数：cpu_offload_enable_mode_a",
            action=argparse.BooleanOptionalAction,
        ),
    ] = False
    cpu_offload_ratio: A[float, "CPU 专家实验参数：cpu_offload_ratio"] = 0.2
    cpu_offload_overflow_policy: A[
        Literal["drop", "error", "block"],
        "CPU 专家实验参数：cpu_offload_overflow_policy",
    ] = "error"
    cpu_offload_max_tokens_per_step: A[
        int, "CPU 专家实验参数：cpu_offload_max_tokens_per_step"
    ] = 65536
    cpu_offload_max_tokens_per_expert: A[
        int, "CPU 专家实验参数：cpu_offload_max_tokens_per_expert"
    ] = 4096
    cpu_offload_num_workers_per_numa: A[
        int, "CPU 专家实验参数：cpu_offload_num_workers_per_numa"
    ] = -1
    cpu_offload_cores_per_numa: A[
        int, "CPU 专家实验参数：cpu_offload_cores_per_numa"
    ] = -1
    cpu_offload_reserved_cores_per_numa: A[
        int, "CPU 专家实验参数：cpu_offload_reserved_cores_per_numa"
    ] = 8
    cpu_offload_min_cores_per_worker: A[
        int, "CPU 专家实验参数：cpu_offload_min_cores_per_worker"
    ] = 1
    cpu_offload_shm_prefix: A[str, "CPU 专家实验参数：cpu_offload_shm_prefix"] = (
        "/dev/shm/sglang_cpu_offload"
    )
    cpu_offload_daemon_proc_name: A[
        str, "CPU 专家实验参数：cpu_offload_daemon_proc_name"
    ] = "cpu_expertd"
    cpu_offload_standalone_daemon: A[
        bool,
        Arg(
            help="CPU 专家实验参数：cpu_offload_standalone_daemon",
            action=argparse.BooleanOptionalAction,
        ),
    ] = False
    cpu_offload_attach_timeout_ms: A[
        int, "CPU 专家实验参数：cpu_offload_attach_timeout_ms"
    ] = 30000
    cpu_offload_amx_enabled: A[
        bool,
        Arg(
            help="CPU 专家实验参数：cpu_offload_amx_enabled",
            action=argparse.BooleanOptionalAction,
        ),
    ] = True
    cpu_offload_enable_cuda_graph_polling: A[
        bool,
        Arg(
            help="CPU 专家实验参数：cpu_offload_enable_cuda_graph_polling",
            action=argparse.BooleanOptionalAction,
        ),
    ] = True
    cpu_offload_timeout_us: A[int, "CPU 专家实验参数：cpu_offload_timeout_us"] = 5000000
    cpu_offload_max_spin: A[int, "CPU 专家实验参数：cpu_offload_max_spin"] = 100000
    cpu_offload_max_hits: A[int, "CPU 专家实验参数：cpu_offload_max_hits"] = 0
    cpu_offload_demote_latency_us: A[
        int, "CPU 专家实验参数：cpu_offload_demote_latency_us"
    ] = 2000
    cpu_offload_demote_window: A[int, "CPU 专家实验参数：cpu_offload_demote_window"] = (
        16
    )
