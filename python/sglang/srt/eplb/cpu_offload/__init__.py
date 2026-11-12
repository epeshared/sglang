"""Dual-loop EPLB-guided CPU offload for SGLang MoE.

See /workspace/dual-loop-eplb-design.md for the full design.

This package is intentionally standalone: it owns CPU offload configuration,
per-rank shadow state, the per-NUMA CPU engine wrapper, a fast-loop EPLB
selector restricted to CPU slots, and dispatch helpers. Integration into
FusedMoE / StandardDispatcher is performed by callers; this package does
not patch them.
"""

from sglang.srt.eplb.cpu_offload.config import (
    CpuOffloadConfig,
    OverflowPolicy,
    load_cpu_offload_config,
)
from sglang.srt.eplb.cpu_offload.dispatch import (
    CpuSplitDecision,
    check_overflow,
    expert_is_local_on_gpu,
    split_topk_for_cpu,
)
from sglang.srt.eplb.cpu_offload.engine import (
    CpuOffloadEngine,
    LayerSpec,
    LayerWeights,
)
from sglang.srt.eplb.cpu_offload.fast_loop import FastLoopScheduler
from sglang.srt.eplb.cpu_offload.metrics import CpuOffloadMetrics
from sglang.srt.eplb.cpu_offload.runtime import (
    CpuOffloadRuntime,
    get_cpu_offload_runtime,
    init_cpu_offload_runtime,
    reset_cpu_offload_runtime_for_tests,
)
from sglang.srt.eplb.cpu_offload.shadow_state import (
    CpuShadowRankState,
    CpuShadowRegistry,
)

__all__ = [
    "CpuOffloadConfig",
    "OverflowPolicy",
    "load_cpu_offload_config",
    "CpuShadowRankState",
    "CpuShadowRegistry",
    "FastLoopScheduler",
    "CpuOffloadMetrics",
    "CpuOffloadEngine",
    "LayerSpec",
    "LayerWeights",
    "CpuSplitDecision",
    "check_overflow",
    "expert_is_local_on_gpu",
    "split_topk_for_cpu",
    "CpuOffloadRuntime",
    "get_cpu_offload_runtime",
    "init_cpu_offload_runtime",
    "reset_cpu_offload_runtime_for_tests",
]
