"""把原来的专家分流、CPU 结果合并和追踪接到新版 MoE 层。"""

from __future__ import annotations

from sglang.srt.eplb.cpu_offload import get_cpu_offload_runtime, step_trace, trace

MOE_TRACE = trace.ENABLED
STEP_TRACE = step_trace.ENABLED
_facts_seen = set()


def begin_layer(layer, hidden_states, topk_output, *, site="moe.fused"):
    """先交给 CPU，再从 GPU 路由里去掉这些分支，防止重复计算。"""
    step = step_trace.begin_layer(layer.layer_id) if STEP_TRACE else None
    if MOE_TRACE:
        trace.hit(site + ".forward")
        if id(layer) not in _facts_seen:
            trace.record_moe_layer_facts(layer)
            _facts_seen.add(id(layer))
        trace.hit(site + ".begin")
    runtime = get_cpu_offload_runtime()
    if runtime is None or not runtime.is_active_locally():
        if MOE_TRACE:
            trace.hit(site + ".begin_passthrough")
        return None, topk_output, step

    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if not TopKOutputChecker.format_is_standard(topk_output):
        raise RuntimeError("CPU 专家分流需要显式专家编号，请使用 Triton MoE 路径")
    from sglang.srt.eplb.cpu_offload.dispatch_hook import begin_cpu_offload

    context = begin_cpu_offload(
        layer_id=layer.layer_id,
        hidden_states=hidden_states,
        topk_ids=topk_output.topk_ids,
        topk_weights=topk_output.topk_weights,
    )
    if context is None:
        if MOE_TRACE:
            trace.hit(site + ".begin_passthrough")
        return None, topk_output, step
    if MOE_TRACE:
        trace.hit(site + ".begin_armed")
    return context, topk_output._replace(topk_ids=context["masked_topk_ids"]), step


def dispatch_done(step, dispatch_output):
    if STEP_TRACE:
        ids = getattr(dispatch_output, "topk_ids", None)
        if ids is None:
            ids = getattr(
                getattr(dispatch_output, "topk_output", None), "topk_ids", None
            )
        step_trace.record_branches_received(step, ids)
        step_trace.mark_dispatch_done(step)


def compute_done(step):
    if STEP_TRACE:
        step_trace.mark_local_moe_done(step)


def combine_cpu(context, output):
    """必须在跨卡求和之前加入 CPU 结果。"""
    if context is not None:
        from sglang.srt.eplb.cpu_offload.dispatch_hook import commit_cpu_offload

        if MOE_TRACE:
            trace.hit("moe.fused.commit")
        return commit_cpu_offload(context, output)
    return output


def end_layer(step):
    if STEP_TRACE:
        step_trace.mark_layer_done(step)
