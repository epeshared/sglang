"""将逐步追踪接到新版解码图后端，保留暖机与正式捕获的边界。"""

import torch

from sglang.srt.eplb.cpu_offload import step_trace


def capture(
    backend, runner, key, run_once, forward_batch, num_tokens, post_warmup_hook
):
    state = step_trace.prepare_graph_capture(
        id(runner), forward_batch.batch_size, num_tokens, runner.device
    )

    def traced_forward():
        # 后端每轮暖机后都会调用回调，不能把该回调当作正式捕获边界。
        if torch.cuda.is_current_stream_capturing():
            step_trace.enter_graph_capture(state)
        return run_once()

    try:
        return backend.capture_one(
            key,
            traced_forward,
            capture_inputs=None,
            post_warmup_hook=post_warmup_hook,
        )
    finally:
        step_trace.leave_graph_capture(state)


def before_replay(runner):
    step_trace.mark_graph_replay(id(runner), runner.bs, runner.raw_num_token)
