"""将逐步追踪接到新版解码图后端，保留暖机与正式捕获的边界。"""

from sglang.srt.eplb.cpu_offload import step_trace


def capture(
    backend, runner, key, run_once, forward_batch, num_tokens, post_warmup_hook
):
    state = step_trace.prepare_graph_capture(
        id(runner), forward_batch.batch_size, num_tokens, runner.device
    )

    def after_warmup(*args, **kwargs):
        if post_warmup_hook is not None:
            post_warmup_hook(*args, **kwargs)
        step_trace.enter_graph_capture(state)

    try:
        return backend.capture_one(
            key,
            run_once,
            capture_inputs=None,
            post_warmup_hook=after_warmup,
        )
    finally:
        step_trace.leave_graph_capture(state)


def before_replay(runner):
    step_trace.mark_graph_replay(id(runner), runner.bs, runner.raw_num_token)
