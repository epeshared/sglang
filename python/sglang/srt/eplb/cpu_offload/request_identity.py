"""沿用 scheduler 的请求编号，不把请求池位置当作请求身份。"""


def from_requests(requests):
    return {
        "request_ids": [
            str(getattr(request, "_step_trace_request_id", request.rid))
            for request in requests
        ],
        "request_decode_steps": [
            int(getattr(request, "_step_trace_decode_step", len(request.output_ids)))
            for request in requests
        ],
        "request_id_source": (
            "synthetic"
            if requests
            and all(
                getattr(request, "_step_trace_identity", "") == "synthetic"
                for request in requests
            )
            else "scheduler"
        ),
    }


def attach(forward_batch, requests):
    """两种批次初始化分支共用同一份身份填充过程。"""
    for name, value in from_requests(requests).items():
        setattr(forward_batch, name, value)
