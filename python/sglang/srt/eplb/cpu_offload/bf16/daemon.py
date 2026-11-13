"""Small-layer validation adapter using the production RemoteMOE CUDA transport without graph capture."""

import torch


class DaemonBf16Experts:
    def __init__(self, weights, layer_id, topk, handle, prefix):
        import moe_cpu_engine
        from cpu_offload_remote_moe import RemoteMOE

        self.engine = moe_cpu_engine
        self.handle = handle
        self.hidden = weights.gate_up.shape[-1]
        self.topk = topk
        self.experts = weights.gate_up.shape[0]
        self.remote = RemoteMOE(
            handle,
            layer_id,
            self.experts,
            topk,
            self.hidden,
            weights.down.shape[-1],
            norm_topk_prob=False,
            shm_prefix=prefix,
        )
        self.remote.set_bf16_weights(
            weights.gate_up, weights.down, weights.swiglu_limit
        )
        self.remote.enable_kernel_path()

    def forward(self, hidden, ids, routing):
        if (
            hidden.dtype != torch.bfloat16
            or hidden.ndim != 2
            or hidden.shape[1] != self.hidden
            or not 0 < len(hidden) <= 256
            or ids.dtype != torch.int32
            or routing.dtype != torch.float32
            or tuple(ids.shape) != (len(hidden), self.topk)
            or ids.shape != routing.shape
        ):
            raise ValueError("BF16 通信验证输入形状、精度或批量非法")
        if bool(((ids < -1) | (ids >= self.experts)).any()):
            raise ValueError("BF16 通信验证专家编号越界")
        h, i, w = [x.cuda().contiguous() for x in (hidden, ids, routing)]
        output = torch.empty_like(h)
        stream = torch.cuda.current_stream()
        before = self.engine.cpu_client_slot_info(self.handle)
        self.remote.forward_experts_kernel(
            stream.cuda_stream,
            h.data_ptr(),
            i.data_ptr(),
            w.data_ptr(),
            output.data_ptr(),
            len(h),
        )
        stream.synchronize()
        after = self.engine.cpu_client_slot_info(self.handle)
        if (
            after["error_code"] != 0
            or after["req_seq"] != after["done_seq"]
            or after["req_seq"] <= before["req_seq"]
            or after["stat_fwd_count"] != before["stat_fwd_count"] + 1
        ):
            raise RuntimeError(f"CPU 通信未完成一次有效计算：{after}")
        return output.cpu()
