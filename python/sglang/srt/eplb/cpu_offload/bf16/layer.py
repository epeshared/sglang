"""原始 BF16 CPU 计算入口；供数值验收和后续守护进程接线复用。"""

import torch

from .weights import ExpertWeights


class Bf16Experts:
    def __init__(self, weights: ExpertWeights):
        import moe_cpu_engine

        if not hasattr(moe_cpu_engine, "fused_experts_bf16_cpu"):
            raise RuntimeError(
                "CPU 后端缺少原始 BF16 入口，请先应用配套 native 补丁并重新编译"
            )
        self.engine = moe_cpu_engine
        self.weights = weights
        self.gate_up = moe_cpu_engine.convert_weight_packed(weights.gate_up)
        self.down = moe_cpu_engine.convert_weight_packed(weights.down)

    def forward(self, hidden, topk_ids, topk_weights):
        """输入局部专家编号；路由权重须已包含模型的缩放系数。"""
        if hidden.dtype != torch.bfloat16 or hidden.device.type != "cpu":
            raise ValueError("CPU 专家输入必须是 CPU BF16 张量")
        return self.engine.fused_experts_bf16_cpu(
            hidden,
            self.gate_up,
            self.down,
            topk_weights,
            topk_ids,
            self.weights.swiglu_limit,
            True,
        )
