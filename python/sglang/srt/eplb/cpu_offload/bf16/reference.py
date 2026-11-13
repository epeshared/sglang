"""独立的逐专家 GPU 参考计算，不调用待测 CPU 内核。"""

import torch
import torch.nn.functional as F


def forward(hidden, gate_up, down, topk_ids, topk_weights, limit):
    result = torch.zeros_like(hidden, dtype=torch.float32)
    clipped = 0
    for expert in range(gate_up.shape[0]):
        rows, slots = torch.where(topk_ids == expert)
        if rows.numel() == 0:
            continue
        gate, up = F.linear(hidden[rows], gate_up[expert]).float().chunk(2, -1)
        clipped += int(((gate > limit) | (up.abs() > limit)).sum().item())
        active = (F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)).to(
            hidden.dtype
        )
        values = F.linear(active, down[expert]).float()
        result.index_add_(0, rows, values * topk_weights[rows, slots, None])
    return result.to(hidden.dtype), clipped


def error_metrics(actual, expected):
    # 只在统计时使用双精度，避免大张量的余弦值因归约舍入略大于 1。
    a, b = actual.detach().double().cpu(), expected.detach().double().cpu()
    return {
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-12)),
        "cosine": float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
        "max_abs": float((a - b).abs().max()),
    }
