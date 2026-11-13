"""Independent per-expert GPU reference without calls to the CPU kernel under test."""

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
    # Use double precision for metrics to avoid cosine values above 1 due to reduction rounding.
    a, b = actual.detach().double().cpu(), expected.detach().double().cpu()
    return {
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-12)),
        "cosine": float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
        "max_abs": float((a - b).abs().max()),
    }
