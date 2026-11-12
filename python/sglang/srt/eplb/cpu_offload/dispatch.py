"""Dispatch-time helpers for CPU offload.

These helpers are pure and CUDA-graph friendly: they only emit tensor
ops on the device that already holds the inputs. They do NOT call into
the CPU engine and do NOT allocate Python objects on the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def expert_is_local_on_gpu(
    rank_id: int,
    expert_id: int,
    *,
    physical_to_logical_map: torch.Tensor,
    layer_id: int,
    num_local_experts: int,
    moe_ep_rank: Optional[int] = None,
) -> bool:
    """Return True if ``expert_id`` (logical) has a physical slot owned by ``rank_id``.

    ``physical_to_logical_map`` has shape ``(num_layers, num_physical_experts)``
    where each physical index 0..num_physical_experts is allocated round-robin
    across EP ranks: rank R owns physical indices
    ``[R*num_local_experts, (R+1)*num_local_experts)``.

    This function is meant to be cheap and called from CPU (not inside the
    CUDA graph). Callers cache its result.
    """
    if moe_ep_rank is None:
        moe_ep_rank = rank_id
    start = moe_ep_rank * num_local_experts
    end = start + num_local_experts
    layer_slice = physical_to_logical_map[layer_id, start:end]
    # Tensor membership; .item() forces a single CPU value.
    return bool((layer_slice == expert_id).any().item())


@dataclass
class CpuSplitDecision:
    """Result of splitting topk_ids for a Mode B CPU handoff.

    ``token_indices`` is a 1-D int tensor (on the same device as topk_ids)
    naming which input tokens have at least one routed expert equal to
    ``cpu_slot_expert_id``. The dispatcher must:

    - copy ``hidden_states[token_indices]`` into the CPU engine input buffer,
    - replace ``topk_ids[token_indices, branch] == cpu_slot`` with -1 BEFORE
      the GPU MoE kernel runs (so GPU does not also compute it),
    - keep ``topk_weights`` untouched (CPU output is already weighted in
      ``forward_experts``; combine adds the result back),
    - launch the CPU forward concurrently with GPU local experts,
    - combine after the GPU polling kernel observes completion.
    """

    cpu_slot_expert_id: int
    n_cpu_tokens: int
    token_indices: Optional[torch.Tensor]
    masked_topk_ids: torch.Tensor
    """``topk_ids`` with CPU-bound branches replaced by -1."""


def split_topk_for_cpu(
    topk_ids: torch.Tensor,
    cpu_slot_expert_id: int,
    *,
    max_tokens: int = -1,
) -> CpuSplitDecision:
    """Split ``topk_ids`` into a GPU branch and a CPU branch (Mode B).

    Parameters
    ----------
    topk_ids:
        ``(num_tokens, topk)`` int tensor with the original (unmasked) expert ids.
    cpu_slot_expert_id:
        The expert id this rank's CPU slot currently holds. ``-1`` disables.
    max_tokens:
        Hard cap on tokens forwarded to CPU. ``-1`` means no cap.

    Returns
    -------
    A ``CpuSplitDecision``. If the slot is -1 or no tokens match,
    ``n_cpu_tokens == 0``, ``token_indices is None``, and
    ``masked_topk_ids is topk_ids`` (no clone).

    Notes
    -----
    This function is pure and emits only standard tensor ops. It is safe
    to call inside a CUDA-graph captured region as long as the caller
    captures the shape with a fixed upper bound; for variable shapes
    (eager mode) it is also fine.
    """
    if cpu_slot_expert_id < 0:
        return CpuSplitDecision(
            cpu_slot_expert_id=-1,
            n_cpu_tokens=0,
            token_indices=None,
            masked_topk_ids=topk_ids,
        )

    if topk_ids.numel() == 0:
        return CpuSplitDecision(
            cpu_slot_expert_id=cpu_slot_expert_id,
            n_cpu_tokens=0,
            token_indices=None,
            masked_topk_ids=topk_ids,
        )

    # branch-level mask: (num_tokens, topk)
    branch_mask = topk_ids == cpu_slot_expert_id

    # NOTE: a previous version checked `branch_mask.any().item()` here as
    # an early-out. That forces a D2H + CUDA sync on EVERY MoE layer
    # call (94 layers x every decode step x N ranks) and dominates decode
    # latency. The downstream `int(token_indices.numel())` already
    # produces a data-dependent sync point we cannot avoid (token_indices
    # comes from torch.nonzero, whose shape depends on the mask). Doing
    # only that one sync saves ~half the GPU pipeline stalls per layer.

    # token-level mask: any branch routes to the CPU slot expert.
    token_mask = branch_mask.any(dim=-1)
    token_indices = torch.nonzero(token_mask, as_tuple=False).squeeze(-1)

    if max_tokens >= 0 and token_indices.numel() > max_tokens:
        token_indices = token_indices[:max_tokens]
        # Recompute branch_mask restricted to the kept tokens so GPU
        # masking is consistent with what CPU will actually receive.
        kept_token_mask = torch.zeros_like(token_mask)
        kept_token_mask[token_indices] = True
        branch_mask = branch_mask & kept_token_mask.unsqueeze(-1)

    masked = topk_ids.masked_fill(branch_mask, -1)
    return CpuSplitDecision(
        cpu_slot_expert_id=cpu_slot_expert_id,
        n_cpu_tokens=int(token_indices.numel()),
        token_indices=token_indices,
        masked_topk_ids=masked,
    )


def split_topk_for_cpu_graph_safe(
    topk_ids: torch.Tensor,
    cpu_slot_expert_id: int,
    max_tokens: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDA-Graph-safe replacement for split_topk_for_cpu.

    Instead of ``torch.nonzero`` (variable shape → not capturable), uses a
    sort-based compact trick that always produces **fixed-shape** tensors
    of size ``max_tokens``.  Padding positions beyond the actual match
    count are filled with ``num_tokens - 1`` (a valid index) and have
    their CPU weight set to 0.0 by the caller, so the daemon's output
    for those rows contributes nothing to the final ``index_add``.

    Returns
    -------
    (token_indices, masked_topk_ids, is_real)
        token_indices : [cap] long tensor — first *cap* entries are sorted
            real indices, rest are padding (clamped to num_tokens-1).
        masked_topk_ids : same shape as *topk_ids* — CPU slot branches
            replaced by -1.
        is_real : [cap] bool tensor — True for real (non-padding) rows,
            False for padding.  Use this to zero out weights for padding.
    """
    num_tokens = topk_ids.shape[0]
    cap = min(max_tokens, num_tokens)

    # branch mask (num_tokens, topk) → token mask (num_tokens,)
    branch_mask = topk_ids == cpu_slot_expert_id
    token_mask = branch_mask.any(dim=-1)

    # Sort-based compact: matching indices get their real id,
    # non-matching get sentinel = num_tokens (will be clamped later).
    indices = torch.arange(num_tokens, device=topk_ids.device, dtype=torch.long)
    sentinel = num_tokens  # one past the last valid index
    tagged = torch.where(
        token_mask,
        indices,
        torch.tensor(sentinel, device=topk_ids.device, dtype=torch.long),
    )
    sorted_idx, _ = torch.sort(tagged)  # sentinels go to the end

    # Take first *cap* entries.
    sorted_idx_raw = sorted_idx[:cap]
    # is_real: True for actual matching tokens, False for sentinels.
    is_real = sorted_idx_raw < num_tokens
    # Clamp sentinels to last valid index (for safe index_select).
    token_indices = sorted_idx_raw.clamp(max=num_tokens - 1)

    # Rebuild the full branch mask for masking topk_ids.
    masked = topk_ids.masked_fill(branch_mask, -1)

    return token_indices, masked, is_real

    # Take first *cap* entries; clamp sentinels to last valid index.
    token_indices = sorted_idx[:cap].clamp(max=num_tokens - 1)

    # Rebuild the full branch mask for masking topk_ids.
    masked = topk_ids.masked_fill(branch_mask, -1)

    # Host-side count: during CUDA Graph capture the tensor shape is
    # fixed, so numel() is a pure Python int (no D2H sync).
    n_cpu_tokens = cap  # will be clamped to actual count by caller

    return token_indices, masked, n_cpu_tokens


def check_overflow(
    decision: CpuSplitDecision,
    *,
    max_per_step: int,
    max_per_expert: int,
    policy: str,
) -> CpuSplitDecision:
    """Enforce per-step / per-expert caps for Mode B handoff.

    On overflow:
    - ``policy == "drop"``: truncate ``token_indices`` and re-mask.
    - ``policy == "error"``: raise ``OverflowError`` (the default).
    - ``policy == "block"``: caller is expected to apply backpressure;
      we surface a flag via the returned decision's ``n_cpu_tokens=-1``.
    """
    n = decision.n_cpu_tokens
    if n <= 0:
        return decision

    cap = n
    if max_per_step > 0:
        cap = min(cap, max_per_step)
    if max_per_expert > 0:
        cap = min(cap, max_per_expert)

    if cap >= n:
        return decision

    if policy == "error":
        raise OverflowError(
            f"CPU offload overflow: {n} tokens for expert "
            f"{decision.cpu_slot_expert_id} exceed caps "
            f"(per_step={max_per_step}, per_expert={max_per_expert})"
        )
    if policy == "block":
        return CpuSplitDecision(
            cpu_slot_expert_id=decision.cpu_slot_expert_id,
            n_cpu_tokens=-1,
            token_indices=None,
            masked_topk_ids=decision.masked_topk_ids,
        )

    # drop
    assert decision.token_indices is not None
    kept = decision.token_indices[:cap]
    return CpuSplitDecision(
        cpu_slot_expert_id=decision.cpu_slot_expert_id,
        n_cpu_tokens=int(kept.numel()),
        token_indices=kept,
        masked_topk_ids=decision.masked_topk_ids,
    )
