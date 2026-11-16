"""BF16 attention-sink normalization using the existing sparse MLA LSE."""

import math

import torch


def validate_sparse_mla_sink_backend(sinks, backend, dcp_size, shards_index_k):
    if sinks is None:
        return
    if backend not in ("flashmla_sparse", "triton"):
        raise RuntimeError(
            f"Learnable attention sinks require flashmla_sparse or triton, got {backend}"
        )
    if backend == "triton" and (dcp_size > 1 or shards_index_k):
        raise NotImplementedError("Triton attention sinks do not support DCP")


def validate_sparse_mla_sinks(q_nope, q_rope, kv, sinks):
    if sinks is None:
        return
    if any(t.dtype != torch.bfloat16 for t in (q_nope, q_rope, kv)):
        raise ValueError("Triton sparse MLA sinks require BF16 query and KV tensors")
    if (
        sinks.shape != (q_nope.shape[1],)
        or sinks.dtype != torch.float32
        or sinks.device != q_nope.device
        or not sinks.is_contiguous()
    ):
        raise ValueError("Triton sparse MLA sinks must be contiguous FP32 [heads]")


def apply_sparse_mla_sinks(output, lse2, sinks, indices, lengths=None):
    """Add zero-value sink mass once, after all split-K partials are merged.

    The existing kernel returns BF16 output and FP32 base-2 LSE. Keep the
    correction in FP32, then restore the output dtype. Explicitly identify empty
    rows: their kernel LSE can be a finite sentinel and their output can be NaN.
    Sink logits are in natural-log units and are not multiplied by sm_scale.
    """
    valid = indices >= 0
    if lengths is not None:
        valid = valid & (
            torch.arange(indices.shape[-1], device=indices.device) < lengths[:, None]
        )
    has_keys = valid.any(dim=-1, keepdim=True)
    base_lse = torch.where(has_keys, lse2.float() * math.log(2.0), -torch.inf)
    sink_lse = torch.logaddexp(base_lse, sinks[None, :])
    factor = torch.where(has_keys, torch.exp(base_lse - sink_lse), 0.0)
    safe_output = torch.where(has_keys[..., None], output.float(), 0.0)
    return (safe_output * factor[..., None]).to(output.dtype), sink_lse / math.log(2.0)
