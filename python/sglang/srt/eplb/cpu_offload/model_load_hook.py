"""Post-weight-load hook: register every MoE layer with the CPU offload engine.

Without this hook the dual-loop daemon backend is a no-op: the
``cpu_expertd`` slots stay attached but no weight-SHM file is ever
populated, so ``begin_cpu_offload`` falls through (``has_layer == False``)
and every token runs on the GPU.

Called once from ``ModelRunner.load_model()`` right after the model
finishes loading and ``process_weights_after_loading`` has run, on every
GPU rank that is in ``cfg.cpu_active_ranks``. Each call:

  1. Walks ``model.modules()`` and picks every MoE module that carries
     a numeric ``layer_id`` plus an ``experts`` submodule whose
     ``w13_weight`` / ``w2_weight`` parameters are materialized.
  2. Builds a ``LayerSpec`` (per-rank EP shard counts) and a
     ``LayerWeights`` bundle (CPU + contiguous copies of the expert
     tensors plus their FP8 scales) and hands them to
     ``runtime.engine.register_layer``.

The first ``register_layer`` triggers ``DaemonBackend.init_runtime``
(attach + open one slot per local NUMA). Weights are bulk-memcpy'd into
the per-(socket, slot, layer) SHM file the daemon mmaps read-only.

Failure of one layer is **fatal** — silently skipping would resurrect
exactly the bug this hook was written to fix (CPU offload appears live
but does nothing).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _gather_layer_weights(experts) -> Optional[dict]:
    """Extract the four required FP8 tensors + optional bias from ``experts``.

    Returns ``None`` if the module does not have the expected fields, in
    which case the layer is silently skipped (treated as "not a MoE
    block we know how to offload"). Anything else raises.
    """
    w13 = getattr(experts, "w13_weight", None)
    w2 = getattr(experts, "w2_weight", None)
    if w13 is None or w2 is None:
        return None

    # FP8 block-quant uses ``*_scale_inv``; channel-quant uses ``*_scale``.
    w13_scale = getattr(experts, "w13_weight_scale_inv", None)
    if w13_scale is None:
        w13_scale = getattr(experts, "w13_weight_scale", None)
    w2_scale = getattr(experts, "w2_weight_scale_inv", None)
    if w2_scale is None:
        w2_scale = getattr(experts, "w2_weight_scale", None)
    if w13_scale is None or w2_scale is None:
        raise RuntimeError(
            "cpu_offload: MoE module has w13/w2 but no FP8 scales — "
            "dual-loop daemon backend only supports FP8 quantized MoE"
        )

    bias = getattr(experts, "e_score_correction_bias", None)
    if bias is None:
        bias = getattr(experts, "correction_bias", None)
    return {
        "w13": w13,
        "w2": w2,
        "w13_scale": w13_scale,
        "w2_scale": w2_scale,
        "bias": bias,
    }


def _to_cpu_contig(t) -> torch.Tensor:
    """Detach, move to CPU, force contiguous. Cheap if already CPU."""
    if isinstance(t, torch.nn.Parameter):
        t = t.data
    if t.device.type != "cpu":
        t = t.detach().to("cpu", non_blocking=False)
    else:
        t = t.detach()
    return t.contiguous()


def _assert_expert_block_sources_agree(module, local_rank: int) -> None:
    """Fail loudly if the two "which experts are mine" answers disagree.

    Two different expressions are in play and only one of them decides what
    the GPU computes:

      * AUTHORITATIVE -- ``moe_ep_rank * num_local_experts``, used by
        ``FusedMoE._map_global_expert_id_to_local_expert_id`` (which expert's
        weights land in which local slot) and by
        ``StandardDispatcher.dispatch`` (which builds ``local_expert_mapping``
        and sends everything outside the block to -1).
      * DERIVED -- ``rank * num_local_physical_experts`` mapped through the
        EPLB ``physical_to_logical_map``, which is what the fast loop's
        candidate filter (``runtime._build_local_expert_set_fn``) and
        ``cuda_graph_offload``'s ``local_expert_lo/hi`` actually read.

    They coincide only while the map is identity, the two expert counts are
    equal, and ``moe_ep_rank == tp_rank``. EPLB rebalancing, expert
    replication, or TP != EP breaks any of those. If they diverge, the fast
    loop restricts cpu_slot to the WRONG set of experts and every offloaded
    token is silently counted twice after the TP all-reduce -- there is no
    symptom other than wrong numbers, which is why this is checked here at
    startup instead of being left to be discovered later.
    """
    try:
        ep_rank = int(getattr(module, "moe_ep_rank"))
        n_local = int(getattr(module, "num_local_experts"))
    except Exception:
        return  # not a module that carries the block parameters
    if n_local <= 0:
        return
    lo, hi = ep_rank * n_local, (ep_rank + 1) * n_local
    authoritative = set(range(lo, hi))

    try:
        from sglang.srt.eplb.expert_location import (
            get_global_expert_location_metadata,
        )

        meta = get_global_expert_location_metadata()
        if meta is None:
            return  # nothing to compare against yet
        n_phys = int(meta.num_local_physical_experts)
        phys = meta.physical_to_logical_map
        row = phys[0, local_rank * n_phys : (local_rank + 1) * n_phys]
        derived = {int(x) for x in row.tolist() if int(x) >= 0}
    except Exception:
        return  # metadata unavailable; nothing to assert

    if derived == authoritative:
        return
    raise RuntimeError(
        "cpu_offload: GPU-expert ownership sources DISAGREE on rank "
        f"{local_rank}.\n"
        f"  authoritative (weight loader + StandardDispatcher): "
        f"moe_ep_rank={ep_rank} num_local_experts={n_local} -> [{lo}, {hi})\n"
        f"  derived (read by fast_loop's candidate filter and "
        f"cuda_graph_offload's local_expert_lo/hi):\n"
        f"    num_local_physical_experts={n_phys}, layer-0 slice -> "
        f"{sorted(derived)[:8]}{'...' if len(derived) > 8 else ''}\n"
        "  The fast loop would restrict cpu_slot to the wrong expert set, so "
        "every offloaded token\n"
        "  would be computed on BOTH this rank's CPU and the owning rank's "
        "GPU and counted twice\n"
        "  after the TP all-reduce, with no symptom other than wrong output. "
        "This happens when the\n"
        "  physical->logical map is not identity (EPLB rebalancing), the two "
        "expert counts differ\n"
        "  (expert replication), or moe_ep_rank != tp_rank (TP != EP). Fix "
        "the placement or teach\n"
        "  runtime._build_local_expert_set_fn to read the authoritative "
        "block before re-enabling."
    )


def register_model_moe_layers(model, runtime) -> int:
    """Iterate the model and upload every MoE layer's weights to the daemon.

    Returns the number of layers successfully registered. Raises on the
    first failure once a candidate layer has been identified.
    """
    if runtime is None:
        return 0

    cfg = runtime.config
    local_rank = runtime.local_rank
    if not cfg.rank_is_active(local_rank):
        logger.info(
            "[cpu_offload] rank %d not in cpu_active_ranks=%s; skipping "
            "model-load weight registration",
            local_rank,
            list(cfg.cpu_active_ranks or ()),
        )
        return 0

    from sglang.srt.eplb.cpu_offload.engine import (
        LayerSpec,
        LayerWeights,
        daemon_loads_weights_from_disk,
    )

    daemon_loads = daemon_loads_weights_from_disk()
    if daemon_loads:
        logger.info(
            "[cpu_offload] CPU_OFFLOAD_DAEMON_LOADS_WEIGHTS=1: skipping "
            "per-rank weight upload; daemon must load full E from disk"
        )

    engine = runtime.engine
    # Ensure the per-NUMA daemon backend is attached + a slot is opened
    # for this rank BEFORE the first register_layer call. ``runtime.py``
    # intentionally defers init_runtime() to first-use; the model-load
    # hook is exactly that first use, so we trigger it here. Idempotent.
    engine.init_runtime()

    registered = 0
    skipped_no_layer_id = 0
    for module in model.modules():
        experts = getattr(module, "experts", None)
        layer_id = getattr(module, "layer_id", None)
        if experts is None:
            continue
        if not isinstance(layer_id, int):
            # MoE block without explicit layer_id — best-effort skip.
            skipped_no_layer_id += 1
            continue

        bundle = _gather_layer_weights(experts)
        if bundle is None:
            continue

        # Per-rank EP shard counts. For TP=EP=N, ``num_local_experts`` is
        # E/N (e.g. 16 for Qwen3-235B E=128 on EP8). FusedMoE without EP
        # exposes only ``num_experts``.
        num_experts = int(
            getattr(experts, "num_local_experts", None)
            or getattr(experts, "num_experts", 0)
        )
        if num_experts <= 0:
            raise RuntimeError(
                f"cpu_offload: layer {layer_id} experts has no usable expert count"
            )
        # In Phase-2 (daemon-loads-from-disk) the CPU slot is a dynamic
        # damper that may host ANY of the E global experts, not just this
        # rank's E/EP shard. We need the FULL expert count for the daemon
        # layer spec.
        if daemon_loads:
            global_experts = int(
                getattr(experts, "num_experts", 0)
                or getattr(experts, "global_num_experts", 0)
                or num_experts
            )
            if global_experts < num_experts:
                global_experts = num_experts
            spec_num_experts = global_experts
        else:
            spec_num_experts = num_experts
        top_k = int(getattr(experts, "top_k", 0))
        if top_k <= 0:
            # Some MoE wrappers stash it under moe_runner_config.
            mrc = getattr(experts, "moe_runner_config", None)
            top_k = int(getattr(mrc, "top_k", 0)) if mrc is not None else 0
        hidden_size = int(getattr(experts, "hidden_size", 0))
        if hidden_size <= 0:
            hidden_size = int(bundle["w13"].shape[-1])
        intermediate = int(
            getattr(experts, "intermediate_size_per_partition", 0)
            or getattr(experts, "intermediate_size", 0)
        )
        if intermediate <= 0:
            # w13 is [E, 2*I, H], so I = shape[1] // 2.
            intermediate = int(bundle["w13"].shape[1]) // 2

        spec = LayerSpec(
            layer_id=layer_id,
            num_experts=spec_num_experts,
            num_experts_per_tok=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate,
            norm_topk_prob=True,
            n_group=1,
            topk_group=1,
        )

        if daemon_loads:
            # Metadata-only registration: don't move GPU tensors to CPU,
            # don't materialize anything. Daemon owns the weight bytes.
            weights = LayerWeights()
        else:
            weights = LayerWeights(
                gate_up_weights=_to_cpu_contig(bundle["w13"]),
                down_weights=_to_cpu_contig(bundle["w2"]),
                gate_up_scales=_to_cpu_contig(bundle["w13_scale"]),
                down_scales=_to_cpu_contig(bundle["w2_scale"]),
                e_score_correction_bias=(
                    _to_cpu_contig(bundle["bias"])
                    if bundle["bias"] is not None
                    else None
                ),
            )

        if registered == 0:
            # Once per rank, on the first MoE layer: the check is about the
            # placement rule, which is identical for every layer.
            _assert_expert_block_sources_agree(experts, local_rank)

        engine.register_layer(spec, weights)
        registered += 1
        if registered == 1 or registered % 16 == 0:
            logger.info(
                "[cpu_offload] uploaded layer %d to daemon "
                "(E_local=%d K=%d H=%d I=%d, total registered so far=%d)",
                layer_id,
                num_experts,
                top_k,
                hidden_size,
                intermediate,
                registered,
            )

    logger.info(
        "[cpu_offload] model-load weight upload complete: rank=%d "
        "layers_registered=%d skipped_no_layer_id=%d",
        local_rank,
        registered,
        skipped_no_layer_id,
    )
    if registered == 0:
        raise RuntimeError(
            "cpu_offload: feature flag is on but ZERO MoE layers were "
            "registered with the daemon. The dual-loop CPU offload "
            "would silently do nothing. Likely causes: (a) the model "
            "isn't MoE, (b) MoE submodules don't expose w13_weight, "
            "(c) layer_id isn't set on the SparseMoeBlock. Fix the "
            "model adapter or disable --enable-cpu-offload-fast-loop."
        )
    return registered
