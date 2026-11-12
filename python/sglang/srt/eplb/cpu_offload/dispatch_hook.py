"""Shared dispatch hook used by both Triton FusedMoE and EP-MoE paths.

Encapsulates the Mode-B CPU-offload entry/exit logic so different MoE
backends can opt in by calling::

    ctx = begin_cpu_offload(layer_id, hidden_states, topk_ids, topk_weights)
    if ctx is not None:
        # GPU experts run normally on ctx["masked_topk_ids"]
        ...
        final_hidden_states = commit_cpu_offload(ctx, final_hidden_states)

The hook is defensive — any exception, missing runtime, or
unregistered layer reduces the call to a no-op so model serving is
never destabilized by CPU-offload misconfiguration.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from sglang.srt.eplb.cpu_offload import trace as _trace

logger = logging.getLogger(__name__)

# Resolved once, at import. Every trace call below sits behind this constant,
# so a normal run pays a module-global load and a not-taken branch per site.
_MOE_TRACE = _trace.ENABLED
_STEP_TRACE_ENABLED = os.environ.get("SGLANG_MOE_STEP_TRACE", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Per-layer load counter forces a D2H + sync (.tolist()) on every MoE
# layer call, which serializes the GPU pipeline. Off by default; the fast
# loop will still pick slots once SGLANG_CPU_OFFLOAD_RECORD_LOAD=1 is set
# (typically during a profiling/warmup run that gathers load stats).
_RECORD_LOAD = os.environ.get("SGLANG_CPU_OFFLOAD_RECORD_LOAD", "0") == "1"
# Sample 1 layer-call out of N when recording, to bound D2H overhead.
try:
    _RECORD_LOAD_SAMPLE = max(
        1, int(os.environ.get("SGLANG_CPU_OFFLOAD_RECORD_LOAD_SAMPLE", "32"))
    )
except ValueError:
    _RECORD_LOAD_SAMPLE = 32

# Debug-only: on rank 0, accumulate per-expert dispatched-token counts and
# dump periodically. Forces a per-call D2H copy; use only for diagnosis.
_IMPACT_DIAG = os.environ.get("CPU_OFFLOAD_IMPACT_DIAG", "0") == "1"
_IMPACT_DIAG_MAX = int(os.environ.get("CPU_OFFLOAD_IMPACT_DIAG_MAX", "80"))
_RANK0_DUMP = os.environ.get("SGLANG_CPU_OFFLOAD_RANK0_DEBUG_DUMP", "0") == "1"
try:
    _RANK0_DUMP_EVERY = max(
        1, int(os.environ.get("SGLANG_CPU_OFFLOAD_RANK0_DEBUG_EVERY", "200"))
    )
except ValueError:
    _RANK0_DUMP_EVERY = 200
# Per-(layer_id) -> list[int] of length num_experts (lazy-allocated on first hit).
_rank0_expert_hits: dict = {}
_rank0_call_cnt: dict = {}


def begin_cpu_offload(
    *,
    layer_id: int,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> Optional[dict]:
    """Mode-B opt-in hook.

    Returns a context dict (consumed by :func:`commit_cpu_offload`) when
    the CPU pipeline was launched, or ``None`` for a pure pass-through.
    The caller is responsible for substituting ``ctx["masked_topk_ids"]``
    for its own topk_ids while running GPU experts.
    """
    if _MOE_TRACE:
        _trace.hit("hook.begin")
        _trace.layer_hit(layer_id, "call")

    # CUDA Graph capture/replay path: use SHM kernel-path offload
    # (all ops are GPU kernels → capturable and replayable).
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        if _MOE_TRACE:
            _trace.hit("hook.begin.capture")
        try:
            from sglang.srt.eplb.cpu_offload.cuda_graph_offload import (
                get_cuda_graph_offload_ctx,
            )

            gctx = get_cuda_graph_offload_ctx()
            from sglang.srt.eplb.cpu_offload import get_cpu_offload_runtime

            rt = get_cpu_offload_runtime()
            if rt is None or not rt.is_active_locally():
                return None
            if gctx is None or not gctx.initialized or not gctx.has_layer(layer_id):
                raise RuntimeError(f"CPU 图 offload 未就绪：layer={layer_id}")
            cpu_slot = rt.local_cpu_slot(layer_id)
            if cpu_slot < 0:
                raise RuntimeError(f"CPU 图捕获前没有分配专家槽位：layer={layer_id}")
            stream = torch.cuda.current_stream().cuda_stream
            _local_lo, _local_hi = -1, -1
            try:
                from sglang.srt.eplb.expert_location import (
                    get_global_expert_location_metadata,
                )

                _meta = get_global_expert_location_metadata()
                if _meta is not None:
                    _n_local = int(_meta.num_local_physical_experts)
                    _local_lo = rt.local_rank * _n_local
                    _local_hi = _local_lo + _n_local - 1
            except Exception:
                pass
            ctx = gctx.begin_offload(
                layer_id,
                hidden_states,
                topk_ids,
                topk_weights,
                cpu_slot,
                stream,
                local_expert_lo=_local_lo,
                local_expert_hi=_local_hi,
            )
            if ctx is None:
                raise RuntimeError(f"CPU 图 offload 捕获失败：layer={layer_id}")
            if _MOE_TRACE:
                _trace.hit("hook.begin.capture_armed")
                _trace.layer_hit(layer_id, "armed")
            ctx["use_cuda_graph_offload"] = True
            ctx["gctx"] = gctx
            rt.metrics.record_pairs_seen(int(topk_ids.numel()))
            rt.metrics.record_offload(rt.local_rank, cpu_slot, ctx.get("num_tokens", 0))
            return ctx
        except Exception as e:
            raise RuntimeError(f"CPU 图 offload 捕获失败：layer={layer_id}") from e
        return None

    # If the graph offload context is initialized, skip the eager path.
    # The graph captures offload kernels during capture and replays them
    # automatically.  The eager path is only for --disable-cuda-graph mode.
    #
    # During graph capture warm-ups (run_once before cudaStreamBeginCapture),
    # is_current_stream_capturing() returns False, but the eager path must
    # still be suppressed because the C++ SHM memcpy is not safe in that
    # context (the daemon may not service requests correctly during the
    # rapid-fire warm-up sequence).
    #
    # For the correctness-test scenario (small batch, e.g. 3 prompts) the
    # caller should pass --batch-size large enough so that cuda_graph_max_bs
    # covers the actual batch size, ensuring graph replay is used instead of
    # the eager path.
    try:
        from sglang.srt.eplb.cpu_offload.cuda_graph_offload import (
            get_cuda_graph_offload_ctx,
        )

        if get_cuda_graph_offload_ctx() is not None:
            if _MOE_TRACE:
                _trace.hit("hook.begin.graph_suppressed")
            # CUDA-graph is initialized: the eager offload path is suppressed,
            # but we still want to collect per-expert load statistics for the
            # fast loop so it can pick a good CPU slot at the next graph
            # re-capture.  This branch is hit during:
            #   - CUDA-graph warm-up runs (capturing=False, eager mode)
            #   - Any decode step whose batch size falls outside the captured
            #     range (rare, but possible).
            # D2H is safe here because these are NOT inside cudaStreamCapture.
            if _RECORD_LOAD:
                try:
                    from sglang.srt.eplb.cpu_offload import (
                        get_cpu_offload_runtime as _get_rt,
                    )

                    _rt = _get_rt()
                    if _rt is not None and _rt.is_active_locally():
                        _cnt = getattr(begin_cpu_offload, "_load_sample_cnt", 0) + 1
                        begin_cpu_offload._load_sample_cnt = _cnt
                        if _cnt % _RECORD_LOAD_SAMPLE == 0 and topk_ids.numel() <= 8192:
                            _ids_cpu = topk_ids.detach().to("cpu", non_blocking=True)
                            _counts = torch.bincount(_ids_cpu.flatten().clamp(min=0))
                            for _eid, _c in enumerate(_counts.tolist()):
                                if _c:
                                    _rt.record_routed(layer_id, _eid, _c)
                except Exception:
                    pass
            return None
    except ImportError:
        pass

    try:
        from sglang.srt.eplb.cpu_offload import get_cpu_offload_runtime
    except ImportError:
        return None

    if _MOE_TRACE:
        _trace.hit("hook.begin.eager")

    rt = get_cpu_offload_runtime()
    if rt is None or not rt.is_active_locally():
        if _MOE_TRACE:
            _trace.reason(
                "hook.begin.eager_skip",
                "runtime is None" if rt is None else "rank not in cpu_active_ranks",
            )
        return None

    # Decode-only gate: skip CPU offload during prefill/extend.
    # Use get_is_extend_in_batch() — the authoritative forward-mode flag —
    # instead of a token-count heuristic that fails for short prompts.
    try:
        if rt.config.cpu_offload_decode_only:
            from sglang.srt.layers.dp_attention import get_is_extend_in_batch

            if get_is_extend_in_batch():
                if _MOE_TRACE:
                    _trace.reason(
                        "hook.begin.eager_skip",
                        "decode-only gate: extend/prefill batch",
                    )
                return None
    except Exception as _dexc:
        if _MOE_TRACE:
            _trace.exc("dispatch_hook.begin_cpu_offload/decode-only-gate", _dexc)

    # Total (token, expert) pairs entering MoE for this layer call ON THIS RANK.
    # topk_ids is the EP-dispatched view: non-local expert slots are already -1,
    # so (topk_ids >= 0).sum() counts ONLY the pairs routed to this rank's experts
    # (both GPU and CPU combined). This is the correct denominator for the CPU
    # hit rate (pairs_to_gpu = pairs_total - pairs_to_cpu).
    #
    # NOTE: this requires a D2H sync (.item()). In CUDA-graph mode this path
    # is skipped entirely (begin_cpu_offload returns early for graph capture/replay),
    # so the sync only occurs in eager mode where it is acceptable.
    try:
        rt.metrics.record_pairs_seen(int((topk_ids >= 0).sum().item()))
    except Exception:
        pass

    # Record per-expert load for the fast loop (off by default — the
    # required D2H copy + .tolist() forces a CUDA sync per MoE layer and
    # serializes the pipeline). Opt in with SGLANG_CPU_OFFLOAD_RECORD_LOAD=1.
    if _RECORD_LOAD:
        try:
            cnt = getattr(begin_cpu_offload, "_load_sample_cnt", 0) + 1
            begin_cpu_offload._load_sample_cnt = cnt
            if cnt % _RECORD_LOAD_SAMPLE == 0 and topk_ids.numel() <= 8192:
                ids_cpu = topk_ids.detach().to("cpu", non_blocking=True)
                counts = torch.bincount(ids_cpu.flatten().clamp(min=0))
                for expert_id, c in enumerate(counts.tolist()):
                    if c:
                        rt.record_routed(layer_id, expert_id, c)
        except Exception:
            pass

    cpu_slot = rt.local_cpu_slot(layer_id)
    # Rank-0 per-expert dispatch dump (debug only, forces D2H).
    if _RANK0_DUMP and rt.local_rank == 0:
        try:
            ids_cpu = topk_ids.detach().to("cpu")
            flat = ids_cpu.flatten()
            flat = flat[flat >= 0]
            if flat.numel() > 0:
                counts = torch.bincount(flat).tolist()
                bucket = _rank0_expert_hits.setdefault(layer_id, [])
                if len(bucket) < len(counts):
                    bucket.extend([0] * (len(counts) - len(bucket)))
                for i, c in enumerate(counts):
                    bucket[i] += c
            cnt = _rank0_call_cnt.get(layer_id, 0) + 1
            _rank0_call_cnt[layer_id] = cnt
            if cnt % _RANK0_DUMP_EVERY == 0:
                bucket = _rank0_expert_hits.get(layer_id, [])
                total = sum(bucket)
                slot_hits = bucket[cpu_slot] if 0 <= cpu_slot < len(bucket) else 0
                top = sorted(enumerate(bucket), key=lambda x: -x[1])[:10]
                top_str = ", ".join(f"e{e}:{c}" for e, c in top if c)
                print(
                    f"[rank0_dump] layer={layer_id} calls={cnt} cpu_slot={cpu_slot} "
                    f"slot_hits={slot_hits}/{total} ({100.0 * slot_hits / max(1, total):.3f}%) "
                    f"top10=[{top_str}]",
                    flush=True,
                )
        except Exception as e:
            print(f"[rank0_dump] error: {e}", flush=True)
    # Periodic debug on rank 0: log first call AND any time slot changes,
    # plus every N=200 calls so we can see steady-state.
    try:
        if rt.local_rank == 0:
            prev = getattr(begin_cpu_offload, "_dbg_prev_slot", -2)
            cnt = getattr(begin_cpu_offload, "_dbg_call_cnt", 0) + 1
            begin_cpu_offload._dbg_call_cnt = cnt
            if cpu_slot != prev or cnt % 200 == 1:
                eng = rt.engine
                init_flag = getattr(eng, "initialized", "?")
                try:
                    has_layer = eng.has_layer(layer_id)
                except Exception as e:
                    has_layer = f"err:{e}"
                print(
                    f"[dbg][cpu_offload][rank=0] call#{cnt} layer={layer_id} "
                    f"cpu_slot={cpu_slot} (prev={prev}) engine.initialized={init_flag} "
                    f"has_layer={has_layer}",
                    flush=True,
                )
                begin_cpu_offload._dbg_prev_slot = cpu_slot
    except Exception:
        pass
    if cpu_slot < 0:
        if _MOE_TRACE:
            _trace.reason("hook.begin.eager_skip", "cpu_slot=-1")
        return None
    if not rt.engine.initialized or not rt.engine.has_layer(layer_id):
        if _MOE_TRACE:
            _trace.reason(
                "hook.begin.eager_skip",
                "engine not initialized"
                if not rt.engine.initialized
                else "layer not registered with engine",
            )
        return None

    cfg = rt.config

    # ------------------------------------------------------------------
    # Eager path: variable-shape, D2H sync allowed, original logic.
    # ------------------------------------------------------------------
    from sglang.srt.eplb.cpu_offload.dispatch import (
        check_overflow,
        split_topk_for_cpu,
    )

    decision = split_topk_for_cpu(
        topk_ids,
        cpu_slot_expert_id=cpu_slot,
        max_tokens=cfg.cpu_offload_max_tokens_per_expert,
    )
    if decision.n_cpu_tokens <= 0:
        if _MOE_TRACE:
            _trace.reason("hook.begin.eager_skip", "no token routed to cpu_slot")
        rt.metrics.record_skip()
        return None

    try:
        decision = check_overflow(
            decision,
            max_per_step=cfg.cpu_offload_max_tokens_per_step,
            max_per_expert=cfg.cpu_offload_max_tokens_per_expert,
            policy=cfg.cpu_offload_overflow_policy.value,
        )
    except OverflowError:
        raise
    if decision.n_cpu_tokens <= 0:
        return None

    token_indices = decision.token_indices
    cpu_input = hidden_states.index_select(0, token_indices).contiguous()

    # GPU-side NaN check: confirm whether GPU hidden_states genuinely has NaN
    # before the D2H copy (not a D2H race). Enable with CPU_OFFLOAD_NAN_CHECK=1.
    if os.environ.get("CPU_OFFLOAD_NAN_CHECK"):
        _nan_chk_cnt = getattr(begin_cpu_offload, "_nan_chk_cnt", 0)
        if _nan_chk_cnt < 20:
            nan_mask = cpu_input.isnan()  # forces GPU sync
            if nan_mask.any():
                begin_cpu_offload._nan_chk_cnt = _nan_chk_cnt + 1
                for _ti in range(cpu_input.shape[0]):
                    _row_nan = nan_mask[_ti]
                    if _row_nan.any():
                        _n = int(_row_nan.sum().item())
                        _nz = _row_nan.nonzero(as_tuple=False)[:, 0].tolist()
                        print(
                            f"[cpu_offload] GPU_INPUT_NAN rank={rt.local_rank}"
                            f" layer={layer_id} tok={_ti}"
                            f" global_tok={int(token_indices[_ti].item())}"
                            f" nan={_n} pos=[{_nz[0]}..{_nz[-1]}]"
                            f" vals={cpu_input[_ti, :4].tolist()}",
                            flush=True,
                        )

    cpu_topk_ids = topk_ids.index_select(0, token_indices).contiguous()
    cpu_topk_weights = topk_weights.index_select(0, token_indices).contiguous()

    # CRITICAL FIX: The daemon's forwardExperts computes ALL branches in
    # topk_ids (the full weighted sum of all selected experts). But the GPU
    # also computes all branches EXCEPT the cpu_slot one (which is masked
    # to -1). To avoid double-counting the non-cpu_slot branches, we must
    # send ONLY the cpu_slot expert to the daemon with its weight at
    # position 0 (required by the C++ n_toks counting logic).
    slot_mask = cpu_topk_ids == cpu_slot  # [n_tokens, topk], one True per row
    cpu_slot_weight = cpu_topk_weights[slot_mask]  # [n_tokens]
    cpu_topk_ids = torch.full_like(cpu_topk_ids, -1)
    cpu_topk_ids[:, 0] = rt.engine.layer_spec(layer_id).cpu_expert_id(cpu_slot)
    cpu_topk_weights = torch.zeros_like(cpu_topk_weights)
    cpu_topk_weights[:, 0] = cpu_slot_weight

    stream = torch.cuda.current_stream().cuda_stream
    cpu_contrib = torch.zeros(
        (decision.n_cpu_tokens, hidden_states.shape[-1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    # CPU_OFFLOAD_PRESYNC=1: force a full CUDA stream sync before the D2H DMA to
    # confirm whether a timing race between GPU tensor creation and the DMA
    # is the root cause of NaN in the SHM payload.
    if os.environ.get("CPU_OFFLOAD_PRESYNC"):
        torch.cuda.current_stream().synchronize()

    step_cpu_events = None
    if _STEP_TRACE_ENABLED:
        from sglang.srt.eplb.cpu_offload.step_trace import begin_cpu_gpu_task

        step_cpu_events = begin_cpu_gpu_task(layer_id)
        if step_cpu_events is not None:
            # eager 路径已经按真实请求裁出了 CPU token，不存在图 padding。
            step_cpu_events.business_offloaded_branches = decision.n_cpu_tokens

    # Try the new split (parallel) path first.  If it fails (e.g. device
    # buffer not allocated on this rank), fall back to the old serial path.
    use_split_path = True
    try:
        rt.engine.launch_submit(
            layer_id,
            dev_input=cpu_input,
            dev_topk_ids=cpu_topk_ids,
            dev_topk_weights=cpu_topk_weights,
            n_tokens=decision.n_cpu_tokens,
            cuda_stream=int(stream),
            trace_events=(
                step_cpu_events.submit_events() if step_cpu_events is not None else None
            ),
        )
    except Exception as exc:
        if _MOE_TRACE:
            _trace.exc("dispatch_hook.begin_cpu_offload/launch_submit", exc)
        logger.warning(
            "[cpu_offload] launch_submit failed (layer=%d): %s; "
            "falling back to serial path",
            layer_id,
            exc,
        )
        use_split_path = False
        if step_cpu_events is not None:
            step_cpu_events.unsupported_reasons.append(
                "eager_serial_fallback_phase_events_unsupported"
            )
        try:
            rt.engine.launch_layer(
                layer_id,
                dev_input=cpu_input,
                dev_topk_ids=cpu_topk_ids,
                dev_topk_weights=cpu_topk_weights,
                dev_output=cpu_contrib,
                n_tokens=decision.n_cpu_tokens,
                cuda_stream=int(stream),
            )
        except Exception as exc2:
            if _MOE_TRACE:
                _trace.exc("dispatch_hook.begin_cpu_offload/launch_layer", exc2)
                _trace.reason(
                    "hook.begin.eager_skip", "launch_layer failed (see sec.5)"
                )
            logger.warning(
                "[cpu_offload] launch_layer also failed (layer=%d): %s; "
                "skipping CPU branch",
                layer_id,
                exc2,
            )
            return None

    if _MOE_TRACE:
        _trace.hit("hook.begin.eager_armed")
        _trace.layer_hit(layer_id, "armed")
    rt.metrics.record_offload(rt.local_rank, cpu_slot, decision.n_cpu_tokens)
    rt.metrics.record_daemon_req_sent()

    # ---- 专家分发诊断打印 (CPU_OFFLOAD_DISP_LAYERS=0,1,2 时生效) ----
    try:
        _disp_env = os.environ.get("CPU_OFFLOAD_DISP_LAYERS", "")
        if _disp_env:
            _disp_layers = {int(x) for x in _disp_env.split(",") if x.strip()}
            if layer_id in _disp_layers:
                _disp_cnt = getattr(begin_cpu_offload, "_disp_cnt", {})
                _call_n = _disp_cnt.get((rt.local_rank, layer_id), 0) + 1
                _disp_cnt[(rt.local_rank, layer_id)] = _call_n
                begin_cpu_offload._disp_cnt = _disp_cnt
                # 只打印前 5 次，避免刷屏
                if _call_n <= 5:
                    ids_host = topk_ids.detach().cpu()  # [n_tok, topk] 原始
                    masked_host = decision.masked_topk_ids.detach().cpu()  # GPU 看到的
                    tok_idx_host = decision.token_indices.detach().cpu().tolist()
                    n_tok = ids_host.shape[0]
                    lines = [
                        f"[DISP][rank={rt.local_rank}] layer={layer_id} call={_call_n} "
                        f"n_tokens={n_tok} cpu_slot={cpu_slot} "
                        f"cpu_hits={decision.n_cpu_tokens} cpu_tok_idx={tok_idx_host}"
                    ]
                    for ti in range(n_tok):
                        orig = ids_host[ti].tolist()
                        gpu = masked_host[ti].tolist()
                        hit = "<<CPU>>" if ti in tok_idx_host else ""
                        lines.append(f"  tok[{ti}] orig={orig} gpu={gpu} {hit}")
                    print("\n".join(lines), flush=True)
    except Exception as _e:
        print(f"[DISP] error: {_e}", flush=True)
    # ---------------------------------------------------------------

    return {
        "runtime": rt,
        "layer_id": layer_id,
        "token_indices": token_indices,
        "n_cpu_tokens": decision.n_cpu_tokens,
        "hidden_size": hidden_states.shape[-1],
        "dtype": hidden_states.dtype,
        "device": hidden_states.device,
        "masked_topk_ids": decision.masked_topk_ids,
        "stream": stream,
        "cpu_contrib": cpu_contrib,
        "use_split_path": use_split_path,
        "step_trace_cpu_events": step_cpu_events,
    }


def commit_cpu_offload(ctx: dict, final_hidden_states: torch.Tensor) -> torch.Tensor:
    """Phase 3: Wait for CPU completion and scatter-add."""
    if _MOE_TRACE:
        _trace.hit("hook.commit")
    # CUDA-Graph-capturable path (SHM kernel path)
    if ctx.get("use_cuda_graph_offload"):
        if _MOE_TRACE:
            _trace.hit("hook.commit.graph")
        gctx = ctx["gctx"]
        result = gctx.commit_offload(ctx, final_hidden_states)
        # Update combine metrics during capture (lower bound).
        n_cpu_tokens = int(ctx.get("num_tokens", 0))
        try:
            from sglang.srt.eplb.cpu_offload import get_cpu_offload_runtime

            _rt = get_cpu_offload_runtime()
            if _rt is not None:
                _rt.metrics.record_combine_success(n_cpu_tokens)
        except Exception:
            pass
        return result

    if _MOE_TRACE:
        _trace.hit("hook.commit.eager")
    rt = ctx["runtime"]
    n_cpu_tokens = int(ctx.get("n_cpu_tokens", 0))
    use_split_path = ctx.get("use_split_path", True)
    step_cpu_events = ctx.get("step_trace_cpu_events")
    try:
        cpu_contrib = ctx["cpu_contrib"]
        if use_split_path:
            # New parallel path: wait kernel + H2D copyout
            if step_cpu_events is not None:
                from sglang.srt.eplb.cpu_offload.step_trace import (
                    record_visible_wait_start,
                )

                record_visible_wait_start(step_cpu_events)
            rt.engine.launch_wait_and_copyout(
                ctx["layer_id"],
                dev_output=cpu_contrib,
                n_tokens=n_cpu_tokens,
                cuda_stream=int(ctx["stream"]),
                trace_events=(
                    step_cpu_events.wait_events()
                    if step_cpu_events is not None
                    else None
                ),
            )
            if step_cpu_events is not None:
                from sglang.srt.eplb.cpu_offload.step_trace import (
                    record_visible_wait_end,
                    record_wait_status,
                )

                record_visible_wait_end(step_cpu_events)
                daemon = getattr(rt.engine, "_daemon", None)
                remote_moe = (
                    daemon._layers.get(ctx["layer_id"]) if daemon is not None else None
                )
                status_buf = getattr(remote_moe, "_status_buf", None)
                if status_buf is not None:
                    record_wait_status(step_cpu_events, status_buf)
        else:
            # Fallback serial path: daemon backend commit_layer is a
            # no-op (forward_experts_kernel already included wait).
            rt.engine.commit_layer(
                ctx["layer_id"],
                dev_output=cpu_contrib,
                n_tokens=n_cpu_tokens,
                cuda_stream=int(ctx["stream"]),
            )

        # Check wait-kernel timeout status. If the CPU daemon did not
        # respond within max_spins, discard the CPU contribution entirely
        # (the buffer may contain stale/incomplete data).
        _cpu_timed_out = False
        try:
            rm = rt.engine._layers.get(ctx["layer_id"])
            if rm is not None and hasattr(rm, "read_last_status_blocking"):
                timeout_flag, error_code = rm.read_last_status_blocking()
                if timeout_flag != 0:
                    if _MOE_TRACE:
                        _trace.hit("hook.commit.timeout")
                    _cpu_timed_out = True
                    rt.metrics.record_combine_fail(n_cpu_tokens)
                    rt.metrics.cpu_offload_timeout_count += 1
                    logger.warning(
                        "[cpu_offload] TIMEOUT layer=%d n_tokens=%d "
                        "(status=%d err=%d); CPU contribution discarded",
                        ctx["layer_id"],
                        n_cpu_tokens,
                        timeout_flag,
                        error_code,
                    )
        except Exception:
            pass

        if _cpu_timed_out:
            return final_hidden_states

        # Sanitize: replace NaN/Inf with 0 so a single bad expert
        # doesn't corrupt the residual stream for all subsequent layers.
        # Functionally equivalent to "CPU contribution lost for that element."
        nan_mask = torch.isnan(cpu_contrib) | torch.isinf(cpu_contrib)
        # Track per-token NaN count so we can split success vs. fail metrics.
        _n_nan_toks = 0
        if nan_mask.any():
            if _MOE_TRACE:
                _trace.hit("hook.commit.nan")
            n_bad = int(nan_mask.sum().item())
            _n_nan_toks = int(nan_mask.any(dim=-1).sum().item())
            total_el = cpu_contrib.numel()
            abs_max = (
                float(cpu_contrib[~nan_mask].abs().max().item())
                if not nan_mask.all()
                else 0.0
            )
            # Print NaN positions (column indices) for first token
            nan_cols = (
                torch.where(nan_mask[0])[0].tolist()
                if cpu_contrib.dim() == 2
                else torch.where(nan_mask)[0].tolist()
            )
            _diag_cnt = getattr(commit_cpu_offload, "_diag_cnt", 0)
            if _diag_cnt < 5:
                # Probe wait-kernel timeout status (requires stream sync)
                timeout_flag, error_code = 0, 0
                try:
                    rm = rt.engine._layers.get(ctx["layer_id"])
                    if rm is not None and hasattr(rm, "read_last_status_blocking"):
                        timeout_flag, error_code = rm.read_last_status_blocking()
                except Exception:
                    pass
                print(
                    f"[cpu_offload][rank={rt.local_rank}] DIAG layer={ctx['layer_id']} "
                    f"n_tokens={n_cpu_tokens} nan_count={n_bad}/{total_el} "
                    f"nan_tokens={_n_nan_toks} abs_max={abs_max:.6e} "
                    f"nan_positions={nan_cols[:40]} "
                    f"TIMEOUT={timeout_flag} ERR={error_code}",
                    flush=True,
                )
                # Dump GPU-side cpu_contrib for comparison with daemon dump
                import os as _os

                if _os.environ.get("CPU_OFFLOAD_DUMP_LAYER"):
                    dump_path = f"/tmp/gpu_dump_layer{ctx['layer_id']}_rank{rt.local_rank}_contrib.pt"
                    torch.save(cpu_contrib.cpu(), dump_path)
                    print(f"  → dumped gpu cpu_contrib to {dump_path}", flush=True)
                commit_cpu_offload._diag_cnt = _diag_cnt + 1
            cpu_contrib = cpu_contrib.masked_fill(nan_mask, 0.0)

        # Check if final_hidden_states (GPU MoE output) itself has NaN
        # BEFORE we add cpu_contrib.  Fires only when CPU_OFFLOAD_NAN_CHECK=1.
        if os.environ.get("CPU_OFFLOAD_NAN_CHECK"):
            _hs_nan = final_hidden_states.isnan()
            if _hs_nan.any():
                _hs_nan_cnt = int(_hs_nan.sum().item())
                _hs_nan_rows = (
                    _hs_nan.any(dim=-1).nonzero(as_tuple=False)[:, 0].tolist()
                )
                print(
                    f"[cpu_offload] GPU_MoE_NAN rank={rt.local_rank}"
                    f" layer={ctx['layer_id']} n_tokens={n_cpu_tokens}"
                    f" hs_nan={_hs_nan_cnt} nan_rows={_hs_nan_rows[:8]}",
                    flush=True,
                )

        if _IMPACT_DIAG:
            _ia_cnt = getattr(commit_cpu_offload, "_ia_print_cnt", 0)
            if _ia_cnt < _IMPACT_DIAG_MAX:
                _contrib_norm = float(cpu_contrib.detach().norm().item())
                _fhs_norm = float(final_hidden_states.detach().norm().item())
                _rel_pct = (_contrib_norm / _fhs_norm * 100) if _fhs_norm > 0 else 0
                _abs_max = float(cpu_contrib.abs().max().item())
                _tidx0 = int(ctx["token_indices"][0].item()) if n_cpu_tokens > 0 else -1
                print(
                    f"[IMPACT][rank={rt.local_rank}] layer={ctx['layer_id']} "
                    f"n_cpu={n_cpu_tokens} "
                    f"contrib_norm={_contrib_norm:.4f} fhs_norm={_fhs_norm:.4f} "
                    f"relative={_rel_pct:.4f}% contrib_max={_abs_max:.6e} "
                    f"tidx0={_tidx0}",
                    flush=True,
                )
                commit_cpu_offload._ia_print_cnt = _ia_cnt + 1
        final_hidden_states = final_hidden_states.index_add(
            0, ctx["token_indices"], cpu_contrib
        )
        # Record success/fail split: NaN tokens count as fail (contribution zeroed).
        _n_ok_toks = n_cpu_tokens - _n_nan_toks
        if _n_ok_toks > 0:
            rt.metrics.record_combine_success(_n_ok_toks)
        if _n_nan_toks > 0:
            rt.metrics.record_combine_fail(_n_nan_toks)
    except Exception as exc:
        # Exception during wait/copyout: daemon did not respond or transfer
        # failed — count as daemon exception (not just NaN combine fail).
        if _MOE_TRACE:
            _trace.exc("dispatch_hook.commit_cpu_offload/wait-copyout", exc)
        rt.metrics.record_daemon_exception()
        rt.metrics.record_combine_fail(n_cpu_tokens)
        logger.warning(
            "[cpu_offload] commit_layer failed (layer=%d): %s; CPU contribution lost",
            ctx["layer_id"],
            exc,
        )
    return final_hidden_states
