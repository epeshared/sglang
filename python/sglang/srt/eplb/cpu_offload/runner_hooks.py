"""把已验证的 CPU 初始化过程接到新版 ModelRunner 生命周期。"""

from __future__ import annotations

import glob
import logging
import os
from contextlib import contextmanager

from sglang.srt.eplb.cpu_offload import step_trace

logger = logging.getLogger(__name__)


def get_global_expert_location_metadata():
    from sglang.srt.eplb.expert_location import (
        get_global_expert_location_metadata as get_metadata,
    )

    return get_metadata()


def decode_graph_capacity():
    """读取新版已解析的捕获批量，不能用旧字段缺失时的 256 代替。"""
    from sglang.srt.runtime_context import get_exec

    config = get_exec().graph.cuda_graph_config.decode
    if config.backend == "disabled":
        return 0
    capacity = max(config.bs) if config.bs else config.max_bs
    if capacity is None or capacity <= 0:
        raise ValueError("CPU 图捕获需要已解析的 decode 批量配置")
    return capacity


class RunnerView:
    """为原初始化过程提供旧版 rank 属性，其余属性仍读取当前 runner。"""

    def __init__(self, runner):
        from sglang.srt.arg_groups.model_override_base import resolved_view

        self.runner = runner
        self.server_args = resolved_view(runner.server_args)

    def __getattr__(self, name):
        return getattr(self.runner, name)

    @property
    def tp_rank(self):
        from sglang.srt.runtime_context import get_parallel

        return get_parallel().tp_rank

    @property
    def tp_size(self):
        from sglang.srt.runtime_context import get_parallel

        return get_parallel().tp_size

    def _init_cuda_graph_offload(self):
        """Initialize CUDA-graph-compatible CPU offload context (方案 B).

        Must be called after register_model_moe_layers() and before
        init_device_graphs() so that SHM kernel-path ops are recorded
        into the CUDA graph during capture.

        Uses the same SHM kernel path (cpu_offload_kernels.cu) as the eager
        daemon backend — all ops are pure GPU kernels, no host callbacks,
        fully CUDA-Graph capturable.
        """
        if not getattr(self.server_args, "enable_cpu_offload_fast_loop", False):
            logger.info(
                "[cuda_graph_offload] skipped: enable_cpu_offload_fast_loop not set"
            )
            return
        max_tokens = decode_graph_capacity()
        if not max_tokens:
            logger.info("[cuda_graph_offload] decode 图捕获已关闭，使用 eager 路径")
            return

        try:
            from sglang.srt.eplb.cpu_offload import get_cpu_offload_runtime
            from sglang.srt.eplb.cpu_offload.cuda_graph_offload import (
                init_cuda_graph_offload,
            )

            rt = get_cpu_offload_runtime()
            if rt is None:
                logger.warning("[cuda_graph_offload] skipped: runtime is None")
                return

            logger.info(
                "[cuda_graph_offload] starting init on rank %d, "
                "engine.initialized=%s, daemon=%s",
                self.tp_rank,
                getattr(rt.engine, "initialized", "?"),
                "present" if getattr(rt.engine, "_daemon", None) else "absent",
            )

            # Get model parameters for buffer sizing
            hf_cfg = self.model_config.hf_config
            hidden_size = getattr(hf_cfg, "hidden_size", 7168)
            topk = getattr(hf_cfg, "num_experts_per_tok", 8)
            _raw_max_hits = getattr(self.server_args, "cpu_offload_max_hits", 0)
            gctx = init_cuda_graph_offload(
                hidden_size=hidden_size,
                topk=topk,
                max_tokens=max_tokens,
                max_spins=getattr(self.server_args, "cpu_offload_max_spin", 50_000_000),
                max_hits=_raw_max_hits if _raw_max_hits > 0 else None,
            )
            if gctx is None:
                logger.warning("[cuda_graph_offload] failed to initialize context")
                return

            # Register layers with the graph offload context by borrowing
            # the RemoteMOE instances from DaemonBackend.
            engine = rt.engine
            if not getattr(engine, "initialized", False):
                logger.info(
                    "[cuda_graph_offload] engine not yet initialized, "
                    "skipping layer registration"
                )
                return

            daemon = getattr(engine, "_daemon", None)
            if daemon is None:
                logger.warning("[cuda_graph_offload] no daemon backend attached")
                return

            n_registered = 0
            for layer_id in engine.registered_layers():
                # Get the RemoteMOE instance from DaemonBackend
                rm = daemon._layers.get(layer_id)
                if rm is not None:
                    if gctx.register_layer_from_daemon(layer_id, rm):
                        n_registered += 1
                else:
                    # Fallback to old API (no-op in new implementation)
                    gctx.register_layer(layer_id, daemon.slot_handle)
                    n_registered += 1

                # Ensure a cpu_slot is assigned BEFORE graph capture so the
                # offload kernels get recorded.  The cpu_slot constant is
                # baked into the CUDA graph at capture time (the comparison
                # kernel ``topk_ids == cpu_slot`` records the scalar value),
                # so it MUST be the correct per-rank expert id — not a shared
                # placeholder like 0 (which would make all ranks filter the
                # same expert and produce identical token counts).
                # This is done per-layer since each MoE layer can have a
                # different hot expert offloaded to CPU.
                if rt.local_cpu_slot(layer_id) < 0:
                    try:
                        state = rt.registry.local_state()
                        if state is not None and state.active:
                            # Derive the per-rank initial slot from the expert
                            # location metadata that was already built during
                            # ModelRunner.__init__ (set_global_expert_location_
                            # metadata is called before init_cuda_graph_offload).
                            # We use the LAST local expert of this rank, which:
                            #   (a) belongs to this rank's physical range, so
                            #       masking it to -1 does not alias with any
                            #       other rank's local computation, and
                            #   (b) is a stable choice that survives across
                            #       EPLB rebalances (it stays in the rank's pool).
                            initial_slot = -1
                            try:
                                meta = get_global_expert_location_metadata()
                                if meta is not None:
                                    num_local = int(meta.num_local_physical_experts)
                                    phys = meta.physical_to_logical_map
                                    # physical_to_logical_map: (num_layers, num_phys)
                                    # rank r owns physical indices [r*num_local, (r+1)*num_local).
                                    # Use this layer's own row (per-layer mapping
                                    # can differ after EPLB rebalancing).
                                    num_layers_meta = phys.shape[0]
                                    row_idx = (
                                        layer_id if layer_id < num_layers_meta else 0
                                    )
                                    layer_row = phys[row_idx].tolist()
                                    phys_idx = self.tp_rank * num_local + (
                                        num_local - 1
                                    )
                                    if 0 <= phys_idx < len(layer_row):
                                        eid = int(layer_row[phys_idx])
                                        # Detect block-initialized (identity) EPLB map:
                                        # if phys[row, phys_idx] == phys_idx, the map is still
                                        # at its default block layout.  The actual model uses
                                        # stride routing so this would point to the WRONG rank's
                                        # GPU expert.  Fall through to the stride formula below.
                                        if eid >= 0 and eid != phys_idx:
                                            initial_slot = eid
                            except Exception:
                                pass

                            if initial_slot < 0:
                                # Metadata unavailable OR block-initialized map — use the
                                # CONTIGUOUS BLOCK layout, which is what actually decides
                                # ownership: rank r owns global expert ids
                                # [r*s_gpu, (r+1)*s_gpu), so its last one is
                                # (r+1)*s_gpu - 1. Both
                                # FusedMoE._map_global_expert_id_to_local_expert_id
                                # (weight placement) and StandardDispatcher.dispatch
                                # (routing) compute exactly this range.
                                #
                                # This previously used the INTERLEAVED formula
                                # ``tp_rank + (s_gpu-1)*ws`` on the belief that expert e
                                # lives on rank e % world_size. That put EVERY rank's
                                # cpu_slot inside the LAST rank's block, so masking it was
                                # a no-op locally while the owning rank still ran it on
                                # GPU -- every offloaded token counted twice after the TP
                                # all-reduce. The block value also equals the ``phys_idx``
                                # the metadata branch above computes, so both branches now
                                # agree instead of contradicting each other.
                                try:
                                    ne = (
                                        getattr(
                                            self.model_config.hf_config,
                                            "num_experts",
                                            0,
                                        )
                                        or 0
                                    )
                                    ws = self.tp_size or 1
                                    s_gpu = ne // ws if (ne > 0 and ws > 0) else 0
                                    if s_gpu > 0:
                                        initial_slot = (self.tp_rank + 1) * s_gpu - 1
                                except Exception:
                                    pass

                            if initial_slot < 0:
                                initial_slot = 0  # last-resort safe fallback

                            state.assign_slot(layer_id, initial_slot)
                            logger.info(
                                "[cuda_graph_offload] pre-assigned initial "
                                "cpu_slot=%d for graph capture (rank=%d, layer=%d)",
                                initial_slot,
                                self.tp_rank,
                                layer_id,
                            )
                    except Exception:
                        pass

            logger.info(
                "[cuda_graph_offload] registered %d layer(s) for "
                "CUDA-graph-capturable CPU offload on rank %d",
                n_registered,
                self.tp_rank,
            )

        except Exception as exc:
            logger.warning(
                "[cuda_graph_offload] initialization failed (continuing "
                "without graph-capturable offload): %s",
                exc,
            )


def validate_runner(runner):
    """移植范围外的组合必须报错，避免看起来启用了 CPU、实际没有参与。"""
    args = runner.server_args
    enabled = bool(getattr(args, "enable_cpu_offload_fast_loop", False))
    if getattr(args, "enable_ep_moe_heto", False) and not enabled:
        raise ValueError(
            "新版基线尚未接入旧静态 Heto 布局；本次迁移的是动态 CPU 专家路径"
        )
    if not enabled or runner.is_draft_worker:
        return False
    from sglang.srt.runtime_context import get_exec, get_parallel, get_spec

    parallel = get_parallel()
    if parallel.pp_size != 1 or parallel.moe_ep_size != parallel.tp_size:
        raise ValueError("动态 CPU 专家要求 PP=1，且专家并行数与张量并行数相同")
    if get_exec().moe.moe_a2a_backend != "none":
        raise ValueError(
            "当前移植保留已测过的无 all-to-all 路径，请设置 --moe-a2a-backend none"
        )
    if get_spec().speculative_algorithm is not None:
        raise ValueError("动态 CPU 专家尚未验证投机解码，请先关闭该功能")
    graphs = get_exec().graph.cuda_graph_config
    if graphs.prefill.backend != "disabled":
        raise ValueError(
            "当前 CPU 图捕获仅接入 decode，请设置 --cuda-graph-backend-prefill disabled"
        )
    if graphs.decode.backend not in ("full", "disabled"):
        raise ValueError("动态 CPU 专家的 decode 图捕获仅支持 full 或 disabled")
    quant = getattr(runner.model_config.hf_config, "quantization_config", None) or {}
    if quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [
        128,
        128,
    ]:
        raise ValueError(
            "现有 CPU 后端只接收 128×128 分块 FP8 权重；"
            "Hy4 BF16 的原始权重加载和 CPU 计算仍需单独适配"
        )
    return True


def prepare_runner(runner):
    view = RunnerView(runner)
    runner._cpu_offload_view = view
    if not validate_runner(runner):
        return
    from sglang.srt.eplb.cpu_offload import init_cpu_offload_runtime

    numa_ids = os.environ.get("CPU_NUMA_NODE_IDS", "").strip()
    if numa_ids:
        count = len([part for part in numa_ids.split(",") if part.strip()])
    else:
        count = (
            int(os.environ.get("SGLANG_NUM_NUMA_NODES", "0"))
            or len(glob.glob("/sys/devices/system/node/node*"))
            or 2
        )
        if (
            os.environ.get("CPU_OFFLOAD_USE_PHYSICAL_SOCKET_SUBNUMA") == "1"
            and count > 2
        ):
            count = 2
    init_cpu_offload_runtime(
        view.server_args,
        local_rank=view.tp_rank,
        world_size=view.tp_size,
        num_numa_nodes=count,
        model_runner=view,
    )


def register_runner_model(runner):
    if runner.is_draft_worker:
        return
    from sglang.srt.eplb.cpu_offload import get_cpu_offload_runtime
    from sglang.srt.eplb.cpu_offload.model_load_hook import register_model_moe_layers

    runtime = get_cpu_offload_runtime()
    if runtime is not None:
        if getattr(runner, "startup_weight_load", None) is not None:
            raise RuntimeError("CPU 专家登记需要完整权重，请关闭启动时的权重加载重叠")
        registered = register_model_moe_layers(runner.model, runtime)
        if runtime.is_active_locally() and not registered:
            raise RuntimeError("已启用 CPU 专家，但模型中没有成功登记的专家层")
    if step_trace.ENABLED:
        step_trace.init_step_trace(runner._cpu_offload_view)


def prepare_graphs(runner):
    if not runner.is_draft_worker:
        runner._cpu_offload_view._init_cuda_graph_offload()


def finish_initialization(runner):
    """pinned 通信沿用原入口的逐 rank 启动线程核验。"""
    if os.environ.get("CPU_OFFLOAD_TRANSPORT") in ("pinned", "anonymous"):
        from sglang.srt.eplb.cpu_offload.transport_threads import record_rank_startup

        record_rank_startup(runner._cpu_offload_view.tp_rank)


@contextmanager
def forward_scope(forward_batch):
    step = None
    if step_trace.ENABLED and forward_batch.forward_mode.is_decode():
        step = step_trace.begin_step(forward_batch)
    try:
        yield
    except BaseException:
        if step is not None:
            step_trace.end_step(step, failed=True)
        raise
    else:
        if step is not None:
            step_trace.end_step(step)


def finish_forward(runner):
    if runner.is_draft_worker or runner.eplb_manager is not None:
        return
    from sglang.srt.eplb.cpu_offload import get_cpu_offload_runtime

    runtime = get_cpu_offload_runtime()
    if runtime is not None:
        runtime.tick_fast_loop()
