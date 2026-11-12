from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from sglang.srt import single_batch_overlap
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
    is_cpu_offload_fast_loop_enabled,
    is_ep_moe_heto_enabled,
    get_ep_moe_heto_gpu_experts,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FlashInferFusedMoE, FusedMoE
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPNormalCombineInput,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz, scaled_fp8_quant
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
from sglang.srt.utils import set_weight_attrs
from sglang.srt.single_batch_overlap import DownGemmOverlapArgs
from sglang.srt.utils import get_bool_env_var, is_hip, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        DispatchOutput,
    )

_is_hip = is_hip()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if not (_is_npu or _is_hip):
    pass

if _use_aiter:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

logger = logging.getLogger(__name__)

# ---- MoE call-path trace (SGLANG_MOE_TRACE=1) -----------------------------
# Read straight from the environment at import: a disabled run pays one
# module-global load plus a not-taken branch per site, and never imports the
# trace module. See sglang/srt/eplb/cpu_offload/trace.py.
_MOE_TRACE = os.environ.get("SGLANG_MOE_TRACE", "").strip().lower() in (
    "1", "true", "yes", "y", "on"
)
_STEP_TRACE_ENABLED = os.environ.get("SGLANG_MOE_STEP_TRACE", "").strip().lower() in (
    "1", "true", "yes", "on"
)
_moe_trace_mod = None
_moe_trace_facts_done = False


def _moe_trace():
    """Import the trace module on first traced call and cache it."""
    global _moe_trace_mod
    if _moe_trace_mod is None:
        from sglang.srt.eplb.cpu_offload import trace as _t

        _moe_trace_mod = _t
    return _moe_trace_mod


# ===== CPU MoE Offload Additional Imports =====
import gc
import json
import os
import numpy as np
from typing import Callable, List as TypingList, Tuple as TypingTuple

# Distributed imports for CPU MoE
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

# Expert location metadata for CPU MoE
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo

# Top-K selection for CPU MoE
from sglang.srt.layers.moe.topk import select_experts, TopKConfig

# Try importing deep_gemm for FP8 acceleration
try:
    from deep_gemm import (
        get_col_major_tma_aligned_tensor,
        m_grouped_gemm_fp8_fp8_bf16_nt_contiguous,
        m_grouped_gemm_fp8_fp8_bf16_nt_masked,
    )
    from sgl_kernel import silu_and_mul
    use_deep_gemm = True
except ImportError:
    use_deep_gemm = False
    logger.info('deep_gemm not available, using fallback')

# Zero-copy support for CPU MoE
import ctypes
libcudart = ctypes.CDLL("libcudart.so")

cudaHostGetDevicePointer = libcudart.cudaHostGetDevicePointer
cudaHostGetDevicePointer.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),  # void **ppDevice
    ctypes.c_void_p,                  # void *pHost
    ctypes.c_uint                     # unsigned int flags
]
cudaHostGetDevicePointer.restype = ctypes.c_int # cudaError_t

def cudaHostGetDevicePointer_wrapper(host_ptr: int, flags: int) -> int:
    """Wrapper for CUDA zero-copy: get device pointer from pinned host memory"""
    device_ptr = ctypes.c_void_p()
    status = cudaHostGetDevicePointer(
        ctypes.byref(device_ptr),
        ctypes.c_void_p(host_ptr),
        flags
    )
    if status != 0:
        raise RuntimeError(f"cudaHostGetDevicePointer failed with error code {status}")
    return device_ptr.value



# ===== Original DeepEPMoE from v0.5.5 =====
class DeepEPMoE(FusedMoE):
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    Mooncake EP shares the same class, as they expose the same interface.
    """

    _has_printed = False

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        num_local_experts: Optional[int] = None,
        local_expert_ids: Optional[TypingList[int]] = None,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            num_local_experts=num_local_experts,
            local_expert_ids=local_expert_ids,
        )

        if _use_aiter or _is_npu:
            self.deprecate_flag = False
        elif deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and isinstance(
            quant_config, Fp8Config
        ):
            self.deprecate_flag = True
        else:
            self.deprecate_flag = False

        if self.deprecate_flag:
            return

        if isinstance(quant_config, Fp8Config):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.use_fp8_w8a8 = True
            self.fp8_dtype = torch.float8_e4m3fn
            self.use_w4afp8 = False
        elif isinstance(quant_config, W4AFp8Config):
            self.use_w4afp8 = True
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
        else:
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.use_w4afp8 = False

        self.deepep_mode = get_deepep_mode()

        if self.deepep_mode.enable_low_latency() and not _is_npu:
            # NPU supports low_latency deepep without deepgemm
            assert (
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            ), f"DeepEP {self.deepep_mode} mode requires deep_gemm"
        if _use_aiter:
            # expert_mask is of size (self.num_local_experts + 1),
            # the extra 1 is for invalid rank_id (in original deepep, the invalid rank_id is -1, but aiter does not allow -1, we use a mask to make those ids invalid)
            # for instance, if we have 4 experts on this rank, we would have a expert_mask like:
            #     self.expert_mask = [1, 1, 1, 1, 0]
            # idx from 0-3 is valid and will be processed, while idx == 4 will be masked out
            self.expert_mask = torch.zeros(
                (self.num_local_experts + 1),
                device=torch.cuda.current_device(),
                dtype=torch.int,
            )
            # the last one is invalid rank_id
            self.expert_mask[:-1] = 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        forward_shared_experts=None,
        alt_stream=None,
        disable_sbo=False,
    ):
        if _MOE_TRACE:
            _moe_trace().hit("moe.deepep.forward")
            global _moe_trace_facts_done
            if not _moe_trace_facts_done:
                _moe_trace_facts_done = True
                _moe_trace().record_moe_layer_facts(self)

        if self.deprecate_flag:
            # Everything below this return is UNREACHABLE in this
            # configuration -- including the CPU-offload begin/commit block.
            # The trace report shows both counters so a reader cannot mistake
            # the dead block for the live path.
            if _MOE_TRACE:
                _moe_trace().hit("moe.deepep.super")
            assert forward_shared_experts is None
            assert alt_stream is None
            return super().forward(
                hidden_states,
                topk_output,
            )

        # CPU-offload Mode-B hook: route the slot-expert's tokens to the
        # standalone cpu_expertd daemon in parallel with SBO. Only the
        # standard TopK format is supported (triton/bypassed formats keep
        # ids opaque). Failure is logged and ignored so SBO still runs.
        cpu_ctx = None
        if _MOE_TRACE:
            _moe_trace().hit("moe.deepep.inclass_begin")
        try:
            from sglang.srt.layers.moe.topk import (
                StandardTopKOutput,
                TopKOutputChecker,
            )
            from sglang.srt.eplb.cpu_offload.dispatch_hook import (
                begin_cpu_offload,
            )

            if TopKOutputChecker.format_is_standard(topk_output):
                cpu_ctx = begin_cpu_offload(
                    layer_id=getattr(self, "layer_id", -1),
                    hidden_states=hidden_states,
                    topk_ids=topk_output.topk_ids,
                    topk_weights=topk_output.topk_weights,
                )
                if cpu_ctx is not None:
                    if _MOE_TRACE:
                        _moe_trace().hit("moe.deepep.inclass_armed")
                    topk_output = StandardTopKOutput(
                        topk_weights=topk_output.topk_weights,
                        topk_ids=cpu_ctx["masked_topk_ids"],
                        router_logits=topk_output.router_logits,
                    )
        except Exception as exc:  # noqa: BLE001
            if _MOE_TRACE:
                _moe_trace().exc("ep_moe.DeepEPMoE.forward/begin", exc)
            logger.warning(
                "[cpu_offload] DeepEPMoE begin failed layer=%s: %s",
                getattr(self, "layer_id", "?"),
                exc,
            )
            cpu_ctx = None

        # We have to call SBO inside MoE to be compatible with hooks used in offloading
        out = single_batch_overlap.execute_sbo(
            hidden_states=hidden_states,
            topk_output=topk_output,
            # SBO args
            experts=self,
            forward_shared_experts=forward_shared_experts,
            alt_stream=alt_stream,
            disable_sbo=disable_sbo,
        )

        if cpu_ctx is not None:
            if _MOE_TRACE:
                _moe_trace().hit("moe.deepep.inclass_commit")
            try:
                from sglang.srt.eplb.cpu_offload.dispatch_hook import (
                    commit_cpu_offload,
                )

                out = commit_cpu_offload(cpu_ctx, out)
            except Exception as exc:  # noqa: BLE001
                if _MOE_TRACE:
                    _moe_trace().exc("ep_moe.DeepEPMoE.forward/commit", exc)
                logger.warning(
                    "[cpu_offload] DeepEPMoE commit failed layer=%s: %s",
                    getattr(self, "layer_id", "?"),
                    exc,
                )

        return out

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        return self.dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

    def run_moe_core(
        self,
        dispatch_output: DispatchOutput,
        down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = None,
    ):

        if self.deprecate_flag:
            assert down_gemm_overlap_args is None
            return super().run_moe_core(
                dispatch_output,
            )

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if _use_aiter:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            # in forward_aiter, we skip token permutation and unpermutation, which have been fused inside aiter kernel
            output = self.forward_aiter(dispatch_output)
        elif _is_npu:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            output = self.forward_npu(dispatch_output)
        elif DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            if self.use_w4afp8:
                output = self.forward_cutlass_w4afp8(dispatch_output)
            else:
                assert False, "forward_deepgemm_contiguous is deprecated"
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if (
                get_moe_runner_backend().is_flashinfer_cutedsl()
                and self.quant_config.get_name() == "modelopt_fp4"
            ):
                output = self.forward_flashinfer_cutedsl(
                    dispatch_output, down_gemm_overlap_args=down_gemm_overlap_args
                )
            elif self.use_w4afp8:
                output = self.forward_cutlass_w4afp8_masked(dispatch_output)
            else:
                assert False, "forward_deepgemm_masked is deprecated"

        combine_input_wrapper = (
            DeepEPNormalCombineInput
            if DispatchOutputChecker.format_is_deepep_normal(dispatch_output)
            else DeepEPLLCombineInput
        )
        return combine_input_wrapper(
            hidden_states=output,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_args: Optional[Dict[str, Any]] = None,
    ):
        return self.dispatcher.combine(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            overlap_args=overlap_args,
        )

    def forward_aiter(
        self,
        dispatch_output: Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput],
    ):
        hidden_states, topk_ids, topk_weights = (
            dispatch_output.hidden_states,
            dispatch_output.topk_ids,
            dispatch_output.topk_weights,
        )
        if hidden_states.shape[0] == 0:
            return hidden_states
        # in original deepep, idx == -1 meaning invalid and will not be processed.
        # aiter does not accept -1, we use a expert mask to make these idx invalid
        # (idx == num_local_experts) meaning not used in aiter fused_moe
        topk_ids_copy = topk_ids.to(torch.int32)
        topk_ids_copy[topk_ids_copy == -1] = self.num_local_experts

        return fused_moe(
            hidden_states,
            self.w13_weight,
            self.w2_weight,
            topk_weights,
            topk_ids_copy,
            w1_scale=self.w13_weight_scale_inv,
            w2_scale=self.w2_weight_scale_inv,
            quant_type=QuantType.per_128x128,
            activation=(
                ActivationType.Silu
                if self.moe_runner_config.activation == "silu"
                else ActivationType.Gelu
            ),
            expert_mask=self.expert_mask,
        )

    def forward_flashinfer_cutedsl(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
        down_gemm_overlap_args: Optional[DownGemmOverlapArgs],
    ):
        hidden_states, hidden_states_scale, _, _, masked_m, _ = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        output = self.quant_method.apply_without_routing_weights(
            layer=self,
            x=(hidden_states, hidden_states_scale),
            masked_m=masked_m,
            moe_runner_config=self.moe_runner_config,
            down_gemm_overlap_args=down_gemm_overlap_args,
        )
        return output

    def forward_cutlass_w4afp8(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_normal(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_cutlass_w4afp8_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        assert get_bool_env_var(
            "SGLANG_DEEPEP_BF16_DISPATCH"
        ), "W4AFP8 does not support FP8 dispatch; please set SGLANG_DEEPEP_BF16_DISPATCH=1."
        return self.quant_method.apply_deepep_ll(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_npu(
        self,
        dispatch_output: Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput],
    ):
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        import torch_npu

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        # NOTE: Ascend's Dispatch & Combine does not support FP16
        output_dtype = torch.bfloat16
        group_list_type = 1

        def _forward_normal(dispatch_output: DeepEPNormalDispatchOutput):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPNormalDispatchOutput)
            hidden_states, hidden_states_scale, _, _, num_recv_tokens_per_expert = (
                dispatch_output
            )

            group_list = torch.tensor(num_recv_tokens_per_expert, dtype=torch.int64).to(
                hidden_states.device
            )
            if self.w13_weight.dtype != torch.int8:
                # gmm1: gate_up_proj
                hidden_states = torch_npu.npu_grouped_matmul(
                    x=[hidden_states],
                    weight=[self.w13_weight.permute(0, 2, 1)],
                    # per_token_scale=[hidden_states_scale],
                    split_item=2,
                    group_list_type=group_list_type,
                    group_type=0,
                    group_list=group_list,
                    output_dtype=output_dtype,
                )[0]
                hidden_states = torch_npu.npu_swiglu(hidden_states)
                # gmm2: down_proj
                hidden_states = torch_npu.npu_grouped_matmul(
                    x=[hidden_states],
                    weight=[self.w2_weight.permute(0, 2, 1)],
                    split_item=2,
                    group_list_type=group_list_type,
                    group_type=0,
                    group_list=group_list,
                    output_dtype=output_dtype,
                )[0]
            else:
                if not get_bool_env_var("DEEP_NORMAL_MODE_USE_INT8_QUANT"):
                    hidden_states, hidden_states_scale = torch_npu.npu_dynamic_quant(
                        hidden_states
                    )
                # gmm1: gate_up_proj
                hidden_states = torch_npu.npu_grouped_matmul(
                    x=[hidden_states],
                    weight=[self.w13_weight],
                    scale=[self.w13_weight_scale.to(output_dtype)],
                    per_token_scale=[hidden_states_scale],
                    split_item=2,
                    group_list_type=group_list_type,
                    group_type=0,
                    group_list=group_list,
                    output_dtype=output_dtype,
                )[0]

                # act_fn: swiglu
                hidden_states = torch_npu.npu_swiglu(hidden_states)
                hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(
                    hidden_states
                )

                # gmm2: down_proj
                hidden_states = torch_npu.npu_grouped_matmul(
                    x=[hidden_states],
                    weight=[self.w2_weight],
                    scale=[self.w2_weight_scale.to(output_dtype)],
                    per_token_scale=[swiglu_out_scale],
                    split_item=2,
                    group_list_type=group_list_type,
                    group_type=0,
                    group_list=group_list,
                    output_dtype=output_dtype,
                )[0]

            return hidden_states

        def _forward_ll(dispatch_output: DeepEPLLDispatchOutput):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPLLDispatchOutput)
            (
                hidden_states,
                hidden_states_scale,
                topk_ids,
                topk_weights,
                group_list,
                _,
            ) = dispatch_output

            group_list = group_list.to(torch.int64)

            if self.w13_weight.dtype != torch.int8:
                # gmm1: gate_up_proj
                hidden_states = torch_npu.npu_grouped_matmul(
                    x=[hidden_states],
                    weight=[self.w13_weight.permute(0, 2, 1)],
                    # per_token_scale=[hidden_states_scale],
                    split_item=2,
                    group_list_type=group_list_type,
                    group_type=0,
                    group_list=group_list,
                    output_dtype=output_dtype,
                )[0]
                hidden_states = torch_npu.npu_swiglu(hidden_states)
                # gmm2: down_proj
                hidden_states = torch_npu.npu_grouped_matmul(
                    x=[hidden_states],
                    weight=[self.w2_weight.permute(0, 2, 1)],
                    split_item=2,
                    group_list_type=group_list_type,
                    group_type=0,
                    group_list=group_list,
                    output_dtype=output_dtype,
                )[0]
            else:
                # gmm1: gate_up_proj
                hidden_states = torch_npu.npu_grouped_matmul(
                    x=[hidden_states],
                    weight=[self.w13_weight],
                    split_item=2,
                    group_list_type=group_list_type,
                    group_type=0,
                    group_list=group_list,
                    output_dtype=torch.int32,
                )[0]

                # act_fn: swiglu
                hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
                    x=hidden_states,
                    weight_scale=self.w13_weight_scale.to(torch.float32),
                    activation_scale=hidden_states_scale,
                    bias=None,
                    quant_scale=None,
                    quant_offset=None,
                    group_index=group_list,
                    activate_left=True,
                    quant_mode=1,
                )

                # gmm2: down_proj
                hidden_states = torch_npu.npu_grouped_matmul(
                    x=[hidden_states],
                    weight=[self.w2_weight],
                    scale=[self.w2_weight_scale.to(output_dtype)],
                    per_token_scale=[swiglu_out_scale],
                    split_item=2,
                    group_list_type=group_list_type,
                    group_type=0,
                    group_list=group_list,
                    output_dtype=output_dtype,
                )[0]

            return hidden_states

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            return _forward_normal(dispatch_output)
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            return _forward_ll(dispatch_output)
        else:
            raise ValueError(f"Not Supported DeepEP format {dispatch_output.format}")


# ===== DeepEPMoEHeto: CPU+GPU Heterogeneous MoE =====
class DeepEPMoEHeto(DeepEPMoE):
    """
    Heterogeneous MoE: Some experts on GPU, others offloaded to CPU.
    Inherits from DeepEPMoE to leverage v0.5.5's proven GPU execution.
    """
    
    RANK_CPU_LIST = ["C0", "C1", "C2", "C3"]
    
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
    ):
        # Get configuration
        self.num_hosts = int(os.getenv("CPU_NUM_HOSTS", "1"))
        self.cpu_ep_size = int(os.getenv("CPU_EP_SIZE", "1"))
        if self.num_hosts == 1:
            self.num_hosts = self.cpu_ep_size
        self.cpu_tp_size = int(self.num_hosts / self.cpu_ep_size)
        
        # Get rank information for is_hosting_cpu() check
        self.ep_rank = get_tensor_model_parallel_rank()
        self.ep_size = get_tensor_model_parallel_world_size()
        self.host_rank = int(self.ep_rank // (self.ep_size / self.num_hosts))
        
        # Save total number of experts for routing
        self.total_num_experts = num_experts

        # Dual-loop fast-loop mode? In that mode JSON / static gpu_experts
        # slicing is irrelevant: GPU placement is the natural EP slice
        # (round-robin) and every CPU daemon will preload all experts.
        self.fast_loop_enabled = is_cpu_offload_fast_loop_enabled()

        if self.fast_loop_enabled:
            expert_map = self._create_fast_loop_expert_map(num_experts)
        else:
            # Try to load expert map from JSON file first, fallback to default
            expert_map = self._try_load_expert_map_from_json(layer_id, num_experts)
            if expert_map is None:
                expert_map = self._create_default_expert_map(num_experts)
        
        # Calculate which experts belong to current rank
        my_global_expert_ids = [expert_id for expert_id, rank_id in expert_map.items() 
                                if isinstance(rank_id, int) and rank_id == self.ep_rank]

        # 打印rank 0每层dispatch后命中的专家id分布
        if self.ep_rank == 0:
            def _print_dispatch_expert_stats(layer_id, topk_ids):
                ids = topk_ids.cpu().numpy().flatten()
                from collections import Counter
                c = Counter(ids)
                print(f"[debug][rank=0][layer={layer_id}] dispatched expert id stats: {dict(c)}")
            self._print_dispatch_expert_stats = _print_dispatch_expert_stats

        # Patch: 把当前rank的cpu_slot_expert_id也加入本rank experts
        from sglang.srt.eplb.cpu_offload import get_cpu_offload_runtime
        rt = get_cpu_offload_runtime()
        cpu_slot_id = None
        if rt is not None and hasattr(rt, 'local_cpu_slot'):
            slot_id = rt.local_cpu_slot(layer_id)
            if slot_id is not None and slot_id >= 0 and slot_id not in my_global_expert_ids:
                my_global_expert_ids.append(slot_id)
        
        # Count actual GPU experts in the map (integer rank IDs)
        actual_gpu_experts = sum(1 for rank_id in expert_map.values() if isinstance(rank_id, int))
        
        # Build is_gpu_expert boolean tensor for fast lookup
        self.is_gpu_expert = torch.zeros(num_experts, dtype=torch.bool)
        for expert_id, rank_id in expert_map.items():
            if isinstance(rank_id, int):
                self.is_gpu_expert[expert_id] = True
        
        num_local_experts = len(my_global_expert_ids)
        logger.info(f"[DeepEPMoEHeto Layer {layer_id}] ep_rank={self.ep_rank}, "
                   f"num_local_experts={num_local_experts}, my_global_expert_ids={my_global_expert_ids}, "
                   f"actual_gpu_experts={actual_gpu_experts}, total_experts={num_experts}, num_hosts={self.num_hosts}")
        
        # Calculate cpu_ep_count for is_hosting_cpu() check
        self.cpu_ep_count = 0
        if actual_gpu_experts < num_experts:
            # Count CPU experts
            for expert_id in range(num_experts):
                if not self.is_gpu_expert[expert_id]:
                    self.cpu_ep_count += 1
        
        # Call parent constructor with local expert information
        # Pass num_local_experts and local_expert_ids to support non-uniform distribution
        # IMPORTANT: num_experts MUST be total expert count (128) for dispatcher's local_expert_mapping
        # - Dispatcher needs full range 0-127 to handle all possible topk_ids from Router
        # - num_local_experts tells it only a subset are actually on this rank
        # - local_expert_ids provides the global IDs for those local experts
        super().__init__(
            num_experts=num_experts,  # Total experts (128) - dispatcher needs this for mapping table size
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            num_local_experts=num_local_experts,  # Actual experts on this rank (15 or 16)
            local_expert_ids=my_global_expert_ids,  # Global IDs for _global_to_local_map
        )
        
        # Don't override dispatcher - we'll bypass DeepEPMoE's SBO path entirely
        # Just use FusedMoE's simpler execution for GPU experts
        
        # Initialize CPU MoE engine for offloaded experts
        self.num_gpu_experts = actual_gpu_experts
        self.cpu_moe = None
        self.expert_map = expert_map  # Store for CPU MoE initialization
        
        # Only create cpu_moe on TP ranks that should host CPU experts
        # This prevents creating duplicate NUMA workers on all TP ranks
        if self.fast_loop_enabled:
            # Dual-loop daemon backend: weight upload + per-layer engine
            # creation is owned by sglang.srt.eplb.cpu_offload's
            # model_load_hook (which talks to the standalone cpu_expertd
            # over SHM). The legacy in-rank EPMoESparseCPUInfer path is
            # redundant in this mode and only inflates startup time +
            # RAM with a second per-layer RemoteMOE shim. Leave
            # self.cpu_moe = None; normal GPU weight loading continues
            # via the parent FusedMoE weight_loader.
            pass
        elif actual_gpu_experts < self.total_num_experts and self.is_hosting_cpu():
            # Need CPU offload and this rank should host it
            self._init_cpu_moe(self.total_num_experts, top_k, hidden_size, intermediate_size, 
                             layer_id, params_dtype, quant_config, prefix)
    
    def _try_load_expert_map_from_json(self, layer_id: int, num_experts: int) -> Optional[Dict[int, Union[int, str]]]:
        """Try to load expert map from JSON file specified by SGLANG_HETOFLOW_EXPERT_MAP.
        
        Args:
            layer_id: Current layer ID
            num_experts: Total number of experts
            
        Returns:
            Dict mapping expert_id to rank_id (int for GPU, str like 'C0' for CPU), or None if no JSON file
        """
        json_path = os.getenv("SGLANG_HETOFLOW_EXPERT_MAP")
        if not json_path:
            return None
        
        try:
            import json
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Get assignment for this layer
            layer_key = str(layer_id)
            if layer_key not in data:
                logger.warning(f"[DeepEPMoEHeto Layer {layer_id}] Layer {layer_key} not found in {json_path}, using default")
                return None
            
            rank_assignments = data[layer_key]
            if len(rank_assignments) != num_experts:
                logger.error(f"[DeepEPMoEHeto Layer {layer_id}] Expert count mismatch: "
                           f"expected {num_experts}, got {len(rank_assignments)} in JSON")
                return None
            
            # Convert list to dict: {expert_id: rank_id}
            expert_map = {expert_id: rank_id for expert_id, rank_id in enumerate(rank_assignments)}
            
            # Count GPU vs CPU experts
            gpu_count = sum(1 for r in rank_assignments if isinstance(r, int))
            cpu_count = sum(1 for r in rank_assignments if isinstance(r, str))
            logger.info(f"[DeepEPMoEHeto Layer {layer_id}] Loaded expert map from {json_path}: "
                       f"{gpu_count} GPU experts, {cpu_count} CPU experts")
            
            return expert_map
            
        except Exception as e:
            logger.error(f"[DeepEPMoEHeto Layer {layer_id}] Failed to load expert map from {json_path}: {e}")
            return None
    
    def _create_fast_loop_expert_map(self, num_experts: int) -> Dict[int, int]:
        """Dual-loop EPLB placement: every expert lives on its natural EP
        slice on GPU. The CPU daemon will preload all experts independently
        (see EPMoESparseCPUInfer.__init__), so no expert is tagged as CPU
        here; the daemon's eligibility is independent of this map.

        CONTIGUOUS BLOCK, matching ``_map_global_expert_id_to_local_expert_id``
        in the parent FusedMoE -- that function is what actually decides which
        expert's weights land in which local slot, and it uses
        ``moe_ep_rank * num_local_experts``. The same block appears in
        ``StandardDispatcher``, in ``_create_default_expert_map``'s fast-loop
        branch, in the fast loop's cpu_slot candidate filter, and in
        ``cuda_graph_offload``'s ``local_expert_lo/hi`` (which is a RANGE and
        so cannot express anything else).

        This used to return ``{eid: eid % ep_size}`` (interleaved), disagreeing
        with every one of those. It computed nothing: its only outputs are
        ``_global_to_local_map`` (written and logged, never read anywhere in
        ``python/``) and a log line. But that log line is the most
        authoritative-looking statement of expert ownership in the whole run,
        and reasoning from it yields the exact opposite of the truth -- it has
        already cost one investigation a full day and produced a confident
        report of "915 of 998 placements out of range" against a system that
        was behaving correctly. A comment saying "this is dead" does not help;
        the log line outlives the comment. So make the dead map agree with the
        live one.

        It also disarms a latent trap. The caller appends ``cpu_slot`` to
        ``my_global_expert_ids`` when it is not already a member, and the
        length of that list becomes ``num_local_experts``. Under the
        interleaved map a block-derived cpu_slot is never a member, so the
        append would fire and push the count to E/EP+1, shifting the weight
        loader's ``[rank*n, (rank+1)*n)`` window and mis-loading every expert.
        It survives only because the slot is still unset this early in startup.
        Under the block map the slot is a member and the append cannot fire.
        """
        if num_experts % self.ep_size != 0:
            raise ValueError(
                f"fast-loop expert map needs num_experts ({num_experts}) "
                f"divisible by ep_size ({self.ep_size}); the weight loader's "
                f"moe_ep_rank * num_local_experts block assumes it."
            )
        num_local = num_experts // self.ep_size
        return {eid: eid // num_local for eid in range(num_experts)}

    def _create_default_expert_map(self, num_experts: int) -> dict[int, Union[int, str]]:
        """Create the default expert placement map.
 
        Two layouts are supported:
        - Legacy cold-expert offload: when ``ep-moe-heto-gpu-experts`` is used,
          place the first N experts on GPU and the remainder on CPU hosts.
        - CPU offload fast loop: evenly place all experts on GPU first, then
          reserve one CPU slot per GPU rank on that rank's host.
        """
        if getattr(self, "fast_loop_enabled", False):
            num_ranks = self.ep_size
            assert num_experts % num_ranks == 0, (
                f"num_experts={num_experts} must be divisible by ep_size={num_ranks}"
            )
 
            experts_per_rank = num_experts // num_ranks
            block = experts_per_rank + 1
            ranks_per_host = max(1, int(self.ep_size / self.num_hosts))
 
            expert_map: dict[int, Union[int, str]] = {}
            # Pass 1: evenly shard all experts across GPU ranks.
            for rank in range(num_ranks):
                base = rank * block
                for offset in range(experts_per_rank):
                    gpu_id = (base + offset) % num_experts
                    expert_map[gpu_id] = rank
 
            # Pass 2: reserve one CPU slot per GPU rank on the corresponding host.
            for rank in range(num_ranks):
                cpu_slot_id = (rank * block + experts_per_rank) % num_experts
                host_idx = 0 if self.cpu_ep_size == 1 else min(
                    rank // ranks_per_host,
                    self.cpu_ep_size - 1,
                )
                expert_map[cpu_slot_id] = self.RANK_CPU_LIST[host_idx]
 
            gpu_count = sum(1 for value in expert_map.values() if isinstance(value, int))
            cpu_count = num_experts - gpu_count
            logger.info(
                "[DeepEPMoEHeto] default expert map (cpu offload mode): "
                "gpu_experts=%d cpu_slots=%d ep_size=%d cpu_ep_size=%d",
                gpu_count,
                cpu_count,
                self.ep_size,
                self.cpu_ep_size,
            )
            return expert_map
 
        num_gpu_experts = get_ep_moe_heto_gpu_experts()
        if num_gpu_experts == -1:
            # Legacy heto default: unspecified means all experts default to CPU.
            num_gpu_experts = 0
 
        if num_gpu_experts < 0:
            logger.warning(
                "[DeepEPMoEHeto] invalid num_gpu_experts=%d, clamped to 0",
                num_gpu_experts,
            )
            num_gpu_experts = 0
        if num_gpu_experts > num_experts:
            logger.warning(
                "[DeepEPMoEHeto] num_gpu_experts=%d exceeds num_experts=%d, clamped",
                num_gpu_experts,
                num_experts,
            )
            num_gpu_experts = num_experts
 
        expert_map: dict[int, Union[int, str]] = {}
        for expert_id in range(num_experts):
            if expert_id < num_gpu_experts:
                expert_map[expert_id] = expert_id % self.ep_size
            else:
                if self.cpu_ep_size == 1:
                    expert_map[expert_id] = self.RANK_CPU_LIST[0]
                else:
                    cpu_expert_idx = expert_id - num_gpu_experts
                    host_idx = cpu_expert_idx % self.cpu_ep_size
                    expert_map[expert_id] = self.RANK_CPU_LIST[host_idx]
 
        gpu_count = sum(1 for value in expert_map.values() if isinstance(value, int))
        cpu_count = num_experts - gpu_count
        logger.info(
            "[DeepEPMoEHeto] default expert map (legacy heto mode): "
            "gpu_experts=%d cpu_experts=%d ep_size=%d cpu_ep_size=%d",
            gpu_count,
            cpu_count,
            self.ep_size,
            self.cpu_ep_size,
        )
        return expert_map

    def _init_cpu_moe(self, num_experts, top_k, hidden_size, intermediate_size, layer_id, params_dtype, quant_config, prefix):
        """Initialize CPU MoE engine"""
        ep_rank = get_tensor_model_parallel_rank()
        ep_size = get_tensor_model_parallel_world_size()
        host_rank = int(ep_rank // (ep_size / self.num_hosts))
        
        # Use the stored expert_map from initialization
        # Initialize CPU MoE using EPMoESparseCPUInfer
        self.cpu_moe = EPMoESparseCPUInfer(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            params_dtype=params_dtype,
            renormalize=True,
            use_grouped_topk=False,
            num_expert_group=None,
            topk_group=None,
            quant_config=quant_config,
            tp_size=ep_size,
            prefix=prefix,
            correction_bias=None,
            custom_routing_function=None,
            activation="silu",
            routed_scaling_factor=None,
            use_per_token_if_dynamic=True,
            expert_map=self.expert_map,  # Use the expert_map computed during init
            ep_rank=self.RANK_CPU_LIST[int(self.host_rank / self.cpu_tp_size)],
            ep_size=self.cpu_ep_size,
            host_rank=host_rank,
            num_hosts=self.num_hosts,
        )
        logger.info(f"[DeepEPMoEHeto Layer {layer_id}] CPU MoE initialized on host_rank={host_rank}")
    
    def is_hosting_cpu(self):
        """Check if this TP rank should create CPU MoE instance.
        
        Returns True only for TP ranks divisible by (ep_size / num_hosts).
        This ensures only a subset of TP ranks fork NUMA workers.
        
        Example with TP=8, NUM_HOSTS=2:
        - ep_size / num_hosts = 8 / 2 = 4
        - Only TP0 (0 % 4 == 0) and TP4 (4 % 4 == 0) return True
        - Creates 2 groups of NUMA workers instead of 8×2=16
        
        For other TP ranks (TP1-3, TP5-7), cpu_moe=None but forward()
        correctly filters CPU expert IDs to prevent out-of-bounds access.
        """
        # Under dual-loop fast loop, the CPU daemon is *always* needed on
        # the hosting rank because it owns the full expert weight set; the
        # legacy `cpu_ep_count` gate (which counts statically-assigned CPU
        # experts) does not apply.
        if getattr(self, "fast_loop_enabled", False):
            return self.ep_rank % (self.ep_size / self.num_hosts) == 0
        return (self.ep_rank % (self.ep_size / self.num_hosts) == 0) and self.cpu_ep_count != 0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        forward_shared_experts=None,
        alt_stream=None,
        disable_sbo=False,
    ):
        if _MOE_TRACE:
            _moe_trace().hit("moe.heto.forward")
            global _moe_trace_facts_done
            if not _moe_trace_facts_done:
                _moe_trace_facts_done = True
                _moe_trace().record_moe_layer_facts(self)
        # 打印dispatch后命中的专家id分布（仅rank 0）。
        # 这里 .detach().cpu().numpy() 是同步D2H，会强制等待所有在飞的CUDA
        # kernel结束，逐层调用会让pipeline串行化（实测让每个decode step
        # 慢一个数量级）。所以默认关闭，仅在排查dispatch时通过环境变量
        # SGLANG_DEBUG_DISPATCHED_EXPERTS=1 打开。
        if self.ep_rank == 0 and os.environ.get("SGLANG_DEBUG_DISPATCHED_EXPERTS", "0") == "1":
            try:
                ids = topk_output.topk_ids.detach().cpu().numpy().flatten()
                from collections import Counter
                c = Counter(ids.tolist())
                print(f"[debug][rank=0][layer={getattr(self,'layer_id','?')}] dispatched expert id stats: {dict(c)}", flush=True)
            except Exception as e:
                print(f"[debug][rank=0] print error: {e}", flush=True)

        # If all experts on GPU, use parent class
        if self.cpu_moe is None:
            # This handles the case where cpu_moe wasn't created (e.g., non-hosting TP ranks)
            # _forward_gpu_experts will filter out CPU expert IDs if needed
            if _MOE_TRACE:
                _moe_trace().hit("moe.heto.gpu_only")
            return self._forward_gpu_experts(hidden_states, topk_output, forward_shared_experts, alt_stream, disable_sbo)

        # Decode-only gate: skip CPU offload during prefill/extend.
        # Use get_is_extend_in_batch() — the authoritative forward-mode flag
        # used by the DeepEP dispatcher — instead of a fragile token-count
        # heuristic that fails for short prompts (< 256 tokens).
        try:
            from sglang.srt.eplb.cpu_offload import get_cpu_offload_runtime
            _rt = get_cpu_offload_runtime()
            if _rt is not None and getattr(_rt.config, "cpu_offload_decode_only", True):
                from sglang.srt.layers.dp_attention import get_is_extend_in_batch
                if get_is_extend_in_batch():
                    if _MOE_TRACE:
                        _moe_trace().hit("moe.heto.decode_gate")
                    return self._forward_gpu_experts(hidden_states, topk_output, forward_shared_experts, alt_stream, disable_sbo)
        except Exception as _hexc:
            if _MOE_TRACE:
                _moe_trace().exc("ep_moe.DeepEPMoEHeto.forward/decode-gate", _hexc)

        # Heterogeneous mode: GPU + CPU
        if _MOE_TRACE:
            _moe_trace().hit("moe.heto.hetero")
        hidden_states_shape = hidden_states.shape
        hidden_states_device = hidden_states.device
        hidden_states_dtype = hidden_states.dtype

        # Mask cpu_slot expert from GPU path to prevent double-counting.
        # Replace cpu_slot ID with -1 in topk_ids (keep original weights).
        # FusedMoE normalises by sum(all 8 weights incl. w_cpu_slot), so
        # GPU computes sum_{k!=cpu_slot} w_k * expert_k(x) correctly (≈ w_k
        # unchanged when sum≈1). CPU adds w_cpu_slot * expert_cpu_slot(x).
        # This mirrors what dispatch_hook does for the non-heto DeepEPMoE path.
        gpu_topk_output = topk_output
        try:
            from sglang.srt.eplb.cpu_offload import get_cpu_offload_runtime
            _rt2 = get_cpu_offload_runtime()
            if _rt2 is not None and hasattr(_rt2, "local_cpu_slot"):
                _cpu_slot = _rt2.local_cpu_slot(self.layer_id)
                if _cpu_slot is not None and _cpu_slot >= 0:
                    if _MOE_TRACE:
                        _moe_trace().hit("moe.heto.hetero_mask")
                    from sglang.srt.layers.moe.topk import StandardTopKOutput
                    _masked_ids = topk_output.topk_ids.clone()
                    _masked_ids[topk_output.topk_ids == _cpu_slot] = -1
                    gpu_topk_output = StandardTopKOutput(
                        topk_weights=topk_output.topk_weights,
                        topk_ids=_masked_ids,
                        router_logits=topk_output.router_logits,
                    )
        except Exception as _mexc:
            if _MOE_TRACE:
                _moe_trace().exc("ep_moe.DeepEPMoEHeto.forward/inclass-mask", _mexc)

        # Prepare and enqueue CPU work (use original topk_output from Router)
        filter_hidden_states, valid_mask = self._forward_cpu_prepare(hidden_states, topk_output)
        cpu_result = self._forward_cpu_enqueue(filter_hidden_states)

        # Execute GPU experts using masked topk_output (cpu_slot excluded)
        gpu_result = self._forward_gpu_experts(hidden_states, gpu_topk_output, forward_shared_experts, alt_stream, disable_sbo)
        
        # Wait for CPU completion
        self._forward_cpu_sync(hidden_states_device)
        
        # Combine GPU and CPU results
        result = self._forward_combine(
            hidden_states_shape,
            hidden_states_device,
            hidden_states_dtype,
            gpu_result,
            cpu_result,
            valid_mask
        )
        
        return result
    
    def _forward_cpu_prepare(self, hidden_states, topk_output):
        """Prepare CPU MoE inputs using Router's pre-computed top-k selection
        
        Key difference from v0.4.6:
        - v0.4.6: CPU calls forward_prepare which does top-k selection, GPU reuses that result
        - v0.5.5: Router already did top-k selection, both CPU and GPU must use the same result
        
        This method extracts topk_ids/weights from topk_output (Router's selection),
        then applies the same data preparation steps as forward_prepare (mapping, sorting, filtering).
        """
        if self.cpu_moe:
            # Extract Router's pre-computed top-k selection (already renormalized)
            topk_weights = topk_output.topk_weights  # [num_tokens, top_k], float32
            topk_ids = topk_output.topk_ids  # [num_tokens, top_k], int32, global expert IDs
            
            # Map global expert IDs to local CPU expert IDs
            topk_ids_local = self.cpu_moe.map_expertid_to_local_experts(topk_ids)
            
            # Sort by expert ID to move -1 (invalid experts) to end if needed
            # In all-CPU mode (all experts on CPU), no -1 exists, so sorting preserves Router's weight-based order
            # _sort_topk_ids only sorts when need_sort_topk=True (i.e., when experts span multiple partitions)
            sorted_topk_weights, sorted_topk_ids = self.cpu_moe._sort_topk_ids(
                topk_weights, topk_ids_local
            )
            
            # Handle -1 expert IDs (experts not on CPU)
            # zero their weights to prevent crashes
            # Note: Always clone to avoid in-place modification, and always apply mask
            # to avoid .any() which breaks CUDA graph capture
            mask_invalid = sorted_topk_ids == -1
            sorted_topk_ids = sorted_topk_ids.clone()
            sorted_topk_weights = sorted_topk_weights.clone()
            # sorted_topk_ids[mask_invalid] = 0
            sorted_topk_weights[mask_invalid] = 0.0
            
            # Prepare tensors for CPU engine (same logic as forward_prepare)
            n_tokens = hidden_states.shape[0]
            if (
                torch.cuda.is_current_stream_capturing()
                or n_tokens <= self.cpu_moe.cached_tensors_size
            ):
                # Cached path: use pre-allocated tensors
                assert n_tokens <= self.cpu_moe.cached_tensors_size
                self.cpu_moe.sorted_topk_weights = sorted_topk_weights
                self.cpu_moe.sorted_topk_ids = sorted_topk_ids
                
                self.cpu_moe._switch_to_next_tensor_set()
                self.cpu_moe.fill_cpu_tensors(hidden_states, sorted_topk_ids, sorted_topk_weights)
                return hidden_states, None
            else:
                # Adhoc path: filter out tokens where all experts are -1
                valid_mask = sorted_topk_ids[:, 0] != -1
                if not valid_mask.any():
                    # No CPU experts needed - return early without processing
                    return None, None
                filter_hidden_states = hidden_states[valid_mask]
                
                # Prepare adhoc tensors
                self.cpu_moe.adhoc_cpu_result = torch.empty_like(
                    filter_hidden_states, device=self.cpu_moe.device, pin_memory=True
                )
                self.cpu_moe.adhoc_hidden_states_cpu = filter_hidden_states.to(
                    self.cpu_moe.device, non_blocking=True
                )
                self.cpu_moe.adhoc_sorted_topk_ids_cpu = sorted_topk_ids[valid_mask].to(
                    self.cpu_moe.device, non_blocking=True
                )
                self.cpu_moe.adhoc_sorted_topk_weights_cpu = sorted_topk_weights[valid_mask].to(
                    self.cpu_moe.device, non_blocking=True
                )
                return filter_hidden_states, valid_mask
        return None, None
    
    def _forward_cpu_enqueue(self, hidden_states):
        """Enqueue CPU MoE computation"""
        if self.cpu_moe and hidden_states is not None:
            return self.cpu_moe.forward_enqueue(hidden_states)
        return None
    
    def _forward_gpu_experts(self, hidden_states, topk_output, forward_shared_experts, alt_stream, disable_sbo):
        """Execute GPU experts using parent DeepEPMoE
        
        Filter topk_output to only include GPU experts (0 to num_gpu_experts-1).
        Other experts will be handled by CPU.
        """
        if self.num_gpu_experts == 0:
            return None
        
        # Extract topk data
        topk_weights = topk_output.topk_weights  # [num_tokens, top_k]
        topk_ids = topk_output.topk_ids  # [num_tokens, top_k]
        router_logits = topk_output.router_logits
        
        # Don't need to filter weights or remap IDs - dispatcher handles everything
        # Dispatcher's local_expert_mapping will map non-local expert IDs to -1
        # MoE kernel automatically skips experts with local_id=-1
        # Result: CPU experts (not in local_expert_ids) are naturally ignored
        
        # Create topk_output for parent class (using original weights and IDs)
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        filtered_topk_output = StandardTopKOutput(
            topk_weights=topk_weights,  # Original weights
            topk_ids=topk_ids,  # Original IDs - dispatcher handles mapping
            router_logits=router_logits,
        )
        
        # Call FusedMoE.forward directly, bypassing DeepEPMoE's SBO path
        # We don't need DeepEP features for local GPU experts
        if _MOE_TRACE:
            _moe_trace().hit("moe.heto.gpu_experts")
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        return FusedMoE.forward(self, hidden_states, filtered_topk_output)
    
    def run_moe_core(
        self,
        dispatch_output: DispatchOutput,
        down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = None,
    ):
        """Override to use FusedMoE's run_moe_core instead of DeepEPMoE's
        
        DeepEPMoE.run_moe_core expects DeepEP dispatch format, but with only 4 GPU experts
        we use standard FusedMoE path instead.
        """
        # Call FusedMoE.run_moe_core directly (skip DeepEPMoE)
        # This uses the quant_method.apply() path which works for small num_experts
        return FusedMoE.run_moe_core(self, dispatch_output)
    
    def _forward_cpu_sync(self, device):
        """Wait for CPU MoE completion"""
        if self.cpu_moe:
            self.cpu_moe.forward_sync(device)
    
    def _forward_combine(self, shape, device, dtype, gpu_result, cpu_result, valid_mask):
        """Combine GPU and CPU results"""
        if gpu_result is None:
            result = torch.zeros(shape, dtype=dtype, device=device)
        else:
            result = gpu_result
        
        if self.cpu_moe and cpu_result is not None:
            if valid_mask is not None:
                cpu_result_on_gpu = self.cpu_moe.forward_to_gpu(device, cpu_result)
                result[valid_mask] += cpu_result_on_gpu
            else:
                if getattr(self.cpu_moe, 'zero_cpy', 0) == 1:
                    result = self.cpu_moe.forward_combine(shape, cpu_result, result)
                else:
                    cpu_result_on_gpu = self.cpu_moe.forward_to_gpu(device, cpu_result)
                    result += cpu_result_on_gpu
        
        return result
    
    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
        """Load weights for both GPU and CPU experts"""
        if self.cpu_moe:
            self.cpu_moe.weight_loader(
                param, loaded_weight, weight_name, shard_id, expert_id
            )

        return super().weight_loader(
            param, loaded_weight, weight_name, shard_id, expert_id
        )


# ===== CPU MoE Classes from v0.4.6.post5 =====
# These classes enable CPU+GPU heterogeneous MoE execution


# ----- EPMoE -----
class EPMoE(torch.nn.Module):
    """
    MoE Expert Parallel Impl


    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        use_per_token_if_dynamic: bool = True,
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = get_tensor_model_parallel_rank()

        self.layer_id = layer_id
        self.num_experts = num_experts
        assert self.num_experts % self.tp_size == 0
        self.num_experts_per_partition = self.num_experts // self.tp_size
        self.start_expert_id = self.tp_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1

        self.top_k = top_k
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.correction_bias = correction_bias
        self.custom_routing_function = custom_routing_function
        self.activation = activation
        self.routed_scaling_factor = routed_scaling_factor
        self.use_per_token_if_dynamic = use_per_token_if_dynamic

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedEPMoEMethod()
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.block_shape = None
            self.activation_scheme = None
        else:
            if isinstance(self, EPMoESparseCPUInfer):
                self.quant_method: Optional[QuantizeMethodBase] = (
                    Fp8EPMoEMethodCPUInfer(quant_config)
                )
            else:
                self.quant_method: Optional[QuantizeMethodBase] = Fp8EPMoEMethod(
                    quant_config
                )
            self.use_fp8_w8a8 = True
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.block_shape = (
                self.quant_method.quant_config.weight_block_size
                if self.use_block_quant
                else None
            )
            self.fp8_dtype = torch.float8_e4m3fn
            self.activation_scheme = quant_config.activation_scheme

        self.quant_method.create_weights(
            layer=self,
            num_experts_per_partition=self.num_experts_per_partition,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

        self.grouped_gemm_runner = None

    def select_experts(self, **kwargs):
        return select_experts(**kwargs)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        # Early return if no GPU experts on this rank
        logger.debug(f"[EPMoE.forward] num_experts_per_partition={self.num_experts_per_partition}")
        if self.num_experts_per_partition == 0:
            logger.info(f"[EPMoE.forward] Skipping GPU path, num_experts_per_partition=0")
            return None
        
        hidden_states_shape = hidden_states.shape
        hidden_states_dtype = hidden_states.dtype
        hidden_states_device = hidden_states.device

        assert self.quant_method is not None
        step_layer = None
        if _STEP_TRACE_ENABLED:
            from sglang.srt.eplb.cpu_offload.step_trace import begin_layer

            step_layer = begin_layer(self.layer_id)

        if self.grouped_gemm_runner is None:
            self.grouped_gemm_runner = GroupedGemmRunner(
                hidden_states.device,
                use_flashinfer=False,  # TODO: use flashinfer
                use_per_token_if_dynamic=self.use_per_token_if_dynamic,
            )

        topk_weights, topk_ids = self.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            use_grouped_topk=self.use_grouped_topk,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            correction_bias=self.correction_bias,
            custom_routing_function=self.custom_routing_function,
            routed_scaling_factor=self.routed_scaling_factor,
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=self.layer_id,
            ),
        )

        # Dual-loop EPLB CPU-offload hook (Mode B). Splits topk so the
        # currently-assigned CPU slot expert id is masked to -1 here and
        # routed through the daemon backend in parallel with the GPU
        # compute below. ``cpu_ctx`` is None when CPU offload is disabled,
        # the slot is unset, or no layer weights are registered.
        cpu_ctx = None
        try:
            from sglang.srt.eplb.cpu_offload.dispatch_hook import begin_cpu_offload

            cpu_ctx = begin_cpu_offload(
                layer_id=self.layer_id,
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )
            if cpu_ctx is not None:
                topk_ids = cpu_ctx["masked_topk_ids"]
        except Exception as exc:
            logger.warning(
                "[cpu_offload] EPMoE.begin_cpu_offload(layer=%d) failed: %s",
                self.layer_id, exc,
            )
            cpu_ctx = None

        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
            topk_ids, self.num_experts
        )
        if _STEP_TRACE_ENABLED:
            from sglang.srt.eplb.cpu_offload.step_trace import (
                mark_dispatch_done,
                record_branches_received,
            )

            record_branches_received(
                step_layer,
                topk_ids,
                local_expert_lo=self.start_expert_id,
                local_expert_hi=self.end_expert_id + 1,
            )
            mark_dispatch_done(step_layer)

        gateup_input = torch.empty(
            (int(hidden_states.shape[0] * self.top_k), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=(
                self.fp8_dtype
                if (self.use_fp8_w8a8 and not self.use_block_quant)
                else hidden_states.dtype
            ),
        )
        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            if self.use_per_token_if_dynamic:
                max_value = torch.max(hidden_states, dim=1).values.to(torch.float32)
                self.w13_input_scale = max_value / torch.finfo(self.fp8_dtype).max
            else:
                max_value = (
                    torch.max(hidden_states)
                    .repeat(self.num_experts_per_partition)
                    .to(torch.float32)
                )
                self.w13_input_scale = max_value / torch.finfo(self.fp8_dtype).max

        # PreReorder
        pre_reorder_triton_kernel[(hidden_states.shape[0],)](
            hidden_states,
            gateup_input,
            src2dst,
            topk_ids,
            self.w13_input_scale,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states.shape[1],
            BLOCK_SIZE=512,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic,
        )
        dispose_tensor(hidden_states)

        if (
            self.activation_scheme == "dynamic"
            and not self.use_block_quant
            and self.use_per_token_if_dynamic
        ):
            scale = torch.empty(
                hidden_states_shape[0] * self.top_k,
                device=hidden_states_device,
                dtype=torch.float32,
            )
            scale[src2dst] = (
                self.w13_input_scale.unsqueeze(1)
                .expand(hidden_states_shape[0], self.top_k)
                .reshape(-1)
            )
            self.w13_input_scale = scale

        seg_indptr_cur_rank = seg_indptr[self.start_expert_id : self.end_expert_id + 2]
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_experts_per_partition,
            device=hidden_states_device,
            dtype=torch.int64,
        )
        # GroupGemm-0
        gateup_output = self.grouped_gemm_runner(
            a=gateup_input,
            b=self.w13_weight,
            c=None,
            c_dtype=hidden_states_dtype,
            batch_size=self.num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=self.use_fp8_w8a8,
            scale_a=self.w13_input_scale,
            scale_b=(
                self.w13_weight_scale_inv
                if self.use_block_quant
                else self.w13_weight_scale
            ),
            block_shape=self.block_shape,
        )
        del gateup_input

        # Act
        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            self.w2_input_scale = None
            down_input = torch.empty(
                gateup_output.shape[0],
                gateup_output.shape[1] // 2,
                device=gateup_output.device,
                dtype=hidden_states_dtype,
            )
        else:
            down_input = torch.empty(
                gateup_output.shape[0],
                gateup_output.shape[1] // 2,
                device=gateup_output.device,
                dtype=(
                    self.fp8_dtype
                    if (self.use_fp8_w8a8 and not self.use_block_quant)
                    else hidden_states_dtype
                ),
            )

        if self.activation == "silu":
            silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                self.start_expert_id,
                self.end_expert_id,
                BLOCK_SIZE=512,
            )
        elif self.activation == "gelu":
            gelu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                self.start_expert_id,
                self.end_expert_id,
                BLOCK_SIZE=512,
            )
        else:
            raise ValueError(f"Unsupported activation: {self.activation=}")
        del gateup_output

        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            if self.use_per_token_if_dynamic:
                down_input, self.w2_input_scale = sglang_per_token_quant_fp8(down_input)
            else:
                self.w2_input_scale = torch.ones(
                    self.num_experts_per_partition,
                    dtype=torch.float32,
                    device=hidden_states_device,
                )

        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            self.w2_weight.shape[1],
            device=hidden_states_device,
            dtype=hidden_states_dtype,
        )
        down_output = self.grouped_gemm_runner(
            a=down_input,
            b=self.w2_weight,
            c=down_output,
            batch_size=self.num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=self.use_fp8_w8a8,
            scale_a=self.w2_input_scale,
            scale_b=(
                self.w2_weight_scale_inv
                if self.use_block_quant
                else self.w2_weight_scale
            ),
            block_shape=self.block_shape,
        )
        del down_input

        # PostReorder
        output = torch.empty(
            hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
        )
        post_reorder_triton_kernel[(hidden_states_shape[0],)](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states_shape[1],
            BLOCK_SIZE=512,
        )
        if _STEP_TRACE_ENABLED:
            from sglang.srt.eplb.cpu_offload.step_trace import mark_local_moe_done

            mark_local_moe_done(step_layer)

        # Merge CPU-offloaded contributions (if the hook fired earlier).
        if cpu_ctx is not None:
            try:
                from sglang.srt.eplb.cpu_offload.dispatch_hook import commit_cpu_offload

                output = commit_cpu_offload(cpu_ctx, output)
            except Exception as exc:
                logger.warning(
                    "[cpu_offload] EPMoE.commit_cpu_offload(layer=%d) failed: %s",
                    self.layer_id, exc,
                )
        if _STEP_TRACE_ENABLED:
            from sglang.srt.eplb.cpu_offload.step_trace import mark_layer_done

            mark_layer_done(step_layer)
        return output

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
    ) -> List[Tuple[str, str, int, str]]:
        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                (
                    "experts.w13_"
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                    else "experts.w2_"
                ),
                f"experts.{expert_id}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        physical_expert_ids = (
            get_global_expert_location_metadata().logical_to_all_physical(
                self.layer_id, expert_id
            )
        )
        for physical_expert_id in physical_expert_ids:
            self._weight_loader_physical(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=physical_expert_id,
            )

    def _weight_loader_physical(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        if expert_id < self.start_expert_id or expert_id > self.end_expert_id:
            return
        expert_id = expert_id - self.start_expert_id

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(
                f"shard_id must be ['w1','w2','w3'] but " f"got {shard_id}."
            )

        # Special case for fp8 scales.
        if "scale" in weight_name:
            self._load_fp8_scale(
                param.data,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
            )
            return

        if shard_id == "w2":
            param.data[expert_id] = loaded_weight
        elif shard_id == "w1":
            param.data[expert_id][: self.intermediate_size, :] = loaded_weight
        elif shard_id == "w3":
            param.data[expert_id][self.intermediate_size :, :] = loaded_weight
        else:
            raise ValueError(f"Expected shard_id w1,w2 or w3 but got {shard_id}")

    def _load_fp8_scale(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        if "input_scale" in weight_name:
            if (
                (shard_id == "w1" or shard_id == "w3")
                and param_data[expert_id] != 1
                and (param_data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param_data[expert_id]} "
                    f"vs. {loaded_weight}"
                )
            param_data[expert_id] = loaded_weight
        # Weight scales
        elif "weight_scale" in weight_name:
            if self.use_block_quant:
                block_n, block_k = self.block_shape[0], self.block_shape[1]
                if shard_id == "w1":
                    param_data[expert_id][
                        : (self.intermediate_size + block_n - 1) // block_n, :
                    ] = loaded_weight
                elif shard_id == "w3":
                    param_data[expert_id][
                        (self.intermediate_size + block_n - 1) // block_n :, :
                    ] = loaded_weight
                else:  # w2
                    param_data[expert_id] = loaded_weight
            # If we are in merged column case (gate_up_proj)
            else:
                if shard_id in ("w1", "w3"):
                    # We have to keep the weight scales of w1 and w3 because
                    # we need to re-quantize w1/w3 weights after weight loading.
                    idx = 0 if shard_id == "w1" else 1
                    param_data[expert_id][idx] = loaded_weight

                # If we are in the row parallel case (down_proj)
                else:
                    param_data[expert_id] = loaded_weight



# ----- EPMoESparse -----
class EPMoESparse(EPMoE):
    # Based on EPMoE, change start_expert_id and end_expert_id to always from 0 to num of expert in this EP rank
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        use_per_token_if_dynamic: bool = True,
        expert_map: dict[int, Union[int, str]] = {},
        ep_rank: Optional[Union[int, str]] = None,
    ):
        expert_map = expert_map or self.create_default_expert_map(num_experts, tp_size)

        self.num_experts = num_experts
        if ep_rank is None:
            self.ep_rank = get_tensor_model_parallel_rank()
        else:
            self.ep_rank = ep_rank

        local_experts = [
            e_id for e_id, e_rank in expert_map.items() if e_rank == self.ep_rank
        ]
        local_expert_count = len(local_experts)
        self.expert_id_to_local = [local_expert_count] * num_experts
        for local_id, expert_id in enumerate(local_experts):
            self.expert_id_to_local[expert_id] = local_id

        self.expert_id_to_local_tensor = None

        super().__init__(
            local_expert_count,
            top_k,
            hidden_size,
            intermediate_size,
            layer_id,
            params_dtype,
            renormalize,
            use_grouped_topk,
            num_expert_group,
            topk_group,
            quant_config,
            1,  # create as if tp_size is 1 weights will be created as local_expert_count
            prefix,
            correction_bias,
            custom_routing_function,
            activation,
            routed_scaling_factor,
            use_per_token_if_dynamic,
        )

        self.num_experts = num_experts
        self.start_expert_id = 0
        self.end_expert_id = local_expert_count - 1
        assert local_expert_count == self.num_experts_per_partition
        # topk_ids is the global expert ids, we need to map them to local expert ids
        self.need_map_topk = self.num_experts_per_partition != self.num_experts

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
        local_expert_id = self.expert_id_to_local[expert_id]
        if self._in_range(local_expert_id):
            super().weight_loader(
                param, loaded_weight, weight_name, shard_id, local_expert_id
            )

    def select_experts(self, **kwargs):
        topk_weights, topk_ids = select_experts(**kwargs)

        if self.need_map_topk:
            topk_ids = self.map_expertid_to_local_experts(topk_ids)
        return topk_weights, topk_ids

    def select_experts_with_original(self, **kwargs):
        # Convert old-style kwargs to TopKConfig for v0.5.5 compatibility
        topk_config = TopKConfig(
            top_k=kwargs.pop('top_k', self.top_k),
            use_grouped_topk=kwargs.pop('use_grouped_topk', False),
            topk_group=kwargs.pop('topk_group', None),
            num_expert_group=kwargs.pop('num_expert_group', None),
            renormalize=kwargs.pop('renormalize', True),
            num_fused_shared_experts=kwargs.pop('num_fused_shared_experts', 0),
            custom_routing_function=kwargs.pop('custom_routing_function', None),
            correction_bias=kwargs.pop('correction_bias', None),
            torch_native=kwargs.pop('torch_native', False),
            routed_scaling_factor=kwargs.pop('routed_scaling_factor', None),
            apply_routed_scaling_factor_on_output=kwargs.pop('apply_routed_scaling_factor_on_output', False),
        )
        
        # Extract required parameters
        hidden_states = kwargs.get('hidden_states')
        router_logits = kwargs.get('router_logits')
        
        # select_experts now takes topk_config instead of individual parameters
        result = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=topk_config,
        )
        topk_weights = result.topk_weights
        topk_ids = result.topk_ids

        topk_weights_original = topk_weights
        topk_ids_original = topk_ids

        if self.need_map_topk:
            topk_ids = self.map_expertid_to_local_experts(topk_ids)
        return topk_weights, topk_ids, topk_weights_original, topk_ids_original

    def map_expertid_to_local_experts(self, topk_ids):
        if self.expert_id_to_local_tensor is None:
            self.expert_id_to_local_tensor = torch.tensor(
                self.expert_id_to_local, device=topk_ids.device, dtype=topk_ids.dtype
            )

        return self.expert_id_to_local_tensor[topk_ids]

    def create_default_expert_map(self, num_experts, tp_size):
        raise NotImplementedError()

    def _in_range(self, id: int):
        return self.start_expert_id <= id <= self.end_expert_id



# ----- CPU Classes -----
class EPMoESparseCPUInterface(EPMoESparse):
    def forward_start(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        raise NotImplementedError()

    def forward_sync(self):
        raise NotImplementedError()

    def forward(self):
        result = self.forward_start()
        self.forward_sync()
        return result


class EPMoESparseCPUInfer(EPMoESparseCPUInterface):
    device = "cpu"
    cpu_infer = None

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        use_per_token_if_dynamic: bool = True,
        expert_map: dict[int, Union[int, str]] = {},
        ep_rank: Optional[Union[int, str]] = None,
        ep_size: Optional[int] = None,
        host_rank: Optional[int] = None,
        num_hosts: Optional[int] = None,
    ):
        if EPMoESparseCPUInfer.cpu_infer is None:
            import moe_cpu_engine as cpu_engine

            # Phase 1/2: opt-in standalone-daemon attach.
            # When CPU_OFFLOAD_STANDALONE_DAEMON=1 (or --cpu-offload-standalone-daemon),
            # the GPU rank does NOT fork NUMA workers. Instead it attaches to
            # an externally launched `cpu_expertd` daemon (see
            # moe-cpu-engine/launch_cpu_expertd.sh) by polling a ready file.
            # See dual-loop-eplb-design.md DAEMON LIFECYCLE.
            standalone = os.getenv(
                "CPU_OFFLOAD_STANDALONE_DAEMON", "0"
            ).lower() in ("1", "true", "yes", "on")
            if standalone:
                if not hasattr(cpu_engine, "cpu_client_attach"):
                    raise RuntimeError(
                        "CPU_OFFLOAD_STANDALONE_DAEMON=1 but moe_cpu_engine "
                        "lacks cpu_client_attach binding; rebuild the extension."
                    )
                shm_prefix = os.getenv(
                    "CPU_OFFLOAD_SHM_PREFIX", "/dev/shm/sglang_cpu_offload"
                )
                timeout_ms = int(
                    os.getenv("CPU_OFFLOAD_ATTACH_TIMEOUT_MS", "30000")
                )
                # host_rank is the socket id this GPU rank should attach to.
                socket_id = int(host_rank) if host_rank is not None else 0
                daemon_pid = cpu_engine.cpu_client_attach(
                    shm_prefix=shm_prefix,
                    socket_id=socket_id,
                    timeout_ms=timeout_ms,
                )
                if daemon_pid < 0:
                    raise RuntimeError(
                        f"[EPMoESparseCPUInfer] failed to attach to cpu_expertd "
                        f"socket={socket_id} (shm_prefix={shm_prefix}, "
                        f"timeout_ms={timeout_ms}). Did launch_cpu_expertd.sh "
                        f"run?"
                    )
                logger.info(
                    f"[EPMoESparseCPUInfer] attached to cpu_expertd "
                    f"socket={socket_id} pid={daemon_pid}"
                )
                # Phase 3.2: open the slot once per rank and stash the
                # handle as a class attribute. RemoteMOE instances built
                # per-layer in _create_cpu_moe_engine reuse this handle.
                slot_id = int(os.getenv("CPU_OFFLOAD_SLOT_ID", "0"))
                handle = cpu_engine.cpu_client_open_slot(
                    shm_prefix=shm_prefix,
                    socket_id=socket_id,
                    slot_id=slot_id,
                    timeout_ms=timeout_ms,
                )
                if handle == 0:
                    raise RuntimeError(
                        f"[EPMoESparseCPUInfer] cpu_client_open_slot failed "
                        f"socket={socket_id} slot={slot_id}"
                    )
                EPMoESparseCPUInfer.cpu_slot_handle = int(handle)
                EPMoESparseCPUInfer.cpu_shm_prefix = shm_prefix
                EPMoESparseCPUInfer.cpu_socket_id = socket_id
                # Standalone-daemon mode: do NOT call numa_initialize.
                # All MoE compute is performed inside the daemon worker;
                # the local CPUInfer is still needed because it owns the
                # CUDA host-callback machinery used by RemoteMOE's
                # forward_experts return value.
                EPMoESparseCPUInfer.cpu_infer = cpu_engine.CPUInfer(0)
                EPMoESparseCPUInfer.cpu_engine = cpu_engine
                EPMoESparseCPUInfer.cpu_standalone = True
                # Phase 3.3: opt-in GPU-kernel submit/wait path. Replaces
                # the host-callback round-trip with a pair of single-thread
                # CUDA kernels operating on the cudaHostRegister'd slot.
                # When this is on, RemoteMOE.forward_experts_kernel is used
                # in place of CPUInfer.submit_with_cuda_stream.
                EPMoESparseCPUInfer.cpu_kernel_path = os.getenv(
                    "CPU_OFFLOAD_KERNEL_PATH", "0"
                ).lower() in ("1", "true", "yes", "on")
                if EPMoESparseCPUInfer.cpu_kernel_path:
                    logger.info(
                        "[EPMoESparseCPUInfer] CPU_OFFLOAD_KERNEL_PATH=1: "
                        "using GPU submit/wait kernels for daemon round-trip"
                    )
                # Dump per-layer CPU-offload token counts at process exit
                # so users can verify how much traffic actually went to CPU.
                # Gated by CPU_OFFLOAD_DUMP_TOKEN_STATS (default on for
                # standalone-daemon mode).
                if os.getenv("CPU_OFFLOAD_DUMP_TOKEN_STATS", "1").lower() in (
                    "1", "true", "yes", "on"
                ):
                    try:
                        import atexit
                        from cpu_offload_remote_moe import dump_token_stats
                        atexit.register(dump_token_stats)
                    except Exception as _e:
                        logger.warning(
                            f"[EPMoESparseCPUInfer] could not register "
                            f"token-stats dump: {_e}"
                        )
            else:
                # Legacy in-rank NUMA fork (MVP shortcut, see design doc).
                # ep_rank is C0 or C1... fetch the number from the string
                cpu_engine.numa_initialize(num_hosts, host_rank, ep_size)
                EPMoESparseCPUInfer.cpu_infer = cpu_engine.CPUInfer(0)
                EPMoESparseCPUInfer.cpu_engine = cpu_engine
                EPMoESparseCPUInfer.cpu_standalone = False

        # === DUAL-LOOP EPLB: per-daemon FULL weight preloading ===
        # Per dual-loop-eplb-design.md, every CPU daemon must preload the full
        # set of MoE expert weights. The user-supplied `expert_map` (static
        # JSON or default placement) only specifies which experts this daemon
        # is currently *eligible* to serve for token dispatch; it must not
        # gate weight allocation. We therefore pass a "full" loading map to
        # the parent (claim every expert as local on this daemon) and install
        # the real dispatch filter at the end of __init__.
        #
        # This only applies under the dual-loop fast-loop mode. The legacy
        # `--enable-ep-moe-heto` (cold-expert static slice) path keeps the
        # original partial loading semantics, where each daemon only loads
        # the experts statically assigned to it.
        self.fast_loop_enabled = is_cpu_offload_fast_loop_enabled()
        dispatch_expert_map = dict(expert_map) if expert_map else {}
        if self.fast_loop_enabled:
            loading_map = {eid: ep_rank for eid in range(num_experts)}
        else:
            loading_map = expert_map

        with torch.device(self.device):
            super().__init__(
                num_experts,
                top_k,
                hidden_size,
                intermediate_size,
                layer_id,
                params_dtype,
                renormalize,
                use_grouped_topk,
                num_expert_group,
                topk_group,
                quant_config,
                tp_size,
                prefix,
                correction_bias,
                custom_routing_function,
                activation,
                routed_scaling_factor,
                use_per_token_if_dynamic,
                loading_map,
                ep_rank,
            )
        cpu_tp_size = int((num_hosts + ep_size - 1) / ep_size)
        if cpu_tp_size > 1:
            # tp w/o comm
            self.tp_size = cpu_tp_size
            self.tp_rank = int(host_rank % cpu_tp_size)
        else:
            # will be set by cpu engine
            self.tp_size = -1
            self.tp_rank = -1
        # CPU expect out of range expert id to be -1
        for expert_id, local_id in enumerate(self.expert_id_to_local):
            if not self._in_range(local_id):
                self.expert_id_to_local[expert_id] = -1

        if isinstance(self.quant_method, Fp8EPMoEMethodCPUInfer):
            # if no need post processing, then delete to speed up.
            if (
                self.quant_method.quant_config.activation_scheme == "static"
                and self.quant_method.quant_config.is_checkpoint_fp8_serialized
            ):
                del self.quant_method
                self.quant_method = None
        self.moe_config = (
            3,  # dummy layer id
            self.num_experts_per_partition,
            self.top_k,
            hidden_size,
            intermediate_size,
            renormalize,
            num_expert_group if num_expert_group is not None else 1,
            topk_group if topk_group is not None else 1,
        )
        self.cpu_moe_engine = None

        # when not all experts are in the partition, we set -1 for experts not in the partition
        # xft requires -1 to be at end of the list, so we need to sort top k
        self.need_sort_topk = self.num_experts_per_partition != self.num_experts

        # CPUInfer doesn't need correction_bias but we need to keep the interface consistent
        self.dummy_correction_bias = torch.zeros(
            self.num_experts, device=self.device, dtype=torch.float
        )
        self.cpu_hidden_states = None
        self.cpu_sorted_topk_ids = None
        self.cpu_sorted_topk_weights = None
        self.cpu_result = None
        self.cpu_eng_stream = torch.cuda.Stream()
        self.cpu_tensor_tbo_subbatch_id = 0
        self.tensor_tbo_pool_size = 2
        # TODO: magic number, mush be more than decoding batch
        self.cached_tensors_size = int(os.getenv("NUM_CACHED_TOKS", "160"))
        self.create_cpu_tensors_if_needed_from_params(
            hidden_size, top_k, torch.bfloat16, self.cached_tensors_size
        )
        self._create_expected_weights_set()

        # === DUAL-LOOP EPLB: install dispatch filter ===
        # Weights for ALL experts have been requested above (see full_loading_map).
        # Per design, every CPU daemon holds the full weight set and may serve
        # any expert that the fast-loop / static map marks as CPU-eligible.
        # The "C0"/"C1" tag in the user map is a NUMA *routing* hint for the
        # requesting GPU rank, NOT per-expert ownership. So both daemons treat
        # every CPU-tagged expert as dispatch-eligible.
        #
        # Only applies under the dual-loop fast loop. The legacy heto path
        # keeps the original (partition-based) dispatch filter that was
        # already installed by the parent class.
        self.dispatch_expert_map = dispatch_expert_map
        if self.fast_loop_enabled:
            self.expert_id_to_local = [eid for eid in range(num_experts)]
            self.expert_id_to_local_tensor = None  # force rebuild on next forward
            self.need_map_topk = True
            self.need_sort_topk = True
            num_dispatch_experts = num_experts
            logger.info(
                f"[EPMoESparseCPUInfer Layer {layer_id}] ep_rank={ep_rank}: "
                f"dual-loop fast loop ON; loaded weights for {num_experts} experts; "
                f"all {num_dispatch_experts} experts CPU-eligible"
            )
        else:
            num_dispatch_experts = sum(
                1 for v in self.expert_id_to_local if v != -1
            )
            logger.info(
                f"[EPMoESparseCPUInfer Layer {layer_id}] ep_rank={ep_rank}: "
                f"legacy heto mode; loaded experts for static partition; "
                f"dispatch-eligible experts: {num_dispatch_experts}"
            )

    def _create_expected_weights_set(self):
        # create a set of weights names to be loaded,
        # so we can check if all weights are loaded
        # after all weights are loaded, we will set weights to CPU
        # so to avoid over occupying memory in one numa node
        self.expected_weights_set = set()
        for expert_id in range(len(self.expert_id_to_local)):
            if not self._in_range(self.expert_id_to_local[expert_id]):
                continue
            for shard_id in ["w1", "w2", "w3"]:
                self.expected_weights_set.add((expert_id, shard_id, "weight"))
                self.expected_weights_set.add((expert_id, shard_id, "weight_scale_inv"))

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        weight_key = (
            expert_id,
            shard_id,
            "weight_scale_inv" if "scale_inv" in weight_name else "weight",
        )
        if weight_key not in self.expected_weights_set:
            return

        if shard_id == "w2":
            param = (
                self.w2_weight_scale_inv
                if "scale_inv" in weight_name
                else self.w2_weight
            )
        elif shard_id in ["w1", "w3"]:
            param = (
                self.w13_weight_scale_inv
                if "scale_inv" in weight_name
                else self.w13_weight
            )
        else:
            raise ValueError(f"Unknown weight: {weight_name}")

        super().weight_loader(param, loaded_weight, weight_name, shard_id, expert_id)
        if weight_key in self.expected_weights_set:
            self.expected_weights_set.remove(weight_key)
            if len(self.expected_weights_set) == 0:
                # all weights are loaded, so we can set weights to CPU
                self._create_cpu_moe_engine()

    def _create_cpu_moe_engine(self):
        if self.cpu_moe_engine is not None:
            raise RuntimeError(
                "CPU MoE engine already created, this should not happen."
            )
        else:
            logger.info(f"Creating CPU MoE engine for layer {self.layer_id}")
            if getattr(EPMoESparseCPUInfer, "cpu_standalone", False):
                # Phase 3.2: build a RemoteMOE shim that proxies set_weights
                # and forward_experts over SHM to cpu_expertd. The shim's
                # surface area matches engine.moe.MOE so the rest of this
                # class's forward path doesn't need to know the difference.
                from cpu_offload_remote_moe import RemoteMOE
                # self.moe_config = (layer_id_dummy=3, expert_num,
                #   top_k, hidden, intermediate, renorm, n_group, topk_group)
                (_dummy_layer, expert_num, top_k, hidden, intermediate,
                 renorm, n_group, topk_group) = self.moe_config
                self.cpu_moe_engine = RemoteMOE(
                    slot_handle=EPMoESparseCPUInfer.cpu_slot_handle,
                    layer_id=int(self.layer_id),
                    expert_num=int(expert_num),
                    num_experts_per_tok=int(top_k),
                    hidden_size=int(hidden),
                    intermediate_size=int(intermediate),
                    norm_topk_prob=bool(renorm),
                    n_group=int(n_group),
                    topk_group=int(topk_group),
                    shm_prefix=EPMoESparseCPUInfer.cpu_shm_prefix,
                )
                if getattr(EPMoESparseCPUInfer, "cpu_kernel_path", False):
                    # cudaHostRegister the slot once per slot handle; the
                    # RemoteMOE caches dev pointers + per-issuer seq counter.
                    self.cpu_moe_engine.enable_kernel_path()
            else:
                self.cpu_moe_engine = self.cpu_engine.moe.MOE(
                    self.cpu_engine.moe.MOEConfig(*self.moe_config)
                )
            self.cpu_moe_engine.set_weights(
                self.w13_weight.data.data_ptr(),
                self.w2_weight.data.data_ptr(),
                self.w13_weight_scale_inv.data.data_ptr(),
                self.w2_weight_scale_inv.data.data_ptr(),
                self.dummy_correction_bias.data.data_ptr(),
                self.tp_size,
                self.tp_rank
            )
            del self.w13_weight
            del self.w2_weight
            del self.w13_weight_scale_inv
            del self.w2_weight_scale_inv
            del self.dummy_correction_bias
            gc.collect()

    def _sort_topk_ids(self, topk_weights: torch.Tensor, topk_ids: torch.Tensor):
        if self.need_sort_topk:
            # Get the sort indices (descending order)
            sorted_topk_ids, sort_indices = torch.sort(topk_ids, dim=1, descending=True)

            # Create batch indices for gathering
            batch_size = topk_ids.shape[0]
            batch_indices = (
                torch.arange(batch_size, device=topk_ids.device)
                .unsqueeze(1)
                .expand_as(sort_indices)
            )

            # Sort both tensors using the same ordering
            sorted_topk_weights = topk_weights[batch_indices, sort_indices]

            return sorted_topk_weights, sorted_topk_ids
        else:
            return topk_weights, topk_ids

    def forward_start(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        _, _, filter_hidden_states, valid_mask = self.forward_prepare(hidden_states, router_logits)
        return self.forward_enqueue(filter_hidden_states), valid_mask

    def forward_prepare(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        topk_weights, topk_ids, topk_weights_original, topk_ids_original = (
            self.select_experts_with_original(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=self.use_grouped_topk,
                renormalize=self.renormalize,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                correction_bias=self.correction_bias,
                custom_routing_function=self.custom_routing_function,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        )

        n_tokens = hidden_states.shape[0]
        if (
            torch.cuda.is_current_stream_capturing()
            or n_tokens <= self.cached_tensors_size
        ):
            assert n_tokens <= self.cached_tensors_size
            self.sorted_topk_weights, self.sorted_topk_ids = self._sort_topk_ids(
                topk_weights, topk_ids
            )
            # we are switching to next set, so prepare and enqueue must be together
            # there cannot be other prepare or enqueue in between
            # please make sure check operation strategies won't break this assumption
            self._switch_to_next_tensor_set()
            self.fill_cpu_tensors(hidden_states, self.sorted_topk_ids, self.sorted_topk_weights)
            return topk_ids_original, topk_weights_original, hidden_states, None
        else:
            sorted_topk_weights, sorted_topk_ids = self._sort_topk_ids(
                topk_weights, topk_ids
            )
            # filter out all -1 expert ids tokens, assuming the topk ids has been sorted
            valid_mask = sorted_topk_ids[:, 0] != -1
            # at least one is True, avoiding memcpy blocking
            if (not valid_mask.any()):
                valid_mask[0] = True
            filter_hidden_states = hidden_states[valid_mask]
            # these are safe as only decode will run bto heto
            self.adhoc_cpu_result = torch.empty_like(
                filter_hidden_states, device=self.device, pin_memory=True
            )
            self.adhoc_hidden_states_cpu = filter_hidden_states.to(
                self.device, non_blocking=True
            )
            self.adhoc_sorted_topk_ids_cpu = sorted_topk_ids[valid_mask].to(
                self.device, non_blocking=True
            )
            self.adhoc_sorted_topk_weights_cpu = sorted_topk_weights[valid_mask].to(
                self.device, non_blocking=True
            )
        return topk_ids_original, topk_weights_original, filter_hidden_states, valid_mask

    def forward_enqueue(self, hidden_states: torch.Tensor):
        if self.cpu_moe_engine is None:
            torch.cuda.current_stream().synchronize()
            self._create_cpu_moe_engine()

        n_tokens = hidden_states.shape[0]
        kernel_path = getattr(EPMoESparseCPUInfer, "cpu_kernel_path", False) \
            and getattr(EPMoESparseCPUInfer, "cpu_standalone", False)
        if (
            torch.cuda.is_current_stream_capturing()
            or n_tokens <= self.cached_tensors_size
        ):
            with torch.cuda.stream(self.cpu_eng_stream):
                if kernel_path:
                    self.cpu_moe_engine.forward_experts_kernel(
                        self.cpu_eng_stream.cuda_stream,
                        self.cpu_hidden_states.data_ptr(),
                        self.cpu_sorted_topk_ids.data_ptr(),
                        self.cpu_sorted_topk_weights.data_ptr(),
                        self.cpu_result.data_ptr(),
                        n_tokens,
                    )
                else:
                    self.cpu_infer.submit_with_cuda_stream(
                        #torch.cuda.current_stream(hidden_states.device).cuda_stream,
                        self.cpu_eng_stream.cuda_stream,
                        self.cpu_moe_engine.forward_experts(
                            self.cpu_hidden_states.data_ptr(),
                            self.cpu_sorted_topk_ids.data_ptr(),
                            self.cpu_sorted_topk_weights.data_ptr(),
                            self.cpu_result.data_ptr(),
                            n_tokens,
                        ),
                    )
            return self.cpu_result[:n_tokens]
        else:
            self.cpu_eng_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.cpu_eng_stream):
                if kernel_path:
                    self.cpu_moe_engine.forward_experts_kernel(
                        self.cpu_eng_stream.cuda_stream,
                        self.adhoc_hidden_states_cpu.data_ptr(),
                        self.adhoc_sorted_topk_ids_cpu.data_ptr(),
                        self.adhoc_sorted_topk_weights_cpu.data_ptr(),
                        self.adhoc_cpu_result.data_ptr(),
                        n_tokens,
                    )
                else:
                    self.cpu_infer.submit_with_cuda_stream(
                        #torch.cuda.current_stream(hidden_states.device).cuda_stream,
                        self.cpu_eng_stream.cuda_stream,
                        self.cpu_moe_engine.forward_experts(
                            self.adhoc_hidden_states_cpu.data_ptr(),
                            self.adhoc_sorted_topk_ids_cpu.data_ptr(),
                            self.adhoc_sorted_topk_weights_cpu.data_ptr(),
                            self.adhoc_cpu_result.data_ptr(),
                            n_tokens,
                        ),
                    )
            return self.adhoc_cpu_result

    def forward_sync(self, hidden_states_device):
        # Kernel-path: the GPU wait kernel already serializes against the
        # daemon on cpu_eng_stream, and the trailing pinned-host memcpy
        # was launched on the same stream. Skip the CPUInfer host-thread
        # drain (there is no host callback queued).
        if getattr(EPMoESparseCPUInfer, "cpu_kernel_path", False) \
                and getattr(EPMoESparseCPUInfer, "cpu_standalone", False):
            return
        with torch.cuda.stream(self.cpu_eng_stream):
            self.cpu_infer.sync_with_cuda_stream(
                self.cpu_eng_stream.cuda_stream
            )

    def forward_to_gpu(self, hidden_states_device, cpu_result):
        n_tokens = cpu_result.shape[0]
        # Check if cpu_result is from cached pool or adhoc allocation
        # Zero-copy only works with cached pool (self.cpu_result), not adhoc_cpu_result
        # adhoc_cpu_result is regular pinned memory, while cpu_result_ptr points to mapped memory
        is_adhoc = hasattr(self, 'adhoc_cpu_result') and self.adhoc_cpu_result is not None and cpu_result.data_ptr() == self.adhoc_cpu_result.data_ptr()
        
        if (
            (torch.cuda.is_current_stream_capturing()
            or n_tokens <= self.cached_tensors_size) and self.zero_cpy == 1 and not is_adhoc
        ):
            # Use zero-copy path for cached tensors
            with torch.cuda.stream(self.cpu_eng_stream):
                result = torch.zeros_like(cpu_result, device=hidden_states_device)
                self.cpu_infer.copy_back_to_device_async(self.cpu_eng_stream.cuda_stream, self.cpu_result_ptr,
                    result.data_ptr(), n_tokens, cpu_result.shape[1])
            torch.cuda.current_stream().wait_stream(self.cpu_eng_stream)
        else:
            # Use standard copy for adhoc tensors or when zero-copy is disabled
            torch.cuda.current_stream().wait_stream(self.cpu_eng_stream)
            result = cpu_result.to(hidden_states_device, non_blocking=True)
        return result

    def forward_combine(self, hidden_states_shape, cpu_result, result):
        self.cpu_eng_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.cpu_eng_stream):
            self.cpu_infer.combine_with_device_async(self.cpu_eng_stream.cuda_stream, self.cpu_result_ptr,
                result.data_ptr(), hidden_states_shape[0], hidden_states_shape[1])
        torch.cuda.current_stream().wait_stream(self.cpu_eng_stream)
        return result

    def forward(self):
        raise NotImplementedError("forward Not implemented")

    def create_cpu_tensors_if_needed_from_params(
        self, hidden_size, topk, dtype, batchsize
    ):
        # zero copy with mapped buffer
        self.zero_cpy = int(os.getenv("CPU_ZERO_COPY", "0"))
        # non-block copy with pin buffer
        pin = (self.zero_cpy == 0)
        if self.cpu_hidden_states is None:
            self.cpu_hidden_states_pool = [
                torch.empty(
                    (batchsize, hidden_size),
                    dtype=dtype,
                    device=self.device,
                    pin_memory=pin,
                )
                for _ in range(self.tensor_tbo_pool_size)
            ]
            self.cpu_sorted_topk_ids_pool = [
                torch.empty(
                    (batchsize, topk),
                    dtype=torch.int32,
                    device=self.device,
                    pin_memory=pin,
                )
                for _ in range(self.tensor_tbo_pool_size)
            ]
            self.cpu_sorted_topk_weights_pool = [
                torch.empty(
                    (batchsize, topk),
                    dtype=torch.float32,
                    device=self.device,
                    pin_memory=pin,
                )
                for _ in range(self.tensor_tbo_pool_size)
            ]
            self.cpu_result_pool = [
                torch.empty(
                    (batchsize, hidden_size),
                    dtype=dtype,
                    device=self.device,
                    pin_memory=pin,
                )
                for _ in range(self.tensor_tbo_pool_size)
            ]

            if not pin:
                # hard code (not been exposed yet): torch.cuda.cudart().cudaHostRegisterMapped | torch.cuda.cudart().cudaHostRegisterPortable
                flags = 0x02 | 0x01
                for tensor in (
                    self.cpu_hidden_states_pool
                    + self.cpu_sorted_topk_ids_pool
                    + self.cpu_sorted_topk_weights_pool
                    + self.cpu_result_pool
                ):
                    status = torch.cuda.cudart().cudaHostRegister(
                        tensor.data_ptr(), tensor.numel() * tensor.element_size(), flags
                    )
                    assert status == 0, "cudaHostRegister failed!"

                self.cpu_hidden_states_ptr_pool = [
                    cudaHostGetDevicePointer_wrapper(
                        tensor.data_ptr(), 0)
                    for tensor in self.cpu_hidden_states_pool
                ]
                self.cpu_sorted_topk_ids_ptr_pool = [
                    cudaHostGetDevicePointer_wrapper(
                        tensor.data_ptr(), 0)
                    for tensor in self.cpu_sorted_topk_ids_pool
                ]
                self.cpu_sorted_topk_weights_ptr_pool = [
                    cudaHostGetDevicePointer_wrapper(
                        tensor.data_ptr(), 0)
                    for tensor in self.cpu_sorted_topk_weights_pool
                ]
                self.cpu_result_ptr_pool = [
                    cudaHostGetDevicePointer_wrapper(
                        tensor.data_ptr(), 0)
                    for tensor in self.cpu_result_pool
                ]

            self._switch_to_next_tensor_set()

    def _switch_to_next_tensor_set(self):
        self.cpu_hidden_states = self.cpu_hidden_states_pool[
            self.cpu_tensor_tbo_subbatch_id
        ]
        self.cpu_sorted_topk_ids = self.cpu_sorted_topk_ids_pool[
            self.cpu_tensor_tbo_subbatch_id
        ]
        self.cpu_sorted_topk_weights = self.cpu_sorted_topk_weights_pool[
            self.cpu_tensor_tbo_subbatch_id
        ]
        self.cpu_result = self.cpu_result_pool[self.cpu_tensor_tbo_subbatch_id]

        if self.zero_cpy == 1:
            self.cpu_hidden_states_ptr = self.cpu_hidden_states_ptr_pool[
                self.cpu_tensor_tbo_subbatch_id
            ]
            self.cpu_sorted_topk_ids_ptr = self.cpu_sorted_topk_ids_ptr_pool[
                self.cpu_tensor_tbo_subbatch_id
            ]
            self.cpu_sorted_topk_weights_ptr = self.cpu_sorted_topk_weights_ptr_pool[
                self.cpu_tensor_tbo_subbatch_id
            ]
            self.cpu_result_ptr = self.cpu_result_ptr_pool[
                self.cpu_tensor_tbo_subbatch_id
            ]

        self.cpu_tensor_tbo_subbatch_id = (
            self.cpu_tensor_tbo_subbatch_id + 1
        ) % self.tensor_tbo_pool_size

    def fill_cpu_tensors(self, hidden_states, sorted_topk_ids, sorted_topk_weights):
        bs = hidden_states.shape[0]
        # TODO: cpu_moe_engine will crash if all topk_ids are -1
        # below is a hack to prevent that, but we should fix it in cpu_moe_engine
        # sorted_topk_ids[:, 0] = torch.max(
        #     sorted_topk_ids[:, 0],
        #     torch.zeros_like(sorted_topk_ids[:, 0]),
        # )
        assert bs <= self.cpu_hidden_states.shape[0]
        # need to wait for current stream to finish computing hidden_states
        self.cpu_eng_stream.wait_stream(torch.cuda.current_stream())
        if (
            (torch.cuda.is_current_stream_capturing()
            or bs <= self.cached_tensors_size) and self.zero_cpy == 1
        ):
            with torch.cuda.stream(self.cpu_eng_stream):
                self.cpu_infer.copy_into_host_async(self.cpu_eng_stream.cuda_stream,
                                        hidden_states.data_ptr(), sorted_topk_ids.data_ptr(), sorted_topk_weights.data_ptr(),
                                        self.cpu_hidden_states_ptr, self.cpu_sorted_topk_ids_ptr, self.cpu_sorted_topk_weights_ptr,
                                        bs, hidden_states.shape[1], self.top_k)
        else:
            with torch.cuda.stream(self.cpu_eng_stream):
                self.cpu_hidden_states[:bs].copy_(hidden_states, non_blocking=True)
                self.cpu_sorted_topk_ids[:bs].copy_(sorted_topk_ids, non_blocking=True)
                self.cpu_sorted_topk_weights[:bs].copy_(sorted_topk_weights, non_blocking=True)



# ----- EPMoEHeto -----
class EPMoEHeto(EPMoESparse):
    # assuming the max number of sockets is 4
    RANK_CPU_LIST = ["C0", "C1", "C2", "C3"]

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        use_per_token_if_dynamic: bool = True,
        expert_map: dict[int, Union[int, str]] = {},
        num_gpu_experts=-1,
    ):
        self.layer_id = layer_id
        # Get the CPU expert size from the environment variable, default to 1 if not set
        self.num_hosts = int(os.getenv("CPU_NUM_HOSTS", "1"))
        self.cpu_ep_size = int(os.getenv("CPU_EP_SIZE", "1"))
        if (self.num_hosts == 1): self.num_hosts = self.cpu_ep_size

        # Get num_gpu_experts from global config if not specified
        if num_gpu_experts == -1:
            num_gpu_experts = get_ep_moe_heto_gpu_experts()
            logger.info(f"[EPMoEHeto Layer {layer_id}] Using configured num_gpu_experts={num_gpu_experts}")

        expert_map = expert_map or self.create_default_expert_map(
            num_experts, tp_size, num_gpu_experts
        )

        super().__init__(
            num_experts,
            top_k,
            hidden_size,
            intermediate_size,
            layer_id,
            params_dtype,
            renormalize,
            use_grouped_topk,
            num_expert_group,
            topk_group,
            quant_config,
            tp_size,
            prefix,
            correction_bias,
            custom_routing_function,
            activation,
            routed_scaling_factor,
            use_per_token_if_dynamic,
            expert_map,
        )
        self.ep_size = tp_size or get_tensor_model_parallel_world_size()
        if (self.num_hosts > len(self.RANK_CPU_LIST) or self.num_hosts > self.ep_size):
            raise ValueError(
                f"CPU_NUM_HOSTS {self.num_hosts} exceeds the max supported {len(self.RANK_CPU_LIST)} \
                or the number of gpus {self.ep_size}"
            )
        if (self.cpu_ep_size != 1 and self.cpu_ep_size != self.num_hosts):
            raise ValueError(
                f"CPU_EP_SIZE {self.cpu_ep_size} must be set as 1 or equal to {self.num_hosts}"
            )
        self.host_rank = int(self.ep_rank // (self.ep_size / self.num_hosts))
        self._init_cpu_resources(expert_map)

        if self.is_hosting_cpu():
            self.cpu_moe = EPMoESparseCPUInfer(
                num_experts,
                top_k,
                hidden_size,
                intermediate_size,
                layer_id,
                params_dtype,
                renormalize,
                use_grouped_topk,
                num_expert_group,
                topk_group,
                quant_config,
                tp_size,
                prefix,
                correction_bias,
                custom_routing_function,
                activation,
                routed_scaling_factor,
                use_per_token_if_dynamic,
                expert_map,
                EPMoEHeto.RANK_CPU_LIST[0 if self.cpu_ep_size == 1 else self.host_rank],
                self.cpu_ep_size,
                self.host_rank,
                self.num_hosts,
            )
        
        # Override num_experts_per_partition to reflect GPU experts only
        # Count how many experts in expert_map belong to current GPU rank
        gpu_expert_count = sum(1 for x in expert_map.values() if isinstance(x, int) and x == self.ep_rank)
        self.num_experts_per_partition = gpu_expert_count
        # logger.info(f"[EPMoEHeto Layer {layer_id}] ep_rank={self.ep_rank}, expert_map={expert_map}, gpu_expert_count={gpu_expert_count}, num_experts_per_partition={self.num_experts_per_partition}")
        
        # Parent class EPMoE/EPMoESparse will create weights based on num_experts_per_partition
        # So only GPU experts (4) will have weights allocated on GPU, not all 128 experts

    def _init_cpu_resources(self, expert_map):
        self.cpu_ep_count = sum(1 for x in expert_map.values()
                if x == self.RANK_CPU_LIST[0 if self.cpu_ep_size == 1 else self.host_rank])
        self.cpu_moe = None
        self.cpu_stream = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        op_shared_experts: Optional[Callable] = None,
    ):
        # Extract router_logits from topk_output
        router_logits = topk_output.router_logits
        
        # hidden_states will be destroyed in the process, so we need to save its properties
        # for later use
        hidden_states_shape = hidden_states.shape
        hidden_states_device = hidden_states.device
        hidden_states_dtype = hidden_states.dtype
        filter_hidden_states, valid_mask = self.forward_routed_experts_prepare(hidden_states, router_logits)
        cpu_result = self.forward_routed_experts_enqueue(filter_hidden_states, router_logits)
        shared_output = None
        if op_shared_experts is not None:
            shared_output = op_shared_experts(hidden_states)
        gpu_result = self.forward_routed_experts_maybe_gpu(
            hidden_states, router_logits  # here hidden_states will be destroyed
        )
        self.forward_routed_experts_sync(
            hidden_states_device,
        )
        result = self.forward_routed_experts_combine(
            hidden_states_shape,
            hidden_states_device,
            hidden_states_dtype,
            gpu_result,
            cpu_result,
            valid_mask
        )
        # Return only result tensor, not tuple
        # (shared_output is not used in current Qwen3 MoE implementation)
        return result

    def forward_routed_experts_prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ):
        if self.cpu_moe:
            topk_ids, topk_weights, filter_hidden_states, valid_mask = self.cpu_moe.forward_prepare(
                hidden_states, router_logits
            )
            self._forward_prepare_result = (topk_weights, topk_ids)
            return filter_hidden_states, valid_mask
        else:
            self._forward_prepare_result = (None, None)
            return None, None

    def forward_routed_experts_enqueue(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ):
        if self.cpu_moe:
            return self.cpu_moe.forward_enqueue(hidden_states)
        return None

    def forward_routed_experts_maybe_gpu(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ):
        if self.num_experts_per_partition == 0:
            return None
            
        # Use FusedMoE's execution flow with parent class weights
        # Parent EPMoE class has created weights for num_experts_per_partition (4) GPU experts
        from sglang.srt.layers.moe.topk import TopKConfig
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
        
        topk_config = TopKConfig(
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
        )
        topk_output = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=topk_config,
        )
        
        # Call fused_moe directly with parent class weights (w13_weight, w2_weight)
        # These weights were created by EPMoE.__init__ for only GPU experts
        return fused_moe(
            hidden_states=hidden_states,
            w1=self.w13_weight,
            w2=self.w2_weight,
            topk_output=topk_output,
            use_fp8_w8a8=True,
            w1_scale=self.w13_weight_scale_inv,
            w2_scale=self.w2_weight_scale_inv,
        )
        return None

    def select_experts(self, **kwargs):
        topk_weights, topk_ids = None, None
        if hasattr(self, "_forward_prepare_result"):
            topk_weights, topk_ids = self._forward_prepare_result
            # release buffer after reuse, currently it will only be used once
            self._forward_prepare_result = (None, None)
        if topk_weights is None or topk_ids is None:
            topk_weights, topk_ids = select_experts(**kwargs)

        if self.need_map_topk:
            topk_ids = self.map_expertid_to_local_experts(topk_ids)
        return topk_weights, topk_ids

    def forward_routed_experts_sync(
        self,
        hidden_states_device,
    ):
        if self.cpu_moe:
            self.cpu_moe.forward_sync(hidden_states_device)

    def forward_routed_experts_combine(
        self,
        hidden_states_shape,
        hidden_states_device,
        hidden_states_dtype,
        gpu_result,
        cpu_result,
        valid_mask
    ):
        if gpu_result is None:
            result = torch.zeros(
                hidden_states_shape,
                dtype=hidden_states_dtype,
                device=hidden_states_device,
            )
        else:
            result = gpu_result

        if self.cpu_moe:
            if valid_mask is not None:
                cpu_result_on_gpu = self.cpu_moe.forward_to_gpu(
                    hidden_states_device, cpu_result
                )
                result[valid_mask] += cpu_result_on_gpu
            else:
                if self.cpu_moe.zero_cpy == 1:
                    result = self.cpu_moe.forward_combine(
                        hidden_states_shape, cpu_result, result)
                else:
                    cpu_result_on_gpu = self.cpu_moe.forward_to_gpu(
                        hidden_states_device, cpu_result
                    )
                    result += cpu_result_on_gpu

        return result

    def is_hosting_cpu(self):
        return (self.ep_rank % (self.ep_size / self.num_hosts) == 0) and self.cpu_ep_count != 0

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
        if self.cpu_moe:
            self.cpu_moe.weight_loader(
                param, loaded_weight, weight_name, shard_id, expert_id
            )

        return super().weight_loader(
            param, loaded_weight, weight_name, shard_id, expert_id
        )

    def _try_load_expert_map(self, num_experts):
        hetomap_env = os.environ.get("SGLANG_HETOFLOW_EXPERT_MAP", "")
        if hetomap_env:
            logger.info(f"Loading expert mapping from {hetomap_env}")
            with open(hetomap_env, "r") as f:
                mapping_plan = json.load(f)
            # mapping_plan: {layer_id: {expert_id: rank_id, ...}, ...}
            # Use self.layer_id to select the mapping for this layer
            layer_map = mapping_plan.get(str(self.layer_id), None)
            if layer_map is None:
                raise ValueError(
                    f"No expert mapping found for layer_id {self.layer_id} in {hetomap_env}"
                )
            # Convert keys to int if needed
            assert isinstance(layer_map, list), "Expected layer_map to be a list"
            assert (
                len(layer_map) == num_experts
            ), f"Expert map length mismatch expected {num_experts}, got {len(layer_map)}"
            expert_map_plan = {i: layer_map[i] for i in range(num_experts)}
            return expert_map_plan
        return None

    def create_default_expert_map(self, num_experts, tp_size, num_gpu_experts):
        expert_map_plan = self._try_load_expert_map(num_experts)
        if expert_map_plan is None:
            # half on CPU, other half set even on GPUs
            ep_size = tp_size or get_tensor_model_parallel_world_size()
            expert_map_plan = dict()
            gpu_in_high_part = False
            if num_gpu_experts < 0:
                gpu_in_high_part = True
                gpu_expert_total = -num_gpu_experts
            else:
                gpu_expert_total = num_gpu_experts
            logger.debug(
                f"create_default_expert_map, gpu_expert_total:{gpu_expert_total}"
            )
            if gpu_in_high_part:
                gpu_range = range(num_experts - gpu_expert_total, num_experts)
                cpu_range = range(0, num_experts - gpu_expert_total)
            else:
                gpu_range = range(0, gpu_expert_total)
                cpu_range = range(gpu_expert_total, num_experts)
            for e_id in cpu_range:
                # Map CPU experts to their ranks
                expert_map_plan[e_id] = self.RANK_CPU_LIST[e_id % self.cpu_ep_size]
            for ep_rank, r in enumerate(np.array_split(list(gpu_range), ep_size)):
                for e_id in r:
                    expert_map_plan[int(e_id)] = ep_rank
        return expert_map_plan




# ===== Modified get_moe_impl_class to support Heto =====

def get_moe_impl_class(quant_config: Optional[QuantizationConfig] = None):
    """
    Get MoE implementation class based on configuration.
    Now supports CPU MoE Offload (Heto mode).
    """
    # Check for Heto mode (CPU+GPU heterogeneous MoE)
    # Either flag routes to DeepEPMoEHeto; they select different runtime behaviors:
    #   * --enable-ep-moe-heto: legacy static cold-expert offload
    #     (uses --ep-moe-heto-gpu-experts to slice experts GPU vs CPU).
    #   * --enable-cpu-offload-fast-loop: dual-loop EPLB CPU offload
    #     (see dual-loop-eplb-design.md). Every CPU daemon preloads the
    #     full expert weight set; GPU placement is the natural EP slice.
    if is_ep_moe_heto_enabled() or is_cpu_offload_fast_loop_enabled():
        if is_cpu_offload_fast_loop_enabled():
            logger.info(
                "Using DeepEPMoEHeto for dual-loop EPLB CPU offload "
                "(fast loop enabled)"
            )
        else:
            logger.info("Using DeepEPMoEHeto for CPU+GPU heterogeneous MoE")
        return DeepEPMoEHeto
    
    # Original v0.5.5 logic
    if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
        return DeepEPMoE

    if get_moe_runner_backend().is_flashinfer_trtllm():
        if quant_config is None:
            return FusedMoE
        try:
            if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
                from sglang.srt.layers.moe.fused_moe_triton.layer import (
                    FlashInferFP4MoE,
                )
                return FlashInferFP4MoE
        except:
            pass

    if get_moe_runner_backend().is_flashinfer_trtllm() and quant_config is not None:
        return FlashInferFusedMoE
    if get_moe_runner_backend().is_flashinfer_cutlass():
        return FusedMoE
    return FusedMoE


# ===== FP8 Quantization Methods for EP MoE =====
class Fp8EPMoEMethod(Fp8MoEMethod):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None
        
    def create_weights(
        self,
        layer: Module,
        num_experts_per_partition: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn

        tp_size = get_tensor_model_parallel_world_size()
        if self.block_quant:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            if intermediate_size % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1:
                if intermediate_size % block_k != 0:
                    raise ValueError(
                        f"The input_size of down's weight = "
                        f"{intermediate_size} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                2 * intermediate_size,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                hidden_size,
                intermediate_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.block_quant:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts_per_partition,
                    2 * ((intermediate_size + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts_per_partition,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.quant_config.activation_scheme == "dynamic"
        else:
            # WEIGHT_SCALES
            # Allocate 2 scales for w1 and w3 respectively.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, 2, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        
        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8."
                )

            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)
            
            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        # If checkpoint is fp16, quantize in place.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            fp8_dtype = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)
            
            layer.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    layer.num_experts_per_partition,
                    dtype=torch.float32,
                    device=w13_weight.device,
                ),
                requires_grad=False,
            )

            for expert in range(layer.num_experts_per_partition):
                w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                    scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                )
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            return

        else:
            if self.quant_config.activation_scheme == "static":
                if layer.w13_input_scale is None or layer.w2_input_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None."
                    )
                layer.w13_weight_scale = torch.nn.Parameter(
                    torch.max(layer.w13_weight_scale, dim=1).values,
                    requires_grad=False,
                )
            return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class Fp8EPMoEMethodCPUInfer(Fp8MoEMethod):
    """MoE method for FP8 for CPU infer.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Difference to Fp8MoEMethod is to create tensor instead of parameters
    as we are not able to release memory if data are stored in parameters
    for unknown reason.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None
        
    def create_weights(
        self,
        layer: Module,
        num_experts_per_partition: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn

        tp_size = get_tensor_model_parallel_world_size()
        if self.block_quant:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            if intermediate_size % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1:
                if intermediate_size % block_k != 0:
                    raise ValueError(
                        f"The input_size of down's weight = "
                        f"{intermediate_size} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )

        # WEIGHTS - Create tensors instead of parameters for CPU
        w13_weight = torch.empty(
            num_experts_per_partition,
            2 * intermediate_size,
            hidden_size,
            dtype=params_dtype,
            device="cpu",
        )
        layer.w13_weight = w13_weight
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.empty(
            num_experts_per_partition,
            hidden_size,
            intermediate_size,
            dtype=params_dtype,
            device="cpu",
        )
        layer.w2_weight = w2_weight
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.block_quant:
            w13_weight_scale = torch.ones(
                num_experts_per_partition,
                2 * ((intermediate_size + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            )
            w2_weight_scale = torch.ones(
                num_experts_per_partition,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size + block_k - 1) // block_k,
                dtype=torch.float32,
            )
            layer.w13_weight_scale_inv = w13_weight_scale
            layer.w2_weight_scale_inv = w2_weight_scale
            assert self.quant_config.activation_scheme == "dynamic"
        else:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, 2, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        
        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8."
                )

            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)
            
            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        # If checkpoint is fp16, quantize in place.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            fp8_dtype = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)
            
            layer.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    layer.num_experts_per_partition,
                    dtype=torch.float32,
                    device=w13_weight.device,
                ),
                requires_grad=False,
            )

            for expert in range(layer.num_experts_per_partition):
                w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                    scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                )
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                )
            layer.w13_weight = w13_weight
            layer.w2_weight = w2_weight
            return

        else:
            if self.quant_config.activation_scheme == "static":
                if layer.w13_input_scale is None or layer.w2_input_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None."
                    )
                layer.w13_weight_scale = torch.nn.Parameter(
                    torch.max(layer.w13_weight_scale, dim=1).values,
                    requires_grad=False,
                )
            return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
