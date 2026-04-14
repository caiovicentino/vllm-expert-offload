# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from enum import Enum

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
)

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    BatchedMarlinExperts,
    MarlinExperts,
    fused_marlin_moe,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_TYPES_MAP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_mxint4_moe import (
    flashinfer_trtllm_mxint4_moe,
    is_flashinfer_mxint4_moe_available,
    prepare_static_weights_for_trtllm_mxint4_moe,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
    marlin_act_int8_process_scales,
    marlin_make_workspace_new,
    marlin_moe_permute_scales,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

logger = init_logger(__name__)


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


class CompressedTensorsWNA16MarlinMoEMethod(CompressedTensorsMoEMethod):
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs | None,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super().__init__(moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        assert weight_quant.symmetric, (
            "Only symmetric quantization is supported for MoE"
        )
        # Extract properties from weight_quant
        self.num_bits = weight_quant.num_bits
        self.packed_factor = 32 // weight_quant.num_bits
        self.strategy = weight_quant.strategy
        self.group_size = weight_quant.group_size
        self.actorder = weight_quant.actorder

        self.quant_type = WNA16_SUPPORTED_TYPES_MAP[self.num_bits]

        self.marlin_input_dtype = get_marlin_input_dtype(layer_name)
        self.use_flashinfer_mxint4_moe = (
            is_flashinfer_mxint4_moe_available()
            and self.group_size == 32
            and weight_quant.num_bits == 4
        )
        self.kernel_backend = (
            "Flashinfer" if self.use_flashinfer_mxint4_moe else "Marlin"
        )
        logger.info_once(
            f"Using {self.kernel_backend} backend for WNA16 MoE "
            f"(group_size={self.group_size}, num_bits={self.num_bits})",
            scope="local",
        )
        # Set in create_weights after we know the layer's expert cache size.
        # Read by maybe_make_prepare_finalize to short-circuit the modular
        # kernel wrap path when the expert LRU cache is active.
        self._cache_active_hint: bool = False

    @property
    def supports_expert_lru_cache(self) -> bool:
        # Cache path is only wired up for the Marlin backend (non-Flashinfer)
        # and for the non-actorder + non-8bit-input variants.  Those are the
        # common config for compressed-tensors INT4 pack-quantized MoE.
        return (
            self.kernel_backend == "Marlin"
            and not self.actorder
            and (
                self.marlin_input_dtype is None
                or self.marlin_input_dtype.itemsize != 1
            )
        )

    def get_weight_shape(
        self,
        weight_name: str,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        num_groups_w2: int | None = None,
        num_groups_w13: int | None = None,
    ) -> tuple[int, int, int]:
        """
        Get the shape of the weight based on the weight name, number of experts
        hidden size, intermediate size per partition, number of groups for w2,
        and number of groups for w13. Pass in num_groups_w2 and num_groups_w13
        for weight scales.
        """
        if weight_name == "w13_scale":
            assert num_groups_w13 is not None, (
                "num_groups_w13 must be provided for weight scales"
            )
        if weight_name == "w2_scale":
            assert num_groups_w2 is not None, (
                "num_groups_w2 must be provided for weight scales"
            )
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1
        shape_map = {
            "w13_weight": {
                "Flashinfer": (
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    hidden_size // self.packed_factor,
                ),
                "Marlin": (
                    num_experts,
                    hidden_size // self.packed_factor,
                    w13_num_shards * intermediate_size_per_partition,
                ),
            },
            "w13_scale": {
                "Flashinfer": (
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    num_groups_w13,
                ),
                "Marlin": (
                    num_experts,
                    num_groups_w13,
                    w13_num_shards * intermediate_size_per_partition,
                ),
            },
            "w2_weight": {
                "Flashinfer": (
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // self.packed_factor,
                ),
                "Marlin": (
                    num_experts,
                    intermediate_size_per_partition // self.packed_factor,
                    hidden_size,
                ),
            },
            "w2_scale": {
                "Flashinfer": (num_experts, hidden_size, num_groups_w2),
                "Marlin": (num_experts, num_groups_w2, hidden_size),
            },
        }
        return shape_map[weight_name][self.kernel_backend]

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        intermediate_size_full = extra_weight_attrs.pop("intermediate_size_full")

        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        is_transposed = self.kernel_backend != "Flashinfer"
        extra_weight_attrs.update(
            {"is_transposed": is_transposed, "quant_method": self.strategy}
        )

        # Expert LRU cache: allocate big per-expert tensors in CPU pinned
        # memory so that checkpoint loading never allocates GPU memory for
        # them.  Process_weights_after_loading then does a per-chunk Marlin
        # repack on a small GPU scratch buffer and initializes the provider.
        # See supports_expert_lru_cache for the supported config subset.
        use_cpu_pinned = (
            self.supports_expert_lru_cache
            and getattr(layer, "_moe_expert_cache_size", 0) > 0
        )
        self._cache_active_hint = use_cpu_pinned

        def _empty_packed(shape):
            if use_cpu_pinned:
                return torch.empty(
                    *shape, dtype=torch.int32, device="cpu"
                ).pin_memory()
            return torch.empty(*shape, dtype=torch.int32)

        w13_weight = torch.nn.Parameter(
            _empty_packed(
                self.get_weight_shape(
                    "w13_weight",
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                )
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            _empty_packed(
                self.get_weight_shape(
                    "w2_weight",
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                )
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # In the case where we have actorder/g_idx,
        # we do not partition the w2 scales
        load_full_w2 = self.actorder and self.group_size != -1
        w2_scales_size = (
            intermediate_size_full if load_full_w2 else intermediate_size_per_partition
        )

        self.is_k_full = (not self.actorder) or (
            intermediate_size_per_partition == intermediate_size_full
        )

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        layer.num_groups_w13 = num_groups_w13
        layer.num_groups_w2 = num_groups_w2

        def _ones_scale(shape):
            if use_cpu_pinned:
                return torch.ones(
                    *shape, dtype=params_dtype, device="cpu"
                ).pin_memory()
            return torch.ones(*shape, dtype=params_dtype)

        w13_scale = torch.nn.Parameter(
            _ones_scale(
                self.get_weight_shape(
                    "w13_scale",
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                    num_groups_w13=num_groups_w13,
                )
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            _ones_scale(
                self.get_weight_shape(
                    "w2_scale",
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                    num_groups_w2=num_groups_w2,
                )
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": load_full_w2})

        w2_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )

        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None
        layer.marlin_state = GPTQMarlinState.REPACK

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_experts = layer.w13_weight_g_idx.shape[0]
        device = layer.w13_weight_g_idx.device

        # Fast path: cache active → Marlin repack on GPU scratch, results
        # back in CPU pinned, then wire up the CachedWeightProvider.
        cache_active = (
            self.supports_expert_lru_cache
            and getattr(layer, "_moe_expert_cache_size", 0) > 0
            and layer.w13_weight_packed.device.type == "cpu"
        )
        if cache_active:
            self._process_weights_after_loading_offloaded(layer)
            return

        if self.kernel_backend == "Flashinfer":
            dict_weights_mxint4 = prepare_static_weights_for_trtllm_mxint4_moe(
                layer.w13_weight_packed,
                layer.w13_weight_scale,
                layer.w2_weight_packed,
                layer.w2_weight_scale,
            )
            replace_parameter(
                layer, "w13_weight_packed", dict_weights_mxint4["gemm1_weights"]
            )
            replace_parameter(
                layer, "w13_weight_scale", dict_weights_mxint4["gemm1_scales"]
            )
            replace_parameter(
                layer, "w2_weight_packed", dict_weights_mxint4["gemm2_weights"]
            )
            replace_parameter(
                layer, "w2_weight_scale", dict_weights_mxint4["gemm2_scales"]
            )
            return None

        is_a_8bit = (
            self.marlin_input_dtype is not None
            and self.marlin_input_dtype.itemsize == 1
        )

        if self.marlin_input_dtype == torch.float8_e4m3fn:
            # NOTE: for non-zp quantization format only
            ops.marlin_int4_fp8_preprocess(layer.w13_weight_packed, inplace=True)
            ops.marlin_int4_fp8_preprocess(layer.w2_weight_packed, inplace=True)
            layer.w13_weight_scale.data = layer.w13_weight_scale.data * 512
            layer.w2_weight_scale.data = layer.w2_weight_scale.data * 512

        # when running models with grouped act order,
        # resort to g_idx values provided in checkpoint
        if self.actorder == "group":
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_weight_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_weight_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_weight_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_weight_g_idx)

            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_weight_g_idx[e]).to(
                    torch.int32
                )
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_weight_g_idx[e]).to(
                    torch.int32
                )
                w13_sorted_g_idx[e] = layer.w13_weight_g_idx[e][
                    w13_g_idx_sort_indices[e]
                ]
                w2_sorted_g_idx[e] = layer.w2_weight_g_idx[e][w2_g_idx_sort_indices[e]]

            replace_parameter(layer, "w13_weight_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_weight_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)

        else:
            layer.w13_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )

        marlin_w13_qweight = ops.gptq_marlin_moe_repack(
            layer.w13_weight_packed,
            layer.w13_g_idx_sort_indices,
            layer.w13_weight_packed.shape[1] * self.packed_factor,
            layer.w13_weight_packed.shape[2],
            self.num_bits,
            is_a_8bit=is_a_8bit,
        )
        replace_parameter(layer, "w13_weight_packed", marlin_w13_qweight)

        marlin_w2_qweight = ops.gptq_marlin_moe_repack(
            layer.w2_weight_packed,
            layer.w2_g_idx_sort_indices,
            layer.w2_weight_packed.shape[1] * self.packed_factor,
            layer.w2_weight_packed.shape[2],
            self.num_bits,
            is_a_8bit=is_a_8bit,
        )
        replace_parameter(layer, "w2_weight_packed", marlin_w2_qweight)

        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            s=layer.w13_weight_scale,
            size_k=layer.w13_weight_packed.shape[2],
            size_n=layer.w13_weight_scale.shape[2],
            group_size=self.group_size,
            is_a_8bit=is_a_8bit,
        )
        if self.marlin_input_dtype == torch.int8 and layer.num_groups_w13 > 1:
            marlin_w13_scales, w13_input_global_scale = marlin_act_int8_process_scales(
                marlin_w13_scales
            )
            layer.register_parameter(
                "w13_input_global_scale",
                torch.nn.Parameter(w13_input_global_scale, requires_grad=False),
            )
        replace_parameter(layer, "w13_weight_scale", marlin_w13_scales)

        marlin_w2_scales = marlin_moe_permute_scales(
            s=layer.w2_weight_scale,
            size_k=layer.w2_weight_scale.shape[1]
            * (self.group_size if self.group_size != -1 else self.packed_factor),
            size_n=layer.w2_weight_scale.shape[2],
            group_size=self.group_size,
            is_a_8bit=is_a_8bit,
        )
        if self.marlin_input_dtype == torch.int8 and layer.num_groups_w2 > 1:
            marlin_w2_scales, w2_input_global_scale = marlin_act_int8_process_scales(
                marlin_w2_scales
            )
            layer.register_parameter(
                "w2_input_global_scale",
                torch.nn.Parameter(w2_input_global_scale, requires_grad=False),
            )
        replace_parameter(layer, "w2_weight_scale", marlin_w2_scales)

        layer.workspace = marlin_make_workspace_new(device, 4)

    def _process_weights_after_loading_offloaded(
        self, layer: torch.nn.Module
    ) -> None:
        """Run Marlin repack on CPU-pinned weights via a GPU scratch copy,
        then instantiate the CachedWeightProvider and release the full CPU
        tensors from the layer parameters (the provider keeps its own refs).
        """
        from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
            CachedWeightProvider,
        )

        num_experts = layer.w13_weight_packed.shape[0]
        gpu = torch.accelerator.current_accelerator()
        assert gpu is not None, (
            "expert offload requires an accelerator to run marlin repack"
        )

        # Empty g_idx + sort_indices on GPU (non-actorder path)
        empty_g_idx = torch.empty(
            (num_experts, 0), dtype=torch.int32, device=gpu
        )
        empty_sort = torch.empty(
            (num_experts, 0), dtype=torch.int32, device=gpu
        )

        # Stage packed weights on GPU, repack, copy back to CPU pinned.
        # The full-expert repack is the simplest correct path; memory is
        # bounded to one layer's worth (~5-10 GB on MiniMax-scale MoE).
        w13_gpu = layer.w13_weight_packed.data.to(gpu, non_blocking=True)
        w2_gpu = layer.w2_weight_packed.data.to(gpu, non_blocking=True)

        marlin_w13 = ops.gptq_marlin_moe_repack(
            w13_gpu,
            empty_sort,
            w13_gpu.shape[1] * self.packed_factor,
            w13_gpu.shape[2],
            self.num_bits,
            is_a_8bit=False,
        )
        marlin_w2 = ops.gptq_marlin_moe_repack(
            w2_gpu,
            empty_sort,
            w2_gpu.shape[1] * self.packed_factor,
            w2_gpu.shape[2],
            self.num_bits,
            is_a_8bit=False,
        )

        # Move repacked to CPU pinned (where the cache provider wants them)
        marlin_w13_cpu = torch.empty(
            marlin_w13.shape, dtype=marlin_w13.dtype, device="cpu"
        ).pin_memory()
        marlin_w13_cpu.copy_(marlin_w13, non_blocking=False)
        marlin_w2_cpu = torch.empty(
            marlin_w2.shape, dtype=marlin_w2.dtype, device="cpu"
        ).pin_memory()
        marlin_w2_cpu.copy_(marlin_w2, non_blocking=False)
        del w13_gpu, w2_gpu, marlin_w13, marlin_w2

        # Stage scales on GPU for permute, then back to CPU pinned
        w13_scale_gpu = layer.w13_weight_scale.data.to(gpu, non_blocking=True)
        w2_scale_gpu = layer.w2_weight_scale.data.to(gpu, non_blocking=True)

        marlin_w13_scales = marlin_moe_permute_scales(
            s=w13_scale_gpu,
            size_k=layer.w13_weight_packed.shape[1] * self.packed_factor,
            size_n=w13_scale_gpu.shape[2],
            group_size=self.group_size,
            is_a_8bit=False,
        )
        marlin_w2_scales = marlin_moe_permute_scales(
            s=w2_scale_gpu,
            size_k=w2_scale_gpu.shape[1]
            * (self.group_size if self.group_size != -1 else self.packed_factor),
            size_n=w2_scale_gpu.shape[2],
            group_size=self.group_size,
            is_a_8bit=False,
        )

        marlin_w13_scales_cpu = torch.empty(
            marlin_w13_scales.shape,
            dtype=marlin_w13_scales.dtype,
            device="cpu",
        ).pin_memory()
        marlin_w13_scales_cpu.copy_(marlin_w13_scales, non_blocking=False)
        marlin_w2_scales_cpu = torch.empty(
            marlin_w2_scales.shape,
            dtype=marlin_w2_scales.dtype,
            device="cpu",
        ).pin_memory()
        marlin_w2_scales_cpu.copy_(marlin_w2_scales, non_blocking=False)
        del w13_scale_gpu, w2_scale_gpu, marlin_w13_scales, marlin_w2_scales

        # Empty g_idx / sort_indices on GPU (non-actorder)
        empty_gpu = torch.empty((num_experts, 0), dtype=torch.int32, device=gpu)
        layer.w13_weight_g_idx = torch.nn.Parameter(
            empty_gpu.clone(), requires_grad=False
        )
        layer.w2_weight_g_idx = torch.nn.Parameter(
            empty_gpu.clone(), requires_grad=False
        )
        layer.w13_g_idx_sort_indices = torch.nn.Parameter(
            empty_gpu.clone(), requires_grad=False
        )
        layer.w2_g_idx_sort_indices = torch.nn.Parameter(
            empty_gpu.clone(), requires_grad=False
        )

        # Marlin workspace (small, stays on GPU)
        layer.workspace = marlin_make_workspace_new(gpu, 4)

        # Instantiate the cache provider directly (bypasses
        # layer._maybe_init_expert_lru_cache which hard-codes w13_weight /
        # w2_weight names and a 1-D scale check that our 3-D scales fail).
        capacity = min(layer._moe_expert_cache_size, num_experts)
        layer.expert_weight_provider = CachedWeightProvider(
            capacity=capacity,
            w13_weight=marlin_w13_cpu,
            w2_weight=marlin_w2_cpu,
            w13_scale=marlin_w13_scales_cpu,
            w2_scale=marlin_w2_scales_cpu,
        )

        # Release full tensors from layer parameters (provider holds refs)
        layer.w13_weight_packed.data = torch.empty(0)
        layer.w2_weight_packed.data = torch.empty(0)
        layer.w13_weight_scale.data = torch.empty(0)
        layer.w2_weight_scale.data = torch.empty(0)

        logger.info_once(
            "CT WNA16 Marlin expert LRU cache: %d/%d experts cached on %s",
            capacity,
            num_experts,
            gpu,
        )

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ):
        # When the expert LRU cache is active we serve MoE via the direct
        # apply() override (which routes through the provider + Marlin).
        # Returning None here prevents maybe_init_modular_kernel from
        # wrapping us in FusedMoEModularMethod, which would otherwise call
        # select_gemm_impl and crash because the packed weights have been
        # released to empty(0) after cache init.
        if self._cache_active_hint:
            return None
        return super().maybe_make_prepare_finalize(routing_tables)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if self.num_bits != 4:
            return None
        return int4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, self.group_size],
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEExpertsModular:
        assert self.num_bits == 4, "only supporting w4"
        if getattr(layer, "expert_weight_provider", None) is not None:
            raise RuntimeError(
                "CT WNA16 Marlin select_gemm_impl is not compatible with the "
                "expert LRU cache. The cache path uses the direct apply() "
                "override and should not go through the modular kernel."
            )
        layer.w13_weight = layer.w13_weight_packed
        layer.w2_weight = layer.w2_weight_packed
        assert all([w is not None for w in [layer.w13_weight, layer.w2_weight]])
        assert self.moe_quant_config is not None
        if (
            prepare_finalize.activation_format
            == mk.FusedMoEActivationFormat.BatchedExperts
        ):
            max_num_tokens_per_rank = prepare_finalize.max_num_tokens_per_rank()
            assert max_num_tokens_per_rank is not None
            return BatchedMarlinExperts(
                max_num_tokens=max_num_tokens_per_rank,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                w13_g_idx=layer.w13_weight_g_idx,
                w2_g_idx=layer.w2_weight_g_idx,
                w13_g_idx_sort_indices=layer.w13_g_idx_sort_indices,
                w2_g_idx_sort_indices=layer.w2_g_idx_sort_indices,
                is_k_full=self.is_k_full,
            )
        else:
            return MarlinExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                w13_g_idx=layer.w13_weight_g_idx,
                w2_g_idx=layer.w2_weight_g_idx,
                w13_g_idx_sort_indices=layer.w13_g_idx_sort_indices,
                w2_g_idx_sort_indices=layer.w2_g_idx_sort_indices,
                is_k_full=self.is_k_full,
            )

    @property
    def is_monolithic(self) -> bool:
        return self.kernel_backend == "Flashinfer"

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        assert self.kernel_backend == "Flashinfer"
        return flashinfer_trtllm_mxint4_moe(
            x=x,
            router_logits=router_logits,
            w13_weight_packed=layer.w13_weight_packed,
            w13_weight_scale=layer.w13_weight_scale,
            w2_weight_packed=layer.w2_weight_packed,
            w2_weight_scale=layer.w2_weight_scale,
            global_num_experts=layer.global_num_experts,
            top_k=layer.top_k,
            intermediate_size_per_partition=layer.intermediate_size_per_partition,
            local_num_experts=layer.local_num_experts,
            ep_rank=layer.ep_rank,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routing_method_type=layer.routing_method_type,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.kernel_backend == "Marlin"

        provider = getattr(layer, "expert_weight_provider", None)
        if provider is not None:
            # Expert LRU cache path: provider streams the requested experts
            # from CPU pinned into a small GPU scratch, returns buffer refs
            # and slot-remapped topk_ids.
            result = provider.prepare(topk_ids)
            return fused_marlin_moe(
                x,
                result.w1,
                result.w2,
                None,
                None,
                result.w1_scale,
                result.w2_scale,
                topk_weights,
                result.topk_ids,
                input_global_scale1=None,
                input_global_scale2=None,
                quant_type_id=self.quant_type.id,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=provider.capacity,
                activation=layer.activation,
                expert_map=None,
                g_idx1=layer.w13_weight_g_idx,
                g_idx2=layer.w2_weight_g_idx,
                sort_indices1=layer.w13_g_idx_sort_indices,
                sort_indices2=layer.w2_g_idx_sort_indices,
                workspace=layer.workspace,
                input_dtype=self.marlin_input_dtype,
                is_k_full=self.is_k_full,
                inplace=not self.moe.disable_inplace,
            )

        return fused_marlin_moe(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            None,
            None,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            input_global_scale1=getattr(layer, "w13_input_global_scale", None),
            input_global_scale2=getattr(layer, "w2_input_global_scale", None),
            quant_type_id=self.quant_type.id,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            activation=layer.activation,
            expert_map=layer.expert_map,
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            workspace=layer.workspace,
            input_dtype=self.marlin_input_dtype,
            is_k_full=self.is_k_full,
            inplace=not self.moe.disable_inplace,
        )
