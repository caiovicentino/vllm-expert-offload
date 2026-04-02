# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ExpertWeightProvider — weight resolution for MoE expert offloading.

The cache is a weight provider, not a special forward path. The kernel
does not know or care where weights came from.
"""

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class ExpertWeightResult:
    """GPU-resident expert weights ready for kernel consumption."""

    w1: torch.Tensor
    w2: torch.Tensor
    topk_ids: torch.Tensor
    w1_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None


class ExpertWeightProvider(ABC):
    """ABC for expert weight resolution. All MoE forward paths use this."""

    @abstractmethod
    def prepare(self, topk_ids: torch.Tensor) -> ExpertWeightResult:
        """Ensure requested experts are GPU-resident."""
        ...


class FullGPUProvider(ExpertWeightProvider):
    """Zero-cost passthrough when all experts fit in GPU."""

    def __init__(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
    ):
        self._w1 = w1
        self._w2 = w2
        self._w1_scale = w1_scale
        self._w2_scale = w2_scale

    def prepare(self, topk_ids: torch.Tensor) -> ExpertWeightResult:
        return ExpertWeightResult(
            w1=self._w1,
            w2=self._w2,
            topk_ids=topk_ids,
            w1_scale=self._w1_scale,
            w2_scale=self._w2_scale,
        )


class CachedWeightProvider(ExpertWeightProvider):
    """GPU LRU cache backed by CPU pinned memory.

    Keeps ``capacity`` expert weight tensors in a fixed-size GPU scratch
    buffer.  All expert weights reside in CPU pinned memory; only the N
    most-recently-used experts are mirrored into the GPU buffer.

    On each forward pass, :meth:`prepare` identifies which experts are
    needed, copies any misses from CPU to GPU (evicting LRU entries when
    the buffer is full), and returns an :class:`ExpertWeightResult` with
    remapped ``topk_ids`` whose values are GPU-buffer slot indices.
    """

    def __init__(
        self,
        capacity: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
    ) -> None:
        num_experts = w13_weight.size(0)

        self.capacity = capacity
        self._num_experts = num_experts
        self.hits = 0
        self.misses = 0
        self._last_log_time: float = 0.0

        if w13_weight.device.type == "cpu":
            cuda_device = torch.accelerator.current_accelerator()
            self._cpu_w13: torch.Tensor = (
                w13_weight if w13_weight.is_pinned() else w13_weight.pin_memory()
            )
            self._cpu_w2: torch.Tensor = (
                w2_weight if w2_weight.is_pinned() else w2_weight.pin_memory()
            )
        else:
            cuda_device = w13_weight.device
            self._cpu_w13 = w13_weight.cpu().pin_memory()
            self._cpu_w2 = w2_weight.cpu().pin_memory()

        self._buf_w13: torch.Tensor = torch.empty(
            capacity,
            *w13_weight.shape[1:],
            dtype=w13_weight.dtype,
            device=cuda_device,
        )
        self._buf_w2: torch.Tensor = torch.empty(
            capacity,
            *w2_weight.shape[1:],
            dtype=w2_weight.dtype,
            device=cuda_device,
        )

        if w13_scale is not None and w2_scale is not None:
            self._cpu_w13_scale: torch.Tensor | None = w13_scale.cpu()
            self._cpu_w2_scale: torch.Tensor | None = w2_scale.cpu()
            self._buf_w13_scale: torch.Tensor | None = torch.empty(
                capacity,
                *w13_scale.shape[1:],
                dtype=w13_scale.dtype,
                device=cuda_device,
            )
            self._buf_w2_scale: torch.Tensor | None = torch.empty(
                capacity,
                *w2_scale.shape[1:],
                dtype=w2_scale.dtype,
                device=cuda_device,
            )
        else:
            self._cpu_w13_scale = None
            self._cpu_w2_scale = None
            self._buf_w13_scale = None
            self._buf_w2_scale = None

        # LRU order: OrderedDict[expert_id, slot_index]
        self._lru: OrderedDict[int, int] = OrderedDict()
        self._free_slots: list[int] = list(range(capacity))

        # Persistent GPU mapping tensor: _mapping[expert_id] = slot.
        self._mapping: torch.Tensor = torch.zeros(
            num_experts, dtype=torch.int32, device=cuda_device
        )

    @property
    def buf_w13(self) -> torch.Tensor:
        return self._buf_w13

    @property
    def buf_w2(self) -> torch.Tensor:
        return self._buf_w2

    @property
    def buf_w13_scale(self) -> torch.Tensor | None:
        return self._buf_w13_scale

    @property
    def buf_w2_scale(self) -> torch.Tensor | None:
        return self._buf_w2_scale

    def invalidate(self, expert_id: int) -> None:
        """Remove *expert_id* from the cache, returning its slot to the free
        list.  No-op if the expert is not currently cached."""
        if expert_id in self._lru:
            slot = self._lru.pop(expert_id)
            self._free_slots.append(slot)

    @torch.compiler.disable
    def prepare(self, topk_ids: torch.Tensor) -> ExpertWeightResult:
        """Populate the GPU buffer and return slot-remapped expert IDs.

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs.

        Returns:
            ExpertWeightResult with remapped topk_ids and GPU buffer refs.

        Raises:
            RuntimeError: If unique experts exceed capacity.
        """
        unique_ids = topk_ids.unique().tolist()
        if len(unique_ids) > self.capacity:
            raise RuntimeError(
                f"CachedWeightProvider.prepare() called with "
                f"{len(unique_ids)} unique experts but capacity is only "
                f"{self.capacity}. Increase --moe-expert-cache-size."
            )

        for expert_id in unique_ids:
            if expert_id in self._lru:
                # Cache hit: move to end (most recently used)
                self._lru.move_to_end(expert_id)
                self.hits += 1
            else:
                # Cache miss: need to load expert
                if self._free_slots:
                    slot = self._free_slots.pop()
                else:
                    # Evict LRU (first item in OrderedDict)
                    victim_id, slot = self._lru.popitem(last=False)

                # Copy expert weights from CPU to GPU slot
                self._buf_w13[slot].copy_(self._cpu_w13[expert_id])
                self._buf_w2[slot].copy_(self._cpu_w2[expert_id])
                if self._buf_w13_scale is not None:
                    assert self._cpu_w13_scale is not None
                    assert self._cpu_w2_scale is not None
                    assert self._buf_w2_scale is not None
                    self._buf_w13_scale[slot].copy_(self._cpu_w13_scale[expert_id])
                    self._buf_w2_scale[slot].copy_(self._cpu_w2_scale[expert_id])

                self._lru[expert_id] = slot
                self._mapping[expert_id] = slot
                self.misses += 1

        now = time.monotonic()
        if now - self._last_log_time >= 60.0:
            self._last_log_time = now
            total = self.hits + self.misses
            if total > 0:
                logger.debug(
                    "Expert cache: %d hits, %d misses (%.1f%% hit rate)",
                    self.hits,
                    self.misses,
                    100.0 * self.hits / total,
                )

        remapped_ids = self._mapping[topk_ids.long()].to(dtype=topk_ids.dtype)

        return ExpertWeightResult(
            w1=self._buf_w13,
            w2=self._buf_w2,
            topk_ids=remapped_ids,
            w1_scale=self._buf_w13_scale,
            w2_scale=self._buf_w2_scale,
        )
