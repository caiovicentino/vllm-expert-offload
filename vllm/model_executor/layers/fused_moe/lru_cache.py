# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache for MoE expert weights with CPU-pinned backing store and pluggable
eviction policies."""

import time

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.cache_policy import (
    ExpertCachePolicy,
    create_cache_policy,
)

logger = init_logger(__name__)


class ExpertLRUCache:
    """GPU buffer cache for MoE expert weights with CPU backing store.

    Keeps ``capacity`` expert weight tensors in a fixed-size GPU scratch
    buffer.  All expert weights reside in CPU pinned memory; only the N
    most-recently/frequently-used rows (depending on the eviction policy)
    are mirrored into the GPU buffer.

    On each forward pass, :meth:`prepare` identifies which experts are
    needed, copies any misses from CPU to GPU (evicting entries according
    to the configured policy when the buffer is full), and returns a remapped
    ``topk_ids`` tensor whose values are GPU-buffer slot indices rather than
    global expert IDs.

    Quantization support:
        Pass ``w13_scale`` and ``w2_scale`` (per-expert weight scale tensors)
        to enable FP8 or other quantized paths.  Scale tensors are small
        (one entry per expert) and are maintained in slot-indexed GPU buffers
        that mirror the weight scratch buffers.  Callers retrieve them via
        :attr:`buf_w13_scale` and :attr:`buf_w2_scale` and pass them to the
        kernel alongside the weight buffers.

    Limitations:
        - Not compatible with CUDA graph capture. Pass ``enforce_eager=True``
          or ``--enforce-eager`` when ``moe_expert_cache_size > 0``.
        - Not compatible with expert parallelism (EP > 1).
    """

    def __init__(
        self,
        capacity: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        policy: str = "lru",
    ) -> None:
        num_experts = w13_weight.size(0)

        self.capacity = capacity
        self._num_experts = num_experts
        self.hits = 0
        self.misses = 0
        self._overflow_warned = False
        self._last_log_time: float = 0.0

        # CPU pinned backing store: all E expert weight rows live here.
        # Weights may arrive already in CPU pinned memory (loaded directly
        # from checkpoint to avoid GPU OOM) or on CUDA (standard path where
        # we move them off GPU after the fact).
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

        # GPU scratch buffers: one slot per cached expert.
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

        # Optional per-expert weight scales (e.g. FP8).  Scale tensors are
        # small so they are kept as slot-indexed GPU buffers; when an expert
        # is loaded into a slot its scale is copied alongside the weights.
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

        # Cache policy for expert eviction (Python-only; outside compile).
        # Not thread-safe: assumes single-threaded model runner (vLLM V1).
        self._policy: ExpertCachePolicy[int, int] = create_cache_policy(
            policy, capacity
        )
        self._free_slots: list[int] = list(range(capacity))

        # Persistent GPU mapping tensor: _mapping[expert_id] = slot.
        # Avoids rebuilding a CPU tensor + H2D transfer on every forward pass.
        # Invalid (uncached) entries hold 0 — safe because we only index with
        # expert IDs that were loaded in the same prepare() call.
        # int32 matches the typical topk_ids dtype, eliminating a type-cast
        # kernel on the hot remap path.
        self._mapping: torch.Tensor = torch.zeros(
            num_experts, dtype=torch.int32, device=cuda_device
        )

    @property
    def buf_w13(self) -> torch.Tensor:
        """GPU scratch buffer for gate/up projection weights."""
        return self._buf_w13

    @property
    def buf_w2(self) -> torch.Tensor:
        """GPU scratch buffer for down projection weights."""
        return self._buf_w2

    @property
    def buf_w13_scale(self) -> torch.Tensor | None:
        """Slot-indexed GPU buffer for gate/up weight scales, or None."""
        return self._buf_w13_scale

    @property
    def buf_w2_scale(self) -> torch.Tensor | None:
        """Slot-indexed GPU buffer for down weight scales, or None."""
        return self._buf_w2_scale

    def invalidate(self, expert_id: int) -> None:
        """Remove *expert_id* from the cache, returning its slot to the free
        list.  No-op if the expert is not currently cached.

        Use this when external code (e.g. EPLB) replaces the backing data
        for an expert; the stale GPU copy must be evicted so it is reloaded
        from the updated CPU backing store on the next :meth:`prepare` call.
        """
        slot = self._policy.remove(expert_id)
        if slot is not None:
            self._free_slots.append(slot)

    @torch.compiler.disable
    def prepare(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Populate the GPU buffer and return slot-remapped expert IDs.

        For each unique expert ID in ``topk_ids``:
          - **Hit**: expert is already in the buffer; update its access pattern
            according to the cache policy.
          - **Miss**: copy the expert's rows from CPU pinned memory to a free
            or evicted GPU slot (synchronous H2D copy).

        The persistent ``_mapping`` tensor is safe iff ``capacity >=
        len(unique_ids)``: when the buffer can hold every expert needed by
        this batch, no mid-loop eviction can displace an already-mapped
        expert.  When overflow occurs (more unique experts than slots), the
        method raises an error and the caller must handle it (e.g. CPU fallback).

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs.

        Returns:
            Tensor with the same shape as ``topk_ids`` where each value is
            replaced by its GPU buffer slot index.
        """
        unique_ids = topk_ids.unique().tolist()
        overflow = len(unique_ids) > self.capacity
        if overflow:
            raise RuntimeError(
                f"ExpertLRUCache.prepare() called with {len(unique_ids)} "
                f"unique experts but capacity is only {self.capacity}.  "
                f"The caller must handle overflow (e.g. CPU fallback) "
                f"instead of calling prepare() when unique > capacity."
            )

        for expert_id in unique_ids:
            slot = self._policy.get(expert_id)
            if slot is not None:
                # Cache hit: expert is already loaded in this slot
                self.hits += 1
            else:
                # Cache miss: need to load expert
                if self._free_slots:
                    # Use a free slot if available
                    slot = self._free_slots.pop()
                else:
                    # Evict according to policy
                    victim_expert_id = self._policy.select_victim()
                    if victim_expert_id is None:
                        raise RuntimeError(
                            "Cache is full but no victim selected for eviction."
                        )
                    slot = self._policy.remove(victim_expert_id)
                    if slot is None:
                        raise RuntimeError(
                            f"Failed to evict expert {victim_expert_id}."
                        )

                # Copy expert weights from CPU to GPU slot
                self._buf_w13[slot].copy_(self._cpu_w13[expert_id])
                self._buf_w2[slot].copy_(self._cpu_w2[expert_id])
                if self._buf_w13_scale is not None:
                    self._buf_w13_scale[slot].copy_(
                        self._cpu_w13_scale[expert_id]  # type: ignore[index]
                    )
                    self._buf_w2_scale[slot].copy_(  # type: ignore[index]
                        self._cpu_w2_scale[expert_id]  # type: ignore[index]
                    )

                # Register expert in policy and update mapping
                self._policy.put(expert_id, slot)
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

        # Vectorized remap via persistent GPU mapping tensor.
        # _mapping is updated in-place for each miss; no allocation per call.
        return self._mapping[topk_ids.long()].to(dtype=topk_ids.dtype)
