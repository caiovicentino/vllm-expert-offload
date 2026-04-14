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
        allow_non_pinned_cpu: bool = False,
    ) -> None:
        num_experts = w13_weight.size(0)

        self.capacity = capacity
        self._num_experts = num_experts
        self.hits = 0
        self.misses = 0
        self._overflow_warned = False
        self._last_log_time: float = 0.0

        if w13_weight.device.type == "cpu":
            # Robust GPU device selection: torch.accelerator.current_accelerator()
            # can return None in some contexts. Fall back to explicit cuda:0.
            cuda_device = torch.accelerator.current_accelerator()
            if cuda_device is None or cuda_device.type == "cpu":
                cuda_device = torch.device("cuda", torch.cuda.current_device())
            logger.warning(
                "CachedWeightProvider init: target GPU device=%s, "
                "w13_weight.device=%s, allow_non_pinned_cpu=%s",
                cuda_device, w13_weight.device, allow_non_pinned_cpu,
            )
            if allow_non_pinned_cpu:
                # Disk-backed (torch.from_file) or regular CPU tensors.
                # Used for models whose full backing store exceeds host RAM;
                # the OS page cache handles eviction. Non-blocking copies
                # to GPU will be slightly slower than pinned DMA.
                self._cpu_w13 = w13_weight
                self._cpu_w2 = w2_weight
            else:
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

    def _prepare_overflow(
        self, topk_ids: torch.Tensor, unique_ids: list[int]
    ) -> ExpertWeightResult:
        """One-shot path for when a single forward needs more unique experts
        than fit in the persistent LRU cache.

        Allocates a per-call GPU buffer big enough for all unique experts in
        this forward, copies them from the CPU backing store, and returns a
        fresh mapping.  The buffers are freed when this ExpertWeightResult
        goes out of scope.

        Slower than the LRU hit path (no reuse across forwards), but
        correct.  The previous behaviour — truncating unique_ids to the last
        ``capacity`` entries — silently produced garbage outputs for every
        token whose chosen expert got dropped, because their mapping entries
        pointed to stale slots from prior calls.
        """
        device = self._buf_w13.device
        n_unique = len(unique_ids)

        buf_w13 = torch.empty(
            n_unique,
            *self._cpu_w13.shape[1:],
            dtype=self._cpu_w13.dtype,
            device=device,
        )
        buf_w2 = torch.empty(
            n_unique,
            *self._cpu_w2.shape[1:],
            dtype=self._cpu_w2.dtype,
            device=device,
        )
        if self._cpu_w13_scale is not None:
            buf_w13_scale = torch.empty(
                n_unique,
                *self._cpu_w13_scale.shape[1:],
                dtype=self._cpu_w13_scale.dtype,
                device=device,
            )
            buf_w2_scale = torch.empty(
                n_unique,
                *self._cpu_w2_scale.shape[1:],
                dtype=self._cpu_w2_scale.dtype,
                device=device,
            )
        else:
            buf_w13_scale = None
            buf_w2_scale = None

        for i, expert_id in enumerate(unique_ids):
            buf_w13[i].copy_(self._cpu_w13[expert_id])
            buf_w2[i].copy_(self._cpu_w2[expert_id])
            if buf_w13_scale is not None:
                buf_w13_scale[i].copy_(self._cpu_w13_scale[expert_id])
                buf_w2_scale[i].copy_(self._cpu_w2_scale[expert_id])

        # Build a one-shot expert_id -> local slot lookup table on GPU.
        lookup = torch.full(
            (self._num_experts,), -1, dtype=torch.int64, device=device
        )
        for i, expert_id in enumerate(unique_ids):
            lookup[expert_id] = i
        remapped_ids = lookup[topk_ids.long()].to(dtype=topk_ids.dtype)

        if not self._overflow_warned:
            logger.warning(
                "CachedWeightProvider.prepare(): unique experts (%d) > "
                "capacity (%d). Falling back to per-forward overflow buffer "
                "(correct but slower than LRU hits). Consider raising "
                "moe-expert-cache-size to >= max-unique-experts-per-forward "
                "for better throughput.",
                n_unique, self.capacity,
            )
            self._overflow_warned = True
        self.overflow_prepares = getattr(self, "overflow_prepares", 0) + 1

        return ExpertWeightResult(
            w1=buf_w13,
            w2=buf_w2,
            topk_ids=remapped_ids,
            w1_scale=buf_w13_scale,
            w2_scale=buf_w2_scale,
        )

    @torch.compiler.disable
    def prepare(self, topk_ids: torch.Tensor) -> ExpertWeightResult:
        """Populate the GPU buffer and return slot-remapped expert IDs.

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs.

        Returns:
            ExpertWeightResult with remapped topk_ids and GPU buffer refs.

        When unique experts in ``topk_ids`` exceed ``self.capacity``, we
        fall back to a per-forward overflow buffer (see
        :meth:`_prepare_overflow`).  This is correct regardless of cache
        capacity, at the cost of not reusing CPU->GPU copies across calls.
        """
        unique_ids = topk_ids.unique().tolist()
        if len(unique_ids) > self.capacity:
            return self._prepare_overflow(topk_ids, unique_ids)

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


class LFRUCachedWeightProvider(CachedWeightProvider):
    """GPU LFRU (Least Frequently + Recently Used) cache for MoE experts.

    Extends CachedWeightProvider with frequency-weighted eviction.
    Standard LRU lets early layers monopolize the cache because they
    execute first every forward pass. LFRU tracks access frequency per
    expert and evicts the one with lowest score = frequency * recency.

    On GPT-OSS-20B benchmarks, LFRU improved deep-layer (18-23) hit rate
    from 0-8% (LRU) to 52-94%.  With 128 experts per layer (Gemma 4,
    Nemotron), the improvement is expected to be even larger.

    Reference: vllm-project/vllm#37190 (e1n00r)
    """

    def __init__(self, *args, decay: float = 0.95, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Frequency counter per expert (decayed over time)
        self._freq: dict[int, float] = {}
        # Monotonic step counter for recency scoring
        self._step: int = 0
        # Last access step per expert
        self._last_access: dict[int, int] = {}
        # Decay factor: controls how fast old frequency decays
        self._decay = decay

    def _score(self, expert_id: int) -> float:
        """Compute eviction score: lower = more likely to evict."""
        freq = self._freq.get(expert_id, 0.0)
        recency = self._step - self._last_access.get(expert_id, 0)
        # Combine frequency and recency: high freq + recent = high score (keep)
        return freq / (1.0 + recency)

    @torch.compiler.disable
    def prepare(self, topk_ids: torch.Tensor) -> ExpertWeightResult:
        self._step += 1

        unique_ids = topk_ids.unique().tolist()
        if len(unique_ids) > self.capacity:
            # Delegate to the base class overflow fallback (allocates a
            # per-forward GPU buffer big enough for all unique experts).
            # Correct regardless of cache size, at the cost of no reuse.
            return self._prepare_overflow(topk_ids, unique_ids)

        for expert_id in unique_ids:
            # Update frequency (decayed)
            self._freq[expert_id] = self._freq.get(expert_id, 0.0) * self._decay + 1.0
            self._last_access[expert_id] = self._step

            if expert_id in self._lru:
                # Cache hit
                self._lru.move_to_end(expert_id)
                self.hits += 1
            else:
                # Cache miss
                if self._free_slots:
                    slot = self._free_slots.pop()
                else:
                    # Evict expert with lowest LFRU score
                    min_score = float("inf")
                    victim_id = next(iter(self._lru))  # fallback to LRU
                    for eid in self._lru:
                        s = self._score(eid)
                        if s < min_score:
                            min_score = s
                            victim_id = eid
                    slot = self._lru.pop(victim_id)

                # Copy from CPU to GPU
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
                    "Expert LFRU cache: %d hits, %d misses (%.1f%% hit rate)",
                    self.hits, self.misses, 100.0 * self.hits / total,
                )

        remapped_ids = self._mapping[topk_ids.long()].to(dtype=topk_ids.dtype)

        return ExpertWeightResult(
            w1=self._buf_w13,
            w2=self._buf_w2,
            topk_ids=remapped_ids,
            w1_scale=self._buf_w13_scale,
            w2_scale=self._buf_w2_scale,
        )


class TieredCachedWeightProvider(LFRUCachedWeightProvider):
    """3-tier expert cache: GPU (LFRU) -> pinned RAM (LRU) -> cold storage.

    Extends :class:`LFRUCachedWeightProvider` by adding an explicit pinned
    RAM tier between the GPU slots and the cold backing store.  The idea:

      GPU tier   -- small, fast, frequency-weighted eviction (inherited)
      RAM tier   -- medium, pinned CPU memory, LRU eviction
      Cold tier  -- large, the tensor passed to the constructor (disk-backed
                    mmap or regular CPU, or eventually a HF safetensors
                    reader callable)

    On a GPU miss we check the RAM tier first.  RAM hits are fast because
    pinned DMA to GPU is 2-3x faster than paged CPU memory, and we avoid
    re-reading the cold tier (which for disk-backed storage may incur
    page-fault reads from SSD).  On a RAM miss we promote cold -> RAM ->
    GPU in one pass and populate both tiers.

    When ``ram_capacity == 0`` this class behaves exactly like the parent
    :class:`LFRUCachedWeightProvider`.

    The cold tier can optionally be replaced at runtime with HF
    safetensors mmap reads via :meth:`set_hf_cold_tier` -- used after the
    CT MoE offloaded path has processed all layers and wants to delete
    its intermediate disk-backed files to free disk space, leaving the
    HF download itself as the source of truth.
    """

    def __init__(
        self,
        capacity: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        allow_non_pinned_cpu: bool = False,
        ram_capacity: int = 0,
    ) -> None:
        super().__init__(
            capacity=capacity,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            allow_non_pinned_cpu=allow_non_pinned_cpu,
        )

        self.ram_capacity = ram_capacity
        self.ram_hits = 0
        self.cold_hits = 0  # GPU misses that also missed the RAM tier
        self._hf_cold_reader = None  # optional HF-backed deep-cold reader

        if ram_capacity > 0:
            self._ram_w13: torch.Tensor | None = torch.empty(
                ram_capacity,
                *w13_weight.shape[1:],
                dtype=w13_weight.dtype,
                device="cpu",
                pin_memory=True,
            )
            self._ram_w2: torch.Tensor | None = torch.empty(
                ram_capacity,
                *w2_weight.shape[1:],
                dtype=w2_weight.dtype,
                device="cpu",
                pin_memory=True,
            )
            if w13_scale is not None and w2_scale is not None:
                self._ram_w13_scale: torch.Tensor | None = torch.empty(
                    ram_capacity,
                    *w13_scale.shape[1:],
                    dtype=w13_scale.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                self._ram_w2_scale: torch.Tensor | None = torch.empty(
                    ram_capacity,
                    *w2_scale.shape[1:],
                    dtype=w2_scale.dtype,
                    device="cpu",
                    pin_memory=True,
                )
            else:
                self._ram_w13_scale = None
                self._ram_w2_scale = None

            self._ram_lru: OrderedDict[int, int] = OrderedDict()
            self._ram_free_slots: list[int] = list(range(ram_capacity))
        else:
            self._ram_w13 = None
            self._ram_w2 = None
            self._ram_w13_scale = None
            self._ram_w2_scale = None
            self._ram_lru = OrderedDict()
            self._ram_free_slots = []

    def set_hf_cold_tier(self, reader) -> None:
        """Install a per-expert HF safetensors reader as the deep cold tier.

        ``reader`` is a callable ``reader(expert_id) -> tuple`` returning
        ``(w13_tensor, w2_tensor, w13_scale, w2_scale)`` already in the
        Marlin-repacked layout expected by the cache buffers.  Used after
        the CT MoE offloaded path has processed all layers and deletes
        its intermediate disk-backed files -- from then on, deep cold
        misses (not in GPU, not in RAM, not in our own cold tensor) will
        hit this reader.  All four return values may be torch CPU tensors;
        None values are ignored for the scale entries.
        """
        self._hf_cold_reader = reader

    def prepopulate_ram_tier(self, num_experts: int) -> int:
        """Fill the RAM tier with every expert from the current cold source.

        Used by process_weights_after_loading to eagerly move all experts
        from the transient disk-backed cold tier into pinned RAM, so that
        the disk-backed files can be deleted immediately afterwards.

        Returns the number of experts actually copied.  If
        ``ram_capacity`` is smaller than ``num_experts`` we copy up to
        ``ram_capacity`` experts and leave the rest on the cold tier.
        """
        if self.ram_capacity == 0:
            return 0
        n = min(num_experts, self.ram_capacity)
        for eid in range(n):
            if eid in self._ram_lru:
                continue
            slot = self._acquire_ram_slot()
            self._copy_cold_to_ram(slot, eid)
            self._ram_lru[eid] = slot
        return n

    def drop_cold_tensors(self) -> None:
        """Release the CPU-side cold-tier references.

        Call this after :meth:`prepopulate_ram_tier` has fully populated
        the RAM tier (ram_capacity >= num_experts) and the caller has
        removed the disk-backed files.  The provider becomes fully
        RAM-backed from this point on.
        """
        self._cpu_w13 = torch.empty(0, dtype=self._cpu_w13.dtype)
        self._cpu_w2 = torch.empty(0, dtype=self._cpu_w2.dtype)
        if self._cpu_w13_scale is not None:
            self._cpu_w13_scale = torch.empty(
                0, dtype=self._cpu_w13_scale.dtype
            )
            self._cpu_w2_scale = torch.empty(
                0, dtype=self._cpu_w2_scale.dtype
            )

    def _acquire_ram_slot(self) -> int:
        """Return a RAM tier slot index, evicting the LRU entry if full."""
        if self._ram_free_slots:
            return self._ram_free_slots.pop()
        _, slot = self._ram_lru.popitem(last=False)
        return slot

    def _copy_ram_to_gpu(self, gpu_slot: int, ram_slot: int) -> None:
        """Async pinned DMA: RAM tier slot -> GPU tier slot."""
        self._buf_w13[gpu_slot].copy_(
            self._ram_w13[ram_slot], non_blocking=True
        )
        self._buf_w2[gpu_slot].copy_(
            self._ram_w2[ram_slot], non_blocking=True
        )
        if self._buf_w13_scale is not None:
            assert self._ram_w13_scale is not None
            assert self._ram_w2_scale is not None
            assert self._buf_w2_scale is not None
            self._buf_w13_scale[gpu_slot].copy_(
                self._ram_w13_scale[ram_slot], non_blocking=True
            )
            self._buf_w2_scale[gpu_slot].copy_(
                self._ram_w2_scale[ram_slot], non_blocking=True
            )

    def _copy_cold_to_ram(self, ram_slot: int, expert_id: int) -> None:
        """Sync read from cold tier -> RAM slot.

        Uses the HF safetensors reader if installed, otherwise reads from
        the CPU-side cold tensor (disk-backed mmap or regular CPU)."""
        if self._hf_cold_reader is not None:
            w13, w2, s13, s2 = self._hf_cold_reader(expert_id)
            self._ram_w13[ram_slot].copy_(w13)
            self._ram_w2[ram_slot].copy_(w2)
            if self._ram_w13_scale is not None and s13 is not None:
                self._ram_w13_scale[ram_slot].copy_(s13)
                self._ram_w2_scale[ram_slot].copy_(s2)
            return

        self._ram_w13[ram_slot].copy_(self._cpu_w13[expert_id])
        self._ram_w2[ram_slot].copy_(self._cpu_w2[expert_id])
        if self._ram_w13_scale is not None:
            assert self._cpu_w13_scale is not None
            assert self._cpu_w2_scale is not None
            self._ram_w13_scale[ram_slot].copy_(self._cpu_w13_scale[expert_id])
            self._ram_w2_scale[ram_slot].copy_(self._cpu_w2_scale[expert_id])

    @torch.compiler.disable
    def prepare(self, topk_ids: torch.Tensor) -> ExpertWeightResult:
        # No RAM tier configured -> fall back to LFRU-only path.
        if self.ram_capacity == 0:
            return super().prepare(topk_ids)

        self._step += 1
        unique_ids = topk_ids.unique().tolist()

        if len(unique_ids) > self.capacity:
            # Overflow: one-shot per-forward GPU buffer from base class.
            return self._prepare_overflow(topk_ids, unique_ids)

        for expert_id in unique_ids:
            self._freq[expert_id] = (
                self._freq.get(expert_id, 0.0) * self._decay + 1.0
            )
            self._last_access[expert_id] = self._step

            if expert_id in self._lru:
                # GPU tier hit.
                self._lru.move_to_end(expert_id)
                if expert_id in self._ram_lru:
                    self._ram_lru.move_to_end(expert_id)
                self.hits += 1
                continue

            # GPU miss: acquire a GPU slot (LFRU eviction if needed).
            if self._free_slots:
                gpu_slot = self._free_slots.pop()
            else:
                min_score = float("inf")
                victim_id = next(iter(self._lru))
                for eid in self._lru:
                    s = self._score(eid)
                    if s < min_score:
                        min_score = s
                        victim_id = eid
                gpu_slot = self._lru.pop(victim_id)

            if expert_id in self._ram_lru:
                ram_slot = self._ram_lru[expert_id]
                self._ram_lru.move_to_end(expert_id)
                self._copy_ram_to_gpu(gpu_slot, ram_slot)
                self.ram_hits += 1
            else:
                ram_slot = self._acquire_ram_slot()
                self._copy_cold_to_ram(ram_slot, expert_id)
                self._ram_lru[expert_id] = ram_slot
                self._copy_ram_to_gpu(gpu_slot, ram_slot)
                self.cold_hits += 1
                self.misses += 1  # keep parent counter in sync

            self._lru[expert_id] = gpu_slot
            self._mapping[expert_id] = gpu_slot

        now = time.monotonic()
        if now - self._last_log_time >= 60.0:
            self._last_log_time = now
            total = self.hits + self.ram_hits + self.cold_hits
            if total > 0:
                logger.info(
                    "Tiered expert cache: GPU %d hits, RAM %d hits, "
                    "cold %d misses (GPU hit %.1f%%, RAM hit %.1f%% of "
                    "GPU misses)",
                    self.hits, self.ram_hits, self.cold_hits,
                    100.0 * self.hits / total,
                    100.0 * self.ram_hits / max(1, self.ram_hits + self.cold_hits),
                )

        remapped_ids = self._mapping[topk_ids.long()].to(dtype=topk_ids.dtype)

        return ExpertWeightResult(
            w1=self._buf_w13,
            w2=self._buf_w2,
            topk_ids=remapped_ids,
            w1_scale=self._buf_w13_scale,
            w2_scale=self._buf_w2_scale,
        )
