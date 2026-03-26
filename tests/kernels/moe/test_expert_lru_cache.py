# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CachedWeightProvider."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
    CachedWeightProvider,
)

NUM_EXPERTS = 8
HIDDEN = 16
INTERMEDIATE = 32


def _make_weights(dtype=torch.bfloat16):
    w13 = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN).to(dtype)
    w2 = torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE).to(dtype)
    return w13, w2


def _topk(ids: list[int]) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int32, device="cuda").unsqueeze(0)


def _make_provider(capacity=4, dtype=torch.bfloat16, with_scales=False):
    w13, w2 = _make_weights(dtype)
    kwargs: dict = dict(capacity=capacity, w13_weight=w13, w2_weight=w2)
    scales = None
    if with_scales:
        w13_s = torch.rand(NUM_EXPERTS, 1, dtype=torch.float32)
        w2_s = torch.rand(NUM_EXPERTS, 1, dtype=torch.float32)
        kwargs.update(w13_scale=w13_s, w2_scale=w2_s)
        scales = (w13_s, w2_s)
    return CachedWeightProvider(**kwargs), w13, w2, scales


# -- hit / miss / eviction ------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cold_miss_fills_slots():
    provider, *_ = _make_provider()
    result = provider.prepare(_topk([0, 1, 2, 3]))
    assert provider.misses == 4
    assert provider.hits == 0
    assert len(result.topk_ids.unique()) == 4
    assert all(0 <= s < 4 for s in result.topk_ids.unique().tolist())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hit_on_repeat_access():
    provider, *_ = _make_provider()
    provider.prepare(_topk([0, 1]))
    provider.prepare(_topk([0, 1]))
    assert provider.hits == 2
    assert provider.misses == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lru_eviction():
    provider, *_ = _make_provider()
    provider.prepare(_topk([0, 1, 2, 3]))
    provider.prepare(_topk([0, 1, 2]))  # makes 3 LRU
    provider.prepare(_topk([4]))  # evicts 3
    assert 3 not in provider._lru
    assert 4 in provider._lru


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalidate_frees_slot():
    provider, *_ = _make_provider()
    provider.prepare(_topk([0, 1, 2, 3]))
    old_slot = provider._lru[2]
    provider.invalidate(2)
    assert 2 not in provider._lru
    assert old_slot in provider._free_slots


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalidate_noop_when_absent():
    provider, *_ = _make_provider()
    provider.invalidate(99)


# -- slot remapping --------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_remapping_matches_internal_dict():
    provider, *_ = _make_provider()
    ids = _topk([0, 2, 4])
    result = provider.prepare(ids)
    for eid, slot in zip(ids.squeeze(0).tolist(), result.topk_ids.squeeze(0).tolist()):
        assert provider._lru[eid] == slot


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_output_dtype_matches_input(dtype):
    provider, *_ = _make_provider()
    ids = torch.tensor([[0, 1]], dtype=dtype, device="cuda")
    result = provider.prepare(ids)
    assert result.topk_ids.dtype == dtype


# -- GPU buffer content ----------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_buffer_matches_source_weights():
    provider, w13, w2, _ = _make_provider()
    result = provider.prepare(_topk([2, 5]))
    for eid in [2, 5]:
        slot = provider._lru[eid]
        torch.testing.assert_close(result.w1[slot].cpu(), w13[eid])
        torch.testing.assert_close(result.w2[slot].cpu(), w2[eid])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_buffer_correct_after_eviction():
    provider, w13, _, _ = _make_provider()
    provider.prepare(_topk([0, 1, 2, 3]))
    provider.prepare(_topk([1, 2, 3]))  # makes 0 LRU
    old_slot = provider._lru[0]
    provider.prepare(_topk([7]))  # evicts 0
    assert provider._lru[7] == old_slot
    torch.testing.assert_close(provider.buf_w13[old_slot].cpu(), w13[7])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpu_backing_is_pinned():
    provider, *_ = _make_provider()
    assert provider._cpu_w13.is_pinned()
    assert provider._cpu_w2.is_pinned()


# -- FP8 scales ------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scale_buffers_allocated():
    provider, _, _, _ = _make_provider(dtype=torch.float8_e4m3fn, with_scales=True)
    assert provider.buf_w13_scale is not None
    assert provider.buf_w2_scale is not None
    assert provider.buf_w13_scale.device.type == "cuda"
    assert provider.buf_w13_scale.shape == (4, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_no_scales_when_not_provided():
    provider, *_ = _make_provider()
    assert provider.buf_w13_scale is None
    assert provider.buf_w2_scale is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scale_copied_with_weights():
    provider, _, _, scales = _make_provider(
        dtype=torch.float8_e4m3fn,
        with_scales=True,
    )
    w13_s, w2_s = scales
    result = provider.prepare(_topk([3, 6]))
    for eid in [3, 6]:
        slot = provider._lru[eid]
        torch.testing.assert_close(result.w1_scale[slot].cpu(), w13_s[eid])
        torch.testing.assert_close(result.w2_scale[slot].cpu(), w2_s[eid])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scale_updated_after_eviction():
    provider, _, _, scales = _make_provider(
        dtype=torch.float8_e4m3fn,
        with_scales=True,
    )
    w13_s, _ = scales
    provider.prepare(_topk([0, 1, 2, 3]))
    provider.prepare(_topk([1, 2, 3]))
    slot_for_0 = provider._lru[0]
    provider.prepare(_topk([7]))
    assert provider._lru[7] == slot_for_0
    torch.testing.assert_close(provider.buf_w13_scale[slot_for_0].cpu(), w13_s[7])


# -- overflow --------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_overflow_warns_and_truncates():
    provider, *_ = _make_provider(capacity=2)
    # 4 unique experts > capacity 2: should warn, not crash
    result = provider.prepare(_topk([0, 1, 2, 3]))
    assert result.topk_ids.shape == (1, 4)
    assert len(provider._lru) <= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_no_overflow_at_capacity():
    provider, *_ = _make_provider(capacity=2)
    result = provider.prepare(_topk([0, 1]))
    assert result.topk_ids.shape == (1, 2)


# -- boundary capacities ---------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cache_size_equals_num_experts():
    provider, *_ = _make_provider(capacity=NUM_EXPERTS)
    provider.prepare(_topk(list(range(NUM_EXPERTS))))
    assert provider.misses == NUM_EXPERTS
    assert provider.hits == 0
    assert len(provider._free_slots) == 0
    assert len(provider._lru) == NUM_EXPERTS
    provider.prepare(_topk(list(range(NUM_EXPERTS))))
    assert provider.hits == NUM_EXPERTS
    assert len(provider._lru) == NUM_EXPERTS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cache_size_one_max_eviction():
    provider, *_ = _make_provider(capacity=1)
    provider.prepare(_topk([0]))
    provider.prepare(_topk([1]))
    provider.prepare(_topk([2]))
    assert provider.hits == 0
    assert provider.misses == 3
    assert len(provider._lru) == 1
    assert 2 in provider._lru


# -- ExpertWeightResult fields ---------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_result_contains_buffer_references():
    provider, *_ = _make_provider()
    result = provider.prepare(_topk([0, 1]))
    assert result.w1 is provider.buf_w13
    assert result.w2 is provider.buf_w2
    assert result.w1_scale is provider.buf_w13_scale
    assert result.w2_scale is provider.buf_w2_scale
