# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ExpertLRUCache."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.lru_cache import ExpertLRUCache

NUM_EXPERTS = 8
HIDDEN = 16
INTERMEDIATE = 32


def _make_weights(dtype=torch.bfloat16):
    w13 = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN).to(dtype)
    w2 = torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE).to(dtype)
    return w13, w2


def _topk(ids: list[int]) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int32, device="cuda").unsqueeze(0)


def _make_cache(capacity=4, dtype=torch.bfloat16, with_scales=False):
    w13, w2 = _make_weights(dtype)
    kwargs: dict = dict(capacity=capacity, w13_weight=w13, w2_weight=w2)
    scales = None
    if with_scales:
        w13_s = torch.rand(NUM_EXPERTS, 1, dtype=torch.float32)
        w2_s = torch.rand(NUM_EXPERTS, 1, dtype=torch.float32)
        kwargs.update(w13_scale=w13_s, w2_scale=w2_s)
        scales = (w13_s, w2_s)
    return ExpertLRUCache(**kwargs), w13, w2, scales


# -- hit / miss / eviction ------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cold_miss_fills_slots():
    cache, *_ = _make_cache()
    slot_ids = cache.prepare(_topk([0, 1, 2, 3]))
    assert cache.misses == 4
    assert cache.hits == 0
    assert len(slot_ids.unique()) == 4
    assert all(0 <= s < 4 for s in slot_ids.unique().tolist())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hit_on_repeat_access():
    cache, *_ = _make_cache()
    cache.prepare(_topk([0, 1]))
    cache.prepare(_topk([0, 1]))
    assert cache.hits == 2
    assert cache.misses == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lru_eviction():
    cache, *_ = _make_cache()
    cache.prepare(_topk([0, 1, 2, 3]))
    cache.prepare(_topk([0, 1, 2]))  # makes 3 LRU
    cache.prepare(_topk([4]))  # evicts 3
    assert 3 not in cache._expert_to_slot
    assert 4 in cache._expert_to_slot


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalidate_frees_slot():
    cache, *_ = _make_cache()
    cache.prepare(_topk([0, 1, 2, 3]))
    old_slot = cache._expert_to_slot[2]
    cache.invalidate(2)
    assert 2 not in cache._expert_to_slot
    assert old_slot in cache._free_slots


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalidate_noop_when_absent():
    cache, *_ = _make_cache()
    cache.invalidate(99)


# -- slot remapping --------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_remapping_matches_internal_dict():
    cache, *_ = _make_cache()
    ids = _topk([0, 2, 4])
    slot_ids = cache.prepare(ids)
    for eid, slot in zip(ids.squeeze(0).tolist(), slot_ids.squeeze(0).tolist()):
        assert cache._expert_to_slot[eid] == slot


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_output_dtype_matches_input(dtype):
    cache, *_ = _make_cache()
    ids = torch.tensor([[0, 1]], dtype=dtype, device="cuda")
    assert cache.prepare(ids).dtype == dtype


# -- GPU buffer content ----------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_buffer_matches_source_weights():
    cache, w13, w2, _ = _make_cache()
    cache.prepare(_topk([2, 5]))
    for eid in [2, 5]:
        slot = cache._expert_to_slot[eid]
        torch.testing.assert_close(cache.buf_w13[slot].cpu(), w13[eid])
        torch.testing.assert_close(cache.buf_w2[slot].cpu(), w2[eid])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_buffer_correct_after_eviction():
    cache, w13, _, _ = _make_cache()
    cache.prepare(_topk([0, 1, 2, 3]))
    cache.prepare(_topk([1, 2, 3]))  # makes 0 LRU
    old_slot = cache._expert_to_slot[0]
    cache.prepare(_topk([7]))  # evicts 0
    assert cache._expert_to_slot[7] == old_slot
    torch.testing.assert_close(cache.buf_w13[old_slot].cpu(), w13[7])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpu_backing_is_pinned():
    cache, *_ = _make_cache()
    assert cache._cpu_w13.is_pinned()
    assert cache._cpu_w2.is_pinned()


# -- FP8 scales ------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scale_buffers_allocated():
    cache, _, _, _ = _make_cache(dtype=torch.float8_e4m3fn, with_scales=True)
    assert cache.buf_w13_scale is not None
    assert cache.buf_w2_scale is not None
    assert cache.buf_w13_scale.device.type == "cuda"
    assert cache.buf_w13_scale.shape == (4, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_no_scales_when_not_provided():
    cache, *_ = _make_cache()
    assert cache.buf_w13_scale is None
    assert cache.buf_w2_scale is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scale_copied_with_weights():
    cache, _, _, scales = _make_cache(
        dtype=torch.float8_e4m3fn,
        with_scales=True,
    )
    w13_s, w2_s = scales
    cache.prepare(_topk([3, 6]))
    for eid in [3, 6]:
        slot = cache._expert_to_slot[eid]
        torch.testing.assert_close(cache.buf_w13_scale[slot].cpu(), w13_s[eid])
        torch.testing.assert_close(cache.buf_w2_scale[slot].cpu(), w2_s[eid])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scale_updated_after_eviction():
    cache, _, _, scales = _make_cache(
        dtype=torch.float8_e4m3fn,
        with_scales=True,
    )
    w13_s, _ = scales
    cache.prepare(_topk([0, 1, 2, 3]))
    cache.prepare(_topk([1, 2, 3]))
    slot_for_0 = cache._expert_to_slot[0]
    cache.prepare(_topk([7]))
    assert cache._expert_to_slot[7] == slot_for_0
    torch.testing.assert_close(cache.buf_w13_scale[slot_for_0].cpu(), w13_s[7])


# -- overflow --------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_overflow_raises():
    cache, *_ = _make_cache(capacity=2)
    with pytest.raises(RuntimeError, match="capacity"):
        cache.prepare(_topk([0, 1, 2, 3]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_no_overflow_at_capacity():
    cache, *_ = _make_cache(capacity=2)
    slot_ids = cache.prepare(_topk([0, 1]))
    assert slot_ids.shape == (1, 2)


# -- boundary capacities ---------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cache_size_equals_num_experts():
    cache, *_ = _make_cache(capacity=NUM_EXPERTS)
    # First pass: all slots fill, no evictions.
    cache.prepare(_topk(list(range(NUM_EXPERTS))))
    assert cache.misses == NUM_EXPERTS
    assert cache.hits == 0
    assert len(cache._free_slots) == 0
    assert len(cache._lru_order) == NUM_EXPERTS
    # Second pass: all hits, nothing evicted.
    cache.prepare(_topk(list(range(NUM_EXPERTS))))
    assert cache.hits == NUM_EXPERTS
    assert len(cache._expert_to_slot) == NUM_EXPERTS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cache_size_one_max_eviction():
    cache, *_ = _make_cache(capacity=1)
    cache.prepare(_topk([0]))
    cache.prepare(_topk([1]))
    cache.prepare(_topk([2]))
    assert cache.hits == 0
    assert cache.misses == 3
    assert len(cache._expert_to_slot) == 1
    assert 2 in cache._expert_to_slot
