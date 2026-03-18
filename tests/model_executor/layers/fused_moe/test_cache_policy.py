# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for pluggable cache policies in MoE expert caching."""

import pytest

from vllm.model_executor.layers.fused_moe.cache_policy import create_cache_policy


def test_lru_policy():
    """Test LRU (Least Recently Used) policy."""
    policy = create_cache_policy("lru", capacity=3)

    # Fill cache
    policy.put(0, 100)
    policy.put(1, 101)
    policy.put(2, 102)

    # Access 1 and 2 to make them recently used
    assert policy.get(1) == 101
    assert policy.get(2) == 102

    # Item 0 should be the least recently used (not accessed after put)
    victim = policy.select_victim()
    assert victim == 0

    # Evict and add new item
    policy.remove(0)
    policy.put(3, 103)

    # Verify state
    assert policy.get(0) is None
    assert policy.get(1) == 101
    assert policy.get(2) == 102
    assert policy.get(3) == 103


def test_lfu_policy():
    """Test LFU (Least Frequently Used) policy."""
    policy = create_cache_policy("lfu", capacity=3)

    # Fill cache
    policy.put(0, 100)
    policy.put(1, 101)
    policy.put(2, 102)

    # Access 1 and 2 multiple times to increase their frequency
    for _ in range(3):
        policy.get(1)
        policy.get(2)

    # Item 0 has lowest frequency (not accessed after put), should be evicted
    victim = policy.select_victim()
    assert victim == 0


def test_fifo_policy():
    """Test FIFO (First In First Out) policy."""
    policy = create_cache_policy("fifo", capacity=3)

    # Fill cache in order
    policy.put(0, 100)
    policy.put(1, 101)
    policy.put(2, 102)

    # Item 0 was added first, should be evicted first
    victim = policy.select_victim()
    assert victim == 0

    # After evicting 0, next victim should be 1
    policy.remove(0)
    policy.put(3, 103)
    victim = policy.select_victim()
    assert victim == 1


def test_slru_policy():
    """Test SLRU (Segmented LRU) policy."""
    # Use capacity 10: 8 protected, 2 probationary
    policy = create_cache_policy("slru", capacity=10)

    # Add item and immediately access it twice to promote
    policy.put(0, 100)
    policy.get(0)  # First access - mark as accessed_once
    policy.get(0)  # Second access - promotes to protected

    # Verify 0 is in protected
    assert policy.get(0) == 100

    # Add more items - they go to probationary
    policy.put(1, 101)
    policy.put(2, 102)

    # Access 1 twice to promote it
    policy.get(1)
    policy.get(1)

    # Both 0 and 1 should be in protected now
    assert policy.get(0) == 100
    assert policy.get(1) == 101

    # Fill more
    for i in range(3, 10):
        policy.put(i, 100 + i)

    # Cache is now full, verify protected items still there
    assert policy.get(0) == 100
    assert policy.get(1) == 101


def test_slru_promotion():
    """Test SLRU promotion from probationary to protected."""
    policy = create_cache_policy("slru", capacity=5)  # 4 protected, 1 probationary

    # Add item to probationary
    policy.put(0, 100)

    # First access: stays in probationary
    val = policy.get(0)
    assert val == 100

    # Second access: promotes to protected
    val = policy.get(0)
    assert val == 100

    # Item 0 should still be accessible (now in protected)
    assert policy.get(0) == 100

    # Add more items to probationary - they won't affect protected item 0
    policy.put(1, 101)
    policy.put(2, 102)  # Evicts 1 from probationary (size 1)

    # Item 0 (in protected) should still be there
    assert policy.get(0) == 100

    # Item 1 was evicted from probationary
    assert policy.get(1) is None

    # Item 2 is in probationary
    assert policy.get(2) == 102


def test_cache_policy_remove():
    """Test explicit removal from cache."""
    policy = create_cache_policy("lru", capacity=3)

    policy.put(0, 100)
    policy.put(1, 101)

    # Remove item
    slot = policy.remove(0)
    assert slot == 100

    # Verify it's gone
    assert policy.get(0) is None

    # Removing non-existent item returns None
    assert policy.remove(99) is None


def test_cache_policy_capacity():
    """Test that cache respects capacity limit."""
    capacity = 5
    policy = create_cache_policy("lru", capacity=capacity)

    # Fill beyond capacity
    for i in range(capacity + 2):
        if i >= capacity:
            victim = policy.select_victim()
            policy.remove(victim)
        policy.put(i, 100 + i)

    # Count items in cache
    items_in_cache = sum(1 for i in range(capacity + 2) if policy.get(i) is not None)
    assert items_in_cache == capacity


def test_invalid_policy_name():
    """Test that invalid policy names raise an error."""
    with pytest.raises(ValueError, match="Unknown cache policy"):
        create_cache_policy("invalid_policy", capacity=10)


def test_cache_policy_get_updates_stats():
    """Test that get() updates policy-specific statistics."""
    lru_policy = create_cache_policy("lru", capacity=3)
    lru_policy.put(0, 100)
    lru_policy.put(1, 101)
    lru_policy.put(2, 102)

    # Access item 1 multiple times
    for _ in range(3):
        lru_policy.get(1)

    # Item 0 should be the victim (least recently used)
    victim = lru_policy.select_victim()
    assert victim == 0

    # For LFU, more accesses should change victim selection
    lfu_policy = create_cache_policy("lfu", capacity=3)
    lfu_policy.put(0, 100)
    lfu_policy.put(1, 101)
    lfu_policy.put(2, 102)

    # Access item 1 frequently
    for _ in range(5):
        lfu_policy.get(1)

    # Item 0 or 2 should be victim (lower frequency)
    victim = lfu_policy.select_victim()
    assert victim in [0, 2]
