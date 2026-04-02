# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache eviction policies for MoE expert weights.

This module provides a pluggable interface for different cache eviction
strategies, leveraging the cachetools library for standard policies.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import cachetools

K = TypeVar("K")  # Key type (expert ID)
V = TypeVar("V")  # Value type (slot index)


class ExpertCachePolicy(ABC, Generic[K, V]):
    """Abstract interface for expert cache eviction policies.

    This interface defines the contract for cache policies that manage
    expert-to-slot mappings in the MoE expert cache.
    """

    @abstractmethod
    def get(self, key: K) -> V | None:
        """Get value for key if it exists (marks as accessed).

        Args:
            key: The expert ID to look up.

        Returns:
            The slot index if key exists, None otherwise.
        """

    @abstractmethod
    def put(self, key: K, value: V) -> None:
        """Insert or update key-value pair.

        Args:
            key: The expert ID.
            value: The slot index to associate with the expert.
        """

    @abstractmethod
    def select_victim(self) -> K | None:
        """Select a victim for eviction.

        Returns:
            The expert ID to evict, or None if cache is not full.
        """

    @abstractmethod
    def remove(self, key: K) -> V | None:
        """Remove key and return its value.

        Args:
            key: The expert ID to remove.

        Returns:
            The slot index if key existed, None otherwise.
        """

    @abstractmethod
    def __contains__(self, key: K) -> bool:
        """Check if key exists in cache."""

    @abstractmethod
    def __len__(self) -> int:
        """Return current number of entries in cache."""

    @property
    @abstractmethod
    def capacity(self) -> int:
        """Return maximum cache capacity."""


class CachetoolsAdapter(ExpertCachePolicy[K, V]):
    """Adapter that wraps cachetools.Cache implementations.

    This adapter provides a consistent interface over various cachetools
    cache types (LRU, LFU, FIFO).
    """

    def __init__(self, cache: cachetools.Cache):
        """Initialize adapter with a cachetools cache instance.

        Args:
            cache: A cachetools.Cache instance (LRUCache, LFUCache, etc.)
        """
        self._cache = cache

    def get(self, key: K) -> V | None:
        """Get value for key, marking it as accessed."""
        return self._cache.get(key)

    def put(self, key: K, value: V) -> None:
        """Insert or update key-value pair."""
        self._cache[key] = value

    def select_victim(self) -> K | None:
        """Select victim for eviction based on the underlying policy.

        Returns:
            Expert ID to evict, or None if not at capacity.
        """
        if len(self._cache) < self._cache.maxsize:
            return None

        # For most cachetools caches, the first item in iteration order
        # is the next victim. This works for LRU, FIFO, and LFU.
        try:
            return next(iter(self._cache))
        except StopIteration:
            return None

    def remove(self, key: K) -> V | None:
        """Remove and return value for key."""
        return self._cache.pop(key, None)

    def __contains__(self, key: K) -> bool:
        """Check if key exists in cache."""
        return key in self._cache

    def __len__(self) -> int:
        """Return current cache size."""
        return len(self._cache)

    @property
    def capacity(self) -> int:
        """Return maximum cache capacity."""
        return int(self._cache.maxsize)


class SLRUPolicy(ExpertCachePolicy[K, V]):
    """Segmented LRU cache policy with protected and probationary segments.

    SLRU divides the cache into two segments:
    - Probationary: New items enter here (20% of capacity by default)
    - Protected: Items promote here on second access (80% of capacity)

    This protects frequently-used items from being evicted by bursts of
    one-time accesses, significantly improving hit rates for many workloads.
    """

    def __init__(self, capacity: int, protected_ratio: float = 0.8):
        """Initialize SLRU policy with two segments.

        Args:
            capacity: Total cache capacity.
            protected_ratio: Fraction of capacity for protected segment (0.0-1.0).
        """
        if not 0.0 <= protected_ratio <= 1.0:
            raise ValueError(
                f"protected_ratio must be in [0, 1], got {protected_ratio}"
            )

        protected_size = max(1, int(capacity * protected_ratio))
        probationary_size = max(1, capacity - protected_size)

        self._protected: cachetools.LRUCache[K, V] = cachetools.LRUCache(
            maxsize=protected_size
        )
        self._probationary: cachetools.LRUCache[K, V] = cachetools.LRUCache(
            maxsize=probationary_size
        )
        self._capacity = capacity
        # Track which probationary items have been accessed once
        self._accessed_once: set[K] = set()

    def get(self, key: K) -> V | None:
        """Get value, promoting from probationary to protected on second access."""
        # Check protected segment first (already promoted items)
        if key in self._protected:
            # Access updates LRU order automatically
            return self._protected[key]

        # Check probationary segment
        if key in self._probationary:
            value = self._probationary[key]  # Access but don't remove yet

            # If this is the second access, promote to protected
            if key in self._accessed_once:
                self._probationary.pop(key)
                self._accessed_once.discard(key)
                # If protected is full, it will auto-evict LRU item
                self._protected[key] = value
            else:
                # First access: mark as accessed once
                self._accessed_once.add(key)

            return value

        return None

    def put(self, key: K, value: V) -> None:
        """Insert new item into probationary segment."""
        # If key already exists in protected, update it there
        if key in self._protected:
            self._protected[key] = value
        # If key exists in probationary, update it there
        elif key in self._probationary:
            self._probationary[key] = value
        else:
            # New key: insert into probationary
            # LRUCache will auto-evict if full
            self._probationary[key] = value

    def select_victim(self) -> K | None:
        """Select LRU item from probationary segment if possible, else protected."""
        if len(self) < self._capacity:
            return None

        # Prefer evicting from probationary (less valuable)
        if len(self._probationary) > 0:
            try:
                return next(iter(self._probationary))
            except StopIteration:
                pass

        # Fall back to protected if probationary is empty
        try:
            return next(iter(self._protected))
        except StopIteration:
            return None

    def remove(self, key: K) -> V | None:
        """Remove key from whichever segment contains it."""
        # Clean up tracking
        self._accessed_once.discard(key)

        # Try protected first
        if key in self._protected:
            return self._protected.pop(key, None)
        # Then probationary
        return self._probationary.pop(key, None)

    def __contains__(self, key: K) -> bool:
        """Check if key exists in either segment."""
        return key in self._protected or key in self._probationary

    def __len__(self) -> int:
        """Return total number of cached items across both segments."""
        return len(self._protected) + len(self._probationary)

    @property
    def capacity(self) -> int:
        """Return total cache capacity."""
        return self._capacity


def create_cache_policy(
    policy_type: str,
    capacity: int,
) -> ExpertCachePolicy[int, int]:
    """Factory function to create cache policies.

    Args:
        policy_type: Cache policy type. One of: "lru", "lfu", "fifo", "slru".
        capacity: Maximum number of items the cache can hold.

    Returns:
        An ExpertCachePolicy instance configured with the specified policy.

    Raises:
        ValueError: If policy_type is not recognized.
    """
    policy_type = policy_type.lower()

    if policy_type == "lru":
        # Least Recently Used: evicts least recently accessed item
        cache: cachetools.Cache[int, int] = cachetools.LRUCache(maxsize=capacity)
        return CachetoolsAdapter(cache)
    elif policy_type == "lfu":
        # Least Frequently Used: evicts least frequently accessed item
        cache = cachetools.LFUCache(maxsize=capacity)
        return CachetoolsAdapter(cache)
    elif policy_type == "fifo":
        # First In First Out: evicts oldest inserted item
        cache = cachetools.FIFOCache(maxsize=capacity)
        return CachetoolsAdapter(cache)
    elif policy_type == "slru":
        # Segmented LRU: two-tier cache with protected and probationary segments
        return SLRUPolicy(capacity=capacity)
    else:
        raise ValueError(
            f"Unknown cache policy: {policy_type}. "
            f"Valid options are: 'lru', 'lfu', 'fifo', 'slru'."
        )
