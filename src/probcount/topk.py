"""Heavy-hitter Top-K tracker backed by a Count-Min Sketch and a min-heap."""

from __future__ import annotations

import heapq
from typing import Self

from probcount.cms import CountMinSketch
from probcount.exact import deep_size_bytes


def _to_str(item: bytes | str) -> str:
    return item.decode("utf-8") if isinstance(item, bytes) else item


class TopKTracker:
    """Track the approximate top-*k* most frequent items in a stream.

    Combines a :class:`~probcount.cms.CountMinSketch` with a bounded
    candidate dictionary of at most *k* items and a min-heap using
    lazy deletion.

    Parameters:
        k: Maximum number of heavy-hitter candidates to retain (``>= 1``).
        sketch: Backing :class:`~probcount.cms.CountMinSketch`.

    Raises:
        ValueError: If ``k < 1``.
    """

    def __init__(self, k: int, sketch: CountMinSketch) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._k = k
        self._sketch = sketch
        self._candidates: dict[str, int] = {}
        self._heap: list[tuple[int, str]] = []

    @property
    def k(self) -> int:
        """Maximum number of tracked candidates."""
        return self._k

    @property
    def sketch(self) -> CountMinSketch:
        """Underlying Count-Min Sketch."""
        return self._sketch

    def _prune_heap(self) -> None:
        """Discard stale heap entries that no longer match ``_candidates``."""
        while self._heap:
            est, key = self._heap[0]
            if self._candidates.get(key) == est:
                break
            heapq.heappop(self._heap)

    def add(self, item: bytes | str, count: int = 1) -> None:
        """Record *count* occurrences of *item* and update the top-*k* heap."""
        key = _to_str(item)
        self._sketch.add(key, count)
        est = self._sketch.estimate(key)

        if key in self._candidates:
            self._candidates[key] = est
            heapq.heappush(self._heap, (est, key))
            if len(self._heap) > 4 * self._k:
                self._heap = [(v, k) for k, v in self._candidates.items()]
                heapq.heapify(self._heap)
            return

        if len(self._candidates) < self._k:
            self._candidates[key] = est
            heapq.heappush(self._heap, (est, key))
            return

        self._prune_heap()
        if self._heap and est > self._heap[0][0]:
            _, evicted = heapq.heappop(self._heap)
            del self._candidates[evicted]
            self._candidates[key] = est
            heapq.heappush(self._heap, (est, key))

    def top(self, k: int | None = None) -> list[tuple[str, int]]:
        """Return up to *k* candidates as ``(item, estimated_count)`` sorted descending."""
        limit = self._k if k is None else min(k, self._k)
        if limit <= 0:
            return []
        ordered = sorted(self._candidates.items(), key=lambda kv: (-kv[1], kv[0]))
        return ordered[:limit]

    def merge(self, other: Self) -> Self:
        """Merge *other* into this tracker and return ``self``.

        Merges the underlying sketches, then re-evaluates the union of both
        candidate sets against the merged sketch and retains the top *k*.

        Raises:
            ValueError: If *other* has a different *k* or incompatible sketch.
        """
        if not isinstance(other, TopKTracker):
            raise ValueError(f"Cannot merge TopKTracker with {type(other).__name__}")
        if self._k != other._k:
            raise ValueError(f"Incompatible k: {self._k} vs {other._k}")

        self._sketch.merge(other._sketch)
        all_keys = set(self._candidates.keys()) | set(other._candidates.keys())
        rescored = [(self._sketch.estimate(key), key) for key in all_keys]
        rescored.sort(key=lambda pair: (-pair[0], pair[1]))
        kept = rescored[: self._k]

        self._candidates = {key: est for est, key in kept}
        self._heap = [(est, key) for est, key in kept]
        heapq.heapify(self._heap)
        return self

    def memory_bytes(self) -> int:
        """Return the approximate memory footprint of sketch + candidate structures."""
        return (
            self._sketch.memory_bytes()
            + deep_size_bytes(self._candidates)
            + deep_size_bytes(self._heap)
        )

    def __repr__(self) -> str:
        return (
            f"TopKTracker(k={self._k}, candidates={len(self._candidates)}, "
            f"sketch={self._sketch!r})"
        )
