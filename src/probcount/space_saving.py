"""Space-Saving heavy-hitter frequency estimator (Metwally et al., 2005).

Provided as a deterministic counter-based baseline for comparison with
:class:`~probcount.topk.TopKTracker` in heavy-hitter experiments.
"""

from __future__ import annotations

from typing import Self

from probcount.base import FrequencyEstimator
from probcount.exact import deep_size_bytes


def _to_str(item: bytes | str) -> str:
    return item.decode("utf-8") if isinstance(item, bytes) else item


class SpaceSaving(FrequencyEstimator):
    """Space-Saving counter keeping at most *k* monitored items.

    Parameters:
        k: Maximum number of monitored item slots (``>= 1``).

    Raises:
        ValueError: If ``k < 1``.
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._k = k
        self._counts: dict[str, int] = {}
        self._errors: dict[str, int] = {}
        self._total: int = 0

    @property
    def k(self) -> int:
        """Maximum number of monitored items."""
        return self._k

    def add(self, item: bytes | str, count: int = 1) -> None:
        """Record *count* occurrences of *item* using the Space-Saving rule."""
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        if count == 0:
            return

        key = _to_str(item)
        self._total += count

        if key in self._counts:
            self._counts[key] += count
            return

        if len(self._counts) < self._k:
            self._counts[key] = count
            self._errors[key] = 0
            return

        min_item = min(self._counts, key=lambda m: self._counts[m])
        min_count = self._counts.pop(min_item)
        self._errors.pop(min_item)
        self._counts[key] = min_count + count
        self._errors[key] = min_count

    def estimate(self, item: bytes | str) -> int:
        """Return the estimated count of *item* (or 0 if unmonitored)."""
        return self._counts.get(_to_str(item), 0)

    def error_bound(self, item: bytes | str) -> int:
        """Return the Space-Saving overestimation error bound for *item*."""
        return self._errors.get(_to_str(item), 0)

    def total(self) -> int:
        """Return total count added."""
        return self._total

    def top_k(self, k: int | None = None) -> list[tuple[str, int]]:
        """Return up to *k* monitored items sorted by estimated count descending."""
        limit = self._k if k is None else min(k, self._k)
        if limit <= 0:
            return []
        return sorted(self._counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]

    def merge(self, other: Self) -> Self:
        """Merge *other* into this SpaceSaving summary and return ``self``."""
        if not isinstance(other, SpaceSaving):
            raise ValueError(f"Cannot merge SpaceSaving with {type(other).__name__}")
        combined: dict[str, int] = dict(self._counts)
        for item, cnt in other._counts.items():
            combined[item] = combined.get(item, 0) + cnt
        top = sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))[: self._k]
        self._counts = dict(top)
        self._errors = {item: 0 for item, _ in top}
        self._total += other._total
        return self

    def memory_bytes(self) -> int:
        """Return approximate memory footprint in bytes."""
        return deep_size_bytes(self._counts) + deep_size_bytes(self._errors)

    def __repr__(self) -> str:
        return f"SpaceSaving(k={self._k}, monitored={len(self._counts)}, total={self._total})"
