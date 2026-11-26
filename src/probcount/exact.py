"""Exact (ground-truth) counters for cardinality and frequency.

These wrappers around Python's built-in ``set`` and
:class:`collections.Counter` implement the abstract interfaces defined in
:mod:`probcount.base`.  They serve as *ground truth* in experiments: we
compare the probabilistic sketches against these exact counters to
measure error.

.. note::

    :func:`deep_size_bytes` uses recursive ``sys.getsizeof`` to estimate
    memory.  It is designed for *comparative* benchmarking (exact vs.
    sketch) rather than absolute measurement.
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from collections.abc import Iterable
from typing import Self

from probcount.base import CardinalityEstimator, FrequencyEstimator

# ---------------------------------------------------------------------------
# Memory-measurement utility
# ---------------------------------------------------------------------------


def deep_size_bytes(obj: object) -> int:
    """Recursively estimate the memory footprint of *obj* in bytes.

    Uses :func:`sys.getsizeof` with a ``seen`` set (by ``id``) to avoid
    double-counting.  Handles ``dict``, ``set``, ``frozenset``, ``list``,
    ``tuple``, :class:`~collections.Counter`, and their contained objects.

    **Limitations**:

    * Does not follow ``__slots__`` or C-extension internal structures.
    * May undercount objects with complex internal references.
    * Best used for *relative* comparison, not absolute measurement.

    Parameters:
        obj: Any Python object.

    Returns:
        Approximate size in bytes.
    """
    seen: set[int] = set()

    def _sizeof(o: object) -> int:
        obj_id = id(o)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        size = sys.getsizeof(o)
        if isinstance(o, dict):
            for k, v in o.items():
                size += _sizeof(k) + _sizeof(v)
        elif isinstance(o, (set, frozenset, list, tuple)):
            for item in o:
                size += _sizeof(item)
        return size

    return _sizeof(obj)


# ---------------------------------------------------------------------------
# Exact cardinality counter
# ---------------------------------------------------------------------------


class ExactUniqueCounter(CardinalityEstimator):
    """Exact distinct-element counter backed by a Python ``set``.

    This always returns the true cardinality; it is used as the ground-truth
    baseline in accuracy experiments.
    """

    def __init__(self) -> None:
        self._items: set[str] = set()

    def add(self, item: bytes | str) -> None:
        """Add a single item, converting ``bytes`` to ``str`` if needed."""
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        self._items.add(item)

    def add_many(self, items: Iterable[bytes | str]) -> None:
        """Add multiple items efficiently."""
        for item in items:
            if isinstance(item, bytes):
                self._items.add(item.decode("utf-8"))
            else:
                self._items.add(item)

    def count(self) -> float:
        """Return the exact number of distinct items."""
        return float(len(self._items))

    def merge(self, other: Self) -> Self:
        """Merge *other* into this counter (set union) and return ``self``."""
        self._items |= other._items
        return self

    def memory_bytes(self) -> int:
        """Return the approximate memory used by the internal set."""
        return deep_size_bytes(self._items)

    def to_bytes(self) -> bytes:
        """Serialize the set to bytes via :mod:`pickle`."""
        return pickle.dumps(self._items)

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        """Deserialize from bytes produced by :meth:`to_bytes`."""
        items: set[str] = pickle.loads(data)  # noqa: S301
        counter = cls()
        counter._items = items
        return counter

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"ExactUniqueCounter(n={len(self._items)})"


# ---------------------------------------------------------------------------
# Exact frequency counter
# ---------------------------------------------------------------------------


class ExactFrequencyCounter(FrequencyEstimator):
    """Exact per-item frequency counter backed by :class:`collections.Counter`.

    Returns true frequencies; used as ground truth for Count-Min Sketch
    accuracy experiments.
    """

    def __init__(self) -> None:
        self._counter: Counter[str] = Counter()

    def add(self, item: bytes | str, count: int = 1) -> None:
        """Record *count* occurrences of *item*."""
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        self._counter[item] += count

    def estimate(self, item: bytes | str) -> int:
        """Return the exact frequency of *item*."""
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        return self._counter[item]

    def total(self) -> int:
        """Return the sum of all recorded counts."""
        return sum(self._counter.values())

    def top_k(self, k: int) -> list[tuple[str, int]]:
        """Return the *k* most frequent items as ``(item, count)`` pairs.

        Parameters:
            k: Number of top items to return.

        Returns:
            List of ``(item, count)`` tuples sorted by count descending.
        """
        return self._counter.most_common(k)

    def merge(self, other: Self) -> Self:
        """Merge *other* into this counter (additive) and return ``self``."""
        self._counter += other._counter
        return self

    def memory_bytes(self) -> int:
        """Return the approximate memory used by the internal Counter."""
        return deep_size_bytes(self._counter)

    def __repr__(self) -> str:
        n_items = len(self._counter)
        total = self.total()
        return f"ExactFrequencyCounter(items={n_items}, total={total})"


def exact_top_k(items: Iterable[bytes | str], k: int) -> list[tuple[str, int]]:
    """Compute exact top-*k* ``(item, count)`` pairs from an iterable of *items*."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    counter = ExactFrequencyCounter()
    for item in items:
        counter.add(item)
    return counter.top_k(k)
