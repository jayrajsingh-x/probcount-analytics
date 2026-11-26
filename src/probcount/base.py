"""Abstract base classes for cardinality and frequency estimators.

These interfaces define the contract that both exact (ground-truth)
counters and probabilistic sketches (HyperLogLog, Count-Min Sketch)
must satisfy.  Using a common interface lets higher-level code (the
API, experiments, tests) be written once and work with any estimator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Self


class CardinalityEstimator(ABC):
    """Interface for data structures that estimate the number of distinct items.

    Implementations include :class:`~probcount.exact.ExactUniqueCounter`
    (Python ``set``) and, in later phases, HyperLogLog.
    """

    @abstractmethod
    def add(self, item: bytes | str) -> None:
        """Record a single item."""

    def add_many(self, items: Iterable[bytes | str]) -> None:
        """Record multiple items.

        The default implementation calls :meth:`add` in a loop.
        Subclasses may override for better performance.
        """
        for item in items:
            self.add(item)

    @abstractmethod
    def count(self) -> float:
        """Return the estimated number of distinct items seen so far."""

    @abstractmethod
    def merge(self, other: Self) -> Self:
        """Merge *other* into this estimator (mutating) and return ``self``.

        After merging, ``self`` represents the union of both input streams.
        """

    @abstractmethod
    def memory_bytes(self) -> int:
        """Return the approximate memory footprint in bytes."""

    @abstractmethod
    def to_bytes(self) -> bytes:
        """Serialize the estimator state to a byte string."""

    @classmethod
    @abstractmethod
    def from_bytes(cls, data: bytes) -> Self:
        """Deserialize an estimator from a byte string produced by :meth:`to_bytes`."""


class FrequencyEstimator(ABC):
    """Interface for data structures that estimate per-item frequencies.

    Implementations include :class:`~probcount.exact.ExactFrequencyCounter`
    (``collections.Counter``) and, in later phases, Count-Min Sketch.
    """

    @abstractmethod
    def add(self, item: bytes | str, count: int = 1) -> None:
        """Record *count* occurrences of *item*."""

    @abstractmethod
    def estimate(self, item: bytes | str) -> int:
        """Return the estimated frequency of *item*."""

    @abstractmethod
    def total(self) -> int:
        """Return the total number of item occurrences recorded."""

    @abstractmethod
    def merge(self, other: Self) -> Self:
        """Merge *other* into this estimator (mutating) and return ``self``."""

    @abstractmethod
    def memory_bytes(self) -> int:
        """Return the approximate memory footprint in bytes."""
