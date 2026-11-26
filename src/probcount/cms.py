"""Count-Min Sketch frequency estimator (Cormode & Muthukrishnan, 2005).

Implements :class:`CountMinSketch` as a :class:`~probcount.base.FrequencyEstimator`
with support for:

* Direct ``(width, depth)`` or error-bound ``from_error(epsilon, delta)`` construction.
* Standard and Conservative Update (Estan & Varghese, 2002) modes.
* Optional Kirsch-Mitzenmacher (2006) double-hashing optimization.
* Vectorised batch updates (``add_many``) and queries (``estimate_many``).
* Merging and validated binary serialization.
"""

from __future__ import annotations

import logging
import math
import struct
from collections.abc import Iterable
from typing import Self

import numpy as np
import numpy.typing as npt

from probcount.base import FrequencyEstimator
from probcount.hashing import derive_seeds, hash64, hash64_many, hash128

logger = logging.getLogger(__name__)

_MAGIC = b"CMS1"
_VERSION = 1
_HEADER_STRUCT = struct.Struct("<4sBBIIqQ")  # magic, ver, flags, width, depth, seed, total
_MASK64 = 0xFFFFFFFFFFFFFFFF


def _to_bytes(item: bytes | str) -> bytes:
    """Convert *item* to UTF-8 bytes if it is a string."""
    return item.encode("utf-8") if isinstance(item, str) else item


class CountMinSketch(FrequencyEstimator):
    """Count-Min Sketch approximate frequency counter.

    Maintains a ``depth x width`` matrix of 64-bit unsigned counters.
    Queries return the minimum across all ``depth`` rows, guaranteeing
    ``estimate(x) >= true_count(x)`` for non-negative updates.

    Parameters:
        width: Number of columns per row (``>= 1``).
        depth: Number of hash rows (``>= 1``).
        seed: Base random seed for deriving independent row hashes.
        conservative: If ``True``, use Conservative Update (only increment
            counters up to ``current_estimate + count``).
        kirsch_mitzenmacher: If ``True``, derive all ``depth`` column indices
            from two base hashes ``(h1 + i * h2) mod width`` instead of
            ``depth`` independent hashes.

    Raises:
        ValueError: If ``width < 1`` or ``depth < 1``.
    """

    def __init__(
        self,
        width: int,
        depth: int,
        seed: int = 0,
        conservative: bool = False,
        kirsch_mitzenmacher: bool = False,
    ) -> None:
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self._width = width
        self._depth = depth
        self._seed = seed
        self._conservative = conservative
        self._kirsch_mitzenmacher = kirsch_mitzenmacher
        self._row_seeds: list[int] = derive_seeds(depth, seed)
        self._table: npt.NDArray[np.uint64] = np.zeros((depth, width), dtype=np.uint64)
        self._total: int = 0

    @classmethod
    def from_error(
        cls,
        epsilon: float,
        delta: float,
        seed: int = 0,
        conservative: bool = False,
        kirsch_mitzenmacher: bool = False,
    ) -> Self:
        """Construct a sketch sized for ``(epsilon, delta)`` guarantees.

        Uses ``width = ceil(e / epsilon)`` and ``depth = ceil(ln(1 / delta))``.

        Raises:
            ValueError: If ``epsilon`` or ``delta`` is not in ``(0, 1)``.
        """
        if not 0.0 < epsilon < 1.0:
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        width = math.ceil(math.e / epsilon)
        depth = math.ceil(math.log(1.0 / delta))
        return cls(
            width=width,
            depth=depth,
            seed=seed,
            conservative=conservative,
            kirsch_mitzenmacher=kirsch_mitzenmacher,
        )

    @property
    def width(self) -> int:
        """Number of columns per row."""
        return self._width

    @property
    def depth(self) -> int:
        """Number of hash rows."""
        return self._depth

    @property
    def seed(self) -> int:
        """Base seed."""
        return self._seed

    @property
    def conservative(self) -> bool:
        """Whether Conservative Update mode is enabled."""
        return self._conservative

    @property
    def kirsch_mitzenmacher(self) -> bool:
        """Whether Kirsch-Mitzenmacher double hashing is enabled."""
        return self._kirsch_mitzenmacher

    @property
    def table(self) -> npt.NDArray[np.uint64]:
        """Read-only view of the underlying ``(depth, width)`` counter table."""
        view = self._table.view()
        view.flags.writeable = False
        return view

    def _columns_for_item(self, item: bytes | str) -> list[int]:
        """Compute the column index in each row for a single *item*."""
        if self._kirsch_mitzenmacher:
            h1, h2 = hash128(_to_bytes(item), self._seed)
            return [((h1 + i * h2) & _MASK64) % self._width for i in range(self._depth)]
        return [hash64(item, s) % self._width for s in self._row_seeds]

    def _columns_for_many(self, items: list[bytes | str]) -> npt.NDArray[np.intp]:
        """Compute a ``(depth, len(items))`` array of column indices."""
        n = len(items)
        cols = np.empty((self._depth, n), dtype=np.intp)
        if self._kirsch_mitzenmacher:
            pairs = [hash128(_to_bytes(x), self._seed) for x in items]
            h1 = np.array([p[0] for p in pairs], dtype=np.uint64)
            h2 = np.array([p[1] for p in pairs], dtype=np.uint64)
            w_u64 = np.uint64(self._width)
            with np.errstate(over="ignore"):
                for i in range(self._depth):
                    cols[i] = ((h1 + np.uint64(i) * h2) % w_u64).astype(np.intp)
        else:
            for i, s in enumerate(self._row_seeds):
                cols[i] = (hash64_many(items, s) % np.uint64(self._width)).astype(np.intp)
        return cols

    def add(self, item: bytes | str, count: int = 1) -> None:
        """Record *count* occurrences of *item*.

        In conservative mode, computes the current estimate ``e`` and sets
        each of the ``depth`` counters to ``max(counter, e + count)``.

        Raises:
            ValueError: If ``count < 0``.
        """
        if count < 0:
            raise ValueError(f"count must be >= 0 (got {count}); negative counts are unsupported")
        if count == 0:
            return

        cols = self._columns_for_item(item)
        if self._conservative:
            current_est = min(int(self._table[i, cols[i]]) for i in range(self._depth))
            target = np.uint64(current_est + count)
            for i in range(self._depth):
                if self._table[i, cols[i]] < target:
                    self._table[i, cols[i]] = target
        else:
            c_u64 = np.uint64(count)
            for i in range(self._depth):
                self._table[i, cols[i]] += c_u64

        self._total += count

    def add_many(
        self,
        items: Iterable[bytes | str],
        counts: Iterable[int] | None = None,
    ) -> None:
        """Record multiple items in batch.

        In standard mode, uses vectorised hashing and ``np.add.at``.
        In conservative mode, processes items sequentially because
        conservative updates are order-dependent.

        Raises:
            ValueError: If any count is negative or ``len(counts) != len(items)``.
        """
        items_list = list(items)
        if not items_list:
            return

        if counts is None:
            counts_arr = np.ones(len(items_list), dtype=np.uint64)
            counts_list: list[int] | None = None
        else:
            counts_list = list(counts)
            if len(counts_list) != len(items_list):
                raise ValueError(
                    f"Length mismatch: {len(items_list)} items vs {len(counts_list)} counts"
                )
            if any(c < 0 for c in counts_list):
                raise ValueError("All counts must be >= 0")
            counts_arr = np.array(counts_list, dtype=np.uint64)

        if self._conservative:
            if counts_list is None:
                for item in items_list:
                    self.add(item, 1)
            else:
                for item, c in zip(items_list, counts_list, strict=True):
                    self.add(item, c)
            return

        cols = self._columns_for_many(items_list)
        for i in range(self._depth):
            np.add.at(self._table[i], cols[i], counts_arr)
        self._total += int(counts_arr.sum())

    def estimate(self, item: bytes | str) -> int:
        """Return the estimated frequency of *item* (minimum across all rows)."""
        cols = self._columns_for_item(item)
        return min(int(self._table[i, cols[i]]) for i in range(self._depth))

    def estimate_many(self, items: Iterable[bytes | str]) -> npt.NDArray[np.uint64]:
        """Vectorised frequency estimation for multiple *items*."""
        items_list = list(items)
        if not items_list:
            return np.empty(0, dtype=np.uint64)

        cols = self._columns_for_many(items_list)
        row_vals = np.empty((self._depth, len(items_list)), dtype=np.uint64)
        for i in range(self._depth):
            row_vals[i] = self._table[i, cols[i]]
        result: npt.NDArray[np.uint64] = np.min(row_vals, axis=0)
        return result

    def total(self) -> int:
        """Return the total count ``N`` added to this sketch."""
        return self._total

    def error_bound(self) -> float:
        """Return the theoretical additive error bound ``(e / width) * N``."""
        return (math.e / self._width) * self._total

    def confidence(self) -> float:
        """Return the theoretical confidence ``1 - exp(-depth)``."""
        return 1.0 - math.exp(-self._depth)

    def merge(self, other: Self) -> Self:
        """Merge *other* into this sketch (element-wise sum) and return ``self``.

        Note:
            Merging two conservative-update sketches remains a valid upper
            bound on true frequencies, but loses the tighter bound that
            sequential conservative updates provide on a single stream.

        Raises:
            ValueError: If *other* has incompatible ``width``, ``depth``,
                ``seed``, or ``kirsch_mitzenmacher`` settings.
        """
        if not isinstance(other, CountMinSketch):
            raise ValueError(f"Cannot merge CountMinSketch with {type(other).__name__}")
        if (
            self._width != other._width
            or self._depth != other._depth
            or self._seed != other._seed
            or self._kirsch_mitzenmacher != other._kirsch_mitzenmacher
        ):
            raise ValueError(
                "Incompatible CountMinSketch parameters for merge: "
                f"({self._width}, {self._depth}, {self._seed}, km={self._kirsch_mitzenmacher}) "
                f"vs ({other._width}, {other._depth}, {other._seed}, "
                f"km={other._kirsch_mitzenmacher})"
            )
        self._table += other._table
        self._total += other._total
        return self

    def memory_bytes(self) -> int:
        """Return the byte size of the counter matrix."""
        return int(self._table.nbytes)

    def to_bytes(self) -> bytes:
        """Serialize header and counter table to a compact binary payload."""
        flags = int(self._conservative) | (int(self._kirsch_mitzenmacher) << 1)
        header = _HEADER_STRUCT.pack(
            _MAGIC,
            _VERSION,
            flags,
            self._width,
            self._depth,
            self._seed,
            self._total,
        )
        return header + self._table.tobytes(order="C")

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        """Deserialize and validate a sketch produced by :meth:`to_bytes`.

        Raises:
            ValueError: If header fields or payload length are invalid.
        """
        if len(data) < _HEADER_STRUCT.size:
            raise ValueError(
                f"Data too short ({len(data)} bytes); expected at least {_HEADER_STRUCT.size}"
            )
        magic, version, flags, width, depth, seed, total = _HEADER_STRUCT.unpack_from(data, 0)
        if magic != _MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic!r}")
        if version != _VERSION:
            raise ValueError(f"Unsupported version: {version}")
        if width < 1 or depth < 1:
            raise ValueError(f"Invalid dimensions: width={width}, depth={depth}")

        expected_payload = width * depth * 8
        payload = data[_HEADER_STRUCT.size :]
        if len(payload) != expected_payload:
            raise ValueError(
                f"Payload length mismatch: got {len(payload)} bytes, expected {expected_payload}"
            )

        conservative = bool(flags & 0x01)
        km = bool(flags & 0x02)
        sketch = cls(
            width=width,
            depth=depth,
            seed=seed,
            conservative=conservative,
            kirsch_mitzenmacher=km,
        )
        sketch._table[:] = np.frombuffer(payload, dtype=np.uint64).reshape((depth, width))
        sketch._total = total
        return sketch

    def __repr__(self) -> str:
        return (
            f"CountMinSketch(width={self._width}, depth={self._depth}, "
            f"total={self._total}, conservative={self._conservative})"
        )
