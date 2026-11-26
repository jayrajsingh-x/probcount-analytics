"""HyperLogLog++ sparse representation for low-cardinality streams.

Stores a compact set of encoded ``(index, rho)`` tokens while the number
of unique buckets is below ``sparse_threshold``, then converts to a dense
:class:`~probcount.hll.HyperLogLog` once the threshold is reached.
"""

from __future__ import annotations

import logging
import math
import struct
from collections.abc import Iterable
from typing import Self

from probcount.base import CardinalityEstimator
from probcount.exact import deep_size_bytes
from probcount.hashing import hash64, split_hash
from probcount.hll import HyperLogLog

logger = logging.getLogger(__name__)

_SPARSE_MAGIC = b"HLS1"
_SPARSE_HEADER = struct.Struct("<4sBBqI")


class SparseHyperLogLog(CardinalityEstimator):
    """HyperLogLog++ sparse-to-dense cardinality estimator.

    For low cardinalities (``<= sparse_threshold`` distinct register indices),
    maintains a compact set of encoded ``(index, rho)`` integers.  Once the
    threshold is exceeded, it converts seamlessly to a dense
    :class:`~probcount.hll.HyperLogLog` with identical register state.
    """

    def __init__(
        self,
        p: int = 14,
        seed: int = 0,
        sparse_threshold: int | None = None,
    ) -> None:
        if not 4 <= p <= 18:
            raise ValueError(f"p must be in [4, 18], got {p}")
        self._p = p
        self._seed = seed
        self._m = 1 << p
        self._threshold = (
            sparse_threshold if sparse_threshold is not None else max(64, self._m // 8)
        )
        self._sparse: dict[int, int] | None = {}
        self._dense: HyperLogLog | None = None

    @property
    def is_sparse(self) -> bool:
        """Return ``True`` if still using the sparse representation."""
        return self._dense is None

    @property
    def p(self) -> int:
        """Precision parameter ``p``."""
        return self._p

    @property
    def seed(self) -> int:
        """Hash seed."""
        return self._seed

    def _convert_to_dense(self) -> None:
        if self._dense is not None:
            return
        dense = HyperLogLog(p=self._p, seed=self._seed)
        if self._sparse:
            regs = dense._registers
            for idx, rho in self._sparse.items():
                if rho > regs[idx]:
                    regs[idx] = rho
        self._dense = dense
        self._sparse = None

    def add(self, item: bytes | str) -> None:
        """Record a single item, converting to dense if threshold is exceeded."""
        if self._dense is not None:
            self._dense.add(item)
            return

        assert self._sparse is not None
        h = hash64(item, self._seed)
        index, w = split_hash(h, self._p)
        rho = (64 - self._p) - w.bit_length() + 1
        prev = self._sparse.get(index, 0)
        if rho > prev:
            self._sparse[index] = rho

        if len(self._sparse) > self._threshold:
            self._convert_to_dense()

    def add_many(self, items: Iterable[bytes | str]) -> None:
        """Record multiple items."""
        if self._dense is not None:
            self._dense.add_many(items)
            return
        for item in items:
            self.add(item)

    def count(self) -> float:
        """Return estimated distinct items (Linear Counting in sparse mode)."""
        if self._dense is not None:
            return self._dense.count()
        assert self._sparse is not None
        k = len(self._sparse)
        if k == 0:
            return 0.0
        v = self._m - k
        if v > 0:
            return float(self._m * math.log(self._m / v))
        return float(k)

    def to_dense(self) -> HyperLogLog:
        """Return a dense :class:`HyperLogLog` equivalent to this sketch."""
        if self._dense is not None:
            return self._dense.copy()
        dense = HyperLogLog(p=self._p, seed=self._seed)
        if self._sparse:
            for idx, rho in self._sparse.items():
                dense._registers[idx] = rho
        return dense

    def merge(self, other: Self) -> Self:
        """Merge *other* into this sketch and return ``self``."""
        if not isinstance(other, SparseHyperLogLog):
            raise ValueError(f"Cannot merge SparseHyperLogLog with {type(other).__name__}")
        if self._p != other._p or self._seed != other._seed:
            raise ValueError("Incompatible SparseHyperLogLog parameters")

        if self._dense is not None or other._dense is not None:
            self._convert_to_dense()
            assert self._dense is not None
            self._dense.merge(other.to_dense())
            return self

        assert self._sparse is not None and other._sparse is not None
        for idx, rho in other._sparse.items():
            if rho > self._sparse.get(idx, 0):
                self._sparse[idx] = rho
        if len(self._sparse) > self._threshold:
            self._convert_to_dense()
        return self

    def memory_bytes(self) -> int:
        """Return packed bytes (dense) or encoded sparse size."""
        if self._dense is not None:
            return self._dense.memory_bytes()
        assert self._sparse is not None
        return deep_size_bytes(self._sparse)

    def to_bytes(self) -> bytes:
        """Serialize sparse or dense state to bytes."""
        if self._dense is not None:
            return b"D" + self._dense.to_bytes()
        assert self._sparse is not None
        header = _SPARSE_HEADER.pack(
            _SPARSE_MAGIC, self._p, 1, self._seed, len(self._sparse)
        )
        pairs = b"".join(
            struct.pack("<IB", idx, rho) for idx, rho in sorted(self._sparse.items())
        )
        return b"S" + header + pairs

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        """Deserialize from bytes produced by :meth:`to_bytes`."""
        if not data:
            raise ValueError("Empty data")
        mode = data[:1]
        body = data[1:]
        if mode == b"D":
            dense = HyperLogLog.from_bytes(body)
            obj = cls(p=dense.p, seed=dense.seed)
            obj._dense = dense
            obj._sparse = None
            return obj
        if mode == b"S":
            if len(body) < _SPARSE_HEADER.size:
                raise ValueError("Sparse HLL payload too short")
            magic, p, ver, seed, count = _SPARSE_HEADER.unpack_from(body, 0)
            if magic != _SPARSE_MAGIC or ver != 1:
                raise ValueError("Invalid sparse HLL header")
            obj = cls(p=p, seed=seed)
            obj._sparse = {}
            offset = _SPARSE_HEADER.size
            for _ in range(count):
                idx, rho = struct.unpack_from("<IB", body, offset)
                obj._sparse[idx] = rho
                offset += 5
            return obj
        raise ValueError(f"Unknown SparseHyperLogLog mode byte: {mode!r}")
