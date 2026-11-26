"""HyperLogLog cardinality estimator (Flajolet et al., 2007).

Implements a mergeable, serializable :class:`HyperLogLog` sketch backed by
a NumPy ``uint8`` register array and the 64-bit MurmurHash3 hash function
from :mod:`probcount.hashing`.
"""

from __future__ import annotations

import logging
import math
import struct
from collections.abc import Iterable
from typing import Self

import numpy as np
import numpy.typing as npt

from probcount.base import CardinalityEstimator
from probcount.hashing import hash64, hash64_many, split_hash

logger = logging.getLogger(__name__)

_MAGIC = b"HLL1"
_VERSION = 1
_HEADER_STRUCT = struct.Struct("<4sBBq")  # magic(4), version(1), p(1), seed(8)
_MIN_P = 4
_MAX_P = 18


def _alpha(m: int) -> float:
    """Return the bias-correction constant alpha_m for *m* registers."""
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / m)


def _pack_6bit(registers: npt.NDArray[np.uint8]) -> bytes:
    """Pack an array of 6-bit register values (length divisible by 4) into bytes."""
    r = registers.reshape(-1, 4).astype(np.uint32)
    b0 = (r[:, 0] | (r[:, 1] << 6)) & 0xFF
    b1 = ((r[:, 1] >> 2) | (r[:, 2] << 4)) & 0xFF
    b2 = ((r[:, 2] >> 4) | (r[:, 3] << 2)) & 0xFF
    return np.column_stack((b0, b1, b2)).astype(np.uint8).tobytes()


def _unpack_6bit(payload: bytes, m: int) -> npt.NDArray[np.uint8]:
    """Unpack *m* 6-bit register values from *payload* bytes."""
    b = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    r0 = b[:, 0] & 0x3F
    r1 = ((b[:, 0] >> 6) | (b[:, 1] << 2)) & 0x3F
    r2 = ((b[:, 1] >> 4) | (b[:, 2] << 4)) & 0x3F
    r3 = (b[:, 2] >> 2) & 0x3F
    regs: npt.NDArray[np.uint8] = np.column_stack((r0, r1, r2, r3)).astype(np.uint8).ravel()
    return regs[:m]


class HyperLogLog(CardinalityEstimator):
    """HyperLogLog probabilistic distinct-element counter.

    Uses ``m = 2**p`` registers to estimate cardinalities with standard error
    ``1.04 / sqrt(m)`` using sub-linear memory.

    Parameters:
        p: Precision parameter in ``[4, 18]`` (default ``14``, i.e. 16,384
           registers and ~0.81% standard error).
        seed: Hash seed passed to :func:`probcount.hashing.hash64`.

    Raises:
        ValueError: If *p* is outside ``[4, 18]``.
    """

    def __init__(self, p: int = 14, seed: int = 0) -> None:
        if not _MIN_P <= p <= _MAX_P:
            raise ValueError(f"p must be in [{_MIN_P}, {_MAX_P}], got {p}")
        self._p = p
        self._seed = seed
        self._m = 1 << p
        self._alpha_m = _alpha(self._m)
        self._registers: npt.NDArray[np.uint8] = np.zeros(self._m, dtype=np.uint8)

    @property
    def p(self) -> int:
        """Precision parameter ``p``."""
        return self._p

    @property
    def m(self) -> int:
        """Number of registers ``2**p``."""
        return self._m

    @property
    def seed(self) -> int:
        """Hash seed used by this sketch."""
        return self._seed

    @property
    def registers(self) -> npt.NDArray[np.uint8]:
        """Read-only view of the underlying register array."""
        view = self._registers.view()
        view.flags.writeable = False
        return view

    def add(self, item: bytes | str) -> None:
        """Record a single item in the sketch."""
        h = hash64(item, self._seed)
        index, w = split_hash(h, self._p)
        rho = (64 - self._p) - w.bit_length() + 1
        if rho > self._registers[index]:
            self._registers[index] = rho

    def add_many(self, items: Iterable[bytes | str]) -> None:
        """Record multiple items using vectorised NumPy operations.

        Produces registers identical to calling :meth:`add` in a loop.
        """
        hashes = hash64_many(items, self._seed)
        if hashes.size == 0:
            return

        rem_bits = 64 - self._p
        indices = hashes >> np.uint64(rem_bits)
        mask = np.uint64((1 << rem_bits) - 1)
        w = hashes & mask

        # Exact integer bit_length in uint64 via binary search over bit widths
        # (avoids float64 precision loss for values above 2**53).
        bit_len = np.zeros(w.shape, dtype=np.uint8)
        tmp = w.copy()
        for shift in (32, 16, 8, 4, 2, 1):
            cond = tmp >= np.uint64(1 << shift)
            bit_len += cond.astype(np.uint8) * np.uint8(shift)
            tmp = np.where(cond, tmp >> np.uint64(shift), tmp)
        bit_len += (w > 0).astype(np.uint8)

        rhos = (np.uint8(rem_bits + 1) - bit_len).astype(np.uint8)
        np.maximum.at(self._registers, indices, rhos)

    def raw_count(self) -> float:
        """Return the uncorrected HyperLogLog harmonic-mean estimate."""
        inv_powers = np.ldexp(1.0, -self._registers.astype(np.int32))
        return float(self._alpha_m * (self._m**2) / np.sum(inv_powers))

    def count(self) -> float:
        """Return the estimated number of distinct items seen so far.

        Applies the Linear Counting small-range correction when the raw
        estimate ``E <= 2.5 * m`` and at least one register is zero.

        Large-range correction is **not** needed because we use a 64-bit
        hash function: hash-space saturation only becomes non-negligible near
        ``2**64``, far beyond practical web-analytics cardinalities (unlike
        classic 32-bit HyperLogLog where correction is needed above
        ``2**32 / 30``).
        """
        estimate = self.raw_count()
        if estimate <= 2.5 * self._m:
            zeros = int(np.count_nonzero(self._registers == 0))
            if zeros > 0:
                return float(self._m * math.log(self._m / zeros))
        return estimate

    def merge(self, other: Self) -> Self:
        """Merge *other* into this sketch (element-wise max) and return ``self``.

        Raises:
            ValueError: If *other* is not a :class:`HyperLogLog` with matching
                ``p`` and ``seed``.
        """
        if not isinstance(other, HyperLogLog):
            raise ValueError(f"Cannot merge HyperLogLog with {type(other).__name__}")
        if self._p != other._p or self._seed != other._seed:
            raise ValueError(
                f"Incompatible sketches: (p={self._p}, seed={self._seed}) vs "
                f"(p={other._p}, seed={other._seed})"
            )
        np.maximum(self._registers, other._registers, out=self._registers)
        return self

    def standard_error(self) -> float:
        """Return the theoretical standard error ``1.04 / sqrt(m)``."""
        return 1.04 / math.sqrt(self._m)

    def memory_bytes(self) -> int:
        """Return the packed 6-bit serialized register size ``ceil(m * 6 / 8)``.

        Each register holds a rank in ``0..(65 - p) <= 61``, which fits in
        6 bits.  In RAM (:meth:`ram_bytes`), NumPy stores each register as
        a full ``uint8`` byte (33% larger) for fast vectorised indexing.
        """
        return math.ceil(self._m * 6 / 8)

    def ram_bytes(self) -> int:
        """Return the actual in-RAM byte size of the NumPy ``uint8`` register array.

        See :meth:`memory_bytes` for the packed 6-bit size comparison.
        """
        return int(self._registers.nbytes)

    def to_bytes(self) -> bytes:
        """Serialize header (magic, version, p, seed) and 6-bit packed registers."""
        header = _HEADER_STRUCT.pack(_MAGIC, _VERSION, self._p, self._seed)
        return header + _pack_6bit(self._registers)

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        """Deserialize and validate a sketch produced by :meth:`to_bytes`.

        Raises:
            ValueError: If the header or payload is malformed or corrupted.
        """
        if len(data) < _HEADER_STRUCT.size:
            raise ValueError(
                f"Data too short ({len(data)} bytes); expected at least {_HEADER_STRUCT.size}"
            )
        magic, version, p, seed = _HEADER_STRUCT.unpack_from(data, 0)
        if magic != _MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic!r}")
        if version != _VERSION:
            raise ValueError(f"Unsupported version: {version}")
        if not _MIN_P <= p <= _MAX_P:
            raise ValueError(f"Invalid precision p={p}; must be in [{_MIN_P}, {_MAX_P}]")

        m = 1 << p
        expected_payload = math.ceil(m * 6 / 8)
        payload = data[_HEADER_STRUCT.size :]
        if len(payload) != expected_payload:
            raise ValueError(
                f"Payload length mismatch: got {len(payload)} bytes, expected {expected_payload}"
            )

        regs = _unpack_6bit(payload, m)
        max_rho = 65 - p
        if int(regs.max()) > max_rho:
            raise ValueError(f"Corrupted register value > {max_rho} for p={p}")

        sketch = cls(p=p, seed=seed)
        sketch._registers[:] = regs
        return sketch

    def copy(self) -> HyperLogLog:
        """Return an independent deep copy of this sketch."""
        clone = HyperLogLog(p=self._p, seed=self._seed)
        clone._registers[:] = self._registers
        return clone

    def is_empty(self) -> bool:
        """Return ``True`` if no items have been added (all registers zero)."""
        return not bool(np.any(self._registers))

    def clear(self) -> None:
        """Reset all registers to zero."""
        self._registers.fill(0)

    def __len__(self) -> int:
        return int(self.count())

    def __repr__(self) -> str:
        return (
            f"HyperLogLog(p={self._p}, m={self._m}, "
            f"est={self.count():.1f}, se={self.standard_error():.4f})"
        )


def union(a: HyperLogLog, b: HyperLogLog) -> HyperLogLog:
    """Return a new :class:`HyperLogLog` representing the union of *a* and *b*."""
    merged = a.copy()
    merged.merge(b)
    return merged
