"""Pure-Python MurmurHash3 x64-128 bit hash function.

Implements the canonical MurmurHash3_x64_128 algorithm by Austin Appleby,
producing a 128-bit hash from which we expose the lower 64 bits via
:func:`hash64`.  All arithmetic is performed with Python arbitrary-precision
integers, masked to 64 bits after every operation to emulate unsigned
64-bit overflow.

Design rationale (see also ``docs/design_notes.md``):

* **Why MurmurHash3?** — It is the hash used by the HyperLogLog++ paper
  (Heule et al., Google 2013).  It has excellent avalanche properties,
  uniform output distribution, and a well-specified reference implementation
  that we can cross-check against the ``mmh3`` C extension.

* **Why 64 bits?** — 64 bits gives a collision probability below 10⁻⁹ for
  cardinalities up to ~10⁸, which is more than sufficient for web-analytics
  workloads.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MASK64: int = 0xFFFFFFFFFFFFFFFF
_C1: int = 0x87C37B91114253D5
_C2: int = 0x4CF5AD432745937F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rotl64(x: int, r: int) -> int:
    """Rotate *x* left by *r* bits within a 64-bit word."""
    return ((x << r) | (x >> (64 - r))) & _MASK64


def _fmix64(k: int) -> int:
    """Finalization mix — force all bits of a hash block to avalanche."""
    k ^= k >> 33
    k = (k * 0xFF51AFD7ED558CCD) & _MASK64
    k ^= k >> 33
    k = (k * 0xC4CEB9FE1A85EC53) & _MASK64
    k ^= k >> 33
    return k


# ---------------------------------------------------------------------------
# Core 128-bit hash
# ---------------------------------------------------------------------------


def hash128(data: bytes, seed: int = 0) -> tuple[int, int]:
    """Compute MurmurHash3 x64-128 of *data* with the given *seed*.

    Parameters:
        data: Raw bytes to hash.
        seed: 32-bit seed value (higher bits are ignored, matching the
              canonical C implementation which takes ``uint32_t``).

    Returns:
        A ``(h1, h2)`` tuple of two unsigned 64-bit integers whose
        concatenation is the full 128-bit hash.
    """
    seed = seed & 0xFFFFFFFF  # canonical uses uint32_t seed
    h1: int = seed
    h2: int = seed

    length = len(data)
    nblocks = length // 16

    # ---------- body ----------
    for i in range(nblocks):
        off = i * 16
        k1 = int.from_bytes(data[off : off + 8], byteorder="little", signed=False)
        k2 = int.from_bytes(data[off + 8 : off + 16], byteorder="little", signed=False)

        k1 = (k1 * _C1) & _MASK64
        k1 = _rotl64(k1, 31)
        k1 = (k1 * _C2) & _MASK64
        h1 ^= k1

        h1 = _rotl64(h1, 27)
        h1 = (h1 + h2) & _MASK64
        h1 = (h1 * 5 + 0x52DCE729) & _MASK64

        k2 = (k2 * _C2) & _MASK64
        k2 = _rotl64(k2, 33)
        k2 = (k2 * _C1) & _MASK64
        h2 ^= k2

        h2 = _rotl64(h2, 31)
        h2 = (h2 + h1) & _MASK64
        h2 = (h2 * 5 + 0x38495AB5) & _MASK64

    # ---------- tail ----------
    tail = data[nblocks * 16 :]
    tlen = len(tail)
    k1 = 0
    k2 = 0

    if tlen >= 15:
        k2 ^= tail[14] << 48
    if tlen >= 14:
        k2 ^= tail[13] << 40
    if tlen >= 13:
        k2 ^= tail[12] << 32
    if tlen >= 12:
        k2 ^= tail[11] << 24
    if tlen >= 11:
        k2 ^= tail[10] << 16
    if tlen >= 10:
        k2 ^= tail[9] << 8
    if tlen >= 9:
        k2 ^= tail[8]
        k2 = (k2 * _C2) & _MASK64
        k2 = _rotl64(k2, 33)
        k2 = (k2 * _C1) & _MASK64
        h2 ^= k2

    if tlen >= 8:
        k1 ^= tail[7] << 56
    if tlen >= 7:
        k1 ^= tail[6] << 48
    if tlen >= 6:
        k1 ^= tail[5] << 40
    if tlen >= 5:
        k1 ^= tail[4] << 32
    if tlen >= 4:
        k1 ^= tail[3] << 24
    if tlen >= 3:
        k1 ^= tail[2] << 16
    if tlen >= 2:
        k1 ^= tail[1] << 8
    if tlen >= 1:
        k1 ^= tail[0]
        k1 = (k1 * _C1) & _MASK64
        k1 = _rotl64(k1, 31)
        k1 = (k1 * _C2) & _MASK64
        h1 ^= k1

    # ---------- finalization ----------
    h1 ^= length
    h2 ^= length

    h1 = (h1 + h2) & _MASK64
    h2 = (h2 + h1) & _MASK64

    h1 = _fmix64(h1)
    h2 = _fmix64(h2)

    h1 = (h1 + h2) & _MASK64
    h2 = (h2 + h1) & _MASK64

    return h1, h2


# ---------------------------------------------------------------------------
# Public convenience API
# ---------------------------------------------------------------------------


def hash64(data: bytes | str, seed: int = 0) -> int:
    """Return the lower 64 bits of MurmurHash3 x64-128 as an unsigned int.

    Parameters:
        data: Input bytes or string (strings are UTF-8 encoded).
        seed: Hash seed (32-bit).

    Returns:
        Unsigned 64-bit integer (``h1`` of the full 128-bit hash).
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    h1, _ = hash128(data, seed)
    return h1


def hash64_many(
    items: Iterable[bytes | str], seed: int = 0
) -> npt.NDArray[np.uint64]:
    """Hash multiple items, returning a NumPy array of ``uint64`` values.

    Parameters:
        items: Iterable of byte strings or text strings.
        seed: Hash seed applied to every item.

    Returns:
        1-D ``numpy.ndarray`` with ``dtype=numpy.uint64``.
    """
    return np.array([hash64(item, seed) for item in items], dtype=np.uint64)


def split_hash(h: int, p: int) -> tuple[int, int]:
    """Split a 64-bit hash into an index and a remaining-bits value.

    Used by HyperLogLog: the most-significant *p* bits determine the
    register index, and the remaining ``64 - p`` bits are the source
    from which the rank (position of the leftmost 1-bit) is computed.

    Parameters:
        h: Unsigned 64-bit hash value.
        p: Number of most-significant bits for the index (must be in 1..63).

    Returns:
        ``(index, remaining_bits)`` where ``0 <= index < 2**p`` and
        ``remaining_bits`` contains the lower ``64 - p`` bits.

    Raises:
        ValueError: If *p* is not in the range ``[1, 63]``.
    """
    if not 1 <= p <= 63:
        raise ValueError(f"p must be in [1, 63], got {p}")
    index = h >> (64 - p)
    remaining = h & ((1 << (64 - p)) - 1)
    return index, remaining


def derive_seeds(n: int, base_seed: int = 0) -> list[int]:
    """Generate *n* independent hash seeds from a single *base_seed*.

    Each seed is produced by hashing the index ``i`` (as an 8-byte
    little-endian integer) with *base_seed*.  This is used to create
    independent hash functions for Count-Min Sketch rows in later phases.

    Parameters:
        n: Number of seeds to generate (must be > 0).
        base_seed: Starting seed value.

    Returns:
        List of *n* distinct unsigned 64-bit seed values.

    Raises:
        ValueError: If *n* is not positive.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    seeds: list[int] = []
    for i in range(n):
        seed_bytes = i.to_bytes(8, byteorder="little")
        seeds.append(hash64(seed_bytes, base_seed))
    return seeds
