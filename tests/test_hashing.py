"""Tests for :mod:`probcount.hashing` — MurmurHash3 x64-128 pure-Python implementation.

Coverage includes:
* Cross-check against the ``mmh3`` C extension on 10,000+ random inputs.
* Known-value tests (empty string, determinism, seed independence).
* Statistical quality: bit uniformity, avalanche, chi-square on top 10 bits.
* Utility helpers: ``split_hash``, ``derive_seeds``, ``hash64_many``.
"""

from __future__ import annotations

import mmh3
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from probcount.hashing import derive_seeds, hash64, hash64_many, hash128, split_hash

# ---------------------------------------------------------------------------
# Cross-check against mmh3
# ---------------------------------------------------------------------------


class TestCrossCheckMmh3:
    """Our pure-Python hash must match the mmh3 C extension exactly."""

    @given(
        data=st.binary(min_size=0, max_size=100),
        seed=st.integers(min_value=0, max_value=0xFFFFFFFF),
    )
    @settings(max_examples=10_000)
    def test_hash64_matches_mmh3(self, data: bytes, seed: int) -> None:
        """hash64 returns the same h1 as mmh3 for random inputs."""
        expected_h1, _ = mmh3.hash64(data, seed, signed=False)
        assert hash64(data, seed) == expected_h1

    @given(
        data=st.binary(min_size=0, max_size=100),
        seed=st.integers(min_value=0, max_value=0xFFFFFFFF),
    )
    @settings(max_examples=5_000)
    def test_hash128_matches_mmh3(self, data: bytes, seed: int) -> None:
        """Full 128-bit hash matches mmh3.hash64 (which returns both halves)."""
        expected = mmh3.hash64(data, seed, signed=False)
        actual = hash128(data, seed)
        assert actual == expected


# ---------------------------------------------------------------------------
# Known-value and determinism tests
# ---------------------------------------------------------------------------


class TestKnownValues:
    """Determinism, seed behaviour, and the empty-string special case."""

    def test_empty_bytes_seed_zero(self) -> None:
        """MurmurHash3 of the empty string with seed 0 is 0."""
        assert hash64(b"", 0) == 0

    def test_empty_string_seed_zero(self) -> None:
        """str variant also returns 0 for empty input."""
        assert hash64("", 0) == 0

    def test_deterministic(self) -> None:
        """Same input + seed always gives the same result."""
        for _ in range(100):
            assert hash64(b"hello world", 42) == hash64(b"hello world", 42)

    def test_string_utf8_encoding(self) -> None:
        """Strings are UTF-8 encoded before hashing."""
        text = "café"
        assert hash64(text, 0) == hash64(text.encode("utf-8"), 0)

    def test_different_seeds_give_different_hashes(self) -> None:
        """Different seeds should almost always produce different outputs."""
        data = b"test input"
        hashes = {hash64(data, seed) for seed in range(1000)}
        # At least 99% should be distinct (allowing a few unlikely collisions)
        assert len(hashes) >= 990

    def test_different_inputs_give_different_hashes(self) -> None:
        """Different inputs with the same seed should produce distinct hashes."""
        hashes = {hash64(f"item_{i}".encode(), 0) for i in range(10_000)}
        assert len(hashes) == 10_000  # zero collisions expected


# ---------------------------------------------------------------------------
# Statistical quality tests
# ---------------------------------------------------------------------------


class TestBitUniformity:
    """Each bit should be 1 roughly 50% of the time."""

    def test_bit_uniformity(self) -> None:
        """Check all 64 bit positions on 100,000 sequential hashes."""
        n = 100_000
        bit_counts = np.zeros(64, dtype=np.int64)
        for i in range(n):
            h = hash64(f"key_{i}".encode(), 0)
            for bit in range(64):
                if h & (1 << bit):
                    bit_counts[bit] += 1

        fractions = bit_counts / n
        # Each bit should be in [0.49, 0.51]
        assert np.all(fractions > 0.49), (
            f"Some bits too low: {fractions[fractions <= 0.49]}"
        )
        assert np.all(fractions < 0.51), (
            f"Some bits too high: {fractions[fractions >= 0.51]}"
        )


class TestAvalanche:
    """Flipping one input bit should flip ~32 output bits on average."""

    def test_avalanche_effect(self) -> None:
        n = 10_000
        total_flipped = 0
        total_bits_tested = 0
        rng = np.random.default_rng(42)

        for _ in range(n):
            # Random 8-byte input
            original = rng.bytes(8)
            h_orig = hash64(original, 0)

            # Flip a random bit
            byte_idx = rng.integers(0, 8)
            bit_idx = rng.integers(0, 8)
            mutated = bytearray(original)
            mutated[byte_idx] ^= 1 << bit_idx
            h_mut = hash64(bytes(mutated), 0)

            diff = h_orig ^ h_mut
            total_flipped += bin(diff).count("1")
            total_bits_tested += 1

        avg_flipped = total_flipped / total_bits_tested
        # Should be ~32 ± 1
        assert 31.0 <= avg_flipped <= 33.0, f"Average bits flipped: {avg_flipped}"


class TestChiSquare:
    """Top 10 bits should be uniformly distributed (chi-square test)."""

    def test_chi_square_top_10_bits(self) -> None:
        """Bucket 100,000 hashes by their top 10 bits and run chi-square."""
        n = 100_000
        n_bins = 1024  # 2^10
        counts = np.zeros(n_bins, dtype=np.int64)

        for i in range(n):
            h = hash64(f"chi_{i}".encode(), 0)
            bucket = h >> 54  # top 10 bits
            counts[bucket] += 1

        expected = n / n_bins
        chi2 = float(np.sum((counts - expected) ** 2 / expected))
        # Degrees of freedom = 1023
        # For p=0.01, critical value ≈ 1118 (χ² table, df=1023)
        assert chi2 < 1200, f"Chi-square too high: {chi2}"


# ---------------------------------------------------------------------------
# split_hash
# ---------------------------------------------------------------------------


class TestSplitHash:
    """Verify split_hash index/remaining decomposition."""

    def test_index_range(self) -> None:
        """Index must be in [0, 2^p)."""
        h = hash64(b"test", 0)
        for p in range(1, 64):
            index, _ = split_hash(h, p)
            assert 0 <= index < (1 << p)

    def test_remaining_range(self) -> None:
        """Remaining bits must be in [0, 2^(64-p))."""
        h = hash64(b"test", 0)
        for p in range(1, 64):
            _, remaining = split_hash(h, p)
            assert 0 <= remaining < (1 << (64 - p))

    def test_reconstruction(self) -> None:
        """Concatenating index and remaining must recover the original hash."""
        h = hash64(b"reconstruct_me", 7)
        for p in range(1, 64):
            index, remaining = split_hash(h, p)
            reconstructed = (index << (64 - p)) | remaining
            assert reconstructed == h, f"Roundtrip failed for p={p}"

    def test_invalid_p_zero(self) -> None:
        with pytest.raises(ValueError, match="p must be"):
            split_hash(42, 0)

    def test_invalid_p_64(self) -> None:
        with pytest.raises(ValueError, match="p must be"):
            split_hash(42, 64)


# ---------------------------------------------------------------------------
# derive_seeds
# ---------------------------------------------------------------------------


class TestDeriveSeeds:
    """Seed derivation must produce distinct, deterministic values."""

    def test_uniqueness(self) -> None:
        seeds = derive_seeds(100, base_seed=0)
        assert len(set(seeds)) == 100

    def test_determinism(self) -> None:
        s1 = derive_seeds(50, base_seed=7)
        s2 = derive_seeds(50, base_seed=7)
        assert s1 == s2

    def test_different_base_seeds(self) -> None:
        s1 = derive_seeds(10, base_seed=0)
        s2 = derive_seeds(10, base_seed=1)
        assert s1 != s2

    def test_invalid_n(self) -> None:
        with pytest.raises(ValueError, match="n must be"):
            derive_seeds(0)


# ---------------------------------------------------------------------------
# hash64_many
# ---------------------------------------------------------------------------


class TestHash64Many:
    """Batch hashing must match individual calls."""

    def test_matches_individual_calls(self) -> None:
        items = [f"item_{i}" for i in range(500)]
        batch = hash64_many(items, seed=0)
        individual = [hash64(item, 0) for item in items]

        assert batch.dtype == np.uint64
        assert len(batch) == len(items)
        np.testing.assert_array_equal(batch, np.array(individual, dtype=np.uint64))

    def test_empty_input(self) -> None:
        result = hash64_many([], seed=0)
        assert len(result) == 0
        assert result.dtype == np.uint64
