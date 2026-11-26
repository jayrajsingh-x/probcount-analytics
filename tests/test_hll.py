"""Tests for :mod:`probcount.hll` — HyperLogLog correctness, properties, and serialization."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from probcount.hll import HyperLogLog, union

# ---------------------------------------------------------------------------
# Basic construction & edge cases
# ---------------------------------------------------------------------------


class TestBasics:
    """Construction, empty state, idempotence, and utility methods."""

    @pytest.mark.parametrize("bad_p", [0, 3, 19, -1])
    def test_invalid_p_raises(self, bad_p: int) -> None:
        with pytest.raises(ValueError, match="p must be"):
            HyperLogLog(p=bad_p)

    def test_empty_sketch(self) -> None:
        hll = HyperLogLog(p=14)
        assert hll.count() == 0.0
        assert len(hll) == 0
        assert hll.is_empty()

    def test_single_item(self) -> None:
        hll = HyperLogLog(p=14)
        hll.add("visitor_1")
        assert not hll.is_empty()
        assert 0.5 <= hll.count() <= 1.5
        assert len(hll) == int(hll.count())

    def test_idempotence_same_item(self) -> None:
        hll = HyperLogLog(p=14)
        for _ in range(1_000):
            hll.add("same_visitor")
        assert 0.5 <= hll.count() <= 1.5

    def test_clear_and_copy(self) -> None:
        hll = HyperLogLog(p=10, seed=7)
        hll.add_many([f"u_{i}" for i in range(100)])
        clone = hll.copy()
        assert np.array_equal(hll.registers, clone.registers)

        hll.clear()
        assert hll.is_empty()
        assert hll.count() == 0.0
        assert not clone.is_empty()

    def test_repr(self) -> None:
        hll = HyperLogLog(p=10)
        text = repr(hll)
        assert "p=10" in text
        assert "est=" in text
        assert "se=" in text

    def test_memory_and_ram_bytes(self) -> None:
        hll = HyperLogLog(p=14)
        assert hll.memory_bytes() == 12_288  # ceil(16384 * 6 / 8)
        assert hll.ram_bytes() == 16_384  # 16384 uint8 registers
        assert math.isclose(hll.standard_error(), 1.04 / math.sqrt(16_384))


# ---------------------------------------------------------------------------
# add_many equivalence
# ---------------------------------------------------------------------------


class TestAddMany:
    """Vectorised add_many must match add() in a loop."""

    def test_add_many_matches_add_loop(self) -> None:
        items: list[bytes | str] = [
            f"user_{i}" if i % 2 == 0 else f"b_{i}".encode() for i in range(5000)
        ]
        h1 = HyperLogLog(p=12, seed=42)
        for item in items:
            h1.add(item)

        h2 = HyperLogLog(p=12, seed=42)
        h2.add_many(items)

        assert np.array_equal(h1.registers, h2.registers)
        assert h1.count() == h2.count()

    def test_add_many_empty_iterable(self) -> None:
        hll = HyperLogLog(p=10)
        hll.add_many([])
        assert hll.is_empty()
        assert hll.count() == 0.0


# ---------------------------------------------------------------------------
# Small-range Linear Counting ablation
# ---------------------------------------------------------------------------


class TestSmallRangeCorrection:
    """Verify Linear Counting significantly improves accuracy for n < 2.5m."""

    def test_linear_counting_vs_raw(self) -> None:
        p = 12  # m = 4096, 2.5m = 10240
        n = 500
        hll = HyperLogLog(p=p, seed=1)
        hll.add_many([f"item_{i}" for i in range(n)])

        raw_err = abs(hll.raw_count() - n) / n
        corrected_err = abs(hll.count() - n) / n

        # Raw HLL formula severely overestimates when V is large (n << m)
        assert raw_err > 0.25
        assert corrected_err < 3 * hll.standard_error()


# ---------------------------------------------------------------------------
# Accuracy across precisions and cardinalities
# ---------------------------------------------------------------------------


def _count_within_3se(p: int, n: int, n_trials: int = 20) -> int:
    """Return how many of *n_trials* fall within 3 * standard_error."""
    hll_ref = HyperLogLog(p=p)
    bound = 3.0 * hll_ref.standard_error()
    items = [f"v_{i}" for i in range(n)]
    within = 0
    for seed in range(n_trials):
        hll = HyperLogLog(p=p, seed=seed)
        hll.add_many(items)
        rel_err = abs(hll.count() - n) / n
        if rel_err <= bound:
            within += 1
    assert within / n_trials >= 0.90, (
        f"p={p}, n={n}: only {within}/{n_trials} within 3*SE ({bound:.4f})"
    )
    return within


class TestAccuracy:
    """Empirical accuracy matches theoretical 1.04 / sqrt(m) bounds."""

    def test_accuracy_grid(self) -> None:
        n_trials = 20
        total_within = 0
        total_trials = 0
        for p in (10, 12, 14, 16):
            for n in (10, 100, 1_000, 10_000, 100_000):
                total_within += _count_within_3se(p=p, n=n, n_trials=n_trials)
                total_trials += n_trials
        assert total_within / total_trials >= 0.99, (
            f"Overall grid: {total_within}/{total_trials} within 3*SE"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("p", [10, 12, 14, 16])
    def test_accuracy_1m(self, p: int) -> None:
        within = _count_within_3se(p=p, n=1_000_000, n_trials=20)
        assert within / 20 >= 0.95


# ---------------------------------------------------------------------------
# Merge laws (Hypothesis property tests)
# ---------------------------------------------------------------------------


class TestMergeLaws:
    """Algebraic laws and union equivalence of HyperLogLog.merge / union."""

    @given(
        stream_a=st.lists(st.text(min_size=1, max_size=20), max_size=200),
        stream_b=st.lists(st.text(min_size=1, max_size=20), max_size=200),
        stream_c=st.lists(st.text(min_size=1, max_size=20), max_size=200),
    )
    @settings(max_examples=50)
    def test_merge_algebraic_laws(
        self, stream_a: list[str], stream_b: list[str], stream_c: list[str]
    ) -> None:
        p = 10
        a = HyperLogLog(p=p, seed=5)
        a.add_many(stream_a)
        b = HyperLogLog(p=p, seed=5)
        b.add_many(stream_b)
        c = HyperLogLog(p=p, seed=5)
        c.add_many(stream_c)

        # Idempotent: union(A, A) == A
        assert np.array_equal(union(a, a).registers, a.registers)

        # Commutative: union(A, B) == union(B, A)
        ab = union(a, b)
        ba = union(b, a)
        assert np.array_equal(ab.registers, ba.registers)

        # Associative: union(union(A, B), C) == union(A, union(B, C))
        ab_c = union(ab, c)
        a_bc = union(a, union(b, c))
        assert np.array_equal(ab_c.registers, a_bc.registers)

        # Exact stream union equivalence
        direct = HyperLogLog(p=p, seed=5)
        direct.add_many(stream_a + stream_b)
        assert np.array_equal(ab.registers, direct.registers)

    def test_overlapping_sets_estimate_union_not_sum(self) -> None:
        shared = [f"shared_{i}" for i in range(5_000)]
        only_a = [f"a_{i}" for i in range(5_000)]
        only_b = [f"b_{i}" for i in range(5_000)]

        a = HyperLogLog(p=14, seed=0)
        a.add_many(shared + only_a)
        b = HyperLogLog(p=14, seed=0)
        b.add_many(shared + only_b)

        merged = union(a, b)
        # True union is 15,000 (not 20,000)
        assert abs(merged.count() - 15_000) / 15_000 < 3 * merged.standard_error()

    def test_merge_mismatch_raises(self) -> None:
        a = HyperLogLog(p=10, seed=0)
        b_diff_p = HyperLogLog(p=12, seed=0)
        b_diff_seed = HyperLogLog(p=10, seed=1)

        with pytest.raises(ValueError, match="Incompatible"):
            a.merge(b_diff_p)
        with pytest.raises(ValueError, match="Incompatible"):
            a.merge(b_diff_seed)
        with pytest.raises(ValueError, match="Cannot merge"):
            a.merge("not_a_sketch")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """to_bytes / from_bytes round-trip and header validation."""

    @pytest.mark.parametrize("p", [4, 10, 14, 18])
    def test_roundtrip_exact(self, p: int) -> None:
        hll = HyperLogLog(p=p, seed=12345)
        hll.add_many([f"k_{i}" for i in range(500)])
        raw = hll.to_bytes()
        assert len(raw) == 14 + hll.memory_bytes()

        restored = HyperLogLog.from_bytes(raw)
        assert restored.p == hll.p
        assert restored.seed == hll.seed
        assert np.array_equal(restored.registers, hll.registers)
        assert restored.count() == hll.count()

    def test_corrupted_bytes_rejected(self) -> None:
        hll = HyperLogLog(p=10, seed=0)
        hll.add("x")
        valid = bytearray(hll.to_bytes())

        # Too short
        with pytest.raises(ValueError, match="too short"):
            HyperLogLog.from_bytes(b"HLL1")

        # Bad magic
        bad_magic = bytearray(valid)
        bad_magic[:4] = b"NOPE"
        with pytest.raises(ValueError, match="magic"):
            HyperLogLog.from_bytes(bytes(bad_magic))

        # Bad version
        bad_ver = bytearray(valid)
        bad_ver[4] = 99
        with pytest.raises(ValueError, match="version"):
            HyperLogLog.from_bytes(bytes(bad_ver))

        # Bad p
        bad_p = bytearray(valid)
        bad_p[5] = 25
        with pytest.raises(ValueError, match="precision"):
            HyperLogLog.from_bytes(bytes(bad_p))

        # Truncated payload
        with pytest.raises(ValueError, match="Payload length"):
            HyperLogLog.from_bytes(bytes(valid[:-1]))

        # Corrupted register value (> 65 - p)
        bad_reg = bytearray(valid)
        bad_reg[14] = 0x3F  # 63 > 65 - 10 = 55
        with pytest.raises(ValueError, match="Corrupted register"):
            HyperLogLog.from_bytes(bytes(bad_reg))
