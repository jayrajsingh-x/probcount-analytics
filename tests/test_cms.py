"""Tests for :mod:`probcount.cms` — Count-Min Sketch properties and guarantees."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from probcount.cms import CountMinSketch


class TestBasics:
    """Construction, empty state, error helpers, and validation."""

    def test_invalid_dimensions_raise(self) -> None:
        with pytest.raises(ValueError, match="width"):
            CountMinSketch(width=0, depth=3)
        with pytest.raises(ValueError, match="depth"):
            CountMinSketch(width=10, depth=0)

    def test_from_error_sizing_and_validation(self) -> None:
        cms = CountMinSketch.from_error(epsilon=0.01, delta=0.01, seed=7)
        assert cms.width == math.ceil(math.e / 0.01)
        assert cms.depth == math.ceil(math.log(1.0 / 0.01))
        assert cms.confidence() >= 0.99

        with pytest.raises(ValueError, match="epsilon"):
            CountMinSketch.from_error(epsilon=0.0, delta=0.01)
        with pytest.raises(ValueError, match="epsilon"):
            CountMinSketch.from_error(epsilon=1.0, delta=0.01)
        with pytest.raises(ValueError, match="delta"):
            CountMinSketch.from_error(epsilon=0.01, delta=0.0)
        with pytest.raises(ValueError, match="delta"):
            CountMinSketch.from_error(epsilon=0.01, delta=1.0)

    def test_empty_sketch(self) -> None:
        cms = CountMinSketch(width=100, depth=4)
        assert cms.total() == 0
        assert cms.estimate("missing") == 0
        assert cms.estimate(b"missing") == 0
        assert cms.error_bound() == 0.0
        assert cms.memory_bytes() == 100 * 4 * 8
        assert "CountMinSketch(" in repr(cms)

    def test_negative_count_rejected(self) -> None:
        std = CountMinSketch(width=50, depth=3, conservative=False)
        cons = CountMinSketch(width=50, depth=3, conservative=True)
        with pytest.raises(ValueError, match="count"):
            std.add("x", -1)
        with pytest.raises(ValueError, match="count"):
            cons.add("x", -1)
        with pytest.raises(ValueError, match=">= 0"):
            std.add_many(["x", "y"], [1, -2])

    def test_zero_count_is_noop(self) -> None:
        cms = CountMinSketch(width=20, depth=3)
        cms.add("x", 0)
        assert cms.total() == 0
        assert cms.estimate("x") == 0


class TestNeverUnderCounts:
    """Hard property: estimate(x) >= true_count(x) for ALL items (zero tolerance)."""

    @given(
        events=st.lists(
            st.tuples(st.text(min_size=1, max_size=12), st.integers(min_value=1, max_value=20)),
            max_size=150,
        ),
        conservative=st.booleans(),
        km=st.booleans(),
    )
    @settings(max_examples=200)
    def test_never_undercounts_property(
        self,
        events: list[tuple[str, int]],
        conservative: bool,
        km: bool,
    ) -> None:
        cms = CountMinSketch(
            width=16,
            depth=3,
            seed=42,
            conservative=conservative,
            kirsch_mitzenmacher=km,
        )
        truth: Counter[str] = Counter()
        for item, cnt in events:
            cms.add(item, cnt)
            truth[item] += cnt

        assert cms.total() == sum(truth.values())
        for item, true_cnt in truth.items():
            assert cms.estimate(item) >= true_cnt


class TestConservativeAndKM:
    """Conservative update tightness and Kirsch-Mitzenmacher mode."""

    def test_conservative_le_standard_ge_truth(self) -> None:
        rng = np.random.default_rng(42)
        ranks = rng.zipf(1.3, size=5_000)
        items = [f"p_{r}" for r in ranks]
        truth = Counter(items)

        std = CountMinSketch(width=64, depth=4, seed=7, conservative=False)
        cons = CountMinSketch(width=64, depth=4, seed=7, conservative=True)
        std.add_many(items)
        cons.add_many(items)

        for item, true_cnt in truth.items():
            e_std = std.estimate(item)
            e_cons = cons.estimate(item)
            assert true_cnt <= e_cons <= e_std

    def test_add_many_equals_add_loop_standard_and_km(self) -> None:
        items = [f"k_{i % 30}" for i in range(500)]
        counts = [(i % 5) + 1 for i in range(500)]
        for km in (False, True):
            c1 = CountMinSketch(width=32, depth=4, seed=9, kirsch_mitzenmacher=km)
            for item, cnt in zip(items, counts, strict=True):
                c1.add(item, cnt)

            c2 = CountMinSketch(width=32, depth=4, seed=9, kirsch_mitzenmacher=km)
            c2.add_many(items, counts)

            assert np.array_equal(c1.table, c2.table)
            assert c1.total() == c2.total()
            est_many = c2.estimate_many(["k_0", "k_1", "missing"])
            assert list(est_many) == [c2.estimate("k_0"), c2.estimate("k_1"), 0]


class TestEpsilonDeltaBound:
    """Verify empirical (epsilon, delta) guarantee."""

    def test_epsilon_delta_guarantee_fast(self) -> None:
        epsilon = 0.005
        delta = 0.02
        rng = np.random.default_rng(123)
        items = [f"item_{r}" for r in rng.zipf(1.2, size=50_000)]
        truth = Counter(items)

        violations = 0
        total_queries = 0
        for seed in range(20):
            cms = CountMinSketch.from_error(epsilon=epsilon, delta=delta, seed=seed)
            cms.add_many(items)
            bound = epsilon * cms.total()
            ests = cms.estimate_many(list(truth.keys()))
            for true_cnt, est in zip(truth.values(), ests, strict=True):
                total_queries += 1
                if (int(est) - true_cnt) > bound:
                    violations += 1

        assert violations / total_queries <= delta + 0.01

    @pytest.mark.slow
    def test_epsilon_delta_guarantee_1m(self) -> None:
        epsilon = 0.001
        delta = 0.01
        rng = np.random.default_rng(7)
        items = [f"u_{r}" for r in rng.zipf(1.2, size=1_000_000)]
        truth = Counter(items)
        keys = list(truth.keys())

        violations = 0
        total_queries = 0
        for seed in range(20):
            cms = CountMinSketch.from_error(epsilon=epsilon, delta=delta, seed=seed)
            cms.add_many(items)
            bound = epsilon * cms.total()
            ests = cms.estimate_many(keys)
            for true_cnt, est in zip(truth.values(), ests, strict=True):
                total_queries += 1
                if (int(est) - true_cnt) > bound:
                    violations += 1

        assert violations / total_queries <= delta + 0.005


class TestMergeAndSerialization:
    """Exact merge in standard mode and validated binary serialization."""

    def test_merge_exact_standard(self) -> None:
        a = [f"a_{i % 20}" for i in range(200)]
        b = [f"b_{i % 25}" for i in range(300)]

        s1 = CountMinSketch(width=64, depth=4, seed=11)
        s1.add_many(a)
        s2 = CountMinSketch(width=64, depth=4, seed=11)
        s2.add_many(b)
        s1.merge(s2)

        direct = CountMinSketch(width=64, depth=4, seed=11)
        direct.add_many(a + b)

        assert np.array_equal(s1.table, direct.table)
        assert s1.total() == direct.total()

    def test_merge_mismatch_raises(self) -> None:
        base = CountMinSketch(width=32, depth=4, seed=0)
        with pytest.raises(ValueError, match="Incompatible"):
            base.merge(CountMinSketch(width=64, depth=4, seed=0))
        with pytest.raises(ValueError, match="Incompatible"):
            base.merge(CountMinSketch(width=32, depth=5, seed=0))
        with pytest.raises(ValueError, match="Incompatible"):
            base.merge(CountMinSketch(width=32, depth=4, seed=1))
        with pytest.raises(ValueError, match="Cannot merge"):
            base.merge("invalid")  # type: ignore[arg-type]

    def test_serialization_roundtrip_and_corruption(self) -> None:
        cms = CountMinSketch(
            width=40, depth=3, seed=55, conservative=True, kirsch_mitzenmacher=True
        )
        cms.add_many(["alpha", "beta", "alpha", "gamma"])
        raw = cms.to_bytes()

        restored = CountMinSketch.from_bytes(raw)
        assert restored.width == cms.width
        assert restored.depth == cms.depth
        assert restored.seed == cms.seed
        assert restored.conservative == cms.conservative
        assert restored.kirsch_mitzenmacher == cms.kirsch_mitzenmacher
        assert restored.total() == cms.total()
        assert np.array_equal(restored.table, cms.table)

        with pytest.raises(ValueError, match="too short"):
            CountMinSketch.from_bytes(b"CMS1")

        bad_magic = bytearray(raw)
        bad_magic[:4] = b"BAD!"
        with pytest.raises(ValueError, match="magic"):
            CountMinSketch.from_bytes(bytes(bad_magic))

        bad_ver = bytearray(raw)
        bad_ver[4] = 9
        with pytest.raises(ValueError, match="version"):
            CountMinSketch.from_bytes(bytes(bad_ver))

        with pytest.raises(ValueError, match="Payload length"):
            CountMinSketch.from_bytes(raw[:-1])
