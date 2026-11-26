"""Tests for :mod:`probcount.setops` — HLL union, intersection, and Jaccard."""

from __future__ import annotations

from probcount.hll import HyperLogLog
from probcount.setops import hll_intersection, hll_jaccard, hll_union


class TestSetOps:
    """Union, inclusion-exclusion intersection, warning flag, and Jaccard."""

    def test_union_accurate(self) -> None:
        a = HyperLogLog(p=14, seed=0)
        b = HyperLogLog(p=14, seed=0)
        a.add_many([f"u_{i}" for i in range(10_000)])
        b.add_many([f"u_{i}" for i in range(5_000, 15_000)])

        res = hll_union(a, b)
        assert abs(res.estimate - 15_000) / 15_000 < 0.03
        assert res.warning is False

    def test_intersection_small_overlap_triggers_warning(self) -> None:
        a = HyperLogLog(p=14, seed=0)
        b = HyperLogLog(p=14, seed=0)
        a.add_many([f"a_{i}" for i in range(10_000)])
        b.add_many([f"b_{i}" for i in range(10_000)])
        # 100 shared items out of ~20,000 union = 0.5% (< 5%)
        shared = [f"s_{i}" for i in range(100)]
        a.add_many(shared)
        b.add_many(shared)

        res = hll_intersection(a, b)
        assert res.warning is True
        assert res.error_estimate > 0
        assert "Small overlap" in res.message

    def test_jaccard_identical_and_disjoint(self) -> None:
        a = HyperLogLog(p=14, seed=0)
        a.add_many([f"x_{i}" for i in range(5_000)])
        j_same = hll_jaccard(a, a.copy())
        assert j_same.estimate > 0.95
        assert j_same.warning is False

        b = HyperLogLog(p=14, seed=0)
        b.add_many([f"y_{i}" for i in range(5_000)])
        j_disjoint = hll_jaccard(a, b)
        assert j_disjoint.estimate < 0.05

        empty = HyperLogLog(p=14, seed=0)
        assert hll_jaccard(empty, empty).estimate == 0.0
