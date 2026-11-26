"""Tests for :mod:`probcount.topk`, :mod:`probcount.space_saving`, and :mod:`probcount.metrics`."""

from __future__ import annotations

import numpy as np
import pytest

from probcount.cms import CountMinSketch
from probcount.exact import exact_top_k
from probcount.metrics import (
    absolute_error,
    overestimation_ratio,
    precision_at_k,
    recall_at_k,
    relative_error,
    spearman_correlation,
)
from probcount.space_saving import SpaceSaving
from probcount.topk import TopKTracker


class TestTopKTracker:
    """Top-K heavy-hitter accuracy, sorting, merging, and memory."""

    def test_invalid_k_raises(self) -> None:
        cms = CountMinSketch(width=64, depth=4)
        with pytest.raises(ValueError, match="k must be"):
            TopKTracker(k=0, sketch=cms)

    def test_zipf_precision_and_recall_at_10(self) -> None:
        rng = np.random.default_rng(42)
        items = [f"page_{r}" for r in rng.zipf(1.3, size=30_000)]

        cms = CountMinSketch(width=256, depth=4, seed=42, conservative=True)
        tracker = TopKTracker(k=25, sketch=cms)
        for item in items:
            tracker.add(item)

        pred = tracker.top(10)
        truth = exact_top_k(items, 10)

        # Sorted descending by estimated count
        counts = [c for _, c in pred]
        assert counts == sorted(counts, reverse=True)

        pred_keys = [k for k, _ in pred]
        truth_keys = [k for k, _ in truth]
        assert precision_at_k(pred_keys, truth_keys, 10) >= 0.9
        assert recall_at_k(pred_keys, truth_keys, 10) >= 0.9
        assert tracker.k == 25
        assert tracker.sketch is cms
        assert tracker.top(0) == []
        assert "TopKTracker(" in repr(tracker)
        assert tracker.memory_bytes() > cms.memory_bytes()

    def test_merge_combines_heavy_hitters(self) -> None:
        rng = np.random.default_rng(7)
        stream_a = [f"p_{r}" for r in rng.zipf(1.25, size=10_000)]
        stream_b = [f"p_{r}" for r in rng.zipf(1.25, size=10_000)]

        t1 = TopKTracker(k=20, sketch=CountMinSketch(width=256, depth=4, seed=1))
        for item in stream_a:
            t1.add(item)

        t2 = TopKTracker(k=20, sketch=CountMinSketch(width=256, depth=4, seed=1))
        for item in stream_b:
            t2.add(item)

        t1.merge(t2)
        pred_keys = [k for k, _ in t1.top(10)]
        truth_keys = [k for k, _ in exact_top_k(stream_a + stream_b, 10)]
        assert recall_at_k(pred_keys, truth_keys, 10) >= 0.8

    def test_merge_mismatch_raises(self) -> None:
        t1 = TopKTracker(k=5, sketch=CountMinSketch(width=32, depth=3))
        t2 = TopKTracker(k=10, sketch=CountMinSketch(width=32, depth=3))
        with pytest.raises(ValueError, match="Incompatible k"):
            t1.merge(t2)
        with pytest.raises(ValueError, match="Cannot merge"):
            t1.merge("bad")  # type: ignore[arg-type]


class TestSpaceSaving:
    """Space-Saving baseline tests."""

    def test_zipf_top_k_and_merge(self) -> None:
        rng = np.random.default_rng(99)
        items = [f"p_{r}" for r in rng.zipf(1.3, size=20_000)]
        ss = SpaceSaving(k=100)
        for item in items:
            ss.add(item)

        assert ss.k == 100
        assert ss.total() == 20_000
        pred_keys = [k for k, _ in ss.top_k(10)]
        truth_keys = [k for k, _ in exact_top_k(items, 10)]
        assert precision_at_k(pred_keys, truth_keys, 10) >= 0.9
        assert recall_at_k(pred_keys, truth_keys, 10) >= 0.9
        assert ss.estimate("p_1") >= 1
        assert ss.error_bound("p_1") >= 0
        assert ss.top_k(0) == []
        assert ss.memory_bytes() > 0
        assert "SpaceSaving(" in repr(ss)

        ss2 = SpaceSaving(k=100)
        ss2.add("p_1", 500)
        ss.merge(ss2)
        assert ss.total() == 20_500

    def test_space_saving_validation(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            SpaceSaving(k=0)
        ss = SpaceSaving(k=5)
        with pytest.raises(ValueError, match="count"):
            ss.add("x", -1)
        ss.add("x", 0)
        assert ss.total() == 0
        with pytest.raises(ValueError, match="Cannot merge"):
            ss.merge("bad")  # type: ignore[arg-type]


class TestMetrics:
    """Accuracy and ranking metric helpers."""

    def test_error_metrics(self) -> None:
        assert absolute_error(12, 10) == 2
        assert relative_error(12, 10) == pytest.approx(0.2)
        assert relative_error(0, 0) == 0.0
        assert overestimation_ratio(15, 10) == pytest.approx(1.5)
        assert overestimation_ratio(0, 0) == 1.0

        with pytest.raises(ValueError):
            relative_error(5, 0)
        with pytest.raises(ValueError):
            overestimation_ratio(5, 0)

    def test_spearman_correlation(self) -> None:
        assert spearman_correlation([10, 20, 30, 40], [1, 2, 3, 4]) == pytest.approx(1.0)
        assert spearman_correlation([10, 20, 30, 40], [4, 3, 2, 1]) == pytest.approx(-1.0)
        with pytest.raises(ValueError):
            spearman_correlation([1, 2], [1])
