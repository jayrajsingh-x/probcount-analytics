"""Tests for :mod:`probcount.windows` — TimeBucketedHLL and TimeBucketedCMS."""

from __future__ import annotations

import numpy as np
import pytest

from probcount.hll import HyperLogLog
from probcount.windows import TimeBucketedCMS, TimeBucketedHLL


class TestTimeBucketedHLL:
    """Bucket alignment, range merge, retention, roll-up, and late events."""

    def test_invalid_bucket_seconds(self) -> None:
        with pytest.raises(ValueError, match="bucket_seconds"):
            TimeBucketedHLL(bucket_seconds=0)
        with pytest.raises(ValueError, match="bucket_seconds"):
            TimeBucketedCMS(bucket_seconds=0)

    def test_query_equals_manual_merge_and_non_mutating(self) -> None:
        tb = TimeBucketedHLL(p=10, seed=1, bucket_seconds=3600, enable_rollup=False)
        tb.add(100.0, "u1")
        tb.add(3700.0, "u2")
        tb.add(7300.0, "u3")

        q = tb.query(0.0, 7200.0)
        manual = HyperLogLog(p=10, seed=1)
        manual.merge(tb.buckets[0])
        manual.merge(tb.buckets[3600])

        assert np.array_equal(q.registers, manual.registers)
        # Stored buckets were not mutated
        assert len(tb.buckets[0]) == 1
        assert len(tb.buckets[3600]) == 1
        assert tb.query(7200.0, 7200.0).is_empty()

    def test_retention_prunes_old_buckets(self) -> None:
        tb = TimeBucketedHLL(
            p=8, bucket_seconds=60, retention_seconds=180, watermark_seconds=1000
        )
        tb.add(0.0, "a")
        tb.add(60.0, "b")
        tb.add(300.0, "c")  # cutoff = 300 - 180 = 120; bucket 0 & 60 pruned
        assert 0 not in tb.buckets
        assert 60 not in tb.buckets
        assert 300 in tb.buckets

    def test_rollup_matches_raw_buckets(self) -> None:
        tb = TimeBucketedHLL(p=10, seed=5, bucket_seconds=60, retention_seconds=10 * 86_400)
        for m in range(130):
            tb.add(float(m * 60), f"u_{m % 40}")
        raw = tb.query(0.0, 7200.0)

        tb.rollup()
        rolled = tb.query(0.0, 7200.0)
        assert np.array_equal(rolled.registers, raw.registers)

    def test_late_event_watermark(self) -> None:
        tb = TimeBucketedHLL(p=8, bucket_seconds=60, watermark_seconds=120.0)
        assert tb.add(1000.0, "u1") is True
        assert tb.add(900.0, "u2") is True  # within 120s watermark
        assert tb.add(800.0, "u3") is False  # 200s late -> dropped
        assert tb.dropped_late == 1
        assert tb.max_timestamp == 1000.0

    def test_last_helper_and_merge(self) -> None:
        tb1 = TimeBucketedHLL(p=10, seed=0, bucket_seconds=3600)
        tb2 = TimeBucketedHLL(p=10, seed=0, bucket_seconds=3600)
        tb1.add(0.0, "a")
        tb2.add(3600.0, "b")
        tb1.merge(tb2)
        assert tb1.last(2, "hour").count() >= 1.5
        assert tb1.memory_bytes() > 0

        # Empty last() and mismatch merge
        empty_tb = TimeBucketedHLL(p=8, bucket_seconds=3600)
        assert empty_tb.last(1, "hour").is_empty()
        with pytest.raises(ValueError, match="Incompatible"):
            tb1.merge(empty_tb)

    def test_day_rollup_from_hour_buckets(self) -> None:
        tb = TimeBucketedHLL(p=10, seed=2, bucket_seconds=3600, retention_seconds=10 * 86_400)
        for h in range(30):
            tb.add(float(h * 3600), f"u_{h % 15}")
        raw = tb.query(0.0, 86_400.0)
        tb.rollup()
        rolled = tb.query(0.0, 86_400.0)
        assert np.array_equal(raw.registers, rolled.registers)


class TestTimeBucketedCMS:
    """TimeBucketedCMS roll-up, watermark, and merge."""

    def test_rollup_and_watermark(self) -> None:
        cms = TimeBucketedCMS(
            width=64, depth=3, seed=0, conservative=False, bucket_seconds=60, watermark_seconds=60
        )
        for m in range(130):
            cms.add(float(m * 60), "home", 1)
        assert cms.add(0.0, "home", 1) is False
        assert cms.dropped_late == 1

        raw = cms.query(0.0, 7200.0)
        cms.rollup()
        rolled = cms.query(0.0, 7200.0)
        assert np.array_equal(raw.table, rolled.table)
        assert cms.last(3, "hour").estimate("home") >= 120
        assert cms.memory_bytes() > 0

        # Day rollup from hour buckets & merge
        cms_h = TimeBucketedCMS(
            width=32, depth=3, seed=0, conservative=False, bucket_seconds=3600
        )
        for h in range(30):
            cms_h.add(float(h * 3600), "p1", 1)
        raw_d = cms_h.query(0.0, 86_400.0)
        cms_h.rollup()
        rolled_d = cms_h.query(0.0, 86_400.0)
        assert np.array_equal(raw_d.table, rolled_d.table)

        cms_other = TimeBucketedCMS(
            width=32, depth=3, seed=0, conservative=False, bucket_seconds=3600
        )
        cms_other.add(float(35 * 3600), "p1", 2)
        cms_h.merge(cms_other)
        assert cms_h.query(0.0, 40 * 3600.0).estimate("p1") >= 32
