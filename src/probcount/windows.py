"""Time-bucketed HyperLogLog and Count-Min Sketch windows.

Supports configurable bucket granularities (minute, hour, day), lazy bucket
creation, retention pruning, multi-resolution roll-up (60m -> 1h -> 1d),
watermark-based late-event rejection, and non-mutating range queries.
"""

from __future__ import annotations

import logging
from typing import Literal, Self

from probcount.cms import CountMinSketch
from probcount.hll import HyperLogLog

logger = logging.getLogger(__name__)

GRANULARITY_SECONDS: dict[str, int] = {
    "minute": 60,
    "hour": 3600,
    "day": 86_400,
}

Unit = Literal["minute", "hour", "day"]


def _bucket_start(ts: float, bucket_seconds: int) -> int:
    return int(ts // bucket_seconds) * bucket_seconds


class TimeBucketedHLL:
    """Time-bucketed collection of :class:`~probcount.hll.HyperLogLog` sketches.

    Query ranges ``[start, end)`` are inclusive of *start* and exclusive of
    *end*; any bucket overlapping ``[start, end)`` is merged into a fresh
    sketch without mutating stored buckets.

    Parameters:
        p: HyperLogLog precision (``4..18``).
        seed: Hash seed.
        bucket_seconds: Primary bucket width in seconds (e.g. 60, 3600, 86400).
        retention_seconds: Drop buckets older than ``max_ts - retention_seconds``.
        watermark_seconds: Reject events older than ``max_ts - watermark_seconds``.
        enable_rollup: Whether to maintain hierarchical hour/day roll-ups.
    """

    def __init__(
        self,
        p: int = 14,
        seed: int = 0,
        bucket_seconds: int = 3600,
        retention_seconds: float = 30 * 86_400.0,
        watermark_seconds: float = 300.0,
        enable_rollup: bool = True,
    ) -> None:
        if bucket_seconds < 1:
            raise ValueError(f"bucket_seconds must be >= 1, got {bucket_seconds}")
        self._p = p
        self._seed = seed
        self._bucket_seconds = bucket_seconds
        self._retention_seconds = retention_seconds
        self._watermark_seconds = watermark_seconds
        self._enable_rollup = enable_rollup

        self._buckets: dict[int, HyperLogLog] = {}
        self._hour_buckets: dict[int, HyperLogLog] = {}
        self._day_buckets: dict[int, HyperLogLog] = {}
        self._max_ts: float | None = None
        self._dropped_late: int = 0

    @property
    def dropped_late(self) -> int:
        """Number of events rejected because they arrived past the watermark."""
        return self._dropped_late

    @property
    def max_timestamp(self) -> float | None:
        """Latest event timestamp accepted."""
        return self._max_ts

    @property
    def bucket_seconds(self) -> int:
        """Primary bucket duration in seconds."""
        return self._bucket_seconds

    @property
    def buckets(self) -> dict[int, HyperLogLog]:
        """Primary ``bucket_start -> HyperLogLog`` mapping."""
        return self._buckets

    def _prune(self) -> None:
        if self._max_ts is None:
            return
        cutoff = self._max_ts - self._retention_seconds
        for store in (self._buckets, self._hour_buckets, self._day_buckets):
            expired = [b for b in store if b + self._bucket_seconds <= cutoff]
            for b in expired:
                del store[b]

    def add(self, timestamp: float, item: bytes | str) -> bool:
        """Add *item* at *timestamp*. Returns ``False`` if dropped as late."""
        if self._max_ts is not None and timestamp < self._max_ts - self._watermark_seconds:
            self._dropped_late += 1
            return False

        self._max_ts = timestamp if self._max_ts is None else max(self._max_ts, timestamp)
        b_start = _bucket_start(timestamp, self._bucket_seconds)
        sketch = self._buckets.get(b_start)
        if sketch is None:
            sketch = HyperLogLog(p=self._p, seed=self._seed)
            self._buckets[b_start] = sketch
        sketch.add(item)
        self._prune()
        return True

    def rollup(self) -> None:
        """Merge completed finer buckets into hour (3600s) and day (86400s) tiers.

        For HyperLogLog, ``max``-merging is lossless: the rolled-up sketch
        is bit-identical to merging all constituent minute/hour buckets.
        """
        if not self._enable_rollup or self._max_ts is None:
            return

        if self._bucket_seconds == 60:
            current_hour = _bucket_start(self._max_ts, 3600)
            for b_start, sketch in self._buckets.items():
                h_start = _bucket_start(float(b_start), 3600)
                if h_start < current_hour:
                    target = self._hour_buckets.get(h_start)
                    if target is None:
                        target = HyperLogLog(p=self._p, seed=self._seed)
                        self._hour_buckets[h_start] = target
                    target.merge(sketch)

        source_for_day = (
            self._hour_buckets
            if self._bucket_seconds == 60
            else (self._buckets if self._bucket_seconds == 3600 else {})
        )
        if source_for_day:
            current_day = _bucket_start(self._max_ts, 86_400)
            for b_start, sketch in source_for_day.items():
                d_start = _bucket_start(float(b_start), 86_400)
                if d_start < current_day:
                    target = self._day_buckets.get(d_start)
                    if target is None:
                        target = HyperLogLog(p=self._p, seed=self._seed)
                        self._day_buckets[d_start] = target
                    target.merge(sketch)

    def query(self, start: float, end: float) -> HyperLogLog:
        """Return a new merged :class:`HyperLogLog` covering ``[start, end)``."""
        result = HyperLogLog(p=self._p, seed=self._seed)
        if start >= end:
            return result

        # Use rolled-up day/hour buckets for fully contained intervals when available
        covered: list[tuple[int, int]] = []
        if self._enable_rollup and self._day_buckets:
            for d_start, sketch in self._day_buckets.items():
                if d_start >= start and d_start + 86_400 <= end:
                    result.merge(sketch)
                    covered.append((d_start, d_start + 86_400))

        if self._enable_rollup and self._hour_buckets:
            for h_start, sketch in self._hour_buckets.items():
                if (
                    h_start >= start
                    and h_start + 3600 <= end
                    and not any(c0 <= h_start and h_start + 3600 <= c1 for c0, c1 in covered)
                ):
                    result.merge(sketch)
                    covered.append((h_start, h_start + 3600))

        for b_start, sketch in self._buckets.items():
            b_end = b_start + self._bucket_seconds
            if b_start < end and b_end > start:
                if any(c0 <= b_start and b_end <= c1 for c0, c1 in covered):
                    continue
                result.merge(sketch)

        return result

    def last(self, n: int, unit: Unit, now: float | None = None) -> HyperLogLog:
        """Query the rolling window of the last *n* *unit*s."""
        ref = self._max_ts if now is None else now
        if ref is None:
            return HyperLogLog(p=self._p, seed=self._seed)
        span = n * GRANULARITY_SECONDS[unit]
        return self.query(ref - span, ref + 1e-9)

    def merge(self, other: Self) -> Self:
        """Merge another :class:`TimeBucketedHLL` into this one."""
        if self._p != other._p or self._seed != other._seed:
            raise ValueError("Incompatible TimeBucketedHLL (p or seed mismatch)")
        if self._bucket_seconds != other._bucket_seconds:
            raise ValueError("Incompatible bucket_seconds")
        for b_start, sketch in other._buckets.items():
            target = self._buckets.get(b_start)
            if target is None:
                self._buckets[b_start] = sketch.copy()
            else:
                target.merge(sketch)
        if other._max_ts is not None:
            self._max_ts = (
                other._max_ts if self._max_ts is None else max(self._max_ts, other._max_ts)
            )
        self._dropped_late += other._dropped_late
        self._prune()
        return self

    def memory_bytes(self) -> int:
        """Return total packed bytes across all stored buckets."""
        return (
            sum(s.memory_bytes() for s in self._buckets.values())
            + sum(s.memory_bytes() for s in self._hour_buckets.values())
            + sum(s.memory_bytes() for s in self._day_buckets.values())
        )


class TimeBucketedCMS:
    """Time-bucketed collection of :class:`~probcount.cms.CountMinSketch` tables."""

    def __init__(
        self,
        width: int = 1024,
        depth: int = 4,
        seed: int = 0,
        conservative: bool = True,
        kirsch_mitzenmacher: bool = False,
        bucket_seconds: int = 3600,
        retention_seconds: float = 30 * 86_400.0,
        watermark_seconds: float = 300.0,
        enable_rollup: bool = True,
    ) -> None:
        if bucket_seconds < 1:
            raise ValueError(f"bucket_seconds must be >= 1, got {bucket_seconds}")
        self._width = width
        self._depth = depth
        self._seed = seed
        self._conservative = conservative
        self._km = kirsch_mitzenmacher
        self._bucket_seconds = bucket_seconds
        self._retention_seconds = retention_seconds
        self._watermark_seconds = watermark_seconds
        self._enable_rollup = enable_rollup

        self._buckets: dict[int, CountMinSketch] = {}
        self._hour_buckets: dict[int, CountMinSketch] = {}
        self._day_buckets: dict[int, CountMinSketch] = {}
        self._max_ts: float | None = None
        self._dropped_late: int = 0

    @property
    def dropped_late(self) -> int:
        """Number of events rejected because they arrived past the watermark."""
        return self._dropped_late

    @property
    def buckets(self) -> dict[int, CountMinSketch]:
        """Primary ``bucket_start -> CountMinSketch`` mapping."""
        return self._buckets

    def _new_cms(self) -> CountMinSketch:
        return CountMinSketch(
            width=self._width,
            depth=self._depth,
            seed=self._seed,
            conservative=self._conservative,
            kirsch_mitzenmacher=self._km,
        )

    def _prune(self) -> None:
        if self._max_ts is None:
            return
        cutoff = self._max_ts - self._retention_seconds
        for store in (self._buckets, self._hour_buckets, self._day_buckets):
            expired = [b for b in store if b + self._bucket_seconds <= cutoff]
            for b in expired:
                del store[b]

    def add(self, timestamp: float, item: bytes | str, count: int = 1) -> bool:
        """Add *count* of *item* at *timestamp*. Returns ``False`` if dropped late."""
        if self._max_ts is not None and timestamp < self._max_ts - self._watermark_seconds:
            self._dropped_late += 1
            return False

        self._max_ts = timestamp if self._max_ts is None else max(self._max_ts, timestamp)
        b_start = _bucket_start(timestamp, self._bucket_seconds)
        sketch = self._buckets.get(b_start)
        if sketch is None:
            sketch = self._new_cms()
            self._buckets[b_start] = sketch
        sketch.add(item, count)
        self._prune()
        return True

    def rollup(self) -> None:
        """Merge completed finer buckets into hour/day tiers."""
        if not self._enable_rollup or self._max_ts is None:
            return

        if self._bucket_seconds == 60:
            current_hour = _bucket_start(self._max_ts, 3600)
            for b_start, sketch in self._buckets.items():
                h_start = _bucket_start(float(b_start), 3600)
                if h_start < current_hour:
                    target = self._hour_buckets.get(h_start)
                    if target is None:
                        target = self._new_cms()
                        self._hour_buckets[h_start] = target
                    target.merge(sketch)

        source_for_day = (
            self._hour_buckets
            if self._bucket_seconds == 60
            else (self._buckets if self._bucket_seconds == 3600 else {})
        )
        if source_for_day:
            current_day = _bucket_start(self._max_ts, 86_400)
            for b_start, sketch in source_for_day.items():
                d_start = _bucket_start(float(b_start), 86_400)
                if d_start < current_day:
                    target = self._day_buckets.get(d_start)
                    if target is None:
                        target = self._new_cms()
                        self._day_buckets[d_start] = target
                    target.merge(sketch)

    def query(self, start: float, end: float) -> CountMinSketch:
        """Return a new merged :class:`CountMinSketch` covering ``[start, end)``."""
        result = self._new_cms()
        if start >= end:
            return result

        covered: list[tuple[int, int]] = []
        if self._enable_rollup and self._day_buckets:
            for d_start, sketch in self._day_buckets.items():
                if d_start >= start and d_start + 86_400 <= end:
                    result.merge(sketch)
                    covered.append((d_start, d_start + 86_400))

        if self._enable_rollup and self._hour_buckets:
            for h_start, sketch in self._hour_buckets.items():
                if (
                    h_start >= start
                    and h_start + 3600 <= end
                    and not any(c0 <= h_start and h_start + 3600 <= c1 for c0, c1 in covered)
                ):
                    result.merge(sketch)
                    covered.append((h_start, h_start + 3600))

        for b_start, sketch in self._buckets.items():
            b_end = b_start + self._bucket_seconds
            if b_start < end and b_end > start:
                if any(c0 <= b_start and b_end <= c1 for c0, c1 in covered):
                    continue
                result.merge(sketch)

        return result

    def last(self, n: int, unit: Unit, now: float | None = None) -> CountMinSketch:
        """Query the rolling window of the last *n* *unit*s."""
        ref = self._max_ts if now is None else now
        if ref is None:
            return self._new_cms()
        span = n * GRANULARITY_SECONDS[unit]
        return self.query(ref - span, ref + 1e-9)

    def merge(self, other: Self) -> Self:
        """Merge another :class:`TimeBucketedCMS` into this one."""
        if (
            self._width != other._width
            or self._depth != other._depth
            or self._seed != other._seed
            or self._bucket_seconds != other._bucket_seconds
        ):
            raise ValueError("Incompatible TimeBucketedCMS parameters")
        for b_start, sketch in other._buckets.items():
            target = self._buckets.get(b_start)
            if target is None:
                clone = self._new_cms()
                clone.merge(sketch)
                self._buckets[b_start] = clone
            else:
                target.merge(sketch)
        if other._max_ts is not None:
            self._max_ts = (
                other._max_ts if self._max_ts is None else max(self._max_ts, other._max_ts)
            )
        self._dropped_late += other._dropped_late
        self._prune()
        return self

    def memory_bytes(self) -> int:
        """Return total bytes across all stored CMS buckets."""
        return (
            sum(s.memory_bytes() for s in self._buckets.values())
            + sum(s.memory_bytes() for s in self._hour_buckets.values())
            + sum(s.memory_bytes() for s in self._day_buckets.values())
        )
