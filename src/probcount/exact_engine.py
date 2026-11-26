"""Exact ground-truth analytics engine for evaluating :class:`AnalyticsEngine`."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from probcount.engine import AnalyticsEngine
from probcount.exact import deep_size_bytes
from probcount.generator import Event
from probcount.windows import GRANULARITY_SECONDS, Unit


class ExactAnalyticsEngine:
    """Exact set-and-Counter baseline matching :class:`AnalyticsEngine`'s query API."""

    def __init__(
        self,
        site: str = "default",
        ignore_bots: bool = True,
        bucket_seconds: int = 3600,
    ) -> None:
        self._site = site
        self._ignore_bots = ignore_bots
        self._bucket_seconds = bucket_seconds

        self._site_visitors: dict[int, set[str]] = defaultdict(set)
        self._page_views: dict[int, Counter[str]] = defaultdict(Counter)
        self._page_visitors: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
        self._events_processed: int = 0

    def process_event(self, event: Event) -> None:
        """Record a single event exactly."""
        if self._ignore_bots and event.is_bot:
            return
        b = int(event.timestamp // self._bucket_seconds) * self._bucket_seconds
        self._site_visitors[b].add(event.visitor_id)
        self._page_views[b][event.page_id] += 1
        self._page_visitors[event.page_id][b].add(event.visitor_id)
        self._events_processed += 1

    def process_batch(self, events: Iterable[Event]) -> None:
        """Record a batch of events."""
        for event in events:
            self.process_event(event)

    def unique_visitors(self, site: str, start: float, end: float) -> float:
        """Exact unique visitors for *site* in ``[start, end)``."""
        if site != self._site:
            return 0.0
        seen: set[str] = set()
        for b, visitors in self._site_visitors.items():
            if b < end and (b + self._bucket_seconds) > start:
                seen |= visitors
        return float(len(seen))

    def unique_visitors_page(self, page: str, start: float, end: float) -> float:
        """Exact unique visitors for *page* in ``[start, end)``."""
        buckets = self._page_visitors.get(page)
        if not buckets:
            return 0.0
        seen: set[str] = set()
        for b, visitors in buckets.items():
            if b < end and (b + self._bucket_seconds) > start:
                seen |= visitors
        return float(len(seen))

    def page_views(self, page: str, start: float, end: float) -> int:
        """Exact view count for *page* in ``[start, end)``."""
        total = 0
        for b, counts in self._page_views.items():
            if b < end and (b + self._bucket_seconds) > start:
                total += counts[page]
        return total

    def top_pages(self, k: int, start: float, end: float) -> list[tuple[str, int]]:
        """Exact top *k* pages in ``[start, end)``."""
        if k < 1:
            return []
        merged: Counter[str] = Counter()
        for b, counts in self._page_views.items():
            if b < end and (b + self._bucket_seconds) > start:
                merged += counts
        return merged.most_common(k)

    def views_per_visitor(self, page: str, start: float, end: float) -> float:
        """Exact average views per visitor for *page* in ``[start, end)``."""
        uv = self.unique_visitors_page(page, start, end)
        if uv <= 0.0:
            return 0.0
        return self.page_views(page, start, end) / uv

    def unique_visitors_series(
        self,
        granularity: Unit,
        start: float,
        end: float,
    ) -> list[tuple[float, float]]:
        """Exact unique-visitor time series across ``[start, end)``."""
        step = GRANULARITY_SECONDS[granularity]
        t = float(int(start // step) * step)
        series: list[tuple[float, float]] = []
        while t < end:
            series.append((t, self.unique_visitors(self._site, t, t + step)))
            t += step
        return series

    def new_visitors(self, start: float, end: float) -> float:
        """Exact count of visitors seen in ``[start, end)`` and never before *start*."""
        window_seen: set[str] = set()
        prior_seen: set[str] = set()
        for b, visitors in self._site_visitors.items():
            if b < start:
                prior_seen |= visitors
            elif b < end and (b + self._bucket_seconds) > start:
                window_seen |= visitors
        return float(len(window_seen - prior_seen))

    def memory_bytes(self) -> int:
        """Return total memory footprint of all stored sets and Counters."""
        return (
            deep_size_bytes(self._site_visitors)
            + deep_size_bytes(self._page_views)
            + deep_size_bytes(self._page_visitors)
        )


def compare(
    engine: AnalyticsEngine,
    exact: ExactAnalyticsEngine,
    query: str,
    *args: Any,
    **kwargs: Any,
) -> dict[str, float]:
    """Compare a scalar query across sketch and exact engines."""
    sketch_val = float(getattr(engine, query)(*args, **kwargs))
    exact_val = float(getattr(exact, query)(*args, **kwargs))
    abs_err = abs(sketch_val - exact_val)
    rel_err = abs_err / exact_val if exact_val > 0 else 0.0
    return {
        "sketch": sketch_val,
        "exact": exact_val,
        "abs_error": abs_err,
        "rel_error": rel_err,
    }
