"""AnalyticsEngine combining HyperLogLog, Count-Min Sketch, and Top-K."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from probcount.cms import CountMinSketch
from probcount.exact import deep_size_bytes
from probcount.generator import Event
from probcount.hll import HyperLogLog
from probcount.setops import SetOpResult, hll_intersection
from probcount.topk import TopKTracker
from probcount.windows import GRANULARITY_SECONDS, TimeBucketedCMS, TimeBucketedHLL, Unit

if TYPE_CHECKING:
    from probcount.exact_engine import ExactAnalyticsEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineConfig:
    """Configuration for :class:`AnalyticsEngine`."""

    site: str = "default"
    hll_p: int = 14
    page_hll_p: int = 10
    cms_width: int = 1024
    cms_depth: int = 4
    cms_conservative: bool = True
    top_k: int = 50
    hot_pages_limit: int = 100
    ignore_bots: bool = True
    bucket_seconds: int = 3600
    retention_seconds: float = 30 * 86_400.0
    watermark_seconds: float = 300.0
    seed: int = 42


class AnalyticsEngine:
    """Probabilistic web-analytics engine backed by HLL, CMS, and Top-K.

    To keep memory strictly bounded regardless of total distinct pages,
    per-page :class:`~probcount.windows.TimeBucketedHLL` trackers are
    maintained **only** for a bounded hot set of at most
    ``config.hot_pages_limit`` pages (ranked by CMS frequency estimate).
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self._cfg = config or EngineConfig()
        self._site_hll = TimeBucketedHLL(
            p=self._cfg.hll_p,
            seed=self._cfg.seed,
            bucket_seconds=self._cfg.bucket_seconds,
            retention_seconds=self._cfg.retention_seconds,
            watermark_seconds=self._cfg.watermark_seconds,
        )
        self._page_cms = TimeBucketedCMS(
            width=self._cfg.cms_width,
            depth=self._cfg.cms_depth,
            seed=self._cfg.seed,
            conservative=self._cfg.cms_conservative,
            bucket_seconds=self._cfg.bucket_seconds,
            retention_seconds=self._cfg.retention_seconds,
            watermark_seconds=self._cfg.watermark_seconds,
        )
        self._global_cms = CountMinSketch(
            width=self._cfg.cms_width,
            depth=self._cfg.cms_depth,
            seed=self._cfg.seed,
            conservative=self._cfg.cms_conservative,
        )
        self._topk = TopKTracker(k=self._cfg.top_k, sketch=self._global_cms)
        self._hot_page_hlls: dict[str, TimeBucketedHLL] = {}
        self._events_processed: int = 0
        self._bots_ignored: int = 0

    @property
    def config(self) -> EngineConfig:
        """Engine configuration."""
        return self._cfg

    @property
    def events_processed(self) -> int:
        """Number of non-ignored events processed."""
        return self._events_processed

    @property
    def bots_ignored(self) -> int:
        """Number of bot events skipped."""
        return self._bots_ignored

    @property
    def hot_pages(self) -> set[str]:
        """Currently tracked hot pages."""
        return set(self._hot_page_hlls.keys())

    def _maybe_track_hot_page(self, event: Event) -> None:
        page = event.page_id
        if page in self._hot_page_hlls:
            self._hot_page_hlls[page].add(event.timestamp, event.visitor_id)
            return

        if len(self._hot_page_hlls) < self._cfg.hot_pages_limit:
            tracker = TimeBucketedHLL(
                p=self._cfg.page_hll_p,
                seed=self._cfg.seed,
                bucket_seconds=self._cfg.bucket_seconds,
                retention_seconds=self._cfg.retention_seconds,
                watermark_seconds=self._cfg.watermark_seconds,
            )
            tracker.add(event.timestamp, event.visitor_id)
            self._hot_page_hlls[page] = tracker
            return

        page_est = self._global_cms.estimate(page)
        min_page = min(self._hot_page_hlls, key=lambda p: self._global_cms.estimate(p))
        if page_est > self._global_cms.estimate(min_page):
            del self._hot_page_hlls[min_page]
            tracker = TimeBucketedHLL(
                p=self._cfg.page_hll_p,
                seed=self._cfg.seed,
                bucket_seconds=self._cfg.bucket_seconds,
                retention_seconds=self._cfg.retention_seconds,
                watermark_seconds=self._cfg.watermark_seconds,
            )
            tracker.add(event.timestamp, event.visitor_id)
            self._hot_page_hlls[page] = tracker

    def process_event(self, event: Event) -> None:
        """Ingest a single :class:`~probcount.generator.Event`."""
        if self._cfg.ignore_bots and event.is_bot:
            self._bots_ignored += 1
            return

        accepted = self._site_hll.add(event.timestamp, event.visitor_id)
        if not accepted:
            return
        self._page_cms.add(event.timestamp, event.page_id, 1)
        self._topk.add(event.page_id, 1)
        if self._cfg.hot_pages_limit > 0:
            self._maybe_track_hot_page(event)
        self._events_processed += 1

    def process_batch(self, events: Iterable[Event]) -> None:
        """Ingest a batch or stream of events."""
        for event in events:
            self.process_event(event)

    def unique_visitors(self, site: str, start: float, end: float) -> float:
        """Estimate unique visitors for *site* in ``[start, end)``."""
        if site != self._cfg.site:
            return 0.0
        return self._site_hll.query(start, end).count()

    def unique_visitors_page(self, page: str, start: float, end: float) -> float:
        """Estimate unique visitors for *page* in ``[start, end)`` (0.0 if untracked)."""
        tracker = self._hot_page_hlls.get(page)
        if tracker is None:
            return 0.0
        return tracker.query(start, end).count()

    def page_views(self, page: str, start: float, end: float) -> int:
        """Estimate view count for *page* in ``[start, end)``."""
        return self._page_cms.query(start, end).estimate(page)

    def top_pages(self, k: int, start: float, end: float) -> list[tuple[str, int]]:
        """Return top *k* pages in ``[start, end)`` scored by the windowed CMS."""
        if k < 1:
            return []
        cms_window = self._page_cms.query(start, end)
        candidates = [page for page, _ in self._topk.top(self._cfg.top_k)]
        scored = [(page, cms_window.estimate(page)) for page in candidates]
        scored = [(p, c) for p, c in scored if c > 0]
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:k]

    def views_per_visitor(self, page: str, start: float, end: float) -> float:
        """Estimate average views per unique visitor for *page* in ``[start, end)``."""
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
        """Return ``[(bucket_start, unique_visitors), ...]`` across ``[start, end)``."""
        step = GRANULARITY_SECONDS[granularity]
        t = float(int(start // step) * step)
        series: list[tuple[float, float]] = []
        while t < end:
            series.append((t, self._site_hll.query(t, t + step).count()))
            t += step
        return series

    def new_visitors(self, start: float, end: float) -> SetOpResult:
        """Estimate new (first-time) visitors in ``[start, end)`` (approximate).

        Computes ``|window| - |window ∩ history_before_start|`` using HLL
        inclusion-exclusion.
        """
        window_hll = self._site_hll.query(start, end)
        prior_hll = self._site_hll.query(float("-inf"), start)
        if prior_hll.is_empty():
            est = window_hll.count()
            return SetOpResult(
                estimate=est,
                error_estimate=est * window_hll.standard_error(),
                warning=False,
                message="Approximate (no prior history before window)",
            )
        overlap = hll_intersection(window_hll, prior_hll)
        new_est = max(0.0, window_hll.count() - overlap.estimate)
        return SetOpResult(
            estimate=new_est,
            error_estimate=overlap.error_estimate,
            warning=overlap.warning,
            message="Approximate via HLL inclusion-exclusion",
        )

    def site_hll_for_window(self, start: float, end: float) -> HyperLogLog:
        """Return the merged site :class:`HyperLogLog` for ``[start, end)``."""
        return self._site_hll.query(start, end)

    def site_cms_for_window(self, start: float, end: float) -> CountMinSketch:
        """Return the merged site :class:`CountMinSketch` for ``[start, end)``."""
        return self._page_cms.query(start, end)

    def merge(self, other: Self) -> Self:
        """Merge another :class:`AnalyticsEngine` into this one."""
        self._site_hll.merge(other._site_hll)
        self._page_cms.merge(other._page_cms)
        self._topk.merge(other._topk)
        for page, tracker in other._hot_page_hlls.items():
            if page in self._hot_page_hlls:
                self._hot_page_hlls[page].merge(tracker)
            elif len(self._hot_page_hlls) < self._cfg.hot_pages_limit:
                clone = TimeBucketedHLL(
                    p=self._cfg.page_hll_p,
                    seed=self._cfg.seed,
                    bucket_seconds=self._cfg.bucket_seconds,
                    retention_seconds=self._cfg.retention_seconds,
                    watermark_seconds=self._cfg.watermark_seconds,
                )
                clone.merge(tracker)
                self._hot_page_hlls[page] = clone
        self._events_processed += other._events_processed
        self._bots_ignored += other._bots_ignored
        return self

    def memory_report(self, exact_engine: ExactAnalyticsEngine | None = None) -> dict[str, int]:
        """Return memory usage in bytes per component and total."""
        site_hll_bytes = self._site_hll.memory_bytes()
        cms_bytes = self._page_cms.memory_bytes()
        topk_bytes = self._topk.memory_bytes()
        hot_pages_bytes = sum(t.memory_bytes() for t in self._hot_page_hlls.values())
        total = site_hll_bytes + cms_bytes + topk_bytes + hot_pages_bytes
        report: dict[str, int] = {
            "site_hll_bytes": site_hll_bytes,
            "cms_bytes": cms_bytes,
            "topk_bytes": topk_bytes,
            "hot_page_hlls_bytes": hot_pages_bytes,
            "total_bytes": total,
        }
        if exact_engine is not None:
            report["exact_bytes"] = exact_engine.memory_bytes()
        else:
            report["exact_bytes"] = deep_size_bytes(self._events_processed)
        return report
