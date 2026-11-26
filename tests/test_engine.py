"""Tests for :mod:`probcount.engine`, :mod:`probcount.exact_engine`,
:mod:`probcount.distributed`, and :mod:`probcount.pipeline`.
"""

from __future__ import annotations

import numpy as np
import pytest

from probcount.config import TrafficConfig
from probcount.distributed import simulate_distributed
from probcount.engine import AnalyticsEngine, EngineConfig
from probcount.exact_engine import ExactAnalyticsEngine, compare
from probcount.generator import Event, TrafficGenerator
from probcount.pipeline import run_pipeline


class TestAnalyticsEngine:
    """End-to-end engine queries, bot filtering, hot pages, and memory."""

    def test_rolling_7day_on_10day_stream(self) -> None:
        start = float(19676 * 86_400)  # aligned to UTC day boundary
        cfg_gen = TrafficConfig(
            n_events=60_000,
            n_unique_visitors=8_000,
            n_pages=30,
            start_time=start,
            duration_seconds=10 * 86_400.0,
            bot_fraction=0.1,
            seed=42,
        )
        events = list(TrafficGenerator(cfg_gen).generate())

        ecfg = EngineConfig(hll_p=14, page_hll_p=10, hot_pages_limit=15, ignore_bots=True)
        engine = AnalyticsEngine(ecfg)
        exact = ExactAnalyticsEngine(ignore_bots=True)
        engine.process_batch(events)
        exact.process_batch(events)

        w_start = start + 3 * 86_400.0
        w_end = start + 10 * 86_400.0
        cmp = compare(engine, exact, "unique_visitors", "default", w_start, w_end)
        se = 1.04 / (1 << 14) ** 0.5
        assert cmp["rel_error"] < 3 * se

        assert engine.bots_ignored > 0
        assert len(engine.hot_pages) <= 15
        assert engine.unique_visitors_page("/nonexistent", w_start, w_end) == 0.0
        assert engine.views_per_visitor("/nonexistent", w_start, w_end) == 0.0
        assert engine.unique_visitors("other_site", w_start, w_end) == 0.0

        top5 = engine.top_pages(5, w_start, w_end)
        assert len(top5) == 5
        assert top5[0][1] >= top5[1][1]

        series = engine.unique_visitors_series("day", w_start, w_end)
        ex_series = exact.unique_visitors_series("day", w_start, w_end)
        assert len(series) == 7
        assert len(ex_series) == 7

        # Hot-page queries & exact parity
        hot_page = top5[0][0]
        assert engine.unique_visitors_page(hot_page, w_start, w_end) > 0.0
        assert engine.page_views(hot_page, w_start, w_end) >= exact.page_views(
            hot_page, w_start, w_end
        )
        assert engine.views_per_visitor(hot_page, w_start, w_end) > 0.0
        assert exact.unique_visitors_page(hot_page, w_start, w_end) > 0.0
        assert exact.views_per_visitor(hot_page, w_start, w_end) > 0.0
        assert exact.unique_visitors_page("/nonexistent", w_start, w_end) == 0.0
        assert exact.views_per_visitor("/nonexistent", w_start, w_end) == 0.0
        assert exact.unique_visitors("other_site", w_start, w_end) == 0.0
        assert len(exact.top_pages(5, w_start, w_end)) == 5

        new_res = engine.new_visitors(w_start, w_end)
        assert new_res.estimate >= 0.0
        assert exact.new_visitors(w_start, w_end) >= 0.0
        # Window before any prior history
        early_new = engine.new_visitors(start, start + 86_400.0)
        assert early_new.estimate > 0.0

        mem = engine.memory_report(exact)
        assert mem["total_bytes"] > 0
        assert mem["exact_bytes"] > 0


class TestDistributedMerge:
    """Distributed merge identity across workers and partitioning modes."""

    @pytest.mark.parametrize("n_workers", [2, 4, 8])
    @pytest.mark.parametrize("mode", ["hash", "round_robin", "random_dup"])
    def test_distributed_identity(self, n_workers: int, mode: str) -> None:
        start = 0.0
        events = [
            Event(
                timestamp=start + (i % 7200),
                visitor_id=f"u_{i % 300}",
                page_id=f"/p/{i % 10}",
                is_bot=False,
            )
            for i in range(1_000)
        ]
        ecfg = EngineConfig(
            hll_p=10,
            page_hll_p=8,
            cms_width=64,
            cms_depth=3,
            cms_conservative=False,
            hot_pages_limit=10,
        )

        single = AnalyticsEngine(ecfg)
        single.process_batch(events)

        res = simulate_distributed(
            events,
            ecfg,
            n_workers=n_workers,
            mode=mode,  # type: ignore[arg-type]
            seed=42,
            dup_prob=0.25,
        )
        # HLL registers must be bit-identical even with duplicates!
        hll_single = single.site_hll_for_window(0.0, 10_000.0)
        hll_merged = res.merged_engine.site_hll_for_window(0.0, 10_000.0)
        assert np.array_equal(hll_single.registers, hll_merged.registers)

        # For disjoint/exact-delivery modes, standard CMS table is also identical
        if mode != "random_dup":
            cms_single = single.site_cms_for_window(0.0, 10_000.0)
            cms_merged = res.merged_engine.site_cms_for_window(0.0, 10_000.0)
            assert np.array_equal(cms_single.table, cms_merged.table)


class TestPipeline:
    """Streaming pipeline runner snapshots."""

    def test_pipeline_snapshots(self) -> None:
        cfg = TrafficConfig(n_events=5_000, n_unique_visitors=1_000, n_pages=10, seed=7)
        gen = TrafficGenerator(cfg)
        res = run_pipeline(
            events=gen.generate(),
            engine_config=EngineConfig(hll_p=12),
            batch_size=1_000,
            snapshot_interval=2_000,
            track_exact=True,
        )
        assert len(res.snapshots) >= 2
        assert res.exact_engine is not None
        assert res.snapshots[-1].rel_error is not None
