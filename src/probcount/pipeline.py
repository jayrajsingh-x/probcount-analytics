"""Streaming ingestion pipeline recording periodic accuracy snapshots."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from probcount.engine import AnalyticsEngine, EngineConfig
from probcount.exact_engine import ExactAnalyticsEngine
from probcount.generator import Event, read_jsonl


@dataclass(frozen=True)
class PipelineSnapshot:
    """Periodic snapshot of sketch vs. exact analytics."""

    timestamp: float
    events_processed: int
    sketch_unique: float
    exact_unique: float | None
    rel_error: float | None
    top_page: str | None


@dataclass(frozen=True)
class PipelineResult:
    """Output of :func:`run_pipeline`."""

    engine: AnalyticsEngine
    exact_engine: ExactAnalyticsEngine | None
    snapshots: list[PipelineSnapshot]


def run_pipeline(
    events: Iterable[Event] | None = None,
    engine_config: EngineConfig | None = None,
    batch_size: int = 10_000,
    snapshot_interval: int = 50_000,
    track_exact: bool = True,
    jsonl_path: str | Path | None = None,
) -> PipelineResult:
    """Stream events from *events* or *jsonl_path* in batches and record snapshots."""
    cfg = engine_config or EngineConfig()
    engine = AnalyticsEngine(cfg)
    exact = (
        ExactAnalyticsEngine(
            site=cfg.site,
            ignore_bots=cfg.ignore_bots,
            bucket_seconds=cfg.bucket_seconds,
        )
        if track_exact
        else None
    )

    stream = read_jsonl(jsonl_path) if jsonl_path is not None else events
    if stream is None:
        raise ValueError("Provide either events or jsonl_path")

    snapshots: list[PipelineSnapshot] = []
    batch: list[Event] = []
    min_ts: float | None = None
    max_ts: float | None = None
    next_snap = snapshot_interval

    def _flush() -> None:
        nonlocal next_snap
        if not batch:
            return
        engine.process_batch(batch)
        if exact is not None:
            exact.process_batch(batch)
        batch.clear()

        if engine.events_processed >= next_snap and min_ts is not None and max_ts is not None:
            sk_u = engine.unique_visitors(cfg.site, min_ts, max_ts + 1.0)
            ex_u = exact.unique_visitors(cfg.site, min_ts, max_ts + 1.0) if exact else None
            rel_err = abs(sk_u - ex_u) / ex_u if ex_u and ex_u > 0 else None
            top = engine.top_pages(1, min_ts, max_ts + 1.0)
            snapshots.append(
                PipelineSnapshot(
                    timestamp=max_ts,
                    events_processed=engine.events_processed,
                    sketch_unique=sk_u,
                    exact_unique=ex_u,
                    rel_error=rel_err,
                    top_page=top[0][0] if top else None,
                )
            )
            next_snap += snapshot_interval

    for event in stream:
        min_ts = event.timestamp if min_ts is None else min(min_ts, event.timestamp)
        max_ts = event.timestamp if max_ts is None else max(max_ts, event.timestamp)
        batch.append(event)
        if len(batch) >= batch_size:
            _flush()

    _flush()
    return PipelineResult(engine=engine, exact_engine=exact, snapshots=snapshots)
