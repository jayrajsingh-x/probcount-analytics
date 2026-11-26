"""FastAPI HTTP service wrapping :class:`~probcount.engine.AnalyticsEngine`.

Concurrency note:
    State mutations and queries are synchronized via a single
    :class:`threading.RLock`.  Because HLL and CMS updates are fast
    in-memory NumPy operations (~10–50 microseconds per batch item),
    holding an RLock avoids the async-queue scheduling overhead while
    remaining safe under both FastAPI's ``TestClient`` and ``uvicorn``
    worker threads.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import math
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from probcount.config import TrafficConfig, TrafficSpike
from probcount.engine import AnalyticsEngine, EngineConfig
from probcount.exact_engine import ExactAnalyticsEngine
from probcount.generator import Event, TrafficGenerator
from probcount.hll import HyperLogLog
from probcount.setops import hll_intersection, hll_jaccard
from probcount.storage import FileSnapshotBackend, InMemoryBackend, StorageBackend
from probcount.windows import GRANULARITY_SECONDS, Unit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class AppSettings(BaseSettings):
    """Configuration loaded from environment variables (``PROBCOUNT_*``)."""

    model_config = SettingsConfigDict(env_prefix="PROBCOUNT_", extra="ignore")

    site: str = "default"
    hll_p: int = 14
    page_hll_p: int = 10
    cms_epsilon: float = 0.0027
    cms_delta: float = 0.02
    cms_conservative: bool = True
    bucket_seconds: int = 3600
    retention_seconds: float = 30 * 86_400.0
    watermark_seconds: float = 300.0
    snapshot_interval_seconds: float = 60.0
    snapshot_dir: str = "data/snapshots"
    use_memory_storage: bool = False
    enable_exact: bool = False
    api_key: str | None = None
    hash_salt: str = "probcount-edge-salt"
    cors_origins: list[str] = ["*"]


# ---------------------------------------------------------------------------
# Pydantic v2 Schemas
# ---------------------------------------------------------------------------


class EventIn(BaseModel):
    """Single visit event payload."""

    visitor_id: str = Field(..., min_length=1, max_length=256, examples=["u_a3f8c1d9e2b0"])
    page_id: str = Field(..., min_length=1, max_length=512, examples=["/products/42"])
    timestamp: float | None = Field(default=None, examples=[1700000000.0])
    is_bot: bool = Field(default=False, examples=[False])


class BatchIn(BaseModel):
    """Batch of up to 10,000 visit events."""

    events: list[EventIn] = Field(..., min_length=1, max_length=10_000)


class IngestResponse(BaseModel):
    """Ingestion acknowledgement."""

    accepted: int
    rejected: int


class UniqueStatsResponse(BaseModel):
    """Unique-visitor query result."""

    estimate: float
    relative_error_bound: float
    exact: float | None = None
    abs_error: float | None = None
    rel_error: float | None = None


class ViewsStatsResponse(BaseModel):
    """Page-view frequency query result."""

    page: str
    estimate: int
    cms_error_bound: float


class TopPageItem(BaseModel):
    """Top page entry."""

    page: str
    views: int
    exact_views: int | None = None
    abs_error: int | None = None
    rel_error: float | None = None


class TopPagesResponse(BaseModel):
    """Top-K pages response."""

    items: list[TopPageItem]


class SeriesPoint(BaseModel):
    """Single time-series point."""

    timestamp: float
    unique_visitors: float
    exact_unique_visitors: float | None = None


class SeriesResponse(BaseModel):
    """Time-series response."""

    granularity: str
    points: list[SeriesPoint]


class OverviewResponse(BaseModel):
    """Headline KPIs."""

    unique_visitors: float
    total_events: int
    bots_ignored: int
    top_page: str | None


class NewVsReturningResponse(BaseModel):
    """New vs. returning visitor approximation."""

    new_visitors: float
    returning_visitors: float
    error_estimate: float
    warning: bool
    message: str


class OverlapResponse(BaseModel):
    """Set intersection and Jaccard similarity between two pages/sets."""

    intersection: float
    jaccard: float
    error_estimate: float
    warning: bool
    message: str


class HLLMergeIn(BaseModel):
    """Base64-encoded HyperLogLog payload for HTTP merge."""

    name: str = Field(..., min_length=1, max_length=128, examples=["worker-1"])
    sketch_b64: str = Field(..., min_length=1)


class HLLSketchResponse(BaseModel):
    """Named HyperLogLog sketch state."""

    name: str
    estimate: float
    sketch_b64: str | None = None


class SimulateIn(BaseModel):
    """Parameters for synthetic traffic simulation."""

    n_events: int = Field(default=10_000, ge=1, le=500_000)
    n_visitors: int = Field(default=2_000, ge=1, le=200_000)
    n_pages: int = Field(default=50, ge=1, le=5_000)
    seed: int = Field(default=42)
    bot_fraction: float = Field(default=0.05, ge=0.0, lt=1.0)
    page_zipf_s: float = Field(default=1.1, gt=0.0, le=3.0)
    spike_multiplier: float = Field(default=1.0, ge=1.0, le=20.0)


# ---------------------------------------------------------------------------
# Service Container
# ---------------------------------------------------------------------------


class ServiceState:
    """Thread-safe container holding engine, exact engine, storage, and metrics."""

    def __init__(
        self,
        settings: AppSettings,
        storage: StorageBackend | None = None,
    ) -> None:
        self.settings = settings
        self.lock = threading.RLock()
        self.start_time = time.time()
        self.rejected_events = 0
        self.named_hlls: dict[str, HyperLogLog] = {}

        cms_w = max(1, math.ceil(math.e / settings.cms_epsilon))
        cms_d = max(1, math.ceil(math.log(1.0 / settings.cms_delta)))
        self.engine_config = EngineConfig(
            site=settings.site,
            hll_p=settings.hll_p,
            page_hll_p=settings.page_hll_p,
            cms_width=cms_w,
            cms_depth=cms_d,
            cms_conservative=settings.cms_conservative,
            bucket_seconds=settings.bucket_seconds,
            retention_seconds=settings.retention_seconds,
            watermark_seconds=settings.watermark_seconds,
        )

        if storage is not None:
            self.storage = storage
        elif settings.use_memory_storage:
            self.storage = InMemoryBackend()
        else:
            self.storage = FileSnapshotBackend(settings.snapshot_dir)

        loaded = self.storage.load()
        self.engine = loaded if loaded is not None else AnalyticsEngine(self.engine_config)
        self.exact: ExactAnalyticsEngine | None = (
            ExactAnalyticsEngine(
                site=settings.site,
                bucket_seconds=settings.bucket_seconds,
            )
            if settings.enable_exact
            else None
        )

    def hash_visitor(self, raw_visitor_id: str) -> str:
        """Edge-hash raw visitor ID with server-side salt so raw IDs are never stored."""
        salted = f"{self.settings.hash_salt}:{raw_visitor_id}".encode()
        return hashlib.blake2b(salted, digest_size=16).hexdigest()

    def resolve_window(
        self,
        start: float | None,
        end: float | None,
        window: str | None = None,
    ) -> tuple[float, float]:
        """Resolve query ``[start, end)`` and validate ``start <= end``."""
        now = self.engine._site_hll.max_timestamp or time.time()
        if window is not None:
            unit_map = {"15m": 900.0, "1h": 3600.0, "24h": 86_400.0, "7d": 7 * 86_400.0}
            span = unit_map.get(window)
            if span is None:
                raise HTTPException(status_code=422, detail=f"Unsupported window: {window}")
            return now - span, now + 1.0

        buckets = self.engine._site_hll.buckets
        earliest = float(min(buckets)) if buckets else now - 3600.0
        s = earliest if start is None else start
        e = now + 1.0 if end is None else end
        if s > e:
            raise HTTPException(status_code=422, detail="start must be <= end")
        return s, e

    def reset(self) -> None:
        """Reset all in-memory engines and counters."""
        with self.lock:
            self.engine = AnalyticsEngine(self.engine_config)
            if self.settings.enable_exact:
                self.exact = ExactAnalyticsEngine(
                    site=self.settings.site,
                    bucket_seconds=self.settings.bucket_seconds,
                )
            self.named_hlls.clear()
            self.rejected_events = 0


def create_app(
    settings: AppSettings | None = None,
    storage: StorageBackend | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = settings or AppSettings()
    state = ServiceState(cfg, storage=storage)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        stop_event = asyncio.Event()

        async def _periodic_snapshot() -> None:
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=max(1.0, cfg.snapshot_interval_seconds),
                    )
                except TimeoutError:
                    with state.lock:
                        state.storage.save(state.engine)
                        state.engine._site_hll._prune()
                        state.engine._page_cms._prune()

        task = asyncio.create_task(_periodic_snapshot())
        try:
            yield
        finally:
            stop_event.set()
            task.cancel()
            with state.lock:
                state.storage.save(state.engine)

    application = FastAPI(
        title="ProbCount Analytics API",
        version="0.1.0",
        description=(
            "Real-time probabilistic web analytics powered by "
            "HyperLogLog & Count-Min Sketch."
        ),
        lifespan=lifespan,
    )
    application.state.probcount = state

    application.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Consistent JSON error format
    @application.exception_handler(HTTPException)
    async def _http_exc_handler(_req: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": f"HTTP_{exc.status_code}", "message": str(exc.detail)},
        )

    @application.exception_handler(RequestValidationError)
    async def _val_exc_handler(_req: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"code": "VALIDATION_ERROR", "message": str(exc.errors())},
        )

    def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if cfg.api_key is not None and x_api_key != cfg.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    @application.post(
        "/events",
        status_code=202,
        response_model=IngestResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def ingest_event(payload: EventIn) -> IngestResponse:
        ts = time.time() if payload.timestamp is None else payload.timestamp
        ev = Event(
            timestamp=ts,
            visitor_id=state.hash_visitor(payload.visitor_id),
            page_id=payload.page_id,
            is_bot=payload.is_bot,
        )
        with state.lock:
            before = state.engine._site_hll.dropped_late
            state.engine.process_event(ev)
            if state.exact is not None:
                state.exact.process_event(ev)
            dropped = state.engine._site_hll.dropped_late - before
            if dropped > 0:
                state.rejected_events += dropped
                return IngestResponse(accepted=0, rejected=1)
        return IngestResponse(accepted=1, rejected=0)

    @application.post(
        "/events/batch",
        status_code=202,
        response_model=IngestResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def ingest_batch(payload: BatchIn) -> IngestResponse:
        now = time.time()
        events = [
            Event(
                timestamp=now if item.timestamp is None else item.timestamp,
                visitor_id=state.hash_visitor(item.visitor_id),
                page_id=item.page_id,
                is_bot=item.is_bot,
            )
            for item in payload.events
        ]
        with state.lock:
            before = state.engine._site_hll.dropped_late
            state.engine.process_batch(events)
            if state.exact is not None:
                state.exact.process_batch(events)
            dropped = state.engine._site_hll.dropped_late - before
            state.rejected_events += dropped
        accepted = len(events) - dropped
        return IngestResponse(accepted=accepted, rejected=dropped)

    # ------------------------------------------------------------------
    # Analytics Queries
    # ------------------------------------------------------------------

    @application.get("/stats/unique", response_model=UniqueStatsResponse)
    def stats_unique(
        site: str = Query(default="default"),
        page: str | None = Query(default=None),
        start: float | None = Query(default=None),
        end: float | None = Query(default=None),
        window: str | None = Query(default=None),
        compare: bool = Query(default=False),
    ) -> UniqueStatsResponse:
        s, e = state.resolve_window(start, end, window)
        with state.lock:
            if page is not None:
                est = state.engine.unique_visitors_page(page, s, e)
                se = 1.04 / math.sqrt(1 << state.engine.config.page_hll_p)
                ex = (
                    state.exact.unique_visitors_page(page, s, e)
                    if (compare and state.exact)
                    else None
                )
            else:
                est = state.engine.unique_visitors(site, s, e)
                se = 1.04 / math.sqrt(1 << state.engine.config.hll_p)
                ex = (
                    state.exact.unique_visitors(site, s, e)
                    if (compare and state.exact)
                    else None
                )

        abs_err = abs(est - ex) if ex is not None else None
        rel_err = (abs_err / ex) if (abs_err is not None and ex and ex > 0) else None
        return UniqueStatsResponse(
            estimate=est,
            relative_error_bound=se,
            exact=ex,
            abs_error=abs_err,
            rel_error=rel_err,
        )

    @application.get("/stats/views", response_model=ViewsStatsResponse)
    def stats_views(
        page: str = Query(..., min_length=1),
        start: float | None = Query(default=None),
        end: float | None = Query(default=None),
    ) -> ViewsStatsResponse:
        s, e = state.resolve_window(start, end)
        with state.lock:
            cms_w = state.engine.site_cms_for_window(s, e)
            return ViewsStatsResponse(
                page=page,
                estimate=cms_w.estimate(page),
                cms_error_bound=cms_w.error_bound(),
            )

    @application.get("/stats/top", response_model=TopPagesResponse)
    def stats_top(
        k: int = Query(default=10, ge=1, le=200),
        start: float | None = Query(default=None),
        end: float | None = Query(default=None),
        compare: bool = Query(default=False),
    ) -> TopPagesResponse:
        s, e = state.resolve_window(start, end)
        with state.lock:
            top_list = state.engine.top_pages(k, s, e)
            items: list[TopPageItem] = []
            for p, v in top_list:
                if compare and state.exact is not None:
                    ex_v = state.exact.page_views(p, s, e)
                    abs_err = abs(v - ex_v)
                    rel_err = abs_err / ex_v if ex_v > 0 else 0.0
                    items.append(
                        TopPageItem(
                            page=p,
                            views=v,
                            exact_views=ex_v,
                            abs_error=abs_err,
                            rel_error=rel_err,
                        )
                    )
                else:
                    items.append(TopPageItem(page=p, views=v))
        return TopPagesResponse(items=items)

    @application.get("/stats/series", response_model=SeriesResponse)
    def stats_series(
        granularity: Annotated[Unit, Query()] = "hour",
        start: float | None = Query(default=None),
        end: float | None = Query(default=None),
        compare: bool = Query(default=False),
    ) -> SeriesResponse:
        if granularity not in GRANULARITY_SECONDS:
            raise HTTPException(status_code=422, detail=f"Invalid granularity: {granularity}")
        s, e = state.resolve_window(start, end)
        with state.lock:
            sk_pts = state.engine.unique_visitors_series(granularity, s, e)
            if compare and state.exact is not None:
                ex_pts = state.exact.unique_visitors_series(granularity, s, e)
                pts = [
                    SeriesPoint(
                        timestamp=t,
                        unique_visitors=v,
                        exact_unique_visitors=ex_v,
                    )
                    for (t, v), (_, ex_v) in zip(sk_pts, ex_pts, strict=True)
                ]
            else:
                pts = [SeriesPoint(timestamp=t, unique_visitors=v) for t, v in sk_pts]
        return SeriesResponse(granularity=granularity, points=pts)

    @application.get("/stats/overview", response_model=OverviewResponse)
    def stats_overview(
        site: str = Query(default="default"),
        start: float | None = Query(default=None),
        end: float | None = Query(default=None),
    ) -> OverviewResponse:
        s, e = state.resolve_window(start, end)
        with state.lock:
            uv = state.engine.unique_visitors(site, s, e)
            top = state.engine.top_pages(1, s, e)
            return OverviewResponse(
                unique_visitors=uv,
                total_events=state.engine.events_processed,
                bots_ignored=state.engine.bots_ignored,
                top_page=top[0][0] if top else None,
            )

    @application.get("/stats/new-vs-returning", response_model=NewVsReturningResponse)
    def stats_new_vs_returning(
        start: float | None = Query(default=None),
        end: float | None = Query(default=None),
    ) -> NewVsReturningResponse:
        s, e = state.resolve_window(start, end)
        with state.lock:
            total_uv = state.engine.unique_visitors(cfg.site, s, e)
            res = state.engine.new_visitors(s, e)
            returning = max(0.0, total_uv - res.estimate)
        return NewVsReturningResponse(
            new_visitors=res.estimate,
            returning_visitors=returning,
            error_estimate=res.error_estimate,
            warning=res.warning,
            message=res.message,
        )

    @application.get("/stats/overlap", response_model=OverlapResponse)
    def stats_overlap(
        a: str = Query(..., min_length=1),
        b: str = Query(..., min_length=1),
        start: float | None = Query(default=None),
        end: float | None = Query(default=None),
    ) -> OverlapResponse:
        s, e = state.resolve_window(start, end)
        with state.lock:
            tb_a = state.engine._hot_page_hlls.get(a)
            tb_b = state.engine._hot_page_hlls.get(b)
            hll_a = tb_a.query(s, e) if tb_a else HyperLogLog(p=cfg.page_hll_p)
            hll_b = tb_b.query(s, e) if tb_b else HyperLogLog(p=cfg.page_hll_p)
            inter = hll_intersection(hll_a, hll_b)
            jacc = hll_jaccard(hll_a, hll_b)
        return OverlapResponse(
            intersection=inter.estimate,
            jaccard=jacc.estimate,
            error_estimate=inter.error_estimate,
            warning=inter.warning,
            message=inter.message,
        )

    # ------------------------------------------------------------------
    # Sketch Merge over HTTP
    # ------------------------------------------------------------------

    @application.post(
        "/sketch/hll/merge",
        response_model=HLLSketchResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def merge_hll(payload: HLLMergeIn) -> HLLSketchResponse:
        try:
            raw = base64.b64decode(payload.sketch_b64)
            incoming = HyperLogLog.from_bytes(raw)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid HLL payload: {exc}") from exc

        with state.lock:
            existing = state.named_hlls.get(payload.name)
            if existing is None:
                state.named_hlls[payload.name] = incoming
            else:
                try:
                    existing.merge(incoming)
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
            est = state.named_hlls[payload.name].count()
        return HLLSketchResponse(name=payload.name, estimate=est)

    @application.get("/sketch/hll/{name}", response_model=HLLSketchResponse)
    def get_hll(name: str) -> HLLSketchResponse:
        with state.lock:
            sketch = state.named_hlls.get(name)
            if sketch is None and name == "site":
                s, e = state.resolve_window(None, None)
                sketch = state.engine.site_hll_for_window(s, e)
            if sketch is None:
                raise HTTPException(status_code=404, detail=f"Sketch '{name}' not found")
            return HLLSketchResponse(
                name=name,
                estimate=sketch.count(),
                sketch_b64=base64.b64encode(sketch.to_bytes()).decode("ascii"),
            )

    @application.get("/sketch/inspect")
    def inspect_sketches() -> dict[str, Any]:
        """Return live HLL register histogram/slice and a CMS slice for teaching UI."""
        import numpy as np

        with state.lock:
            s, e = state.resolve_window(None, None)
            hll = state.engine.site_hll_for_window(s, e)
            regs = hll.registers
            vals, counts = np.unique(regs, return_counts=True)
            hist = {int(v): int(c) for v, c in zip(vals, counts, strict=True)}
            cms = state.engine.site_cms_for_window(s, e)
            cms_slice = cms.table[:, :32].tolist()
            return {
                "hll_p": hll.p,
                "hll_m": hll.m,
                "hll_estimate": hll.count(),
                "register_histogram": hist,
                "first_64_registers": [int(x) for x in regs[:64]],
                "cms_width": cms.width,
                "cms_depth": cms.depth,
                "cms_slice": cms_slice,
            }

    # ------------------------------------------------------------------
    # Admin & Observability
    # ------------------------------------------------------------------

    @application.post("/admin/snapshot", dependencies=[Depends(_require_api_key)])
    def admin_snapshot() -> dict[str, str]:
        with state.lock:
            snap_id = state.storage.save(state.engine)
        return {"snapshot_id": snap_id}

    @application.post("/admin/simulate", dependencies=[Depends(_require_api_key)])
    def admin_simulate(payload: SimulateIn) -> dict[str, Any]:
        t0 = time.perf_counter()
        start_ts = time.time() - 86_400.0
        spike = (
            TrafficSpike(
                start=start_ts + 36_000.0,
                end=start_ts + 43_200.0,
                multiplier=payload.spike_multiplier,
            )
            if payload.spike_multiplier > 1.0
            else None
        )
        tcfg = TrafficConfig(
            n_events=payload.n_events,
            n_unique_visitors=min(payload.n_visitors, payload.n_events),
            n_pages=payload.n_pages,
            start_time=start_ts,
            bot_fraction=payload.bot_fraction,
            page_zipf_s=payload.page_zipf_s,
            traffic_spike=spike,
            seed=payload.seed,
        )
        events = [
            Event(
                timestamp=e.timestamp,
                visitor_id=state.hash_visitor(e.visitor_id),
                page_id=e.page_id,
                is_bot=e.is_bot,
            )
            for e in TrafficGenerator(tcfg).generate()
        ]
        with state.lock:
            state.engine.process_batch(events)
            if state.exact is not None:
                state.exact.process_batch(events)
        return {
            "events_generated": len(events),
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }

    @application.post("/admin/reset", dependencies=[Depends(_require_api_key)])
    def admin_reset() -> dict[str, str]:
        state.reset()
        return {"status": "reset"}

    @application.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "uptime_seconds": round(time.time() - state.start_time, 3)}

    @application.get("/metrics")
    def metrics() -> dict[str, Any]:
        with state.lock:
            mem = state.engine.memory_report(state.exact)
            return {
                "events_processed": state.engine.events_processed,
                "rejected": state.rejected_events,
                "dropped_late": state.engine._site_hll.dropped_late,
                "bots_ignored": state.engine.bots_ignored,
                "memory": mem,
                "sketch_params": {
                    "hll_p": state.engine.config.hll_p,
                    "cms_width": state.engine.config.cms_width,
                    "cms_depth": state.engine.config.cms_depth,
                },
                "uptime_seconds": round(time.time() - state.start_time, 3),
            }

    return application


app = create_app()
