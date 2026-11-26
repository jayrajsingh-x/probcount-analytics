"""End-to-end integration test: generator -> API -> queries -> snapshot -> restart -> parity."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from probcount.api import AppSettings, create_app
from probcount.config import TrafficConfig
from probcount.generator import TrafficGenerator
from probcount.storage import FileSnapshotBackend


def test_e2e_generator_api_snapshot_restart_parity(tmp_path: Path) -> None:
    snap_dir = tmp_path / "snapshots"
    settings = AppSettings(
        hll_p=14,
        page_hll_p=10,
        use_memory_storage=False,
        snapshot_dir=str(snap_dir),
        enable_exact=False,
    )

    cfg = TrafficConfig(n_events=2_500, n_unique_visitors=500, n_pages=20, seed=7)
    events = [
        {
            "visitor_id": e.visitor_id,
            "page_id": e.page_id,
            "timestamp": e.timestamp,
            "is_bot": e.is_bot,
        }
        for e in TrafficGenerator(cfg).generate()
    ]

    with TestClient(create_app(settings=settings, storage=FileSnapshotBackend(snap_dir))) as c1:
        r = c1.post("/events/batch", json={"events": events})
        assert r.status_code == 202
        snap = c1.post("/admin/snapshot")
        assert snap.status_code == 200

        pre_unique = c1.get("/stats/unique").json()
        pre_top = c1.get("/stats/top", params={"k": 5}).json()
        pre_overview = c1.get("/stats/overview").json()

    # Restart a fresh app instance pointing to the same snapshot directory
    with TestClient(create_app(settings=settings, storage=FileSnapshotBackend(snap_dir))) as c2:
        post_unique = c2.get("/stats/unique").json()
        post_top = c2.get("/stats/top", params={"k": 5}).json()
        post_overview = c2.get("/stats/overview").json()

        assert post_unique["estimate"] == pre_unique["estimate"]
        assert post_top == pre_top
        assert post_overview == pre_overview
