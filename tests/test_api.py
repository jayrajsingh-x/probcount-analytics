"""Tests for :mod:`probcount.api` and :mod:`probcount.client`."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from probcount.api import AppSettings, create_app
from probcount.hll import HyperLogLog
from probcount.storage import FileSnapshotBackend, InMemoryBackend


def _make_client(
    api_key: str | None = None,
    enable_exact: bool = False,
    storage_dir: Path | None = None,
) -> TestClient:
    settings = AppSettings(
        hll_p=14,
        page_hll_p=10,
        use_memory_storage=(storage_dir is None),
        snapshot_dir=str(storage_dir) if storage_dir else "data/snapshots",
        enable_exact=enable_exact,
        api_key=api_key,
    )
    storage = FileSnapshotBackend(storage_dir) if storage_dir else InMemoryBackend()
    return TestClient(create_app(settings=settings, storage=storage))


class TestAPIValidationAndAuth:
    """Validation errors, auth, and consistent JSON error format."""

    def test_health_and_metrics(self) -> None:
        with _make_client() as client:
            r = client.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"

            m = client.get("/metrics")
            assert m.status_code == 200
            assert m.json()["events_processed"] == 0

    def test_validation_errors(self) -> None:
        with _make_client() as client:
            # Empty visitor_id -> 422 JSON error
            r = client.post("/events", json={"visitor_id": "", "page_id": "/a"})
            assert r.status_code == 422
            assert r.json()["code"] == "VALIDATION_ERROR"

            # Batch > 10,000 -> 422
            too_many = [{"visitor_id": "u1", "page_id": "/a"}] * 10_001
            r2 = client.post("/events/batch", json={"events": too_many})
            assert r2.status_code == 422

            # start > end -> 422
            r3 = client.get("/stats/unique", params={"start": 200.0, "end": 100.0})
            assert r3.status_code == 422
            assert r3.json()["code"] == "HTTP_422"

            # Unknown HLL sketch -> 404
            r4 = client.get("/sketch/hll/missing")
            assert r4.status_code == 404

    def test_api_key_auth(self) -> None:
        with _make_client(api_key="secret123") as client:
            r_unauth = client.post("/admin/snapshot")
            assert r_unauth.status_code == 401

            r_ev_unauth = client.post("/events", json={"visitor_id": "u1", "page_id": "/x"})
            assert r_ev_unauth.status_code == 401

            r_ok = client.post(
                "/events",
                headers={"X-API-Key": "secret123"},
                json={"visitor_id": "u1", "page_id": "/x"},
            )
            assert r_ok.status_code == 202


class TestAPIEndpointsAndAccuracy:
    """All stats endpoints, compare=true on 200k events, merge, and restart."""

    def test_all_endpoints_and_compare(self) -> None:
        with _make_client(enable_exact=True) as client:
            sim = client.post(
                "/admin/simulate",
                json={"n_events": 20_000, "n_visitors": 4_000, "n_pages": 30, "seed": 42},
            )
            assert sim.status_code == 200
            assert sim.json()["events_generated"] == 20_000

            uniq = client.get("/stats/unique", params={"compare": "true"})
            assert uniq.status_code == 200
            body = uniq.json()
            assert body["exact"] > 0
            assert body["rel_error"] < 0.03

            top = client.get("/stats/top", params={"k": 5})
            assert top.status_code == 200
            top_page = top.json()["items"][0]["page"]

            views = client.get("/stats/views", params={"page": top_page})
            assert views.status_code == 200
            assert views.json()["estimate"] > 0

            series = client.get("/stats/series", params={"granularity": "hour"})
            assert series.status_code == 200
            assert len(series.json()["points"]) > 0

            ov = client.get("/stats/overview")
            assert ov.status_code == 200
            assert ov.json()["total_events"] > 0

            nvr = client.get("/stats/new-vs-returning")
            assert nvr.status_code == 200

            second_page = top.json()["items"][1]["page"]
            lap = client.get("/stats/overlap", params={"a": top_page, "b": second_page})
            assert lap.status_code == 200

            # Reset
            rst = client.post("/admin/reset")
            assert rst.status_code == 200
            assert client.get("/metrics").json()["events_processed"] == 0

    @pytest.mark.slow
    def test_compare_200k_accuracy(self) -> None:
        with _make_client(enable_exact=True) as client:
            sim = client.post(
                "/admin/simulate",
                json={"n_events": 200_000, "n_visitors": 25_000, "n_pages": 30, "seed": 42},
            )
            assert sim.status_code == 200
            uniq = client.get("/stats/unique", params={"compare": "true"})
            assert uniq.status_code == 200
            assert uniq.json()["rel_error"] < 0.03

    def test_hll_merge_over_http(self) -> None:
        h1 = HyperLogLog(p=10, seed=0)
        h1.add_many([f"a_{i}" for i in range(500)])
        h2 = HyperLogLog(p=10, seed=0)
        h2.add_many([f"b_{i}" for i in range(500)])

        b1 = base64.b64encode(h1.to_bytes()).decode("ascii")
        b2 = base64.b64encode(h2.to_bytes()).decode("ascii")

        with _make_client() as client:
            r1 = client.post("/sketch/hll/merge", json={"name": "site", "sketch_b64": b1})
            assert r1.status_code == 200
            r2 = client.post("/sketch/hll/merge", json={"name": "site", "sketch_b64": b2})
            assert r2.status_code == 200

            got = client.get("/sketch/hll/site")
            assert got.status_code == 200
            local = h1.copy().merge(h2)
            assert got.json()["estimate"] == local.count()

    def test_restart_persistence(self, tmp_path: Path) -> None:
        with _make_client(storage_dir=tmp_path) as c1:
            c1.post(
                "/events/batch",
                json={
                    "events": [
                        {"visitor_id": f"u_{i}", "page_id": "/home", "timestamp": 1_700_000_000.0}
                        for i in range(100)
                    ]
                },
            )
            before = c1.get("/stats/unique").json()["estimate"]

        # New app instance loads snapshot saved on graceful shutdown
        with _make_client(storage_dir=tmp_path) as c2:
            after = c2.get("/stats/unique").json()["estimate"]
            assert after == before

    def test_concurrent_batch_posts_no_lost_events(self) -> None:
        settings = AppSettings(hll_p=14, page_hll_p=10, use_memory_storage=True)
        application = create_app(settings=settings, storage=InMemoryBackend())

        async def _run() -> int:
            transport = httpx.ASGITransport(app=application)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                async def _send(worker_id: int) -> int:
                    batch = [
                        {
                            "visitor_id": f"w{worker_id}_u{i}",
                            "page_id": "/concurrent",
                            "timestamp": 1_700_000_000.0,
                        }
                        for i in range(250)
                    ]
                    r = await client.post("/events/batch", json={"events": batch})
                    assert r.status_code == 202
                    return int(r.json()["accepted"])

                results = await asyncio.gather(*(_send(w) for w in range(16)))
                m = await client.get("/metrics")
                assert m.json()["events_processed"] == 4_000
                return sum(results)

        assert asyncio.run(_run()) == 4_000
