"""Unit tests for :mod:`dashboard.data` using mocked httpx responses."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock

import httpx
from dashboard.data import (
    compute_memory_saved_pct,
    fetch_health,
    fetch_metrics,
    fetch_overview,
    fetch_series,
    fetch_site_hll_bytes,
    fetch_sketch_inspect,
    fetch_top_pages,
    metrics_to_csv,
    trigger_reset,
    trigger_simulation,
    trigger_snapshot,
)

from probcount.hll import HyperLogLog


def _mock_client(json_data: dict[str, object] | None = None, status: int = 200) -> MagicMock:
    client = MagicMock(spec=httpx.Client)
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.text = "error"
    resp.json.return_value = json_data or {}
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    client.get.return_value = resp
    client.post.return_value = resp
    return client


def test_fetch_helpers_success() -> None:
    c = _mock_client({"status": "ok", "uptime_seconds": 12.0})
    assert fetch_health("http://x", client=c).ok is True
    assert fetch_metrics("http://x", client=c).ok is True
    assert fetch_overview("http://x", client=c).ok is True
    assert fetch_series("http://x", compare=True, client=c).ok is True
    assert fetch_top_pages("http://x", k=5, start=0.0, end=10.0, client=c).ok is True
    assert fetch_sketch_inspect("http://x", client=c).ok is True
    assert trigger_snapshot("http://x", client=c).ok is True
    assert trigger_reset("http://x", client=c).ok is True
    assert (
        trigger_simulation("http://x", 100, 10, 5, 1.1, 0.05, 1.0, client=c).ok is True
    )


def test_fetch_site_hll_bytes() -> None:
    hll = HyperLogLog(p=10)
    hll.add("u1")
    b64 = base64.b64encode(hll.to_bytes()).decode("ascii")
    c = _mock_client({"sketch_b64": b64})
    raw = fetch_site_hll_bytes("http://x", client=c)
    assert raw == hll.to_bytes()

    c_err = _mock_client({}, status=404)
    assert fetch_site_hll_bytes("http://x", client=c_err) is None


def test_error_and_timeout_handling() -> None:
    c_500 = _mock_client({}, status=500)
    res_500 = fetch_overview("http://x", client=c_500)
    assert res_500.ok is False
    assert "HTTP 500" in res_500.error

    c_to = MagicMock(spec=httpx.Client)
    c_to.get.side_effect = httpx.ReadTimeout("timed out")
    c_to.post.side_effect = httpx.ReadTimeout("timed out")
    res_to = fetch_health("http://x", client=c_to)
    assert res_to.ok is False
    assert "timed out" in res_to.error

    res_post_to = trigger_snapshot("http://x", client=c_to)
    assert res_post_to.ok is False
    assert "timed out" in res_post_to.error


def test_memory_saved_and_csv() -> None:
    assert compute_memory_saved_pct({"total_bytes": 1000, "exact_bytes": 100_000}) == 99.0
    assert compute_memory_saved_pct({"total_bytes": 1000, "exact_bytes": 0}) == 0.0
    csv_str = metrics_to_csv({"events_processed": 10, "memory": {"total_bytes": 128}})
    assert "events_processed,10" in csv_str
    assert "memory.total_bytes,128" in csv_str
