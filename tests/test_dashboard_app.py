"""Streamlit AppTest smoke tests for all dashboard pages."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import patch

import pytest
from dashboard.data import ApiResult
from streamlit.testing.v1 import AppTest

from probcount.hll import HyperLogLog

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"


def _ok(data: dict[str, object]) -> ApiResult:
    return ApiResult(ok=True, data=data)


@pytest.fixture()
def mock_dashboard_api():
    hll = HyperLogLog(p=10)
    hll.add("u1")
    hll_bytes = hll.to_bytes()

    with (
        patch("dashboard.data.fetch_health", return_value=_ok({"status": "ok", "uptime_seconds": 5.0})),
        patch(
            "dashboard.data.fetch_overview",
            return_value=_ok(
                {
                    "unique_visitors": 500.0,
                    "total_events": 2000,
                    "bots_ignored": 50,
                    "top_page": "/products/1",
                }
            ),
        ),
        patch(
            "dashboard.data.fetch_metrics",
            return_value=_ok(
                {
                    "events_processed": 2000,
                    "memory": {"total_bytes": 50_000, "exact_bytes": 2_000_000},
                    "sketch_params": {"hll_p": 14, "cms_width": 1000, "cms_depth": 4},
                }
            ),
        ),
        patch(
            "dashboard.data.fetch_series",
            return_value=_ok(
                {
                    "granularity": "hour",
                    "points": [
                        {
                            "timestamp": 1700000000.0,
                            "unique_visitors": 100.0,
                            "exact_unique_visitors": 101.0,
                        },
                        {
                            "timestamp": 1700003600.0,
                            "unique_visitors": 150.0,
                            "exact_unique_visitors": 149.0,
                        },
                    ],
                }
            ),
        ),
        patch(
            "dashboard.data.fetch_top_pages",
            return_value=_ok(
                {
                    "items": [
                        {
                            "page": "/products/1",
                            "views": 500,
                            "exact_views": 495,
                            "abs_error": 5,
                            "rel_error": 0.01,
                        }
                    ]
                }
            ),
        ),
        patch(
            "dashboard.data.fetch_sketch_inspect",
            return_value=_ok(
                {
                    "hll_p": 14,
                    "hll_m": 16384,
                    "hll_estimate": 500.0,
                    "register_histogram": {"0": 15900, "1": 250, "2": 130},
                    "first_64_registers": [0] * 64,
                    "cms_width": 1024,
                    "cms_depth": 4,
                    "cms_slice": [[0] * 32 for _ in range(4)],
                }
            ),
        ),
        patch("dashboard.data.fetch_site_hll_bytes", return_value=hll_bytes),
        patch("dashboard.data._get", return_value=_ok({"sketch_b64": base64.b64encode(hll_bytes).decode()})),
    ):
        yield


@pytest.mark.parametrize(
    "script_rel",
    [
        "app.py",
        "pages/1_Top_Pages.py",
        "pages/2_Accuracy_Lab.py",
        "pages/3_Parameter_Playground.py",
        "pages/4_Sketch_Inspector.py",
        "pages/5_Merge_Demo.py",
        "pages/6_Simulator.py",
        "pages/7_Admin.py",
    ],
)
def test_page_renders_without_exception(mock_dashboard_api: object, script_rel: str) -> None:
    at = AppTest.from_file(str(DASHBOARD_DIR / script_rel), default_timeout=15)
    at.run()
    assert not at.exception, f"{script_rel} raised: {at.exception}"
