"""Pure-Python data fetching and transformation layer for the dashboard.

Contains **no Streamlit imports** so it can be unit-tested cleanly with
mocked ``httpx`` responses.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class ApiResult:
    """Wrapper for API responses with graceful error handling."""

    ok: bool
    data: dict[str, Any]
    error: str = ""


def _get(
    base_url: str,
    path: str,
    params: dict[str, Any] | None = None,
    timeout: float = 10.0,
    client: httpx.Client | None = None,
) -> ApiResult:
    try:
        if client is not None:
            resp = client.get(path, params=params)
        else:
            with httpx.Client(base_url=base_url, timeout=timeout) as c:
                resp = c.get(path, params=params)
        resp.raise_for_status()
        return ApiResult(ok=True, data=dict(resp.json()))
    except httpx.TimeoutException:
        return ApiResult(ok=False, data={}, error="Request timed out while contacting API.")
    except httpx.HTTPStatusError as exc:
        return ApiResult(
            ok=False,
            data={},
            error=f"API returned HTTP {exc.response.status_code}: {exc.response.text[:200]}",
        )
    except httpx.HTTPError as exc:
        return ApiResult(ok=False, data={}, error=f"Cannot reach API at {base_url}: {exc}")


def _post(
    base_url: str,
    path: str,
    json_body: dict[str, Any] | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
    client: httpx.Client | None = None,
) -> ApiResult:
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        if client is not None:
            resp = client.post(path, json=json_body, headers=headers)
        else:
            with httpx.Client(base_url=base_url, timeout=timeout) as c:
                resp = c.post(path, json=json_body, headers=headers)
        resp.raise_for_status()
        return ApiResult(ok=True, data=dict(resp.json()))
    except httpx.TimeoutException:
        return ApiResult(ok=False, data={}, error="Request timed out while contacting API.")
    except httpx.HTTPStatusError as exc:
        return ApiResult(
            ok=False,
            data={},
            error=f"API returned HTTP {exc.response.status_code}: {exc.response.text[:200]}",
        )
    except httpx.HTTPError as exc:
        return ApiResult(ok=False, data={}, error=f"Cannot reach API at {base_url}: {exc}")


def fetch_health(base_url: str, client: httpx.Client | None = None) -> ApiResult:
    """Check API health."""
    return _get(base_url, "/health", timeout=3.0, client=client)


def fetch_metrics(base_url: str, client: httpx.Client | None = None) -> ApiResult:
    """Fetch service metrics and memory breakdown."""
    return _get(base_url, "/metrics", client=client)


def fetch_overview(base_url: str, client: httpx.Client | None = None) -> ApiResult:
    """Fetch headline overview KPIs."""
    return _get(base_url, "/stats/overview", client=client)


def fetch_series(
    base_url: str,
    granularity: str = "hour",
    compare: bool = False,
    client: httpx.Client | None = None,
) -> ApiResult:
    """Fetch unique-visitor time series."""
    return _get(
        base_url,
        "/stats/series",
        params={"granularity": granularity, "compare": str(compare).lower()},
        client=client,
    )


def fetch_top_pages(
    base_url: str,
    k: int = 10,
    start: float | None = None,
    end: float | None = None,
    compare: bool = False,
    client: httpx.Client | None = None,
) -> ApiResult:
    """Fetch top-K pages with optional exact comparison."""
    params: dict[str, Any] = {"k": k, "compare": str(compare).lower()}
    if start is not None:
        params["start"] = start
    if end is not None:
        params["end"] = end
    return _get(base_url, "/stats/top", params=params, client=client)


def fetch_sketch_inspect(base_url: str, client: httpx.Client | None = None) -> ApiResult:
    """Fetch HLL register histogram/slice and CMS table slice."""
    return _get(base_url, "/sketch/inspect", client=client)


def fetch_site_hll_bytes(base_url: str, client: httpx.Client | None = None) -> bytes | None:
    """Fetch serialized site HyperLogLog binary bytes."""
    res = _get(base_url, "/sketch/hll/site", client=client)
    if not res.ok or not res.data.get("sketch_b64"):
        return None
    return base64.b64decode(str(res.data["sketch_b64"]))


def trigger_simulation(
    base_url: str,
    n_events: int,
    n_visitors: int,
    n_pages: int,
    page_zipf_s: float,
    bot_fraction: float,
    spike_multiplier: float,
    seed: int = 42,
    api_key: str | None = None,
    client: httpx.Client | None = None,
) -> ApiResult:
    """Trigger synthetic traffic simulation on the API."""
    return _post(
        base_url,
        "/admin/simulate",
        json_body={
            "n_events": n_events,
            "n_visitors": n_visitors,
            "n_pages": n_pages,
            "page_zipf_s": page_zipf_s,
            "bot_fraction": bot_fraction,
            "spike_multiplier": spike_multiplier,
            "seed": seed,
        },
        api_key=api_key,
        client=client,
    )


def trigger_snapshot(
    base_url: str, api_key: str | None = None, client: httpx.Client | None = None
) -> ApiResult:
    """Force an immediate snapshot save."""
    return _post(base_url, "/admin/snapshot", api_key=api_key, client=client)


def trigger_reset(
    base_url: str, api_key: str | None = None, client: httpx.Client | None = None
) -> ApiResult:
    """Reset the engine on the API."""
    return _post(base_url, "/admin/reset", api_key=api_key, client=client)


def compute_memory_saved_pct(memory_report: dict[str, Any]) -> float:
    """Compute percentage of memory saved by sketches vs exact structures."""
    total = float(memory_report.get("total_bytes", 0))
    exact = float(memory_report.get("exact_bytes", 0))
    if exact <= 0:
        return 0.0
    return max(0.0, (1.0 - total / exact) * 100.0)


def metrics_to_csv(metrics: dict[str, Any]) -> str:
    """Flatten metrics dict into a simple two-column CSV string."""
    lines = ["metric,value"]
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                lines.append(f"{k}.{sub_k},{sub_v}")
        else:
            lines.append(f"{k},{v}")
    return "\n".join(lines) + "\n"
