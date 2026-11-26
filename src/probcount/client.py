"""Lightweight Python HTTP client for the ProbCount Analytics API."""

from __future__ import annotations

from typing import Any

import httpx


class ProbCountClient:
    """Synchronous HTTP client wrapping the ProbCount FastAPI service."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def post_event(
        self,
        visitor_id: str,
        page_id: str,
        timestamp: float | None = None,
        is_bot: bool = False,
    ) -> dict[str, Any]:
        """Submit a single visit event."""
        resp = self._client.post(
            "/events",
            json={
                "visitor_id": visitor_id,
                "page_id": page_id,
                "timestamp": timestamp,
                "is_bot": is_bot,
            },
        )
        resp.raise_for_status()
        return dict(resp.json())

    def post_batch(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit a batch of up to 10,000 events."""
        resp = self._client.post("/events/batch", json={"events": events})
        resp.raise_for_status()
        return dict(resp.json())

    def unique_visitors(
        self,
        site: str = "default",
        page: str | None = None,
        compare: bool = False,
    ) -> dict[str, Any]:
        """Query unique visitors."""
        params: dict[str, Any] = {"site": site, "compare": compare}
        if page is not None:
            params["page"] = page
        resp = self._client.get("/stats/unique", params=params)
        resp.raise_for_status()
        return dict(resp.json())

    def top_pages(self, k: int = 10) -> dict[str, Any]:
        """Query top-*k* pages."""
        resp = self._client.get("/stats/top", params={"k": k})
        resp.raise_for_status()
        return dict(resp.json())

    def metrics(self) -> dict[str, Any]:
        """Fetch service metrics."""
        resp = self._client.get("/metrics")
        resp.raise_for_status()
        return dict(resp.json())
