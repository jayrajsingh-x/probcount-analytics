"""Async HTTP load test for the ProbCount FastAPI service.

Usage:
    python scripts/load_test.py --url http://127.0.0.1:8000 --batches 20 --batch-size 5000
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx
import numpy as np


async def run_load_test(
    url: str,
    n_batches: int,
    batch_size: int,
    concurrency: int,
    api_key: str | None = None,
) -> tuple[float, float, float, float]:
    """Post *n_batches* of *batch_size* events and return (eps, p50_ms, p95_ms, p99_ms)."""
    headers = {"X-API-Key": api_key} if api_key else {}
    sem = asyncio.Semaphore(concurrency)
    latencies_ms: list[float] = []

    payload = {
        "events": [
            {
                "visitor_id": f"u_{i % 2000}",
                "page_id": f"/products/{i % 50}",
                "is_bot": False,
            }
            for i in range(batch_size)
        ]
    }

    async with httpx.AsyncClient(base_url=url, headers=headers, timeout=60.0) as client:

        async def _post_one() -> None:
            async with sem:
                t0 = time.perf_counter()
                resp = await client.post("/events/batch", json=payload)
                resp.raise_for_status()
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        t_start = time.perf_counter()
        await asyncio.gather(*(_post_one() for _ in range(n_batches)))
        total_sec = time.perf_counter() - t_start

    total_events = n_batches * batch_size
    eps = total_events / total_sec
    arr = np.array(latencies_ms)
    return (
        eps,
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ProbCount async HTTP load test")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    eps, p50, p95, p99 = asyncio.run(
        run_load_test(args.url, args.batches, args.batch_size, args.concurrency, args.api_key)
    )
    print(f"Total events: {args.batches * args.batch_size:,}")
    print(f"Throughput:   {eps:,.1f} events/sec")
    print(f"Latency p50:  {p50:.2f} ms | p95: {p95:.2f} ms | p99: {p99:.2f} ms")


if __name__ == "__main__":
    main()
