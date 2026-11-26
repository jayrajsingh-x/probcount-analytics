"""Launch the ProbCount API, seed 500,000 events, and start Streamlit dashboard."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import httpx


def main() -> None:
    env = os.environ.copy()
    env.setdefault("PROBCOUNT_ENABLE_EXACT", "true")
    env.setdefault("PROBCOUNT_USE_MEMORY_STORAGE", "true")
    env.setdefault("PROBCOUNT_API_URL", "http://127.0.0.1:8000")

    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "probcount.api:app", "--host", "127.0.0.1", "--port", "8000"],
        env=env,
    )
    try:
        for _ in range(40):
            try:
                if httpx.get("http://127.0.0.1:8000/health", timeout=1.0).status_code == 200:
                    break
            except Exception:
                time.sleep(0.25)

        print("Seeding 500,000 simulated events...", flush=True)
        resp = httpx.post(
            "http://127.0.0.1:8000/admin/simulate",
            json={"n_events": 500_000, "n_visitors": 50_000, "n_pages": 100, "seed": 42},
            timeout=180.0,
        )
        resp.raise_for_status()
        print(f"Seeded: {resp.json()}", flush=True)

        ui_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "dashboard/app.py",
                "--server.port",
                "8501",
                "--server.headless",
                "true",
            ],
            env=env,
        )
        ui_proc.wait()
    finally:
        api_proc.terminate()


if __name__ == "__main__":
    main()
