"""Local dataset loaders (HTTP access logs & CSV clickstreams) with generator fallback."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from probcount.config import TrafficConfig
from probcount.generator import Event, TrafficGenerator

_CLF_PATTERN = re.compile(
    r'^(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<time>[^\]]+)\]\s+"(?P<method>\S+)\s+(?P<path>\S+)[^"]*"\s+\d+\s+\S+(?:\s+"[^"]*"\s+"(?P<ua>[^"]*)")?'
)


def load_access_log(path: str | Path) -> Iterator[Event]:
    """Parse a local Common/Combined HTTP access log into :class:`Event` objects."""
    with open(Path(path), encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _CLF_PATTERN.match(line.strip())
            if not m:
                continue
            ts_str = m.group("time")
            try:
                ts = datetime.strptime(ts_str, "%d/%b/%Y:%H:%M:%S %z").timestamp()
            except ValueError:
                ts = 0.0
            ua = (m.group("ua") or "").lower()
            is_bot = any(tok in ua for tok in ("bot", "crawler", "spider", "curl"))
            yield Event(
                timestamp=ts,
                visitor_id=m.group("ip"),
                page_id=m.group("path"),
                is_bot=is_bot,
            )


def load_clickstream_csv(path: str | Path) -> Iterator[Event]:
    """Parse a local CSV clickstream with columns ``timestamp,visitor_id,page_id,is_bot``."""
    with open(Path(path), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield Event(
                timestamp=float(row["timestamp"]),
                visitor_id=row["visitor_id"],
                page_id=row["page_id"],
                is_bot=row.get("is_bot", "false").lower() in ("true", "1", "yes"),
            )


def load_or_fallback(
    path: str | Path | None,
    fallback_config: TrafficConfig | None = None,
) -> Iterator[Event]:
    """Load events from a local file if it exists; otherwise stream from TrafficGenerator."""
    if path is not None:
        p = Path(path)
        if p.exists():
            if p.suffix.lower() == ".csv":
                yield from load_clickstream_csv(p)
                return
            yield from load_access_log(p)
            return

    cfg = fallback_config or TrafficConfig(
        n_events=100_000, n_unique_visitors=20_000, n_pages=100, seed=42
    )
    yield from TrafficGenerator(cfg).generate()
