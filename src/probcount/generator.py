"""Synthetic traffic generator for web-analytics simulation.

Produces a reproducible stream of :class:`Event` objects that model
realistic website traffic with:

* Zipf-distributed page popularity (a few pages get most views).
* Zipf-distributed visitor activity (some visitors are heavy hitters).
* Optional diurnal (time-of-day) variation.
* Optional traffic spikes in configurable time windows.
* Configurable bot-traffic fraction.

The generator is **streaming**: events are yielded lazily, so memory
usage does not grow with ``n_events``.  Ground-truth statistics are
collected on-the-fly during generation and exposed via
:attr:`TrafficGenerator.ground_truth` after the stream is fully consumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from probcount.config import TrafficConfig

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """A single visit event.

    Attributes:
        timestamp: Seconds since epoch.
        visitor_id: Unique visitor identifier (e.g. ``'u_a3f8c1d9e2b0'``).
        page_id: Page path (e.g. ``'/products/42'``).
        is_bot: Whether this event was generated as bot traffic.
    """

    timestamp: float
    visitor_id: str
    page_id: str
    is_bot: bool


@dataclass(frozen=True)
class GroundTruth:
    """Exact statistics recorded during traffic generation.

    Attributes:
        total_events: Total number of events generated.
        unique_visitors: True number of distinct visitor IDs.
        unique_visitors_per_page: Mapping page → distinct visitor count.
        unique_visitors_per_hour: Mapping hour-index → distinct visitor count.
    """

    total_events: int
    unique_visitors: int
    unique_visitors_per_page: dict[str, int]
    unique_visitors_per_hour: dict[int, int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_pages(n_pages: int) -> list[str]:
    """Create *n_pages* realistic-looking page paths."""
    pages: list[str] = []
    n_products = max(1, int(n_pages * 0.6))
    n_blog = max(1, int(n_pages * 0.25))
    n_static = max(0, n_pages - n_products - n_blog)

    for i in range(1, n_products + 1):
        pages.append(f"/products/{i}")
    for i in range(1, n_blog + 1):
        pages.append(f"/blog/{i}")

    static = [
        "/about", "/contact", "/faq", "/pricing", "/terms",
        "/privacy", "/careers", "/support", "/login", "/signup",
        "/help", "/docs", "/api", "/status", "/features",
    ]
    for i in range(n_static):
        pages.append(static[i] if i < len(static) else f"/page/{i}")

    return pages[:n_pages]


def _zipf_weights(n: int, s: float) -> np.ndarray:
    """Return normalized Zipf probabilities for ranks 1 … *n*."""
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, s)
    weights /= weights.sum()
    return weights


def _generate_timestamps(
    config: TrafficConfig,
    rng_np: np.random.Generator,
) -> np.ndarray:
    """Pre-generate and sort all event timestamps.

    Timestamps are sampled per-hour with weights that account for the
    optional diurnal curve and traffic spike.  Memory usage is *O(n)*
    in ``float64`` values (~8 bytes each), which is acceptable and bounded.
    """
    n = config.n_events
    duration = config.duration_seconds
    start = config.start_time
    n_hours = max(1, math.ceil(duration / 3600))

    # --- hourly weights ---
    if config.diurnal:
        hours_of_day = np.arange(n_hours, dtype=np.float64) % 24
        weights = 1.0 + 0.8 * np.sin(2 * np.pi * (hours_of_day - 6.0) / 24.0)
    else:
        weights = np.ones(n_hours, dtype=np.float64)

    # --- spike overlay ---
    if config.traffic_spike is not None:
        spike = config.traffic_spike
        for h in range(n_hours):
            hour_start = start + h * 3600
            hour_end = hour_start + 3600
            if hour_start < spike.end and hour_end > spike.start:
                weights[h] *= spike.multiplier

    weights /= weights.sum()

    # --- sample in chunks to limit peak numpy memory ---
    chunk_size = 500_000
    timestamps = np.empty(n, dtype=np.float64)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        sz = end - i
        hours = rng_np.choice(n_hours, size=sz, p=weights)
        timestamps[i:end] = start + hours * 3600 + rng_np.uniform(0, 3600, size=sz)

    # Clip & sort so events are chronological
    np.clip(timestamps, start, start + duration, out=timestamps)
    timestamps.sort()
    return timestamps


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------


class TrafficGenerator:
    """Streaming synthetic traffic generator.

    Usage::

        gen = TrafficGenerator(config)
        for event in gen.generate():
            process(event)
        truth = gen.ground_truth

    The ``ground_truth`` attribute is available only after the iterator
    returned by :meth:`generate` has been fully consumed.
    """

    def __init__(self, config: TrafficConfig) -> None:
        self.config = config
        self._ground_truth: GroundTruth | None = None

    @property
    def ground_truth(self) -> GroundTruth:
        """Exact statistics collected during generation.

        Raises:
            RuntimeError: If the generator has not been fully consumed yet.
        """
        if self._ground_truth is None:
            raise RuntimeError(
                "Ground truth is available only after the generator "
                "stream has been fully consumed."
            )
        return self._ground_truth

    def generate(self) -> Iterator[Event]:
        """Yield events lazily in chronological order.

        Events are produced one at a time so that memory usage does not
        grow with ``n_events``.  Ground-truth statistics are accumulated
        in internal sets/counters that are bounded by
        ``n_unique_visitors × n_pages`` (not by ``n_events``).
        """
        cfg = self.config
        rng = random.Random(cfg.seed)
        rng_np = np.random.default_rng(cfg.seed)

        # --- pre-generate fixed pools ---
        visitor_ids = [
            f"u_{rng.getrandbits(48):012x}" for _ in range(cfg.n_unique_visitors)
        ]
        pages = _generate_pages(cfg.n_pages)

        page_weights = _zipf_weights(cfg.n_pages, cfg.page_zipf_s)
        visitor_weights = _zipf_weights(cfg.n_unique_visitors, cfg.visitor_zipf_s)

        # --- pre-generate timestamps (bounded float64 array) ---
        timestamps = _generate_timestamps(cfg, rng_np)

        # --- ground-truth accumulators ---
        seen_visitors: set[str] = set()
        visitors_per_page: dict[str, set[str]] = defaultdict(set)
        visitors_per_hour: dict[int, set[str]] = defaultdict(set)

        # --- streaming event loop (chunked RNG draws) ---
        chunk = 100_000
        idx = 0
        n = cfg.n_events

        while idx < n:
            end = min(idx + chunk, n)
            sz = end - idx

            # Visitor selection: mix of Zipf and uniform draws
            use_zipf = rng_np.random(size=sz) < cfg.returning_visitor_ratio
            n_zipf = int(use_zipf.sum())
            n_uniform = sz - n_zipf

            v_indices = np.empty(sz, dtype=np.intp)
            if n_zipf > 0:
                v_indices[use_zipf] = rng_np.choice(
                    cfg.n_unique_visitors, size=n_zipf, p=visitor_weights,
                )
            if n_uniform > 0:
                v_indices[~use_zipf] = rng_np.integers(
                    0, cfg.n_unique_visitors, size=n_uniform,
                )

            p_indices = rng_np.choice(cfg.n_pages, size=sz, p=page_weights)
            bot_flags = rng_np.random(size=sz) < cfg.bot_fraction

            for j in range(sz):
                vid = visitor_ids[int(v_indices[j])]
                pid = pages[int(p_indices[j])]
                ts = float(timestamps[idx])
                is_bot = bool(bot_flags[j])

                # ground-truth tracking
                seen_visitors.add(vid)
                visitors_per_page[pid].add(vid)
                hour = int((ts - cfg.start_time) / 3600)
                visitors_per_hour[hour].add(vid)

                yield Event(timestamp=ts, visitor_id=vid, page_id=pid, is_bot=is_bot)
                idx += 1

        self._ground_truth = GroundTruth(
            total_events=n,
            unique_visitors=len(seen_visitors),
            unique_visitors_per_page={p: len(v) for p, v in visitors_per_page.items()},
            unique_visitors_per_hour={h: len(v) for h, v in visitors_per_hour.items()},
        )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def generate_events(config: TrafficConfig) -> Iterator[Event]:
    """Convenience wrapper: yield events from a :class:`TrafficGenerator`.

    .. note::

        Ground truth is not accessible through this function.  Use
        :class:`TrafficGenerator` directly if you need it.
    """
    gen = TrafficGenerator(config)
    yield from gen.generate()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_jsonl(path: str | Path, events: Iterator[Event] | list[Event]) -> None:
    """Write events to a JSONL (JSON Lines) file.

    Parent directories are created automatically.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(asdict(event)) + "\n")


def read_jsonl(path: str | Path) -> Iterator[Event]:
    """Lazily read events from a JSONL file."""
    with open(Path(path), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            yield Event(**d)


def write_csv(path: str | Path, events: Iterator[Event] | list[Event]) -> None:
    """Write events to a CSV file with a header row.

    Parent directories are created automatically.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "visitor_id", "page_id", "is_bot"])
        for event in events:
            writer.writerow([event.timestamp, event.visitor_id, event.page_id, event.is_bot])


def read_csv(path: str | Path) -> Iterator[Event]:
    """Lazily read events from a CSV file."""
    with open(Path(path), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield Event(
                timestamp=float(row["timestamp"]),
                visitor_id=row["visitor_id"],
                page_id=row["page_id"],
                is_bot=row["is_bot"].lower() in ("true", "1", "yes"),
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="probcount.generator",
        description="Generate synthetic web-analytics traffic data.",
    )
    parser.add_argument("--events", type=int, default=100_000, help="Number of events")
    parser.add_argument("--visitors", type=int, default=10_000, help="Unique visitors")
    parser.add_argument("--pages", type=int, default=100, help="Number of pages")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out",
        type=str,
        default="data/sample.jsonl",
        help="Output file (.jsonl or .csv)",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "csv"],
        default=None,
        help="Output format (inferred from extension if omitted)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the traffic generator."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = TrafficConfig(
        n_events=args.events,
        n_unique_visitors=args.visitors,
        n_pages=args.pages,
        seed=args.seed,
    )

    gen = TrafficGenerator(config)
    events = gen.generate()

    fmt = args.format
    if fmt is None:
        fmt = "csv" if args.out.endswith(".csv") else "jsonl"

    print(f"Generating {config.n_events:,} events "
          f"({config.n_unique_visitors:,} unique visitors, "
          f"{config.n_pages} pages) …")

    if fmt == "csv":
        write_csv(args.out, events)
    else:
        write_jsonl(args.out, events)

    truth = gen.ground_truth
    print(f"Done. {truth.total_events:,} events written to {args.out}")
    print(f"Ground truth — unique visitors: {truth.unique_visitors:,}")
    print(f"Ground truth — pages with traffic: {len(truth.unique_visitors_per_page)}")


if __name__ == "__main__":
    main()
