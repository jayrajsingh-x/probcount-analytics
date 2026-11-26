"""Distributed multi-worker sketch simulation and coordinator merge."""

from __future__ import annotations

import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from probcount.engine import AnalyticsEngine, EngineConfig
from probcount.generator import Event
from probcount.hashing import hash64

PartitionMode = Literal["hash", "round_robin", "random_dup"]


@dataclass(frozen=True)
class DistributedResult:
    """Output of a distributed simulation run."""

    merged_engine: AnalyticsEngine
    worker_engines: list[AnalyticsEngine]
    merge_seconds: float


def simulate_distributed(
    events: Iterable[Event],
    config: EngineConfig,
    n_workers: int = 4,
    mode: PartitionMode = "hash",
    seed: int = 42,
    dup_prob: float = 0.2,
) -> DistributedResult:
    """Partition *events* across *n_workers* engines and merge at a coordinator.

    Supported partitioning modes:
    * ``hash``: route by ``hash64(event.visitor_id) % n_workers`` (disjoint visitors).
    * ``round_robin``: route event ``i`` to worker ``i % n_workers`` (overlapping visitors).
    * ``random_dup``: route randomly with optional duplicate delivery (``dup_prob``).
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")

    workers = [AnalyticsEngine(config) for _ in range(n_workers)]
    rng = random.Random(seed)

    for idx, event in enumerate(events):
        if mode == "hash":
            w = hash64(event.visitor_id, seed) % n_workers
            workers[w].process_event(event)
        elif mode == "round_robin":
            w = idx % n_workers
            workers[w].process_event(event)
        elif mode == "random_dup":
            w = rng.randrange(n_workers)
            workers[w].process_event(event)
            if n_workers > 1 and rng.random() < dup_prob:
                w2 = (w + 1) % n_workers
                workers[w2].process_event(event)
        else:
            raise ValueError(f"Unknown partition mode: {mode}")

    t0 = time.perf_counter()
    merged = AnalyticsEngine(config)
    for worker in workers:
        merged.merge(worker)
    merge_seconds = time.perf_counter() - t0

    return DistributedResult(
        merged_engine=merged,
        worker_engines=workers,
        merge_seconds=merge_seconds,
    )
