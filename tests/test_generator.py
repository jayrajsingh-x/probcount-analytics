"""Tests for :mod:`probcount.generator` — synthetic traffic generation.

Coverage includes reproducibility, Zipf distribution check, unique-visitor
bound, spike behaviour, streaming memory, ground-truth consistency,
I/O round-trips (JSONL and CSV), and CLI smoke test.
"""

from __future__ import annotations

import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from probcount.config import TrafficConfig, TrafficSpike
from probcount.exact import ExactUniqueCounter
from probcount.generator import (
    Event,
    TrafficGenerator,
    generate_events,
    main,
    read_csv,
    read_jsonl,
    write_csv,
    write_jsonl,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Same config + seed must yield an identical stream."""

    def test_same_seed_same_stream(self, small_config: TrafficConfig) -> None:
        gen1 = TrafficGenerator(small_config)
        gen2 = TrafficGenerator(small_config)
        events1 = list(gen1.generate())
        events2 = list(gen2.generate())
        assert events1 == events2

    def test_different_seed_different_stream(self) -> None:
        cfg1 = TrafficConfig(n_events=200, n_unique_visitors=50, n_pages=5, seed=1)
        cfg2 = TrafficConfig(n_events=200, n_unique_visitors=50, n_pages=5, seed=2)
        gen1 = list(TrafficGenerator(cfg1).generate())
        gen2 = list(TrafficGenerator(cfg2).generate())
        assert gen1 != gen2


# ---------------------------------------------------------------------------
# Distribution checks
# ---------------------------------------------------------------------------


class TestZipfDistribution:
    """Page frequencies should follow a Zipf-like power law."""

    def test_rank_frequency_slope(self, medium_config: TrafficConfig) -> None:
        """Log-log slope of rank vs. frequency should be close to -page_zipf_s."""
        gen = TrafficGenerator(medium_config)
        page_counts: dict[str, int] = {}
        for event in gen.generate():
            page_counts[event.page_id] = page_counts.get(event.page_id, 0) + 1

        # Sort by frequency descending → rank order
        freqs = sorted(page_counts.values(), reverse=True)
        ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)
        log_ranks = np.log(ranks)
        log_freqs = np.log(np.array(freqs, dtype=np.float64))

        # Linear regression in log-log space
        slope, _ = np.polyfit(log_ranks, log_freqs, 1)  # type: ignore[call-overload]

        # Slope should be approximately -page_zipf_s (within tolerance)
        assert abs(slope + medium_config.page_zipf_s) < 0.4, (
            f"Zipf slope {slope:.3f}, expected ~{-medium_config.page_zipf_s:.1f}"
        )


# ---------------------------------------------------------------------------
# Unique visitor bound
# ---------------------------------------------------------------------------


class TestUniqueVisitorBound:
    """The generator must not produce more visitors than configured."""

    def test_visitor_count_bounded(self) -> None:
        cfg = TrafficConfig(
            n_events=10_000, n_unique_visitors=100, n_pages=10, seed=42,
        )
        gen = TrafficGenerator(cfg)
        visitor_ids = {e.visitor_id for e in gen.generate()}
        assert len(visitor_ids) <= cfg.n_unique_visitors


# ---------------------------------------------------------------------------
# Spike window
# ---------------------------------------------------------------------------


class TestTrafficSpike:
    """Events in the spike window should have a higher rate."""

    def test_spike_increases_rate(self) -> None:
        start = 0.0
        duration = 86_400.0  # 24 hours
        spike_start = 36_000.0  # hour 10
        spike_end = 43_200.0  # hour 12 (2-hour window)
        multiplier = 5.0

        cfg = TrafficConfig(
            n_events=100_000,
            n_unique_visitors=10_000,
            n_pages=20,
            start_time=start,
            duration_seconds=duration,
            diurnal=False,  # disable diurnal to isolate spike effect
            traffic_spike=TrafficSpike(spike_start, spike_end, multiplier),
            seed=42,
        )
        gen = TrafficGenerator(cfg)
        in_spike = 0
        outside = 0
        for event in gen.generate():
            if spike_start <= event.timestamp < spike_end:
                in_spike += 1
            else:
                outside += 1

        # Spike window is 2h out of 24h = 1/12 of duration
        # With multiplier=5, expected fraction in spike ≈ 5/12 / (5/12 + 11/12)
        # ≈ 5/16 ≈ 31.25%
        # Rate in spike ≈ 5× rate outside
        spike_hours = (spike_end - spike_start) / 3600  # 2
        outside_hours = (duration - (spike_end - spike_start)) / 3600  # 22
        rate_in = in_spike / spike_hours
        rate_out = outside / outside_hours

        ratio = rate_in / rate_out
        # Should be close to multiplier (within 50% tolerance)
        assert ratio > multiplier * 0.5, f"Spike ratio {ratio:.2f} too low"
        assert ratio < multiplier * 1.5, f"Spike ratio {ratio:.2f} too high"


# ---------------------------------------------------------------------------
# Streaming memory
# ---------------------------------------------------------------------------


class TestStreamingMemory:
    """The generator must use bounded memory regardless of n_events."""

    def test_bounded_memory(self) -> None:
        """Generate 1M events and verify peak memory stays under 200 MB."""
        cfg = TrafficConfig(
            n_events=1_000_000,
            n_unique_visitors=10_000,
            n_pages=50,
            seed=42,
            diurnal=False,
        )
        gen = TrafficGenerator(cfg)
        tracemalloc.start()
        count = 0
        for _ in gen.generate():
            count += 1
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert count == 1_000_000
        # 200 MB threshold — generous but far below what list-accumulation
        # would use (~300–400 MB for 1M Event objects)
        assert peak < 200 * 1024 * 1024, (
            f"Peak memory {peak / 1024 / 1024:.1f} MB exceeds 200 MB"
        )


# ---------------------------------------------------------------------------
# Ground truth consistency
# ---------------------------------------------------------------------------


class TestGroundTruth:
    """Generator ground truth must match an independent exact counter."""

    def test_matches_exact_counter(self) -> None:
        cfg = TrafficConfig(
            n_events=50_000,
            n_unique_visitors=5_000,
            n_pages=20,
            seed=99,
        )
        gen = TrafficGenerator(cfg)
        exact = ExactUniqueCounter()
        for event in gen.generate():
            exact.add(event.visitor_id)

        truth = gen.ground_truth
        assert truth.total_events == 50_000
        assert truth.unique_visitors == int(exact.count())

    def test_ground_truth_raises_before_consume(self) -> None:
        gen = TrafficGenerator(
            TrafficConfig(n_events=100, n_unique_visitors=10, n_pages=5, seed=0)
        )
        with pytest.raises(RuntimeError, match="fully consumed"):
            _ = gen.ground_truth


# ---------------------------------------------------------------------------
# I/O round-trips
# ---------------------------------------------------------------------------


class TestIOJsonl:
    """JSONL write → read round-trip."""

    def test_roundtrip(self, small_config: TrafficConfig, tmp_path: Path) -> None:
        gen = TrafficGenerator(small_config)
        events = list(gen.generate())[:100]
        path = tmp_path / "events.jsonl"
        write_jsonl(path, iter(events))
        restored = list(read_jsonl(path))
        assert restored == events


class TestIOCsv:
    """CSV write → read round-trip."""

    def test_roundtrip(self, small_config: TrafficConfig, tmp_path: Path) -> None:
        gen = TrafficGenerator(small_config)
        events = list(gen.generate())[:100]
        path = tmp_path / "events.csv"
        write_csv(path, iter(events))
        restored = list(read_csv(path))

        # Floating-point round-trip via CSV may lose precision
        for orig, rest in zip(events, restored, strict=True):
            assert orig.visitor_id == rest.visitor_id
            assert orig.page_id == rest.page_id
            assert orig.is_bot == rest.is_bot
            assert abs(orig.timestamp - rest.timestamp) < 1e-6


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestCLI:
    """The module should be runnable via ``python -m probcount.generator``."""

    def test_cli_produces_file(self, tmp_path: Path) -> None:
        out = tmp_path / "cli_output.jsonl"
        result = subprocess.run(
            [
                sys.executable, "-m", "probcount.generator",
                "--events", "500",
                "--visitors", "50",
                "--pages", "5",
                "--seed", "7",
                "--out", str(out),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 500

    def test_main_inprocess_csv_and_generate_events(
        self, small_config: TrafficConfig, tmp_path: Path
    ) -> None:
        events = list(generate_events(small_config))
        assert len(events) == small_config.n_events

        out_csv = tmp_path / "cli_out.csv"
        main(
            [
                "--events", "200",
                "--visitors", "20",
                "--pages", "5",
                "--seed", "11",
                "--out", str(out_csv),
            ]
        )
        assert out_csv.exists()
        restored = list(read_csv(out_csv))
        assert len(restored) == 200


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------


class TestEvent:
    """Basic Event properties."""

    def test_frozen(self) -> None:
        e = Event(timestamp=1.0, visitor_id="u_abc", page_id="/x", is_bot=False)
        with pytest.raises(AttributeError):
            e.visitor_id = "other"  # type: ignore[misc]

    def test_equality(self) -> None:
        e1 = Event(1.0, "u_abc", "/x", False)
        e2 = Event(1.0, "u_abc", "/x", False)
        assert e1 == e2
