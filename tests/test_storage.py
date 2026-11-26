"""Tests for :mod:`probcount.storage` — atomic snapshots, checksums, and fallback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from probcount.engine import AnalyticsEngine, EngineConfig
from probcount.generator import Event
from probcount.storage import (
    FileSnapshotBackend,
    InMemoryBackend,
    deserialize_engine,
    serialize_engine,
)


def _sample_engine() -> AnalyticsEngine:
    eng = AnalyticsEngine(EngineConfig(hll_p=10, page_hll_p=8, cms_width=64, cms_depth=3))
    for i in range(200):
        eng.process_event(Event(1_700_000_000.0 + i, f"u_{i % 50}", f"/p/{i % 10}", False))
    return eng


class TestSnapshotSerialization:
    """Binary PCS1 serialization & validation."""

    def test_roundtrip_query_equality(self) -> None:
        eng = _sample_engine()
        raw = serialize_engine(eng)
        restored = deserialize_engine(raw)

        assert restored.unique_visitors("default", 0.0, 2_000_000_000.0) == eng.unique_visitors(
            "default", 0.0, 2_000_000_000.0
        )
        assert restored.top_pages(5, 0.0, 2_000_000_000.0) == eng.top_pages(
            5, 0.0, 2_000_000_000.0
        )
        assert restored.events_processed == eng.events_processed

    def test_corrupted_bytes_rejected(self) -> None:
        raw = bytearray(serialize_engine(_sample_engine()))
        with pytest.raises(ValueError, match="too short"):
            deserialize_engine(b"PCS1")

        bad = bytearray(raw)
        bad[20] ^= 0xFF
        with pytest.raises(ValueError, match="checksum"):
            deserialize_engine(bytes(bad))


class TestBackends:
    """InMemoryBackend & FileSnapshotBackend atomic write, retention, and fallback."""

    def test_in_memory_backend_fallback_and_retention(self) -> None:
        backend = InMemoryBackend(max_snapshots=2)
        eng = _sample_engine()
        backend.save(eng)
        eng.process_event(Event(1_700_000_500.0, "extra", "/p/1", False))
        backend.save(eng)
        backend.save(eng)
        assert len(backend.list_snapshots()) == 2

        # Corrupt newest snapshot -> falls back to previous
        sid, _ = backend._snapshots[-1]
        backend._snapshots[-1] = (sid, b"corrupted")
        loaded = backend.load()
        assert loaded is not None
        assert loaded.events_processed == 201

    def test_file_backend_atomic_crash_and_fallback(self, tmp_path: Path) -> None:
        backend = FileSnapshotBackend(tmp_path, max_snapshots=2)
        eng = _sample_engine()
        s1 = backend.save(eng)
        eng.process_event(Event(1_700_000_500.0, "u_new", "/p/0", False))
        s2 = backend.save(eng)

        # Simulate crash between temp write and os.replace
        with (
            patch("probcount.storage.os.replace", side_effect=OSError("simulated crash")),
            pytest.raises(OSError, match="simulated crash"),
        ):
            backend.save(eng)

        # Previous valid snapshot s2 is intact; now corrupt s2 to test fallback to s1
        (tmp_path / s2).write_bytes(b"corrupted_data")
        loaded = backend.load()
        assert loaded is not None
        assert loaded.events_processed == 200
        assert s1 in backend.list_snapshots()
