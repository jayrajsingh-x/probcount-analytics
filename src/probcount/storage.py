"""Snapshot persistence backends for :class:`~probcount.engine.AnalyticsEngine`.

Provides a :class:`StorageBackend` interface with :class:`InMemoryBackend`
and :class:`FileSnapshotBackend`.  Snapshots use a versioned binary
container (``PCS1``) with an outer header, serialized HLL/CMS sketches,
and a SHA-256 integrity checksum.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import struct
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any

from probcount.cms import CountMinSketch
from probcount.engine import AnalyticsEngine, EngineConfig
from probcount.hll import HyperLogLog
from probcount.windows import TimeBucketedCMS, TimeBucketedHLL

logger = logging.getLogger(__name__)

_MAGIC = b"PCS1"
_VERSION = 1
_HEADER_STRUCT = struct.Struct("<4sBdI")  # magic(4), version(1), timestamp(8), payload_len(4)
_CHECKSUM_LEN = 32  # SHA-256


def _serialize_tb_hll(tb: TimeBucketedHLL) -> dict[str, Any]:
    return {
        "p": tb._p,
        "seed": tb._seed,
        "bucket_seconds": tb._bucket_seconds,
        "retention_seconds": tb._retention_seconds,
        "watermark_seconds": tb._watermark_seconds,
        "enable_rollup": tb._enable_rollup,
        "max_ts": tb._max_ts,
        "dropped_late": tb._dropped_late,
        "buckets": {
            str(k): base64.b64encode(v.to_bytes()).decode("ascii")
            for k, v in tb._buckets.items()
        },
        "hour_buckets": {
            str(k): base64.b64encode(v.to_bytes()).decode("ascii")
            for k, v in tb._hour_buckets.items()
        },
        "day_buckets": {
            str(k): base64.b64encode(v.to_bytes()).decode("ascii")
            for k, v in tb._day_buckets.items()
        },
    }


def _deserialize_tb_hll(data: dict[str, Any]) -> TimeBucketedHLL:
    tb = TimeBucketedHLL(
        p=int(data["p"]),
        seed=int(data["seed"]),
        bucket_seconds=int(data["bucket_seconds"]),
        retention_seconds=float(data["retention_seconds"]),
        watermark_seconds=float(data["watermark_seconds"]),
        enable_rollup=bool(data["enable_rollup"]),
    )
    tb._max_ts = float(data["max_ts"]) if data["max_ts"] is not None else None
    tb._dropped_late = int(data["dropped_late"])
    for k, b64 in data["buckets"].items():
        tb._buckets[int(k)] = HyperLogLog.from_bytes(base64.b64decode(b64))
    for k, b64 in data["hour_buckets"].items():
        tb._hour_buckets[int(k)] = HyperLogLog.from_bytes(base64.b64decode(b64))
    for k, b64 in data["day_buckets"].items():
        tb._day_buckets[int(k)] = HyperLogLog.from_bytes(base64.b64decode(b64))
    return tb


def _serialize_tb_cms(tb: TimeBucketedCMS) -> dict[str, Any]:
    return {
        "width": tb._width,
        "depth": tb._depth,
        "seed": tb._seed,
        "conservative": tb._conservative,
        "km": tb._km,
        "bucket_seconds": tb._bucket_seconds,
        "retention_seconds": tb._retention_seconds,
        "watermark_seconds": tb._watermark_seconds,
        "enable_rollup": tb._enable_rollup,
        "max_ts": tb._max_ts,
        "dropped_late": tb._dropped_late,
        "buckets": {
            str(k): base64.b64encode(v.to_bytes()).decode("ascii")
            for k, v in tb._buckets.items()
        },
        "hour_buckets": {
            str(k): base64.b64encode(v.to_bytes()).decode("ascii")
            for k, v in tb._hour_buckets.items()
        },
        "day_buckets": {
            str(k): base64.b64encode(v.to_bytes()).decode("ascii")
            for k, v in tb._day_buckets.items()
        },
    }


def _deserialize_tb_cms(data: dict[str, Any]) -> TimeBucketedCMS:
    tb = TimeBucketedCMS(
        width=int(data["width"]),
        depth=int(data["depth"]),
        seed=int(data["seed"]),
        conservative=bool(data["conservative"]),
        kirsch_mitzenmacher=bool(data["km"]),
        bucket_seconds=int(data["bucket_seconds"]),
        retention_seconds=float(data["retention_seconds"]),
        watermark_seconds=float(data["watermark_seconds"]),
        enable_rollup=bool(data["enable_rollup"]),
    )
    tb._max_ts = float(data["max_ts"]) if data["max_ts"] is not None else None
    tb._dropped_late = int(data["dropped_late"])
    for k, b64 in data["buckets"].items():
        tb._buckets[int(k)] = CountMinSketch.from_bytes(base64.b64decode(b64))
    for k, b64 in data["hour_buckets"].items():
        tb._hour_buckets[int(k)] = CountMinSketch.from_bytes(base64.b64decode(b64))
    for k, b64 in data["day_buckets"].items():
        tb._day_buckets[int(k)] = CountMinSketch.from_bytes(base64.b64decode(b64))
    return tb


def serialize_engine(engine: AnalyticsEngine, timestamp: float | None = None) -> bytes:
    """Serialize *engine* into a checksummed ``PCS1`` binary snapshot."""
    ts = time.time() if timestamp is None else timestamp
    state = {
        "config": asdict(engine.config),
        "events_processed": engine._events_processed,
        "bots_ignored": engine._bots_ignored,
        "site_hll": _serialize_tb_hll(engine._site_hll),
        "page_cms": _serialize_tb_cms(engine._page_cms),
        "global_cms": base64.b64encode(engine._global_cms.to_bytes()).decode("ascii"),
        "topk_candidates": engine._topk._candidates,
        "hot_page_hlls": {
            page: _serialize_tb_hll(tb) for page, tb in engine._hot_page_hlls.items()
        },
    }
    payload = json.dumps(state, separators=(",", ":")).encode("utf-8")
    header = _HEADER_STRUCT.pack(_MAGIC, _VERSION, ts, len(payload))
    body = header + payload
    checksum = hashlib.sha256(body).digest()
    return body + checksum


def deserialize_engine(data: bytes) -> AnalyticsEngine:
    """Validate and deserialize a ``PCS1`` binary snapshot into an :class:`AnalyticsEngine`."""
    min_len = _HEADER_STRUCT.size + _CHECKSUM_LEN
    if len(data) < min_len:
        raise ValueError(f"Snapshot too short ({len(data)} bytes)")

    body = data[:-_CHECKSUM_LEN]
    expected_sum = data[-_CHECKSUM_LEN:]
    actual_sum = hashlib.sha256(body).digest()
    if actual_sum != expected_sum:
        raise ValueError("Snapshot SHA-256 checksum mismatch")

    magic, version, _, payload_len = _HEADER_STRUCT.unpack_from(body, 0)
    if magic != _MAGIC:
        raise ValueError(f"Invalid snapshot magic: {magic!r}")
    if version != _VERSION:
        raise ValueError(f"Unsupported snapshot version: {version}")

    payload = body[_HEADER_STRUCT.size :]
    if len(payload) != payload_len:
        raise ValueError(f"Snapshot payload length mismatch: {len(payload)} != {payload_len}")

    state = json.loads(payload.decode("utf-8"))
    cfg = EngineConfig(**state["config"])
    engine = AnalyticsEngine(cfg)
    engine._events_processed = int(state["events_processed"])
    engine._bots_ignored = int(state["bots_ignored"])
    engine._site_hll = _deserialize_tb_hll(state["site_hll"])
    engine._page_cms = _deserialize_tb_cms(state["page_cms"])
    engine._global_cms = CountMinSketch.from_bytes(base64.b64decode(state["global_cms"]))
    engine._topk._sketch = engine._global_cms
    engine._topk._candidates = {str(k): int(v) for k, v in state["topk_candidates"].items()}
    engine._topk._heap = [(v, k) for k, v in engine._topk._candidates.items()]
    engine._hot_page_hlls = {
        str(page): _deserialize_tb_hll(tb_data)
        for page, tb_data in state["hot_page_hlls"].items()
    }
    return engine


class StorageBackend(ABC):
    """Abstract snapshot persistence interface."""

    @abstractmethod
    def save(self, engine: AnalyticsEngine) -> str:
        """Persist *engine* and return a snapshot identifier."""

    @abstractmethod
    def load(self) -> AnalyticsEngine | None:
        """Load the latest valid snapshot, or ``None`` if none exist."""

    @abstractmethod
    def list_snapshots(self) -> list[str]:
        """Return snapshot identifiers ordered newest-first."""


class InMemoryBackend(StorageBackend):
    """In-memory snapshot store for testing and ephemeral runs."""

    def __init__(self, max_snapshots: int = 5) -> None:
        self._max_snapshots = max_snapshots
        self._snapshots: list[tuple[str, bytes]] = []

    def save(self, engine: AnalyticsEngine) -> str:
        snap_id = f"mem_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        self._snapshots.append((snap_id, serialize_engine(engine)))
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots :]
        return snap_id

    def load(self) -> AnalyticsEngine | None:
        for snap_id, raw in reversed(self._snapshots):
            try:
                return deserialize_engine(raw)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                logger.warning("Skipping corrupted snapshot %s: %s", snap_id, exc)
        return None

    def list_snapshots(self) -> list[str]:
        return [snap_id for snap_id, _ in reversed(self._snapshots)]


class FileSnapshotBackend(StorageBackend):
    """Atomic filesystem snapshot backend with checksum verification and fallback."""

    def __init__(self, directory: str | Path, max_snapshots: int = 5) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_snapshots = max(1, max_snapshots)

    def save(self, engine: AnalyticsEngine) -> str:
        self._dir.mkdir(parents=True, exist_ok=True)
        ts_ns = time.time_ns()
        snap_name = f"snapshot_{ts_ns}_{uuid.uuid4().hex[:6]}.snap"
        final_path = self._dir / snap_name
        tmp_path = self._dir / f".tmp_{snap_name}"

        raw = serialize_engine(engine)
        with open(tmp_path, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)

        # Prune older snapshots beyond max_snapshots
        snaps = self._sorted_paths()
        for old in snaps[self._max_snapshots :]:
            old.unlink(missing_ok=True)

        return snap_name

    def _sorted_paths(self) -> list[Path]:
        if not self._dir.exists():
            return []
        paths = [p for p in self._dir.glob("snapshot_*.snap") if p.is_file()]
        paths.sort(key=lambda p: p.name, reverse=True)
        return paths

    def load(self) -> AnalyticsEngine | None:
        for path in self._sorted_paths():
            try:
                return deserialize_engine(path.read_bytes())
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                logger.warning("Corrupted snapshot %s, trying previous: %s", path.name, exc)
        return None

    def list_snapshots(self) -> list[str]:
        return [p.name for p in self._sorted_paths()]
