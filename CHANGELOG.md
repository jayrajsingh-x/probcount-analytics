# Changelog

All notable changes to **ProbCount** are documented in this file.

## [1.0.0] — 2026-11-25 (Phases 1–8 Complete)

### Phase 1 — Foundation
- `src`-layout package `probcount` with `pyproject.toml` and GitHub Actions CI.
- Pure-Python MurmurHash3 x64-128 (`hash64`, `hash64_many`, `split_hash`, `derive_seeds`), cross-checked against `mmh3`.
- Abstract interfaces (`CardinalityEstimator`, `FrequencyEstimator`) and exact baselines (`ExactUniqueCounter`, `ExactFrequencyCounter`, `deep_size_bytes`).
- Streaming synthetic `TrafficGenerator` with Zipf skew, diurnal curve, traffic spikes, and JSONL/CSV CLI.

### Phase 2 — HyperLogLog
- `HyperLogLog(p, seed)` with vectorised `add_many`, Linear Counting small-range correction, `merge`, `union`, and 6-bit packed binary serialization (`to_bytes`/`from_bytes`).
- Experiments E1–E6 and `docs/hll_theory.md`.

### Phase 3 — Count-Min Sketch & Top-K
- `CountMinSketch` with `from_error(epsilon, delta)`, Conservative Update, Kirsch-Mitzenmacher mode, `merge`, and binary serialization.
- `TopKTracker` (bounded min-heap with lazy deletion), `SpaceSaving` baseline, and `metrics.py`.
- Experiments C1–C6 and `docs/cms_theory.md`.

### Phase 4 — Analytics Engine & Time Windows
- `TimeBucketedHLL` and `TimeBucketedCMS` with lazy buckets, retention pruning, `60m -> 1h -> 1d` roll-up, and late-event watermark.
- `AnalyticsEngine`, `ExactAnalyticsEngine`, `setops.py`, `distributed.py`, and `pipeline.py`.
- Experiments A1–A6 and `docs/engine_design.md`.

### Phase 5 — FastAPI Service & Durable Storage
- `StorageBackend`, `InMemoryBackend`, `FileSnapshotBackend` (`PCS1` binary container with SHA-256 checksum, atomic write, and corrupted-snapshot fallback).
- FastAPI service (`api.py`) with Pydantic v2, `pydantic-settings`, edge BLAKE2b hashing, `RLock` concurrency, `X-API-Key` auth, `ProbCountClient`, and async `scripts/load_test.py`.

### Phase 6 — Streamlit Dashboard
- Multipage Streamlit UI (`dashboard/app.py` + 7 pages), pure-Python `dashboard/data.py`, `AppTest` smoke suite, and `scripts/run_demo.py`.

### Phase 7 — Research Benchmark Suite
- Unified `python -m experiments.run_all` runner, `stats.py`, `plot_style.py`, `env_info.py`, `datasets.py`, and `SparseHyperLogLog`.
- Studies S1–S8, `docs/results.md`, and `docs/findings.md`.

### Phase 8 — Hardening, Packaging & Submission
- End-to-end integration test, `bandit` & `pip-audit` security checks, `Dockerfile` + `docker-compose.yml`, `requirements-lock.txt`, `docs/report.md`, `scripts/verify_report_numbers.py`, `docs/slides.md`, `docs/demo_script.md`, `docs/viva_qa.md`, `docs/glossary.md`, and `docs/SUBMISSION_CHECKLIST.md`.
