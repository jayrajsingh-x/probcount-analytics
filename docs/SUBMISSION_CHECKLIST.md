# Submission Checklist (v1.0.0)

- [x] **Core Library (`src/probcount/`)**: Pure-Python MurmurHash3, `HyperLogLog`, `SparseHyperLogLog`, `CountMinSketch`, `TopKTracker`, `SpaceSaving`, `TimeBucketedHLL`/`CMS`, `AnalyticsEngine`, `StorageBackend`, FastAPI `api.py`, `ProbCountClient`
- [x] **Tests**: Unit, Hypothesis property, `AppTest` dashboard smoke, and E2E integration tests passing with ≥ 90% overall coverage and ≥ 95% on `hll.py`, `cms.py`, `topk.py`, `windows.py`
- [x] **Lint & Typecheck**: `ruff` and `mypy` pass cleanly
- [x] **Security & Dependency Audit**: `bandit` and `pip-audit` executed with zero unresolved high-severity findings
- [x] **Reproducibility**: `experiments/results/*.csv`, `environment.json`, `docs/results.md`, and `scripts/verify_report_numbers.py` passing
- [x] **Academic Report**: `docs/report.md` (IEEE references, verified numbers)
- [x] **Presentation & Viva**: `docs/slides.md`, `docs/demo_script.md`, `docs/viva_qa.md`, `docs/glossary.md`
- [x] **Packaging & Docs**: `Dockerfile`, `docker-compose.yml`, `requirements-lock.txt`, `README.md`, `CHANGELOG.md`, `LICENSE`
