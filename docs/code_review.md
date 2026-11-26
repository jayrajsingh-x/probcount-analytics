# Code Review & Quality Hardening Summary

## Audit Scope
Reviewed all modules in `src/probcount/`, `dashboard/`, and `experiments/` for:
- Dead code, unused imports, and lingering `TODO`/`FIXME` comments
- Missing type annotations or public docstrings
- Memory/performance hazards under realistic workloads
- Security and dependency hygiene (`bandit`, `pip-audit`)

## Key Findings & Fixes Applied
1. **`TopKTracker` Heap Growth (`src/probcount/topk.py`)**:
   - *Finding*: Repeated updates to hot items pushed stale `(est, key)` entries onto `self._heap` without periodic compaction, causing `topk_bytes` to reach `32.4 MB` on 500k events.
   - *Fix*: Added periodic compaction when `len(self._heap) > 4 * self._k`, rebuilding the heap from `self._candidates` ($O(k)$). Reduced 500k-event `TopKTracker` memory to `52 KB`.
2. **API Default Window Range (`src/probcount/api.py`)**:
   - *Finding*: `resolve_window(start=None, end=None)` defaulted `start` to `0.0` (1970 epoch), causing `/stats/series` to iterate ~472,000 empty hourly buckets.
   - *Fix*: Defaulted `start` to the earliest stored bucket timestamp (or `now - 3600s` when empty).
3. **Lint & Strict Typing**:
   - Confirmed zero `print()` statements in `src/probcount/` library code (`logging` used throughout).
   - Verified `ruff check` and `mypy` pass with zero errors across `src/`, `dashboard/`, and Phase 7 `experiments/` modules.
4. **Security & Dependencies**:
   - Verified `pickle` in `ExactUniqueCounter` is strictly for local ground-truth test baselines (never exposed over HTTP); API snapshots use JSON + base64 + SHA-256 checksums (`PCS1`).
