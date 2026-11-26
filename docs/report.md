# Probabilistic Counting of Unique Visitors using HyperLogLog and Count-Min Sketch

**Version:** `1.0.0` | **Python:** `3.13.15` (`>= 3.11`)

---

## Abstract

Modern web-analytics pipelines must track millions of unique visitors and page-view frequencies across sliding time windows under strict memory and latency budgets. Exact hash-set and hash-table counters scale linearly ($O(n)$) with distinct visitors and URLs, quickly exhausting RAM. This project designs, implements from scratch, and evaluates **ProbCount**, an end-to-end streaming analytics engine combining **HyperLogLog (HLL)** for cardinality estimation, **Count-Min Sketch (CMS)** with Conservative Update for frequency estimation, a bounded min-heap **Top-K** tracker, multi-resolution time-bucketed windows (`60m → 1h → 1d`), a **FastAPI** service with atomic SHA-256 snapshots, and an interactive **Streamlit** dashboard. Empirical evaluation demonstrates **0.491%–0.623%** MAPE at $p=14$ (12 KB packed), **43.6%–52.7%** over-estimation reduction via Conservative Update, **12.0 KB** flat HLL memory vs **96.22 MB** exact sets at 5M events, and **7,738.9 events/sec** HTTP batch ingestion.

---

## 1. Introduction & Motivation

Counting distinct visitors (*F₀* / cardinality) and heavy-hitter page views (*F₁* / frequency) are foundational to web analytics. An exact `set[str]` storing 200,000 visitor IDs consumes tens of megabytes; across 168 hourly buckets and thousands of pages, exact counters easily exceed hundreds of megabytes. Probabilistic sketches trade a tiny, mathematically bounded approximation error for **constant $O(1)$ sub-linear space** and **lossless distributed merging**.

---

## 2. Background & Streaming Model

In the turnstile/cash-register streaming model, events $(t_i, v_i, p_i)$ arrive sequentially and must be processed in a single pass. Both HLL and CMS rely on uniform 64-bit hashing. We implement **MurmurHash3 x64-128** [5] in pure Python (`src/probcount/hashing.py`), cross-checked against the `mmh3` C reference across 10,000 random Hypothesis inputs.

---

## 3. HyperLogLog

Following Flajolet et al. [1], `HyperLogLog(p, seed)` (`src/probcount/hll.py`) partitions a 64-bit hash into a $p$-bit register index ($m = 2^p$) and a $(64-p)$-bit tail $w$, recording $\rho(w) = \text{clz}(w) + 1$. The harmonic-mean estimator $E = \alpha_m m^2 / \sum_{j=0}^{m-1} 2^{-M[j]}$ achieves theoretical standard error $\sigma \approx 1.04/\sqrt{m}$ (`0.812%` at $p=14$). Small-range bias ($E \le 2.5m, V > 0$) is corrected with Linear Counting $m \ln(m/V)$. Because we use 64-bit hashes, large-range 32-bit correction is unnecessary. Two sketches with matching $(p, \text{seed})$ merge losslessly via element-wise $\max$. We also implement `SparseHyperLogLog` [2] (`src/probcount/hll_sparse.py`) for low cardinalities.

---

## 4. Count-Min Sketch & Top-K

Following Cormode and Muthukrishnan [3], `CountMinSketch(width, depth)` (`src/probcount/cms.py`) maintains a $d \times w$ `uint64` matrix with $w = \lceil e/\varepsilon \rceil$ and $d = \lceil \ln(1/\delta) \rceil$, guaranteeing $\hat{f}(x) \ge f(x)$ and $\Pr[\hat{f}(x) - f(x) > \varepsilon N] \le \delta$. Enabling **Conservative Update** sets each counter to $\max(\text{table}[i, h_i(x)], \hat{f}(x) + c)$, cutting collision inflation. `TopKTracker` (`src/probcount/topk.py`) pairs CMS with a $k$-bounded min-heap (with periodic compaction) and is compared against Metwally et al.'s `SpaceSaving` [4].

---

## 5. System Design

- **Time Windows (`src/probcount/windows.py`)**: `TimeBucketedHLL` and `TimeBucketedCMS` align events to `minute`/`hour`/`day` buckets, support `60m → 1h → 1d` roll-up, prune expired buckets, and drop events past `watermark_seconds`.
- **Analytics Engine (`src/probcount/engine.py`)**: Tracks site HLL, CMS, Top-K, and per-page HLLs only for the top `hot_pages_limit` pages.
- **Storage & API (`src/probcount/storage.py`, `src/probcount/api.py`)**: Edge BLAKE2b salted hashing protects visitor privacy; `FileSnapshotBackend` writes `PCS1` containers atomically (`temp` + `fsync` + `os.replace`) with SHA-256 checksums.
- **Dashboard (`dashboard/`)**: Thin Streamlit multipage UI backed by `dashboard/data.py`.

---

## 6. Implementation Notes & Testing Strategy

- **Verification**: `ruff`, `mypy` (strict), Hypothesis property tests (200-case CMS never-undercount, HLL merge algebraic laws, 10,000-case `mmh3` hash parity), Streamlit `AppTest` smoke tests, and E2E API restart persistence tests.
- **Optimization**: Periodic heap compaction in `TopKTracker` reduced 500k-event tracker memory from 32.4 MB to 52 KB.

---

## 7. Experimental Evaluation

All benchmarks were executed via `python -m experiments.run_all --config experiments/config.yaml --quick` (see `docs/results.md`):

- **HLL Accuracy (`S1`)**: At $p=14$, MAPE is **0.491%** ($n=1{,}000$), **0.607%** ($n=10{,}000$), and **0.623%** ($n=50{,}000$), closely matching the theoretical **0.812%** SE.
- **Parameter Grid (`S2`)**: Packed HLL sizes range from **0.19 KB** ($p=8$, `7.091%` MAPE) to **12.0 KB** ($p=14$, `0.799%` MAPE) and **192.0 KB** ($p=18$, `0.095%` MAPE).
- **Conservative Update (`S3`)**: Reduces CMS mean over-estimation by **52.6%** ($s=0.8$), **51.4%** ($s=1.0$), **43.6%** ($s=1.2$), and **52.7%** ($s=1.5$).
- **Hash Ablation (`S4`)**: Weak `sum(bytes)` suffers **99.99%** error on sequential IDs vs **1.52%** for MurmurHash3.
- **Scalability (`S5`)**: At 5,000,000 events, HLL remains **12.0 KB** (`114,803.9 ev/s`) while `ExactUniqueCounter` grows to **96.22 MB**.
- **HTTP Throughput**: `scripts/load_test.py` achieved **7,738.9 events/sec** (`p50=990.95 ms`).

---

## 8. Limitations & Threats to Validity

1. **Inclusion-Exclusion Cancellation**: `hll_intersection` exhibits high error (`81.65%` at 1% overlap) when intersection is small relative to union.
2. **Cold-Page Exclusion**: Pages outside `hot_pages_limit` return `0.0` for per-page unique visitors.
3. **Pure-Python Hashing Speed**: Pure-Python MurmurHash3 is slower than C/Rust extensions (`~115k ev/s` batch HLL).

---

## 9. Future Work

- HyperLogLog++ empirical bias-correction tables for the $E \in [2.5m, 5m]$ transition.
- Count Sketch ($\pm 1$ sign hashes) for unbiased turnstile deletions.
- Cython/Rust SIMD acceleration for `hash64_many` and Redis `PFADD`/`PFCOUNT` backend integration.

---

## 10. Conclusion

ProbCount demonstrates that from-scratch HyperLogLog and Count-Min Sketch implementations can power a complete, durable, real-time web-analytics engine with sub-percent error and >95% memory savings at scale.

---

## References

[1] P. Flajolet, É. Fusy, O. Gandouet, and F. Meunier, "HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm," in *Proc. Discrete Mathematics and Theoretical Computer Science (DMTCS)*, 2007, pp. 127–146.

[2] S. Heule, M. Nunkesser, and A. Hall, "HyperLogLog in practice: algorithmic engineering of a state of the art cardinality estimation algorithm," in *Proc. 16th Int. Conf. Extending Database Technology (EDBT)*, 2013, pp. 683–692.

[3] G. Cormode and S. Muthukrishnan, "An improved data stream summary: the count-min sketch and its applications," *Journal of Algorithms*, vol. 55, no. 1, pp. 58–75, 2005.

[4] A. Metwally, D. Agrawal, and A. El Abbadi, "Efficient computation of frequent and top-k elements in data streams," in *Proc. 10th Int. Conf. Database Theory (ICDT)*, 2005, pp. 398–412.

[5] A. Appleby, "MurmurHash3," SMHasher, 2011.

---

## Appendix

- **Parameter Table**: `p=14` (12 KB packed, 0.812% SE); `cms_width=1024, cms_depth=4` (32 KB, $\varepsilon \approx 0.00265, \delta \approx 0.0183$).
- **API Reference**: See [docs/api.md](api.md).
- **Reproduction**: Run `make reproduce` (`python -m experiments.run_all --config experiments/config.yaml --quick && python scripts/verify_report_numbers.py`).
