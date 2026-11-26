# Viva / Examiner Q&A (42 Questions)

## A. Theory (1–12)
1. **Why does HyperLogLog use the harmonic mean instead of the arithmetic mean?** Arithmetic mean is dominated by single lucky outliers (e.g., one hash with 30 leading zeros); harmonic mean dampens large outliers.
2. **What is stochastic averaging?** Splitting the stream into $m = 2^p$ independent buckets so variance drops by $1/\sqrt{m}$ without hashing each item $m$ times.
3. **Why is the standard error $1.04/\sqrt{m}$?** Derived from Flajolet et al.'s Mellin-transform asymptotic analysis of the harmonic-mean estimator variance.
4. **Why do we apply Linear Counting when $E \le 2.5m$?** When many registers are still zero ($V > 0$), $2^{-0} = 1$ biases the harmonic mean upward; Linear Counting $m\ln(m/V)$ is exact in this regime.
5. **Why is large-range correction omitted?** 32-bit HLL saturates near $2^{32}/30$; our 64-bit hash only saturates near $2^{64}$, far beyond web-analytics cardinalities.
6. **How many bits does each HLL register need?** For 64-bit hashes and $p \ge 4$, $\rho \le 65 - 4 = 61 < 64$, fitting in **6 bits** (12 KB at $p=14$).
7. **Why does Count-Min Sketch take the minimum across rows?** Non-negative updates only add non-negative collision noise, so every row is an upper bound; the minimum is the tightest upper bound.
8. **State the CMS $(\varepsilon, \delta)$ guarantee.** With $w = \lceil e/\varepsilon \rceil$ and $d = \lceil \ln(1/\delta) \rceil$, $\Pr[\hat{f}(x) - f(x) > \varepsilon N] \le \delta$.
9. **Why does Markov's inequality apply to CMS?** Collision noise in each row is non-negative with expectation $\le N/w$.
10. **What is Conservative Update?** Setting each row counter to $\max(\text{counter}, \hat{f}(x) + c)$ instead of adding $c$ blindly, cutting collision inflation by ~50%.
11. **Why do weak hashes break HLL?** Sequential IDs (`user_000001`) cluster into a few buckets and biased bit patterns, causing 99.99% error (Study S4).
12. **What is the Kirsch-Mitzenmacher optimization?** Generating $d$ row hashes from just two base hashes $(h_1 + i \cdot h_2) \bmod w$, giving 4.7× faster batch updates.

## B. Implementation (13–24)
13. **Why implement MurmurHash3 from scratch?** To avoid opaque black-box dependencies and verify bit-level correctness against `mmh3` on 10,000 Hypothesis inputs.
14. **How does `add_many` avoid `float64` precision loss?** It computes exact integer `bit_length` in `uint64` via 6 vectorised bit-shift steps rather than `np.log2()`.
15. **Why store HLL registers as `uint8` in RAM but 6-bit packed on disk?** `uint8` enables fast NumPy indexing (`ram_bytes=16 KB`), while 6-bit packing saves 25% on wire/disk (`memory_bytes=12 KB`).
16. **How does `TopKTracker` stay $O(k)$ memory?** It uses a bounded dict + min-heap with lazy deletion and periodic heap compaction when `len(heap) > 4*k`.
17. **What happens if `count < 0` in CMS?** Raises `ValueError` to prevent `uint64` underflow and preserve the non-negative upper-bound invariant.
18. **Why is Conservative Update order-dependent?** Each update depends on the current estimate $\hat{f}(x)$ at arrival time.
19. **How does `SparseHyperLogLog` work?** Stores `(index, rho)` pairs in a dict below `sparse_threshold`, then converts to dense `HyperLogLog` with identical registers.
20. **How are snapshots made crash-safe?** Written to a temp file, flushed, `fsync`'d, atomically renamed via `os.replace`, and verified with a SHA-256 trailer (`PCS1`).
21. **What happens if the newest snapshot is corrupted?** `FileSnapshotBackend.load()` logs a warning and falls back to the previous valid snapshot.
22. **Why `threading.RLock` instead of an `asyncio.Queue`?** In-memory NumPy updates take microseconds; an RLock avoids queue-scheduling latency while keeping concurrent batches lossless.
23. **How are `TimeBucketedHLL` roll-ups lossless?** HLL merge is element-wise `max`, which is associative, commutative, and idempotent (`60m → 1h → 1d` gives identical registers).
24. **How are late events handled?** Events older than `max_ts - watermark_seconds` are rejected and counted in `dropped_late`.

## C. Evaluation (25–33)
25. **How is memory measured fairly?** `deep_size_bytes()` recursively sums `sys.getsizeof()` with an `id`-based seen set so shared Python strings aren't double-counted.
26. **What Zipf skew was used?** Default $s = 1.1$ for pages, plus ablations across $s \in [0.8, 1.5]$.
27. **How were statistical CIs computed?** Across 10–30 independent seeded trials, reporting bias, std dev, RMSE, MAPE, max error, and 95% CI ($\bar{x} \pm 1.96\sigma/\sqrt{n}$).
28. **Does window merging accumulate error?** No — Study A2 shows error stays flat (~0.1%–0.6%) from 1 hour to 14 days.
29. **When does inclusion-exclusion fail?** When set overlap is `< 5%` of the union (`81.65%` error at 1% overlap in Study A3).
30. **How does `TopKTracker` compare to `SpaceSaving`?** At equal small slot counts, `TopKTracker` with CMS retains higher recall (`1.0` vs `0.4` at $k=10$ in C5) because CMS remembers tail counts.
31. **What HTTP throughput did the API achieve?** `7,738.9 events/sec` on localhost (`batches=20, batch_size=2000, concurrency=4`).
32. **How are report numbers kept in sync with CSVs?** `scripts/verify_report_numbers.py` checks quoted metrics in `docs/report.md` against `experiments/results/*.csv`.
33. **How is reproducibility guaranteed?** Fixed seeds, `environment.json`, and a unit test verifying byte-identical CSV output across runs.

## D. Design Decisions (34–42)
34. **Why bound per-page HLLs to `hot_pages_limit`?** Tracking an HLL per URL on sites with millions of URLs would make memory $O(P \cdot 2^p)$; bounding to top-$N$ hot pages keeps memory $O(1)$.
35. **What does `unique_visitors_page` return for cold pages?** Documented sentinel `0.0`.
36. **How is visitor privacy protected?** Raw `visitor_id` is salted-hashed with `BLAKE2b` at the API edge and never stored; sketches are non-invertible.
37. **Why keep `probcount` runtime dependencies to `numpy` only?** Keeps the core algorithmic library lightweight and embeddable without FastAPI/Streamlit bloat.
38. **Why separate `dashboard/data.py` from Streamlit pages?** Allows unit-testing HTTP fetching and error handling with mocked `httpx` without needing a browser.
39. **Why use `AppTest` in addition to unit tests?** Verifies all 8 Streamlit pages render end-to-end with zero runtime exceptions.
40. **Why support both `hash`, `round_robin`, and `random_dup` in distributed simulation?** Proves HLL `max`-merge is invariant to duplicate delivery (`random_dup`), whereas CMS `sum`-merge requires disjoint delivery.
41. **Why not implement Count Sketch deletions?** Out of scope for append-only web analytics; listed in Future Work.
42. **How would you scale this to 1M+ events/sec in production?** Replace pure-Python MurmurHash3 with a C/Rust extension, shard workers by `visitor_id`, and merge HLLs via `/sketch/hll/merge`.
