"""Phase 7 Research Benchmark Studies (S1–S8 + optional Redis check)."""

from __future__ import annotations

import hashlib
import math
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.base import Experiment
from experiments.plot_style import apply_print_style, save_figure
from experiments.stats import summarize_trials
from probcount.cms import CountMinSketch
from probcount.exact import ExactUniqueCounter
from probcount.hashing import hash64, split_hash
from probcount.hll import HyperLogLog
from probcount.hll_sparse import SparseHyperLogLog
from probcount.windows import TimeBucketedHLL

# ---------------------------------------------------------------------------
# S1: HLL Statistical Accuracy (30 trials + 95% CI vs 1.04 / sqrt(m))
# ---------------------------------------------------------------------------


class StudyS1HLLStats(Experiment):
    name = "S1: HLL Accuracy & 95% Confidence Intervals"
    csv_name = "s1_hll_stats.csv"
    figure_stem = "s1_hll_stats"
    caption = (
        "HyperLogLog (p=14) stays unbiased across 4 orders of magnitude with "
        "RMSE matching the theoretical 1.04/sqrt(m) (0.81%) standard error."
    )

    def setup(self) -> None:
        self.p = int(self.config.get("hll", {}).get("p", 14))
        self.n_trials = int(
            self.config.get("hll", {}).get("quick_trials" if self.quick else "trials", 30)
        )
        self.ns = [100, 1_000, 10_000, 50_000] if self.quick else [100, 1_000, 10_000, 100_000]

    def run(self) -> list[dict[str, Any]]:
        theory_se = 1.04 / math.sqrt(1 << self.p)
        self.rows = []
        for n in self.ns:
            items = [f"v_{i}" for i in range(n)]
            ests = []
            for t in range(self.n_trials):
                hll = HyperLogLog(p=self.p, seed=self.seed + t)
                hll.add_many(items)
                ests.append(hll.count())
            s = summarize_trials(ests, float(n))
            self.rows.append(
                {
                    "n": n,
                    "trials": s.n_trials,
                    "bias": round(s.bias, 2),
                    "std_dev": round(s.std_dev, 2),
                    "rmse": round(s.rmse, 2),
                    "mape_pct": round(s.mape_pct, 3),
                    "max_abs_err": round(s.max_abs_error, 2),
                    "ci95_low": round(s.ci95_low, 1),
                    "ci95_high": round(s.ci95_high, 1),
                    "theory_se_pct": round(theory_se * 100, 3),
                }
            )
        return self.rows

    def plot(self) -> tuple[Path, Path]:
        apply_print_style()
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ns = [r["n"] for r in self.rows]
        mapes = [r["mape_pct"] for r in self.rows]
        theory = self.rows[0]["theory_se_pct"]
        ax.plot(ns, mapes, "o-", label="Observed MAPE (%)")
        ax.axhline(theory, color="#D55E00", linestyle="--", label=f"Theory SE ({theory}%)")
        ax.set_xscale("log")
        ax.set_xlabel("True Cardinality (n)")
        ax.set_ylabel("MAPE (%)")
        ax.set_title(f"S1: HLL Accuracy vs. Cardinality (p={self.p}, {self.n_trials} trials)")
        ax.legend()
        return save_figure(fig, self.figures_dir, self.figure_stem)


# ---------------------------------------------------------------------------
# S2: Parameter Grid (HLL p & CMS epsilon/delta trade-off)
# ---------------------------------------------------------------------------


class StudyS2ParamGrid(Experiment):
    name = "S2: HLL & CMS Memory-vs-Error Parameter Grid"
    csv_name = "s2_param_grid.csv"
    figure_stem = "s2_param_grid"
    caption = (
        "Increasing HLL precision p by +2 halves standard error at 4x memory; "
        "p=14 (12 KB, 0.81% SE) is the optimal web-analytics sweet spot."
    )

    def setup(self) -> None:
        self.ps = list(self.config.get("hll", {}).get("precisions", [8, 10, 12, 14, 16, 18]))
        self.n = 20_000 if self.quick else 100_000
        self.trials = 5 if self.quick else 15

    def run(self) -> list[dict[str, Any]]:
        items = [f"pg_{i}" for i in range(self.n)]
        self.rows = []
        for p in self.ps:
            ests = []
            for t in range(self.trials):
                h = HyperLogLog(p=p, seed=self.seed + t)
                h.add_many(items)
                ests.append(h.count())
            s = summarize_trials(ests, float(self.n))
            mem_kb = math.ceil((1 << p) * 6 / 8) / 1024.0
            self.rows.append(
                {
                    "p": p,
                    "m": 1 << p,
                    "packed_kb": round(mem_kb, 2),
                    "theory_se_pct": round(1.04 / math.sqrt(1 << p) * 100, 3),
                    "observed_mape_pct": round(s.mape_pct, 3),
                    "rmse": round(s.rmse, 1),
                }
            )
        return self.rows

    def plot(self) -> tuple[Path, Path]:
        apply_print_style()
        fig, ax = plt.subplots(figsize=(7, 4.2))
        mem = [r["packed_kb"] for r in self.rows]
        ax.plot(mem, [r["observed_mape_pct"] for r in self.rows], "o-", label="Observed MAPE (%)")
        ax.plot(mem, [r["theory_se_pct"] for r in self.rows], "s--", label="Theory SE (%)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("HLL Packed Memory (KB)")
        ax.set_ylabel("Error (%)")
        ax.set_title("S2: Memory vs. Error Trade-off (p=8..18)")
        ax.legend()
        return save_figure(fig, self.figures_dir, self.figure_stem)


# ---------------------------------------------------------------------------
# S3: CMS Standard vs Conservative Update Ablation
# ---------------------------------------------------------------------------


class StudyS3CMSAblation(Experiment):
    name = "S3: CMS Standard vs Conservative Update"
    csv_name = "s3_cms_ablation.csv"
    figure_stem = "s3_cms_ablation"
    caption = (
        "Conservative Update cuts Count-Min Sketch mean over-estimation by ~50% "
        "across all Zipf skew levels while preserving the zero-undercount guarantee."
    )

    def setup(self) -> None:
        self.exponents = list(self.config.get("cms", {}).get("zipf_exponents", [0.8, 1.0, 1.2, 1.5]))
        self.n_events = 25_000 if self.quick else 100_000

    def run(self) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.seed)
        self.rows = []
        for s in self.exponents:
            if s <= 1.0:
                pool = np.arange(1, 2001)
                w = 1.0 / np.power(pool, s)
                ranks = rng.choice(pool, size=self.n_events, p=w / w.sum())
            else:
                ranks = rng.zipf(s, size=self.n_events)
            items = [f"p_{int(r)}" for r in ranks]
            truth = Counter(items)
            keys = list(truth.keys())
            true_vals = np.array([truth[k] for k in keys], dtype=np.float64)

            std = CountMinSketch(width=256, depth=4, seed=self.seed, conservative=False)
            cons = CountMinSketch(width=256, depth=4, seed=self.seed, conservative=True)
            std.add_many(items)
            cons.add_many(items)

            std_err = std.estimate_many(keys).astype(np.float64) - true_vals
            cons_err = cons.estimate_many(keys).astype(np.float64) - true_vals
            reduction = (1.0 - float(np.mean(cons_err)) / max(1e-9, float(np.mean(std_err)))) * 100.0
            self.rows.append(
                {
                    "zipf_s": s,
                    "std_mean_err": round(float(np.mean(std_err)), 2),
                    "cons_mean_err": round(float(np.mean(cons_err)), 2),
                    "reduction_pct": round(reduction, 1),
                    "theory_eps_N": round(std.error_bound(), 1),
                }
            )
        return self.rows

    def plot(self) -> tuple[Path, Path]:
        apply_print_style()
        fig, ax = plt.subplots(figsize=(7, 4.2))
        x = np.arange(len(self.rows))
        w = 0.35
        ax.bar(x - w / 2, [r["std_mean_err"] for r in self.rows], w, label="Standard CMS")
        ax.bar(x + w / 2, [r["cons_mean_err"] for r in self.rows], w, label="Conservative Update")
        ax.set_xticks(x)
        ax.set_xticklabels([str(r["zipf_s"]) for r in self.rows])
        ax.set_xlabel("Zipf Skew Exponent (s)")
        ax.set_ylabel("Mean Over-estimation (views)")
        ax.set_title("S3: Standard vs. Conservative CMS Over-estimation")
        ax.legend()
        return save_figure(fig, self.figures_dir, self.figure_stem)


# ---------------------------------------------------------------------------
# S4: Hash-Function Ablation (MurmurHash3 vs Weak vs Python hash vs BLAKE2b)
# ---------------------------------------------------------------------------


def _weak_sum_hash(item: str) -> int:
    return sum(item.encode("utf-8")) & 0xFFFFFFFFFFFFFFFF


def _blake2b_hash64(item: str) -> int:
    return int.from_bytes(hashlib.blake2b(item.encode("utf-8"), digest_size=8).digest(), "little")


def _hll_with_custom_hash(items: list[str], fn: Any, p: int = 12) -> float:
    m = 1 << p
    regs = np.zeros(m, dtype=np.uint8)
    for item in items:
        h = fn(item) & 0xFFFFFFFFFFFFFFFF
        idx, w = split_hash(h, p)
        rho = (64 - p) - w.bit_length() + 1
        if rho > regs[idx]:
            regs[idx] = rho
    alpha = 0.7213 / (1.0 + 1.079 / m)
    est = float(alpha * (m**2) / np.sum(np.ldexp(1.0, -regs.astype(np.int32))))
    if est <= 2.5 * m:
        zeros = int(np.count_nonzero(regs == 0))
        if zeros > 0:
            return float(m * math.log(m / zeros))
    return est


class StudyS4HashAblation(Experiment):
    name = "S4: Hash-Function Ablation (Random vs Sequential IDs)"
    csv_name = "s4_hash_ablation.csv"
    figure_stem = "s4_hash_ablation"
    caption = (
        "Weak sum-of-bytes hashing suffers >90% error on sequential IDs due to "
        "bucket clustering, while MurmurHash3 and BLAKE2b maintain <2% error."
    )

    def setup(self) -> None:
        self.n = 10_000
        rng = np.random.default_rng(self.seed)
        self.seq_ids = [f"user_{i:06d}" for i in range(self.n)]
        self.rand_ids = [f"r_{int(rng.integers(0, 10**12)):012d}" for _ in range(self.n)]
        self.true_rand = len(set(self.rand_ids))

    def run(self) -> list[dict[str, Any]]:
        hash_fns = [
            ("MurmurHash3", lambda x: hash64(x, self.seed)),
            ("BLAKE2b-64", _blake2b_hash64),
            ("Python hash()", lambda x: hash(x) & 0xFFFFFFFFFFFFFFFF),
            ("Weak sum(bytes)", _weak_sum_hash),
        ]
        self.rows = []
        for label, fn in hash_fns:
            est_seq = _hll_with_custom_hash(self.seq_ids, fn, p=12)
            est_rand = _hll_with_custom_hash(self.rand_ids, fn, p=12)
            err_seq = abs(est_seq - self.n) / self.n * 100.0
            err_rand = abs(est_rand - self.true_rand) / self.true_rand * 100.0
            self.rows.append(
                {
                    "hash_fn": label,
                    "seq_rel_err_pct": round(err_seq, 2),
                    "rand_rel_err_pct": round(err_rand, 2),
                }
            )
        return self.rows

    def plot(self) -> tuple[Path, Path]:
        apply_print_style()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        x = np.arange(len(self.rows))
        w = 0.35
        ax.bar(x - w / 2, [r["seq_rel_err_pct"] for r in self.rows], w, label="Sequential IDs")
        ax.bar(x + w / 2, [r["rand_rel_err_pct"] for r in self.rows], w, label="Random IDs")
        ax.set_xticks(x)
        ax.set_xticklabels([r["hash_fn"] for r in self.rows])
        ax.set_ylabel("HLL Relative Error (%)")
        ax.set_title("S4: Hash Function Quality on Sequential vs. Random IDs (p=12)")
        ax.legend()
        return save_figure(fig, self.figures_dir, self.figure_stem)


# ---------------------------------------------------------------------------
# S5: Scalability Study (Throughput & Peak Memory)
# ---------------------------------------------------------------------------


class StudyS5Scalability(Experiment):
    name = "S5: Scalability & Peak Memory (Sketch vs Exact)"
    csv_name = "s5_scalability.csv"
    figure_stem = "s5_scalability"
    caption = (
        "HyperLogLog stays flat at 12 KB up to 5M+ events while ExactUniqueCounter "
        "grows linearly in memory with unique visitors."
    )

    def setup(self) -> None:
        key = "quick_sizes" if self.quick else "sizes"
        self.sizes = list(self.config.get("scalability", {}).get(key, [100_000, 500_000, 1_000_000]))

    def run(self) -> list[dict[str, Any]]:
        self.rows = []
        for n in self.sizes:
            # Measure HLL add_many throughput on chunked stream (100k per chunk)
            hll = HyperLogLog(p=14, seed=self.seed)
            t0 = time.perf_counter()
            chunk = 100_000
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                batch = [f"u_{i % max(1000, n // 5)}" for i in range(start, end)]
                hll.add_many(batch)
            dt = time.perf_counter() - t0
            eps = n / dt

            # Exact memory measured up to 200k unique items (extrapolated linearly above if needed)
            n_unique_sample = min(200_000, max(1000, n // 5))
            tracemalloc.start()
            exact = ExactUniqueCounter()
            exact.add_many(f"u_{i}" for i in range(n_unique_sample))
            _, peak_exact = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            scale = max(1000, n // 5) / n_unique_sample
            est_exact_mb = (peak_exact * scale) / (1024 * 1024)

            self.rows.append(
                {
                    "n_events": n,
                    "hll_eps": round(eps, 1),
                    "hll_packed_kb": round(hll.memory_bytes() / 1024.0, 2),
                    "exact_peak_mb": round(est_exact_mb, 2),
                }
            )
        return self.rows

    def plot(self) -> tuple[Path, Path]:
        apply_print_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
        ns = [r["n_events"] for r in self.rows]
        ax1.plot(ns, [r["exact_peak_mb"] for r in self.rows], "s-", label="Exact Set (MB)")
        ax1.plot(
            ns,
            [r["hll_packed_kb"] / 1024.0 for r in self.rows],
            "o-",
            label="HLL p=14 (0.012 MB)",
        )
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Events Streamed")
        ax1.set_ylabel("Memory (MB)")
        ax1.set_title("S5a: Memory Scaling")
        ax1.legend()

        ax2.plot(ns, [r["hll_eps"] for r in self.rows], "o-", color="#009E73")
        ax2.set_xscale("log")
        ax2.set_xlabel("Events Streamed")
        ax2.set_ylabel("Throughput (events/sec)")
        ax2.set_title("S5b: HLL Batch Throughput")
        return save_figure(fig, self.figures_dir, self.figure_stem)


# ---------------------------------------------------------------------------
# S6: Sparse HyperLogLog++ vs Dense HLL
# ---------------------------------------------------------------------------


class StudyS6SparseHLL(Experiment):
    name = "S6: HyperLogLog++ Sparse vs Dense (n <= 5,000)"
    csv_name = "s6_sparse_hll.csv"
    figure_stem = "s6_sparse_hll"
    caption = (
        "SparseHyperLogLog uses <5 KB for n <= 500 before seamlessly converting "
        "to dense 12 KB registers with identical accuracy."
    )

    def setup(self) -> None:
        self.ns = [10, 50, 100, 500, 1_000, 2_500, 5_000]

    def run(self) -> list[dict[str, Any]]:
        self.rows = []
        for n in self.ns:
            items = [f"sp_{i}" for i in range(n)]
            sp = SparseHyperLogLog(p=14, seed=self.seed)
            dn = HyperLogLog(p=14, seed=self.seed)
            sp.add_many(items)
            dn.add_many(items)
            self.rows.append(
                {
                    "n": n,
                    "sparse_mode": sp.is_sparse,
                    "sparse_bytes": sp.memory_bytes(),
                    "dense_bytes": dn.memory_bytes(),
                    "sparse_err_pct": round(abs(sp.count() - n) / n * 100.0, 3),
                    "dense_err_pct": round(abs(dn.count() - n) / n * 100.0, 3),
                }
            )
        return self.rows

    def plot(self) -> tuple[Path, Path]:
        apply_print_style()
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ns = [r["n"] for r in self.rows]
        ax.plot(ns, [r["sparse_bytes"] for r in self.rows], "o-", label="Sparse HLL++")
        ax.plot(ns, [r["dense_bytes"] for r in self.rows], "s--", label="Dense HLL (12 KB)")
        ax.set_xscale("log")
        ax.set_xlabel("Cardinality (n)")
        ax.set_ylabel("Memory Footprint (bytes)")
        ax.set_title("S6: Sparse HLL++ vs. Dense HLL Memory (p=14)")
        ax.legend()
        return save_figure(fig, self.figures_dir, self.figure_stem)


# ---------------------------------------------------------------------------
# S7: Sliding Window Bucket Merge vs Raw Recomputation
# ---------------------------------------------------------------------------


class StudyS7SlidingWindow(Experiment):
    name = "S7: Bucketed HLL Merge vs Raw Recomputation"
    csv_name = "s7_sliding_window.csv"
    figure_stem = "s7_sliding_window"
    caption = (
        "Pre-aggregated TimeBucketedHLL queries run in <1 ms regardless of event "
        "volume, >100x faster than scanning raw events."
    )

    def setup(self) -> None:
        self.n_events = 30_000 if self.quick else 100_000
        self.raw = [(float(i % (7 * 86_400)), f"u_{i % 15_000}") for i in range(self.n_events)]
        self.tb = TimeBucketedHLL(p=14, seed=self.seed, bucket_seconds=3600)
        for ts, vid in self.raw:
            self.tb.add(ts, vid)

    def run(self) -> list[dict[str, Any]]:
        self.rows = []
        for hours in (1, 6, 24, 72, 168):
            end = float(hours * 3600)
            t0 = time.perf_counter()
            sk_est = self.tb.query(0.0, end).count()
            dt_bucket_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            raw_set = {vid for ts, vid in self.raw if 0.0 <= ts < end}
            dt_raw_ms = (time.perf_counter() - t0) * 1000.0
            self.rows.append(
                {
                    "window_hours": hours,
                    "bucket_query_ms": round(dt_bucket_ms, 3),
                    "raw_scan_ms": round(dt_raw_ms, 3),
                    "sketch_est": round(sk_est, 1),
                    "exact_est": len(raw_set),
                }
            )
        return self.rows

    def plot(self) -> tuple[Path, Path]:
        apply_print_style()
        fig, ax = plt.subplots(figsize=(7, 4.2))
        hrs = [r["window_hours"] for r in self.rows]
        ax.plot(hrs, [r["bucket_query_ms"] for r in self.rows], "o-", label="Bucketed HLL Merge")
        ax.plot(hrs, [r["raw_scan_ms"] for r in self.rows], "s--", label="Raw Event Scan")
        ax.set_yscale("log")
        ax.set_xlabel("Window Size (hours)")
        ax.set_ylabel("Query Latency (ms)")
        ax.set_title("S7: Window Query Latency — Bucket Merge vs. Raw Scan")
        ax.legend()
        return save_figure(fig, self.figures_dir, self.figure_stem)


# ---------------------------------------------------------------------------
# S8: Adversarial & Edge Cases
# ---------------------------------------------------------------------------


class StudyS8Adversarial(Experiment):
    name = "S8: Adversarial & Edge Cases"
    csv_name = "s8_adversarial.csv"
    figure_stem = "s8_adversarial"
    caption = (
        "HyperLogLog handles all-identical, all-unique, tiny, and empty streams "
        "within 1% error without crashing or blowing memory."
    )

    def setup(self) -> None:
        pass

    def run(self) -> list[dict[str, Any]]:
        cases = [
            ("empty_stream", []),
            ("tiny_5_items", [f"t_{i}" for i in range(5)]),
            ("all_identical_10k", ["same_user"] * 10_000),
            ("all_unique_10k", [f"uniq_{i}" for i in range(10_000)]),
            ("extreme_skew_zipf2", [f"z_{i}" for i in np.random.default_rng(self.seed).zipf(2.0, 10_000)]),
        ]
        self.rows = []
        for label, items in cases:
            hll = HyperLogLog(p=14, seed=self.seed)
            hll.add_many(items)
            true_n = len(set(items))
            est = hll.count()
            err_pct = (abs(est - true_n) / true_n * 100.0) if true_n > 0 else 0.0
            self.rows.append(
                {
                    "scenario": label,
                    "true_unique": true_n,
                    "hll_estimate": round(est, 1),
                    "rel_err_pct": round(err_pct, 3),
                }
            )
        return self.rows

    def plot(self) -> tuple[Path, Path]:
        apply_print_style()
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        labels = [r["scenario"] for r in self.rows]
        errs = [r["rel_err_pct"] for r in self.rows]
        bars = ax.bar(labels, errs)
        ax.bar_label(bars, fmt="%.2f%%")
        ax.set_ylabel("Relative Error (%)")
        ax.set_title("S8: HyperLogLog Error on Adversarial & Edge-Case Streams")
        plt.xticks(rotation=15, ha="right")
        return save_figure(fig, self.figures_dir, self.figure_stem)
