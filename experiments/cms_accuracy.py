"""Phase 3 Experiments: Count-Min Sketch & Top-K Heavy Hitters.

Produces CSV results in ``experiments/results/`` and report-ready PNG
figures in ``docs/figures/``.

Usage:
    python experiments/cms_accuracy.py --quick --seed 42
    python experiments/cms_accuracy.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from probcount.cms import CountMinSketch  # noqa: E402
from probcount.exact import ExactFrequencyCounter  # noqa: E402
from probcount.metrics import precision_at_k, recall_at_k  # noqa: E402
from probcount.space_saving import SpaceSaving  # noqa: E402
from probcount.topk import TopKTracker  # noqa: E402

RESULTS_DIR = Path("experiments/results")
FIGURES_DIR = Path("docs/figures")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def _zipf_stream(n_events: int, s: float, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    ranks = rng.zipf(max(1.01, s), size=n_events) if s <= 1.0 else rng.zipf(s, size=n_events)
    if s < 1.0:
        # Finite Zipf over 2000 items for s < 1.0
        pool = np.arange(1, 2001)
        weights = 1.0 / np.power(pool, s)
        weights /= weights.sum()
        ranks = rng.choice(pool, size=n_events, p=weights)
    return [f"page_{int(r)}" for r in ranks]


# ---------------------------------------------------------------------------
# C1: Error vs Width at fixed Depth
# ---------------------------------------------------------------------------


def run_c1(seed: int, quick: bool) -> None:
    n_events = 50_000 if quick else 200_000
    items = _zipf_stream(n_events, s=1.2, seed=seed)
    truth = Counter(items)
    keys = list(truth.keys())
    true_vals = np.array([truth[k] for k in keys], dtype=np.float64)

    widths = [64, 128, 256, 512, 1024, 2048]
    depth = 4
    rows: list[dict[str, object]] = []

    for w in widths:
        cms = CountMinSketch(width=w, depth=depth, seed=seed)
        cms.add_many(items)
        ests = cms.estimate_many(keys).astype(np.float64)
        mean_abs = float(np.mean(ests - true_vals))
        bound = cms.error_bound()
        rows.append(
            {
                "width": w,
                "depth": depth,
                "mean_abs_err": round(mean_abs, 2),
                "theory_bound": round(bound, 2),
            }
        )

    _write_csv(
        RESULTS_DIR / "c1_error_vs_width.csv",
        ["width", "depth", "mean_abs_err", "theory_bound"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(widths, [r["mean_abs_err"] for r in rows], "o-", label="Observed mean error")
    ax.plot(widths, [r["theory_bound"] for r in rows], "r--", label="Theory bound (e/w)*N")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sketch Width (w)")
    ax.set_ylabel("Additive Error")
    ax.set_title(f"C1: Error vs. Width (depth={depth}, N={n_events:,})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "c1_error_vs_width.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# C2: Error vs Depth at fixed Width
# ---------------------------------------------------------------------------


def run_c2(seed: int, quick: bool) -> None:
    n_events = 50_000 if quick else 200_000
    items = _zipf_stream(n_events, s=1.2, seed=seed)
    truth = Counter(items)
    keys = list(truth.keys())
    true_vals = np.array([truth[k] for k in keys], dtype=np.float64)

    width = 256
    depths = [1, 2, 3, 4, 6, 8, 10]
    rows: list[dict[str, object]] = []

    for d in depths:
        cms = CountMinSketch(width=width, depth=d, seed=seed)
        cms.add_many(items)
        ests = cms.estimate_many(keys).astype(np.float64)
        mean_abs = float(np.mean(ests - true_vals))
        p95_abs = float(np.percentile(ests - true_vals, 95))
        rows.append(
            {
                "depth": d,
                "width": width,
                "mean_abs_err": round(mean_abs, 2),
                "p95_abs_err": round(p95_abs, 2),
                "confidence": round(cms.confidence(), 5),
            }
        )

    _write_csv(
        RESULTS_DIR / "c2_error_vs_depth.csv",
        ["depth", "width", "mean_abs_err", "p95_abs_err", "confidence"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(depths, [r["mean_abs_err"] for r in rows], "o-", label="Mean error")
    ax.plot(depths, [r["p95_abs_err"] for r in rows], "s--", label="95th percentile error")
    ax.set_xlabel("Sketch Depth (d)")
    ax.set_ylabel("Additive Error")
    ax.set_title(f"C2: Diminishing Returns with Depth (width={width}, N={n_events:,})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "c2_error_vs_depth.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# C3: Standard vs Conservative Update across Zipf exponents
# ---------------------------------------------------------------------------


def run_c3(seed: int, quick: bool) -> None:
    n_events = 30_000 if quick else 100_000
    exponents = [0.8, 1.0, 1.2, 1.5]
    rows: list[dict[str, object]] = []

    for s in exponents:
        items = _zipf_stream(n_events, s=s, seed=seed)
        truth = Counter(items)
        keys = list(truth.keys())
        true_vals = np.array([truth[k] for k in keys], dtype=np.float64)

        std = CountMinSketch(width=256, depth=4, seed=seed, conservative=False)
        cons = CountMinSketch(width=256, depth=4, seed=seed, conservative=True)
        std.add_many(items)
        cons.add_many(items)

        std_err = std.estimate_many(keys).astype(np.float64) - true_vals
        cons_err = cons.estimate_many(keys).astype(np.float64) - true_vals
        rows.append(
            {
                "zipf_s": s,
                "std_mean_err": round(float(np.mean(std_err)), 2),
                "cons_mean_err": round(float(np.mean(cons_err)), 2),
                "std_max_err": int(np.max(std_err)),
                "cons_max_err": int(np.max(cons_err)),
            }
        )

    _write_csv(
        RESULTS_DIR / "c3_conservative_vs_standard.csv",
        ["zipf_s", "std_mean_err", "cons_mean_err", "std_max_err", "cons_max_err"],
        rows,
    )

    x = np.arange(len(exponents))
    w_bar = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w_bar / 2, [r["std_mean_err"] for r in rows], w_bar, label="Standard CMS")
    ax.bar(x + w_bar / 2, [r["cons_mean_err"] for r in rows], w_bar, label="Conservative Update")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in exponents])
    ax.set_xlabel("Zipf Exponent (s)")
    ax.set_ylabel("Mean Over-estimation")
    ax.set_title("C3: Standard vs. Conservative Update across Skew")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "c3_conservative_vs_standard.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# C4: Heavy Hitters vs Long Tail
# ---------------------------------------------------------------------------


def run_c4(seed: int, quick: bool) -> None:
    n_events = 50_000 if quick else 200_000
    items = _zipf_stream(n_events, s=1.2, seed=seed)
    truth = Counter(items)
    ranked = truth.most_common()

    cms = CountMinSketch(width=512, depth=4, seed=seed, conservative=True)
    cms.add_many(items)

    rows: list[dict[str, object]] = []
    for idx, (item, true_cnt) in enumerate(ranked[:200], start=1):
        est = cms.estimate(item)
        rel_err = abs(est - true_cnt) / true_cnt
        bucket = "heavy_top20" if idx <= 20 else "long_tail"
        rows.append(
            {
                "rank": idx,
                "true_count": true_cnt,
                "est_count": est,
                "rel_err": round(rel_err, 4),
                "bucket": bucket,
            }
        )

    _write_csv(
        RESULTS_DIR / "c4_heavy_vs_tail.csv",
        ["rank", "true_count", "est_count", "rel_err", "bucket"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(
        [r["true_count"] for r in rows],
        [float(r["rel_err"]) * 100 for r in rows],  # type: ignore[arg-type]
        alpha=0.65,
        s=22,
        label="Items (ranks 1..200)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True Frequency")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("C4: CMS Relative Error — Heavy Hitters vs. Long Tail")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "c4_heavy_vs_tail.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# C5: Top-K Quality (CMS+Heap vs SpaceSaving)
# ---------------------------------------------------------------------------


def run_c5(seed: int, quick: bool) -> None:
    n_events = 30_000 if quick else 100_000
    items = _zipf_stream(n_events, s=1.25, seed=seed)
    truth = Counter(items)
    truth_ranked = [item for item, _ in truth.most_common()]

    widths = [64, 128, 256, 512]
    k = 10
    rows: list[dict[str, object]] = []

    ss = SpaceSaving(k=k * 2)
    for item in items:
        ss.add(item)
    ss_keys = [item for item, _ in ss.top_k(k)]
    ss_prec = precision_at_k(ss_keys, truth_ranked, k)
    ss_rec = recall_at_k(ss_keys, truth_ranked, k)

    for w in widths:
        tracker = TopKTracker(
            k=k * 2,
            sketch=CountMinSketch(width=w, depth=4, seed=seed, conservative=True),
        )
        for item in items:
            tracker.add(item)
        cms_keys = [item for item, _ in tracker.top(k)]
        rows.append(
            {
                "width": w,
                "k": k,
                "cms_precision": round(precision_at_k(cms_keys, truth_ranked, k), 4),
                "cms_recall": round(recall_at_k(cms_keys, truth_ranked, k), 4),
                "ss_precision": round(ss_prec, 4),
                "ss_recall": round(ss_rec, 4),
            }
        )

    _write_csv(
        RESULTS_DIR / "c5_topk_quality.csv",
        ["width", "k", "cms_precision", "cms_recall", "ss_precision", "ss_recall"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(widths, [r["cms_recall"] for r in rows], "o-", label="CMS + Min-Heap Recall@10")
    ax.axhline(ss_rec, color="g", linestyle="--", label=f"SpaceSaving Recall@10 ({ss_rec:.2f})")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("CMS Width (w)")
    ax.set_ylabel("Recall@10")
    ax.set_title("C5: Top-10 Heavy-Hitter Recall vs. Sketch Width")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "c5_topk_quality.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# C6: Memory vs Exact Counter + Throughput (Standard vs KM)
# ---------------------------------------------------------------------------


def run_c6(seed: int, quick: bool) -> None:
    distinct_sizes = (
        [1_000, 10_000, 50_000, 100_000] if quick else [1_000, 10_000, 100_000, 500_000]
    )
    cms_mem = CountMinSketch.from_error(epsilon=0.001, delta=0.01, seed=seed)
    cms_bytes = cms_mem.memory_bytes()

    mem_rows: list[dict[str, object]] = []
    for n in distinct_sizes:
        exact = ExactFrequencyCounter()
        for i in range(n):
            exact.add(f"u_{i:010x}")
        ex_bytes = exact.memory_bytes()
        mem_rows.append(
            {
                "n_distinct": n,
                "cms_bytes": cms_bytes,
                "exact_bytes": ex_bytes,
                "saved_pct": round((1.0 - cms_bytes / ex_bytes) * 100.0, 2),
            }
        )

    _write_csv(
        RESULTS_DIR / "c6_memory.csv",
        ["n_distinct", "cms_bytes", "exact_bytes", "saved_pct"],
        mem_rows,
    )

    # Throughput comparison: add() vs add_many() (standard vs Kirsch-Mitzenmacher)
    n_tp = 20_000 if quick else 100_000
    tp_items = [f"k_{i % 500}" for i in range(n_tp)]
    tp_rows: list[dict[str, object]] = []

    for label, km, batch in [
        ("add() std", False, False),
        ("add_many() std", False, True),
        ("add_many() KM", True, True),
    ]:
        s = CountMinSketch(width=1024, depth=5, seed=seed, kirsch_mitzenmacher=km)
        t0 = time.perf_counter()
        if batch:
            s.add_many(tp_items)
        else:
            for item in tp_items:
                s.add(item)
        dt = time.perf_counter() - t0
        tp_rows.append(
            {
                "mode": label,
                "n": n_tp,
                "seconds": round(dt, 4),
                "events_per_sec": round(n_tp / dt, 1),
            }
        )

    _write_csv(
        RESULTS_DIR / "c6_throughput.csv",
        ["mode", "n", "seconds", "events_per_sec"],
        tp_rows,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(distinct_sizes, [r["exact_bytes"] for r in mem_rows], "s-", label="Exact Counter")
    ax1.plot(distinct_sizes, [r["cms_bytes"] for r in mem_rows], "o-", label="CMS (eps=0.001)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Distinct Items")
    ax1.set_ylabel("Memory (bytes)")
    ax1.set_title("C6a: Memory Footprint")
    ax1.legend()

    modes = [str(r["mode"]) for r in tp_rows]
    eps_vals = [float(r["events_per_sec"]) for r in tp_rows]  # type: ignore[arg-type]
    bars = ax2.bar(modes, eps_vals, color=["#4C72B0", "#55A868", "#C44E52"])
    ax2.bar_label(bars, fmt="%.0f")
    ax2.set_ylabel("Events / sec")
    ax2.set_title("C6b: Throughput (Standard vs. KM)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "c6_memory_and_throughput.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 CMS & Top-K experiments.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--quick", action="store_true", help="Run fast subset")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _setup_style()

    t_start = time.perf_counter()
    for name, fn in [
        ("C1 (Error vs Width)", run_c1),
        ("C2 (Error vs Depth)", run_c2),
        ("C3 (Conservative vs Standard)", run_c3),
        ("C4 (Heavy vs Tail)", run_c4),
        ("C5 (Top-K Quality)", run_c5),
        ("C6 (Memory & Throughput)", run_c6),
    ]:
        t0 = time.perf_counter()
        fn(args.seed, args.quick)
        print(f"Completed {name} in {time.perf_counter() - t0:.2f}s")

    print(f"All CMS experiments finished in {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
