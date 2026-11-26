"""Phase 2 Experiments: HyperLogLog Accuracy, Memory, Throughput, Merge & Ablation.

Produces CSV results in ``experiments/results/`` and report-ready PNG
figures in ``docs/figures/``.

Usage:
    python experiments/hll_accuracy.py --quick --seed 42
    python experiments/hll_accuracy.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from probcount.config import TrafficConfig  # noqa: E402
from probcount.exact import ExactUniqueCounter  # noqa: E402
from probcount.generator import TrafficGenerator  # noqa: E402
from probcount.hll import HyperLogLog  # noqa: E402

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


# ---------------------------------------------------------------------------
# E1: Error vs Cardinality
# ---------------------------------------------------------------------------


def run_e1(seed: int, quick: bool) -> None:
    p = 14
    n_trials = 5 if quick else 30
    if quick:
        cardinalities = [10, 50, 200, 1_000, 5_000, 25_000, 100_000]
    else:
        cardinalities = [int(x) for x in np.geomspace(10, 10_000_000, 13)]

    se_theory = 1.04 / math.sqrt(1 << p)
    rows: list[dict[str, object]] = []

    for n in cardinalities:
        items = [f"v_{i}" for i in range(n)]
        errs = []
        for t in range(n_trials):
            hll = HyperLogLog(p=p, seed=seed + t)
            hll.add_many(items)
            errs.append(abs(hll.count() - n) / n)
        arr = np.array(errs)
        mean_err = float(arr.mean())
        p95_low, p95_high = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        rows.append(
            {
                "n": n,
                "mean_rel_err": round(mean_err, 6),
                "p2_5": round(p95_low, 6),
                "p97_5": round(p95_high, 6),
                "theory_se": round(se_theory, 6),
            }
        )

    _write_csv(
        RESULTS_DIR / "e1_error_vs_cardinality.csv",
        ["n", "mean_rel_err", "p2_5", "p97_5", "theory_se"],
        rows,
    )

    ns = [r["n"] for r in rows]
    means = [float(r["mean_rel_err"]) * 100 for r in rows]  # type: ignore[arg-type]
    lows = [float(r["p2_5"]) * 100 for r in rows]  # type: ignore[arg-type]
    highs = [float(r["p97_5"]) * 100 for r in rows]  # type: ignore[arg-type]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, means, "o-", label="Mean relative error")
    ax.fill_between(ns, lows, highs, alpha=0.25, label="95% empirical band")
    ax.axhline(
        se_theory * 100, color="r", linestyle="--", label=f"Theory SE ({se_theory * 100:.2f}%)"
    )
    ax.set_xscale("log")
    ax.set_xlabel("True Cardinality (n)")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title(f"E1: HyperLogLog Error vs. Cardinality (p={p}, {n_trials} trials)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "e1_error_vs_cardinality.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# E2: Error vs Precision
# ---------------------------------------------------------------------------


def run_e2(seed: int, quick: bool) -> None:
    n = 20_000 if quick else 1_000_000
    n_trials = 5 if quick else 20
    ps = list(range(4, 19))
    items = [f"p_item_{i}" for i in range(n)]

    rows: list[dict[str, object]] = []
    for p in ps:
        rel_errs = []
        for t in range(n_trials):
            hll = HyperLogLog(p=p, seed=seed + t)
            hll.add_many(items)
            rel_errs.append((hll.count() - n) / n)
        obs_se = float(np.std(rel_errs))
        theory_se = 1.04 / math.sqrt(1 << p)
        rows.append(
            {
                "p": p,
                "m": 1 << p,
                "observed_se": round(obs_se, 6),
                "theory_se": round(theory_se, 6),
            }
        )

    _write_csv(
        RESULTS_DIR / "e2_error_vs_precision.csv",
        ["p", "m", "observed_se", "theory_se"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ps, [float(r["observed_se"]) * 100 for r in rows], "o-", label="Observed SE")  # type: ignore[arg-type]
    ax.plot(ps, [float(r["theory_se"]) * 100 for r in rows], "r--", label="Theory 1.04 / sqrt(m)")  # type: ignore[arg-type]
    ax.set_yscale("log")
    ax.set_xlabel("Precision p")
    ax.set_ylabel("Standard Error (%)")
    ax.set_title(f"E2: Standard Error vs. Precision (n={n:,})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "e2_error_vs_precision.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# E3: Memory Comparison
# ---------------------------------------------------------------------------


def run_e3(seed: int, quick: bool) -> None:
    ns = (
        [1_000, 10_000, 100_000, 250_000]
        if quick
        else [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    )
    p = 14
    hll = HyperLogLog(p=p, seed=seed)
    hll_bytes = hll.memory_bytes()

    rows: list[dict[str, object]] = []
    for n in ns:
        exact = ExactUniqueCounter()
        exact.add_many(f"u_{i:012x}" for i in range(n))
        exact_bytes = exact.memory_bytes()
        saved_pct = (1.0 - hll_bytes / exact_bytes) * 100.0
        rows.append(
            {
                "n": n,
                "hll_packed_bytes": hll_bytes,
                "exact_bytes": exact_bytes,
                "memory_saved_pct": round(saved_pct, 4),
            }
        )

    _write_csv(
        RESULTS_DIR / "e3_memory.csv",
        ["n", "hll_packed_bytes", "exact_bytes", "memory_saved_pct"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, [r["exact_bytes"] for r in rows], "s-", label="ExactUniqueCounter (Python set)")
    ax.plot(
        ns,
        [r["hll_packed_bytes"] for r in rows],
        "o-",
        label=f"HyperLogLog p={p} (12 KB packed)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Unique Visitors (n)")
    ax.set_ylabel("Memory Footprint (bytes)")
    ax.set_title("E3: Memory Usage — HyperLogLog vs. Exact Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "e3_memory.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# E4: Throughput
# ---------------------------------------------------------------------------


def run_e4(seed: int, quick: bool) -> None:
    n = 20_000 if quick else 500_000
    items = [f"u_{i:012x}" for i in range(n)]

    h1 = HyperLogLog(p=14, seed=seed)
    t0 = time.perf_counter()
    for item in items:
        h1.add(item)
    dt_add = time.perf_counter() - t0

    h2 = HyperLogLog(p=14, seed=seed)
    t0 = time.perf_counter()
    h2.add_many(items)
    dt_many = time.perf_counter() - t0

    eps_add = n / dt_add
    eps_many = n / dt_many

    rows = [
        {
            "method": "add() loop",
            "n": n,
            "seconds": round(dt_add, 4),
            "events_per_sec": round(eps_add, 1),
        },
        {
            "method": "add_many() batch",
            "n": n,
            "seconds": round(dt_many, 4),
            "events_per_sec": round(eps_many, 1),
        },
    ]
    _write_csv(
        RESULTS_DIR / "e4_throughput.csv",
        ["method", "n", "seconds", "events_per_sec"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    methods = [str(r["method"]) for r in rows]
    throughputs = [float(r["events_per_sec"]) for r in rows]  # type: ignore[arg-type]
    bars = ax.bar(methods, throughputs, color=["#4C72B0", "#55A868"])
    ax.bar_label(bars, fmt="%.0f ev/s")
    ax.set_ylabel("Throughput (events / sec)")
    ax.set_title(f"E4: Ingestion Throughput (p=14, n={n:,})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "e4_throughput.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# E5: Distributed Merge Accuracy
# ---------------------------------------------------------------------------


def run_e5(seed: int, quick: bool) -> None:
    n_events = 50_000 if quick else 1_000_000
    n_visitors = 15_000 if quick else 200_000
    cfg = TrafficConfig(n_events=n_events, n_unique_visitors=n_visitors, n_pages=50, seed=seed)
    gen = TrafficGenerator(cfg)
    stream = [e.visitor_id for e in gen.generate()]
    true_unique = gen.ground_truth.unique_visitors

    server_counts = [2, 4, 8, 16]
    rows: list[dict[str, object]] = []
    for k in server_counts:
        chunks = np.array_split(np.array(stream, dtype=object), k)
        merged = HyperLogLog(p=14, seed=seed)
        for chunk in chunks:
            local = HyperLogLog(p=14, seed=seed)
            local.add_many(chunk.tolist())
            merged.merge(local)
        est = merged.count()
        rel_err = abs(est - true_unique) / true_unique
        rows.append(
            {
                "servers": k,
                "true_unique": true_unique,
                "merged_estimate": round(est, 1),
                "rel_err_pct": round(rel_err * 100, 4),
            }
        )

    _write_csv(
        RESULTS_DIR / "e5_merge_accuracy.csv",
        ["servers", "true_unique", "merged_estimate", "rel_err_pct"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(
        [r["servers"] for r in rows],
        [r["rel_err_pct"] for r in rows],
        "o-",
        label="Merged HLL relative error",
    )
    ax.axhline(
        1.04 / math.sqrt(1 << 14) * 100,
        color="r",
        linestyle="--",
        label="Theory SE (0.81%)",
    )
    ax.set_xticks(server_counts)
    ax.set_xlabel("Number of Simulated Servers")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title(f"E5: Distributed Merge Accuracy ({n_events:,} events)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "e5_merge_accuracy.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# E6: Estimator Ablation (Raw vs Linear Counting)
# ---------------------------------------------------------------------------


def run_e6(seed: int, quick: bool) -> None:
    p = 12  # m = 4096
    m = 1 << p
    ns = [int(x) for x in np.linspace(50, 3 * m, 20 if quick else 30)]
    n_trials = 5 if quick else 15

    rows: list[dict[str, object]] = []
    for n in ns:
        items = [f"ab_{i}" for i in range(n)]
        raw_errs = []
        lc_errs = []
        for t in range(n_trials):
            hll = HyperLogLog(p=p, seed=seed + t)
            hll.add_many(items)
            raw_errs.append(abs(hll.raw_count() - n) / n)
            lc_errs.append(abs(hll.count() - n) / n)
        rows.append(
            {
                "n": n,
                "raw_mean_err_pct": round(float(np.mean(raw_errs)) * 100, 3),
                "corrected_mean_err_pct": round(float(np.mean(lc_errs)) * 100, 3),
            }
        )

    _write_csv(
        RESULTS_DIR / "e6_ablation.csv",
        ["n", "raw_mean_err_pct", "corrected_mean_err_pct"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, [r["raw_mean_err_pct"] for r in rows], "s--", label="Raw HLL (no Linear Counting)")
    ax.plot(ns, [r["corrected_mean_err_pct"] for r in rows], "o-", label="HLL + Linear Counting")
    ax.axvline(2.5 * m, color="gray", linestyle=":", label=f"2.5m threshold ({int(2.5*m)})")
    ax.set_xlabel("True Cardinality (n)")
    ax.set_ylabel("Mean Relative Error (%)")
    ax.set_title(f"E6: Small-Range Ablation (p={p}, m={m})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "e6_ablation.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 HyperLogLog experiments.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--quick", action="store_true", help="Run fast subset for verification")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _setup_style()

    t_start = time.perf_counter()
    for name, fn in [
        ("E1 (Error vs Cardinality)", run_e1),
        ("E2 (Error vs Precision)", run_e2),
        ("E3 (Memory)", run_e3),
        ("E4 (Throughput)", run_e4),
        ("E5 (Merge Accuracy)", run_e5),
        ("E6 (Ablation)", run_e6),
    ]:
        t0 = time.perf_counter()
        fn(args.seed, args.quick)
        print(f"Completed {name} in {time.perf_counter() - t0:.2f}s")

    print(f"All experiments finished in {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
