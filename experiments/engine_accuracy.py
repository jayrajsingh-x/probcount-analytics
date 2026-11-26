"""Phase 4 Experiments: AnalyticsEngine Accuracy, Memory, Spikes & Distributed Scaling.

Produces CSVs in ``experiments/results/`` and PNGs in ``docs/figures/``.

Usage:
    python experiments/engine_accuracy.py --quick --seed 42
    python experiments/engine_accuracy.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from probcount.config import TrafficConfig, TrafficSpike  # noqa: E402
from probcount.distributed import simulate_distributed  # noqa: E402
from probcount.engine import AnalyticsEngine, EngineConfig  # noqa: E402
from probcount.exact_engine import ExactAnalyticsEngine  # noqa: E402
from probcount.generator import TrafficGenerator  # noqa: E402
from probcount.hll import HyperLogLog  # noqa: E402
from probcount.setops import hll_intersection  # noqa: E402

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
# A1: 7-day hourly unique-visitor time series
# ---------------------------------------------------------------------------


def run_a1(seed: int, quick: bool) -> None:
    start = 1_700_000_000.0
    n_events = 25_000 if quick else 500_000
    cfg = TrafficConfig(
        n_events=n_events,
        n_unique_visitors=8_000 if quick else 25_000,
        n_pages=50,
        start_time=start,
        duration_seconds=7 * 86_400.0,
        seed=seed,
    )
    events = list(TrafficGenerator(cfg).generate())
    engine = AnalyticsEngine(EngineConfig(hll_p=14, seed=seed))
    exact = ExactAnalyticsEngine()
    engine.process_batch(events)
    exact.process_batch(events)

    sk_series = engine.unique_visitors_series("hour", start, start + 7 * 86_400.0)
    ex_series = exact.unique_visitors_series("hour", start, start + 7 * 86_400.0)

    rows: list[dict[str, object]] = []
    for i, ((_, sk_v), (_, ex_v)) in enumerate(zip(sk_series, ex_series, strict=True)):
        rel = abs(sk_v - ex_v) / ex_v * 100.0 if ex_v > 0 else 0.0
        rows.append(
            {
                "hour": i,
                "sketch_uv": round(sk_v, 1),
                "exact_uv": int(ex_v),
                "rel_err_pct": round(rel, 3),
            }
        )

    _write_csv(
        RESULTS_DIR / "a1_hourly_series.csv",
        ["hour", "sketch_uv", "exact_uv", "rel_err_pct"],
        rows,
    )

    hours = [r["hour"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(hours, [r["exact_uv"] for r in rows], label="Exact")
    ax1.plot(hours, [r["sketch_uv"] for r in rows], "--", label="Sketch (HLL p=14)")
    ax1.set_ylabel("Unique Visitors / hr")
    ax1.set_title("A1: 7-Day Hourly Unique Visitors — Sketch vs. Exact")
    ax1.legend()

    ax2.plot(hours, [r["rel_err_pct"] for r in rows], color="tab:orange")
    ax2.axhline(0.81, color="r", linestyle="--", label="Theory SE (0.81%)")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Relative Error (%)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "a1_hourly_series.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# A2: Window-size study (1 hour to 14 days)
# ---------------------------------------------------------------------------


def run_a2(seed: int, quick: bool) -> None:
    start = 1_700_000_000.0
    n_events = 30_000 if quick else 500_000
    cfg = TrafficConfig(
        n_events=n_events,
        n_unique_visitors=10_000 if quick else 30_000,
        n_pages=40,
        start_time=start,
        duration_seconds=14 * 86_400.0,
        seed=seed,
    )
    events = list(TrafficGenerator(cfg).generate())
    engine = AnalyticsEngine(EngineConfig(hll_p=14, seed=seed))
    exact = ExactAnalyticsEngine()
    engine.process_batch(events)
    exact.process_batch(events)

    window_hours = [1, 6, 24, 72, 168, 336]
    rows: list[dict[str, object]] = []
    for h in window_hours:
        end = start + h * 3600.0
        sk_v = engine.unique_visitors("default", start, end)
        ex_v = exact.unique_visitors("default", start, end)
        rel = abs(sk_v - ex_v) / ex_v * 100.0 if ex_v > 0 else 0.0
        rows.append(
            {
                "window_hours": h,
                "sketch_uv": round(sk_v, 1),
                "exact_uv": int(ex_v),
                "rel_err_pct": round(rel, 4),
            }
        )

    _write_csv(
        RESULTS_DIR / "a2_window_size.csv",
        ["window_hours", "sketch_uv", "exact_uv", "rel_err_pct"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(window_hours, [r["rel_err_pct"] for r in rows], "o-", label="Observed rel error")
    ax.axhline(0.81, color="r", linestyle="--", label="Theory SE (0.81%)")
    ax.set_xscale("log")
    ax.set_xlabel("Window Length (hours)")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("A2: Error vs. Window Length (Merging Does Not Accumulate Error)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "a2_window_size.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# A3: Inclusion-Exclusion Intersection Study
# ---------------------------------------------------------------------------


def run_a3(seed: int, quick: bool) -> None:
    overlaps = [0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    n_each = 20_000
    n_trials = 5 if quick else 15
    rows: list[dict[str, object]] = []

    for ov in overlaps:
        n_shared = int(n_each * ov)
        n_only = n_each - n_shared
        shared = [f"s_{i}" for i in range(n_shared)]
        only_a = [f"a_{i}" for i in range(n_only)]
        only_b = [f"b_{i}" for i in range(n_only)]

        errs = []
        warned = 0
        for t in range(n_trials):
            ha = HyperLogLog(p=14, seed=seed + t)
            hb = HyperLogLog(p=14, seed=seed + t)
            ha.add_many(shared + only_a)
            hb.add_many(shared + only_b)
            res = hll_intersection(ha, hb)
            errs.append(abs(res.estimate - n_shared) / max(1, n_shared) * 100.0)
            warned += int(res.warning)

        rows.append(
            {
                "overlap_frac": ov,
                "true_inter": n_shared,
                "mean_rel_err_pct": round(float(np.mean(errs)), 2),
                "warn_rate": round(warned / n_trials, 2),
            }
        )

    _write_csv(
        RESULTS_DIR / "a3_intersection.csv",
        ["overlap_frac", "true_inter", "mean_rel_err_pct", "warn_rate"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(
        [float(r["overlap_frac"]) * 100 for r in rows],  # type: ignore[arg-type]
        [r["mean_rel_err_pct"] for r in rows],
        "o-",
        label="Inclusion-Exclusion Rel Error",
    )
    ax.axvline(5.0, color="r", linestyle="--", label="5% warning threshold")
    ax.set_yscale("log")
    ax.set_xlabel("True Overlap (%)")
    ax.set_ylabel("Mean Relative Error (%)")
    ax.set_title("A3: HLL Inclusion-Exclusion Intersection Error vs. Overlap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "a3_intersection.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# A4: Memory Budget vs Exact Engine
# ---------------------------------------------------------------------------


def run_a4(seed: int, quick: bool) -> None:
    sizes = (
        [5_000, 15_000, 30_000, 60_000]
        if quick
        else [100_000, 500_000, 1_000_000, 2_000_000]
    )
    rows: list[dict[str, object]] = []

    for n in sizes:
        cfg = TrafficConfig(
            n_events=n,
            n_unique_visitors=max(1_000, n // 5),
            n_pages=200,
            start_time=0.0,
            duration_seconds=86_400.0,
            seed=seed,
        )
        events = TrafficGenerator(cfg).generate()
        engine = AnalyticsEngine(
            EngineConfig(hll_p=14, page_hll_p=10, hot_pages_limit=50, seed=seed)
        )
        exact = ExactAnalyticsEngine()
        for e in events:
            engine.process_event(e)
            exact.process_event(e)
        rep = engine.memory_report(exact)
        ratio_pct = rep["total_bytes"] / rep["exact_bytes"] * 100.0
        rows.append(
            {
                "n_events": n,
                "engine_bytes": rep["total_bytes"],
                "exact_bytes": rep["exact_bytes"],
                "engine_pct_of_exact": round(ratio_pct, 2),
            }
        )

    _write_csv(
        RESULTS_DIR / "a4_memory_budget.csv",
        ["n_events", "engine_bytes", "exact_bytes", "engine_pct_of_exact"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(sizes, [r["exact_bytes"] for r in rows], "s-", label="ExactAnalyticsEngine")
    ax.plot(sizes, [r["engine_bytes"] for r in rows], "o-", label="AnalyticsEngine (Sketch)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Events Processed")
    ax.set_ylabel("Memory (bytes)")
    ax.set_title("A4: Engine Memory vs. Exact Engine")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "a4_memory_budget.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# A5: Traffic Spike & Bot-Flood Scenario
# ---------------------------------------------------------------------------


def run_a5(seed: int, quick: bool) -> None:
    start = 0.0
    n_events = 20_000 if quick else 200_000
    spike = TrafficSpike(start=36_000.0, end=43_200.0, multiplier=8.0)
    cfg = TrafficConfig(
        n_events=n_events,
        n_unique_visitors=15_000,
        n_pages=30,
        start_time=start,
        duration_seconds=86_400.0,
        bot_fraction=0.25,
        traffic_spike=spike,
        seed=seed,
    )
    events = list(TrafficGenerator(cfg).generate())

    eng_filtered = AnalyticsEngine(EngineConfig(hll_p=14, ignore_bots=True, seed=seed))
    eng_raw = AnalyticsEngine(EngineConfig(hll_p=14, ignore_bots=False, seed=seed))
    eng_filtered.process_batch(events)
    eng_raw.process_batch(events)

    s_filt = eng_filtered.unique_visitors_series("hour", start, 86_400.0)
    s_raw = eng_raw.unique_visitors_series("hour", start, 86_400.0)

    rows = [
        {"hour": i, "filtered_uv": round(fv, 1), "with_bots_uv": round(rv, 1)}
        for i, ((_, fv), (_, rv)) in enumerate(zip(s_filt, s_raw, strict=True))
    ]
    _write_csv(
        RESULTS_DIR / "a5_spike_and_bots.csv",
        ["hour", "filtered_uv", "with_bots_uv"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(7, 4.2))
    hours = [r["hour"] for r in rows]
    ax.plot(hours, [r["with_bots_uv"] for r in rows], "--", label="All traffic (incl. 25% bots)")
    ax.plot(hours, [r["filtered_uv"] for r in rows], "o-", label="Human only (ignore_bots=True)")
    ax.axvspan(10, 12, color="red", alpha=0.12, label="8x Traffic Spike")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Unique Visitors / hr")
    ax.set_title("A5: Traffic Spike & Bot Filtering")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "a5_spike_and_bots.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# A6: Distributed Scaling
# ---------------------------------------------------------------------------


def run_a6(seed: int, quick: bool) -> None:
    n_events = 15_000 if quick else 200_000
    cfg_gen = TrafficConfig(
        n_events=n_events,
        n_unique_visitors=10_000,
        n_pages=25,
        start_time=0.0,
        duration_seconds=6 * 3600.0,
        seed=seed,
    )
    events = list(TrafficGenerator(cfg_gen).generate())
    ecfg = EngineConfig(hll_p=14, cms_conservative=False, seed=seed)

    exact = ExactAnalyticsEngine()
    exact.process_batch(events)
    true_uv = exact.unique_visitors("default", 0.0, 6 * 3600.0)

    worker_counts = [2, 4, 8, 16]
    rows: list[dict[str, object]] = []
    for k in worker_counts:
        res = simulate_distributed(events, ecfg, n_workers=k, mode="hash", seed=seed)
        est_uv = res.merged_engine.unique_visitors("default", 0.0, 6 * 3600.0)
        rel_err = abs(est_uv - true_uv) / true_uv * 100.0
        rows.append(
            {
                "workers": k,
                "true_uv": int(true_uv),
                "merged_uv": round(est_uv, 1),
                "rel_err_pct": round(rel_err, 4),
                "merge_ms": round(res.merge_seconds * 1000.0, 2),
            }
        )

    _write_csv(
        RESULTS_DIR / "a6_distributed.csv",
        ["workers", "true_uv", "merged_uv", "rel_err_pct", "merge_ms"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(
        worker_counts,
        [r["merge_ms"] for r in rows],
        "o-",
        label="Coordinator merge time (ms)",
    )
    ax.set_xticks(worker_counts)
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Merge Time (ms)")
    ax.set_title("A6: Distributed Coordinator Merge Time vs. Workers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "a6_distributed.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 AnalyticsEngine experiments.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--quick", action="store_true", help="Run fast subset")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _setup_style()

    t_start = time.perf_counter()
    for name, fn in [
        ("A1 (Hourly Series)", run_a1),
        ("A2 (Window Size)", run_a2),
        ("A3 (Intersection)", run_a3),
        ("A4 (Memory Budget)", run_a4),
        ("A5 (Spike & Bots)", run_a5),
        ("A6 (Distributed)", run_a6),
    ]:
        t0 = time.perf_counter()
        fn(args.seed, args.quick)
        print(f"Completed {name} in {time.perf_counter() - t0:.2f}s", flush=True)

    print(f"All Engine experiments finished in {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
