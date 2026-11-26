"""Verify key numbers quoted in docs/report.md against experiments/results/*.csv."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    report_text = Path("docs/report.md").read_text(encoding="utf-8")
    results_dir = Path("experiments/results")

    s1 = _read_csv(results_dir / "s1_hll_stats.csv")
    s2 = _read_csv(results_dir / "s2_param_grid.csv")
    s3 = _read_csv(results_dir / "s3_cms_ablation.csv")
    s4 = _read_csv(results_dir / "s4_hash_ablation.csv")
    s5 = _read_csv(results_dir / "s5_scalability.csv")

    expected_tokens = [
        s1[1]["mape_pct"],   # 0.491
        s1[2]["mape_pct"],   # 0.607
        s1[3]["mape_pct"],   # 0.623
        s1[0]["theory_se_pct"],  # 0.812
        s2[0]["packed_kb"],  # 0.19
        s2[3]["packed_kb"],  # 12.0
        s2[5]["packed_kb"],  # 192.0
        s3[0]["reduction_pct"],  # 52.6
        s3[2]["reduction_pct"],  # 43.6
        s4[0]["seq_rel_err_pct"],  # 1.52
        s4[3]["seq_rel_err_pct"],  # 99.99
        s5[-1]["exact_peak_mb"],   # 96.22
    ]

    missing = [tok for tok in expected_tokens if tok not in report_text]
    if missing:
        print(f"Report number verification FAILED; missing tokens: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Verified {len(expected_tokens)} key metrics in docs/report.md against CSVs.")


if __name__ == "__main__":
    main()
