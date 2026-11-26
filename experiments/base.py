"""Abstract base class for reproducible Phase 7 benchmark experiments."""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Experiment(ABC):
    """Base class for benchmark experiments with setup, run, save, and plot."""

    name: str
    csv_name: str
    figure_stem: str
    caption: str

    def __init__(self, config: dict[str, Any], quick: bool = False) -> None:
        self.config = config
        self.quick = quick
        self.seed = int(config.get("seed", 42))
        self.results_dir = Path(config.get("results_dir", "experiments/results"))
        self.figures_dir = Path(config.get("figures_dir", "docs/figures"))
        self.rows: list[dict[str, Any]] = []

    @abstractmethod
    def setup(self) -> None:
        """Prepare inputs or configurations before running."""

    @abstractmethod
    def run(self) -> list[dict[str, Any]]:
        """Execute the benchmark and return result rows."""

    def save(self) -> Path:
        """Persist ``self.rows`` to ``self.results_dir / self.csv_name``."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        out = self.results_dir / self.csv_name
        if not self.rows:
            return out
        fieldnames = list(self.rows[0].keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return out

    @abstractmethod
    def plot(self) -> tuple[Path, Path]:
        """Generate ``.png`` and ``.svg`` figures and return their paths."""
