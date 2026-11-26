"""Consistent publication-ready Matplotlib styling and PNG+SVG saver."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.figure
from cycler import cycler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Colorblind-safe Okabe-Ito palette
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
]


def apply_print_style() -> None:
    """Apply consistent print-ready Matplotlib rcParams."""
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.framealpha": 0.9,
            "axes.prop_cycle": cycler(color=PALETTE),
        }
    )


def save_figure(fig: matplotlib.figure.Figure, figures_dir: Path, stem: str) -> tuple[Path, Path]:
    """Save *fig* as both ``.png`` and ``.svg`` in *figures_dir*."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{stem}.png"
    svg_path = figures_dir / f"{stem}.svg"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path
