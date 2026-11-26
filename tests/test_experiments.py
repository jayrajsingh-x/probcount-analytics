"""Tests for Phase 7 benchmark infrastructure (datasets, env_info, plot_style, reproducibility)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from experiments.datasets import load_access_log, load_clickstream_csv, load_or_fallback
from experiments.env_info import write_environment_json
from experiments.plot_style import apply_print_style, save_figure
from experiments.studies import StudyS8Adversarial


def test_access_log_and_csv_loaders(tmp_path: Path) -> None:
    log_file = tmp_path / "access.log"
    log_file.write_text(
        '192.168.1.1 - - [10/Oct/2023:13:55:36 +0000] "GET /home HTTP/1.1" 200 1024 "-" "Mozilla/5.0"\n'
        '10.0.0.5 - - [10/Oct/2023:13:56:00 +0000] "GET /robots.txt HTTP/1.1" 200 64 "-" "Googlebot/2.1"\n',
        encoding="utf-8",
    )
    events = list(load_access_log(log_file))
    assert len(events) == 2
    assert events[0].visitor_id == "192.168.1.1"
    assert events[0].is_bot is False
    assert events[1].is_bot is True

    csv_file = tmp_path / "clicks.csv"
    csv_file.write_text(
        "timestamp,visitor_id,page_id,is_bot\n100.0,u_1,/p/1,false\n101.0,u_2,/p/2,true\n",
        encoding="utf-8",
    )
    c_events = list(load_clickstream_csv(csv_file))
    assert len(c_events) == 2
    assert c_events[1].is_bot is True

    # Fallback when file does not exist
    fb = list(load_or_fallback(tmp_path / "missing.log"))
    assert len(fb) > 0


def test_environment_and_plot_style(tmp_path: Path) -> None:
    env_path = tmp_path / "env.json"
    env = write_environment_json(env_path)
    assert env_path.exists()
    assert "python_version" in env
    assert "packages" in env

    apply_print_style()
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    png, svg = save_figure(fig, tmp_path / "figs", "sample")
    assert png.exists() and svg.exists()


def test_reproducibility_byte_identical_csv(tmp_path: Path) -> None:
    cfg1 = {"seed": 42, "results_dir": str(tmp_path / "r1"), "figures_dir": str(tmp_path / "f1")}
    cfg2 = {"seed": 42, "results_dir": str(tmp_path / "r2"), "figures_dir": str(tmp_path / "f2")}

    e1 = StudyS8Adversarial(cfg1, quick=True)
    e1.setup()
    e1.run()
    p1 = e1.save()

    e2 = StudyS8Adversarial(cfg2, quick=True)
    e2.setup()
    e2.run()
    p2 = e2.save()

    assert p1.read_bytes() == p2.read_bytes()
