"""Tests for :mod:`experiments.stats`."""

from __future__ import annotations

import pytest
from experiments.stats import summarize_trials


def test_summarize_trials_exact() -> None:
    s = summarize_trials([100.0, 100.0, 100.0], 100.0)
    assert s.bias == 0.0
    assert s.std_dev == 0.0
    assert s.rmse == 0.0
    assert s.mape_pct == 0.0
    assert s.max_abs_error == 0.0
    assert s.ci95_low == 100.0
    assert s.ci95_high == 100.0


def test_summarize_trials_metrics_and_ci() -> None:
    s = summarize_trials([98.0, 102.0, 104.0, 96.0], 100.0)
    assert s.n_trials == 4
    assert s.bias == pytest.approx(0.0)
    assert s.max_abs_error == pytest.approx(4.0)
    assert s.mape_pct == pytest.approx(3.0)
    assert s.ci95_low < 100.0 < s.ci95_high


def test_summarize_trials_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize_trials([], 10.0)
