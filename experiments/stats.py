"""Statistical summary utilities for multi-trial benchmark analysis."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrialSummary:
    """Summary statistics across repeated estimation trials."""

    n_trials: int
    true_value: float
    mean_estimate: float
    bias: float
    std_dev: float
    rmse: float
    mape_pct: float
    max_abs_error: float
    ci95_low: float
    ci95_high: float


def summarize_trials(estimates: Sequence[float], true_value: float) -> TrialSummary:
    """Compute bias, std dev, RMSE, MAPE, max error, and 95% CI.

    Raises:
        ValueError: If *estimates* is empty.
    """
    if not estimates:
        raise ValueError("estimates must not be empty")

    arr = np.asarray(estimates, dtype=np.float64)
    n = len(arr)
    mean_est = float(np.mean(arr))
    signed_err = arr - true_value
    abs_err = np.abs(signed_err)

    bias = float(np.mean(signed_err))
    std_dev = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    rmse = float(np.sqrt(np.mean(signed_err**2)))
    mape_pct = (
        float(np.mean(abs_err / abs(true_value)) * 100.0) if true_value != 0.0 else 0.0
    )
    max_abs = float(np.max(abs_err))

    # 95% confidence interval for the mean estimate
    margin = 1.96 * (std_dev / math.sqrt(n)) if n > 1 else 0.0
    return TrialSummary(
        n_trials=n,
        true_value=true_value,
        mean_estimate=mean_est,
        bias=bias,
        std_dev=std_dev,
        rmse=rmse,
        mape_pct=mape_pct,
        max_abs_error=max_abs,
        ci95_low=mean_est - margin,
        ci95_high=mean_est + margin,
    )
