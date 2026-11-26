"""Evaluation metrics for frequency estimation and Top-K heavy hitters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def absolute_error(estimated: float, true: float) -> float:
    """Return ``|estimated - true|``."""
    return abs(estimated - true)


def relative_error(estimated: float, true: float) -> float:
    """Return ``|estimated - true| / true`` (or ``0.0`` if both are zero).

    Raises:
        ValueError: If ``true == 0`` and ``estimated != 0``.
    """
    if true == 0:
        if estimated == 0:
            return 0.0
        raise ValueError("Cannot compute relative error when true value is 0")
    return abs(estimated - true) / abs(true)


def overestimation_ratio(estimated: float, true: float) -> float:
    """Return ``estimated / true`` (or ``1.0`` if both are zero).

    Raises:
        ValueError: If ``true == 0`` and ``estimated != 0``.
    """
    if true == 0:
        if estimated == 0:
            return 1.0
        raise ValueError("Cannot compute overestimation ratio when true value is 0")
    return estimated / true


def precision_at_k(
    predicted: Sequence[str],
    truth: Sequence[str],
    k: int,
) -> float:
    """Return fraction of top-*k* predicted items that appear in top-*k* truth."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    pred_k = set(predicted[:k])
    truth_k = set(truth[:k])
    if not pred_k:
        return 0.0
    return len(pred_k & truth_k) / len(pred_k)


def recall_at_k(
    predicted: Sequence[str],
    truth: Sequence[str],
    k: int,
) -> float:
    """Return fraction of top-*k* truth items recovered in top-*k* predicted."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    pred_k = set(predicted[:k])
    truth_k = set(truth[:k])
    if not truth_k:
        return 1.0
    return len(pred_k & truth_k) / len(truth_k)


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    """Compute 1-based average ranks (handling ties) in pure NumPy."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for idx in range(i, j + 1):
            ranks[order[idx]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Spearman rank correlation between *x* and *y* in pure NumPy.

    Raises:
        ValueError: If lengths differ or fewer than 2 elements are provided.
    """
    if len(x) != len(y):
        raise ValueError(f"Length mismatch: {len(x)} vs {len(y)}")
    if len(x) < 2:
        raise ValueError("Need at least 2 elements for Spearman correlation")

    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx_c**2) * np.sum(ry_c**2)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(rx_c * ry_c) / denom)
