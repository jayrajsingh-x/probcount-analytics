"""Shared test fixtures and helpers."""

from __future__ import annotations

import pytest

from probcount.config import TrafficConfig


@pytest.fixture()
def small_config() -> TrafficConfig:
    """A small deterministic traffic configuration for fast tests."""
    return TrafficConfig(
        n_events=1_000,
        n_unique_visitors=100,
        n_pages=10,
        seed=42,
        diurnal=False,
        bot_fraction=0.05,
    )


@pytest.fixture()
def medium_config() -> TrafficConfig:
    """A medium-sized configuration for statistical tests."""
    return TrafficConfig(
        n_events=100_000,
        n_unique_visitors=10_000,
        n_pages=50,
        seed=123,
        diurnal=True,
        bot_fraction=0.05,
    )
