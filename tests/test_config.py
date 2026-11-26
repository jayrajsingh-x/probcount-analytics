"""Tests for :mod:`probcount.config` — validation and defaults."""

from __future__ import annotations

import pytest

from probcount.config import TrafficConfig, TrafficSpike


class TestTrafficSpike:
    """Validation of TrafficSpike."""

    def test_valid_spike(self) -> None:
        spike = TrafficSpike(start=100.0, end=200.0, multiplier=3.0)
        assert spike.start == 100.0
        assert spike.multiplier == 3.0

    def test_start_ge_end_raises(self) -> None:
        with pytest.raises(ValueError, match="start.*before.*end"):
            TrafficSpike(start=200.0, end=100.0, multiplier=2.0)

    def test_start_eq_end_raises(self) -> None:
        with pytest.raises(ValueError, match="start.*before.*end"):
            TrafficSpike(start=100.0, end=100.0, multiplier=2.0)

    def test_multiplier_lt_one_raises(self) -> None:
        with pytest.raises(ValueError, match="multiplier"):
            TrafficSpike(start=100.0, end=200.0, multiplier=0.5)


class TestTrafficConfig:
    """Validation of TrafficConfig."""

    def test_valid_defaults(self) -> None:
        config = TrafficConfig()
        assert config.n_events == 100_000
        assert config.n_unique_visitors == 10_000

    def test_n_events_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_events"):
            TrafficConfig(n_events=0)

    def test_n_events_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="n_events"):
            TrafficConfig(n_events=-1)

    def test_n_unique_visitors_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_unique_visitors.*> 0"):
            TrafficConfig(n_events=100, n_unique_visitors=0)

    def test_n_unique_visitors_exceeds_n_events_raises(self) -> None:
        with pytest.raises(ValueError, match="n_unique_visitors.*<= n_events"):
            TrafficConfig(n_events=100, n_unique_visitors=200)

    def test_n_pages_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_pages"):
            TrafficConfig(n_pages=0)

    def test_page_zipf_s_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="page_zipf_s"):
            TrafficConfig(page_zipf_s=-1.0)

    def test_page_zipf_s_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="page_zipf_s"):
            TrafficConfig(page_zipf_s=0.0)

    def test_visitor_zipf_s_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="visitor_zipf_s"):
            TrafficConfig(visitor_zipf_s=-0.5)

    def test_bot_fraction_one_raises(self) -> None:
        with pytest.raises(ValueError, match="bot_fraction"):
            TrafficConfig(bot_fraction=1.0)

    def test_bot_fraction_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="bot_fraction"):
            TrafficConfig(bot_fraction=-0.1)

    def test_duration_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="duration_seconds"):
            TrafficConfig(duration_seconds=0)

    def test_returning_visitor_ratio_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="returning_visitor_ratio"):
            TrafficConfig(returning_visitor_ratio=1.5)

    def test_valid_custom_config(self) -> None:
        spike = TrafficSpike(start=1000.0, end=2000.0, multiplier=5.0)
        config = TrafficConfig(
            n_events=50_000,
            n_unique_visitors=5_000,
            n_pages=20,
            page_zipf_s=1.5,
            returning_visitor_ratio=0.8,
            visitor_zipf_s=1.2,
            start_time=0.0,
            duration_seconds=43_200.0,
            diurnal=False,
            bot_fraction=0.1,
            traffic_spike=spike,
            seed=99,
        )
        assert config.traffic_spike is not None
        assert config.traffic_spike.multiplier == 5.0
