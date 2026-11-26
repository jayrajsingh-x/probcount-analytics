"""Central configuration dataclasses with defaults and validation.

All configurable parameters for the traffic generator and future
probabilistic data structures are defined here as frozen dataclasses.
Validation is performed in ``__post_init__`` and raises ``ValueError``
for any invalid combination of parameters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrafficSpike:
    """A temporary traffic spike within a time window.

    Attributes:
        start: Start timestamp of the spike (seconds since epoch).
        end: End timestamp of the spike (seconds since epoch).
        multiplier: Factor by which event rate is multiplied during the spike.
    """

    start: float
    end: float
    multiplier: float

    def __post_init__(self) -> None:
        """Validate spike parameters."""
        if self.start >= self.end:
            raise ValueError(
                f"Spike start ({self.start}) must be before end ({self.end})"
            )
        if self.multiplier < 1.0:
            raise ValueError(
                f"Spike multiplier must be >= 1.0, got {self.multiplier}"
            )


def _default_start_time() -> float:
    """Return the current time as the default start timestamp."""
    return time.time()


@dataclass(frozen=True)
class TrafficConfig:
    """Configuration for the synthetic traffic generator.

    Attributes:
        n_events: Total number of visit events to generate (> 0).
        n_unique_visitors: Size of the visitor pool (> 0, <= n_events).
        n_pages: Number of distinct pages (> 0).
        page_zipf_s: Zipf exponent for page popularity (> 0).
        returning_visitor_ratio: Probability that an event's visitor is drawn
            from the Zipf-weighted (heavy-hitter) distribution rather than
            uniformly.  Higher values create more repeat visitors.  Range [0, 1].
        visitor_zipf_s: Zipf exponent for visitor frequency (> 0).
            Higher values make a few visitors dominate traffic.
        start_time: Start timestamp in seconds since epoch.
        duration_seconds: Length of the simulated time window (> 0).
        diurnal: Whether to apply a diurnal (time-of-day) pattern.
        bot_fraction: Fraction of events marked as bot traffic.  Range [0, 1).
        traffic_spike: Optional spike configuration.
        seed: Random seed for reproducibility.
    """

    n_events: int = 100_000
    n_unique_visitors: int = 10_000
    n_pages: int = 100
    page_zipf_s: float = 1.1
    returning_visitor_ratio: float = 0.6
    visitor_zipf_s: float = 1.0
    start_time: float = field(default_factory=_default_start_time)
    duration_seconds: float = 86_400.0
    diurnal: bool = True
    bot_fraction: float = 0.05
    traffic_spike: TrafficSpike | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate all configuration parameters."""
        if self.n_events <= 0:
            raise ValueError(f"n_events must be > 0, got {self.n_events}")
        if self.n_unique_visitors <= 0:
            raise ValueError(
                f"n_unique_visitors must be > 0, got {self.n_unique_visitors}"
            )
        if self.n_unique_visitors > self.n_events:
            raise ValueError(
                f"n_unique_visitors ({self.n_unique_visitors}) must be "
                f"<= n_events ({self.n_events})"
            )
        if self.n_pages <= 0:
            raise ValueError(f"n_pages must be > 0, got {self.n_pages}")
        if self.page_zipf_s <= 0:
            raise ValueError(
                f"page_zipf_s must be > 0, got {self.page_zipf_s}"
            )
        if not 0.0 <= self.returning_visitor_ratio <= 1.0:
            raise ValueError(
                f"returning_visitor_ratio must be in [0, 1], "
                f"got {self.returning_visitor_ratio}"
            )
        if self.visitor_zipf_s <= 0:
            raise ValueError(
                f"visitor_zipf_s must be > 0, got {self.visitor_zipf_s}"
            )
        if self.duration_seconds <= 0:
            raise ValueError(
                f"duration_seconds must be > 0, got {self.duration_seconds}"
            )
        if not 0.0 <= self.bot_fraction < 1.0:
            raise ValueError(
                f"bot_fraction must be in [0, 1), got {self.bot_fraction}"
            )
