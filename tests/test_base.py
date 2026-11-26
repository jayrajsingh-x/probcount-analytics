"""Tests for :mod:`probcount.base` — abstract interface contracts."""

from __future__ import annotations

import pytest

from probcount.base import CardinalityEstimator, FrequencyEstimator


class TestCardinalityEstimatorABC:
    """Cannot instantiate or partially implement CardinalityEstimator."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            CardinalityEstimator()  # type: ignore[abstract]

    def test_partial_implementation_raises(self) -> None:
        """Defining only some abstract methods still raises TypeError."""

        class Partial(CardinalityEstimator):  # type: ignore[abstract]
            def add(self, item: bytes | str) -> None: ...
            def count(self) -> float:
                return 0.0

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]


class TestFrequencyEstimatorABC:
    """Cannot instantiate or partially implement FrequencyEstimator."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            FrequencyEstimator()  # type: ignore[abstract]

    def test_partial_implementation_raises(self) -> None:

        class Partial(FrequencyEstimator):  # type: ignore[abstract]
            def add(self, item: bytes | str, count: int = 1) -> None: ...

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]
