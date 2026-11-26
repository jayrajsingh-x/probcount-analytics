"""Tests for :mod:`probcount.hll_sparse` — HyperLogLog++ sparse representation."""

from __future__ import annotations

import numpy as np
import pytest

from probcount.hll import HyperLogLog
from probcount.hll_sparse import SparseHyperLogLog


class TestSparseHyperLogLog:
    """Sparse mode exactness, dense conversion, merge, and serialization."""

    def test_invalid_p_raises(self) -> None:
        with pytest.raises(ValueError, match="p must be"):
            SparseHyperLogLog(p=3)

    def test_small_set_sparse_and_exact(self) -> None:
        shll = SparseHyperLogLog(p=14, seed=0)
        assert shll.is_sparse is True
        assert shll.count() == 0.0

        shll.add_many([f"u_{i}" for i in range(50)])
        assert shll.is_sparse is True
        assert abs(shll.count() - 50.0) <= 1.0
        assert shll.memory_bytes() < HyperLogLog(p=14).memory_bytes()

    def test_conversion_matches_dense_registers(self) -> None:
        items = [f"item_{i}" for i in range(500)]
        shll = SparseHyperLogLog(p=10, seed=42, sparse_threshold=100)
        dense = HyperLogLog(p=10, seed=42)

        shll.add_many(items)
        dense.add_many(items)

        assert shll.is_sparse is False
        assert np.array_equal(shll.to_dense().registers, dense.registers)
        assert shll.count() == dense.count()

    def test_merge_and_serialization(self) -> None:
        s1 = SparseHyperLogLog(p=12, seed=1)
        s2 = SparseHyperLogLog(p=12, seed=1)
        s1.add_many(["a", "b", "c"])
        s2.add_many(["c", "d"])
        s1.merge(s2)
        assert abs(s1.count() - 4.0) <= 0.5

        raw_s = s1.to_bytes()
        restored_s = SparseHyperLogLog.from_bytes(raw_s)
        assert restored_s.count() == s1.count()

        # Dense roundtrip
        s_dense = SparseHyperLogLog(p=8, seed=1, sparse_threshold=10)
        s_dense.add_many([f"x_{i}" for i in range(200)])
        assert s_dense.is_sparse is False
        restored_d = SparseHyperLogLog.from_bytes(s_dense.to_bytes())
        assert restored_d.is_sparse is False
        assert restored_d.count() == s_dense.count()
