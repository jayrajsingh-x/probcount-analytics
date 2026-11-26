"""Tests for :mod:`probcount.exact` — exact ground-truth counters."""

from __future__ import annotations

from probcount.exact import (
    ExactFrequencyCounter,
    ExactUniqueCounter,
    deep_size_bytes,
)

# ---------------------------------------------------------------------------
# ExactUniqueCounter
# ---------------------------------------------------------------------------


class TestExactUniqueCounter:
    """Correctness of the set-backed cardinality counter."""

    def test_count_empty(self) -> None:
        c = ExactUniqueCounter()
        assert c.count() == 0.0

    def test_add_and_count(self) -> None:
        c = ExactUniqueCounter()
        c.add("alice")
        c.add("bob")
        c.add("charlie")
        assert c.count() == 3.0

    def test_duplicates_ignored(self) -> None:
        c = ExactUniqueCounter()
        c.add("alice")
        c.add("alice")
        assert c.count() == 1.0

    def test_add_bytes(self) -> None:
        c = ExactUniqueCounter()
        c.add(b"hello")
        assert c.count() == 1.0

    def test_add_many(self) -> None:
        c = ExactUniqueCounter()
        c.add_many(["a", "b", "c", "a"])
        assert c.count() == 3.0

    def test_merge(self) -> None:
        c1 = ExactUniqueCounter()
        c1.add_many(["a", "b"])
        c2 = ExactUniqueCounter()
        c2.add_many(["b", "c"])
        c1.merge(c2)
        assert c1.count() == 3.0  # union of {a,b} and {b,c}

    def test_merge_returns_self(self) -> None:
        c1 = ExactUniqueCounter()
        c2 = ExactUniqueCounter()
        result = c1.merge(c2)
        assert result is c1

    def test_to_bytes_from_bytes_roundtrip(self) -> None:
        c = ExactUniqueCounter()
        c.add_many(["x", "y", "z"])
        data = c.to_bytes()
        restored = ExactUniqueCounter.from_bytes(data)
        assert restored.count() == 3.0

    def test_memory_bytes_positive(self) -> None:
        c = ExactUniqueCounter()
        c.add_many([f"item_{i}" for i in range(100)])
        assert c.memory_bytes() > 0

    def test_memory_bytes_grows(self) -> None:
        c = ExactUniqueCounter()
        c.add("one")
        small = c.memory_bytes()
        c.add_many([f"item_{i}" for i in range(1000)])
        large = c.memory_bytes()
        assert large > small


# ---------------------------------------------------------------------------
# ExactFrequencyCounter
# ---------------------------------------------------------------------------


class TestExactFrequencyCounter:
    """Correctness of the Counter-backed frequency estimator."""

    def test_estimate_unseen(self) -> None:
        c = ExactFrequencyCounter()
        assert c.estimate("unknown") == 0

    def test_add_and_estimate(self) -> None:
        c = ExactFrequencyCounter()
        c.add("page_a")
        c.add("page_a")
        c.add("page_b")
        assert c.estimate("page_a") == 2
        assert c.estimate("page_b") == 1

    def test_add_with_count(self) -> None:
        c = ExactFrequencyCounter()
        c.add("page_a", count=5)
        assert c.estimate("page_a") == 5

    def test_add_bytes(self) -> None:
        c = ExactFrequencyCounter()
        c.add(b"page", count=3)
        assert c.estimate("page") == 3

    def test_total(self) -> None:
        c = ExactFrequencyCounter()
        c.add("a", 10)
        c.add("b", 20)
        assert c.total() == 30

    def test_top_k(self) -> None:
        c = ExactFrequencyCounter()
        c.add("rare", 1)
        c.add("common", 100)
        c.add("medium", 10)
        result = c.top_k(2)
        assert result == [("common", 100), ("medium", 10)]

    def test_merge(self) -> None:
        c1 = ExactFrequencyCounter()
        c1.add("a", 5)
        c1.add("b", 3)

        c2 = ExactFrequencyCounter()
        c2.add("b", 7)
        c2.add("c", 2)

        c1.merge(c2)
        assert c1.estimate("a") == 5
        assert c1.estimate("b") == 10
        assert c1.estimate("c") == 2

    def test_merge_returns_self(self) -> None:
        c1 = ExactFrequencyCounter()
        c2 = ExactFrequencyCounter()
        result = c1.merge(c2)
        assert result is c1

    def test_memory_bytes_positive(self) -> None:
        c = ExactFrequencyCounter()
        c.add("item", 1)
        assert c.memory_bytes() > 0


# ---------------------------------------------------------------------------
# deep_size_bytes
# ---------------------------------------------------------------------------


class TestDeepSizeBytes:
    """Recursive memory estimation utility."""

    def test_returns_positive_for_set(self) -> None:
        s = {"a", "b", "c"}
        assert deep_size_bytes(s) > 0

    def test_returns_positive_for_dict(self) -> None:
        d = {"key": "value"}
        assert deep_size_bytes(d) > 0

    def test_larger_object_larger_size(self) -> None:
        small = {f"item_{i}" for i in range(10)}
        large = {f"item_{i}" for i in range(10_000)}
        assert deep_size_bytes(large) > deep_size_bytes(small)

    def test_no_double_counting(self) -> None:
        """Shared references should not be counted twice."""
        shared = "a_shared_string_that_is_reused"
        lst: list[object] = [shared, shared, shared]
        size_list = deep_size_bytes(lst)
        # Should count the string only once
        size_string = deep_size_bytes(shared)
        # The list itself plus one copy of the string
        assert size_list < deep_size_bytes([shared]) + 3 * size_string
