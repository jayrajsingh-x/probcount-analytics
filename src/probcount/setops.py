"""Set operations on HyperLogLog sketches (union, intersection, Jaccard).

Note:
    Intersection uses the inclusion-exclusion principle:
    ``|A ∩ B| ≈ |A| + |B| - |A ∪ B|``.  When the true intersection is
    small relative to ``|A ∪ B|``, the estimation errors of the three
    sketches can dominate the difference, yielding high relative error.
    Functions here return a :class:`SetOpResult` with an ``error_estimate``
    and a ``warning`` flag whenever the estimated overlap ratio is below 5%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from probcount.hll import HyperLogLog, union


@dataclass(frozen=True)
class SetOpResult:
    """Result of an approximate set operation on HyperLogLog sketches."""

    estimate: float
    error_estimate: float
    warning: bool
    message: str = ""


def hll_union(a: HyperLogLog, b: HyperLogLog) -> SetOpResult:
    """Estimate ``|A ∪ B|`` by merging *a* and *b*."""
    merged = union(a, b)
    est = merged.count()
    err = est * merged.standard_error()
    return SetOpResult(estimate=est, error_estimate=err, warning=False)


def hll_intersection(a: HyperLogLog, b: HyperLogLog) -> SetOpResult:
    """Estimate ``|A ∩ B|`` via inclusion-exclusion ``|A| + |B| - |A ∪ B|``.

    Sets ``warning=True`` when the estimated overlap is small relative to the
    union (``< 5%``), where inclusion-exclusion error can be large.
    """
    est_a = a.count()
    est_b = b.count()
    merged = union(a, b)
    est_u = merged.count()

    est_i = max(0.0, est_a + est_b - est_u)
    se = a.standard_error()
    err = math.sqrt((se * est_a) ** 2 + (se * est_b) ** 2 + (merged.standard_error() * est_u) ** 2)

    overlap_ratio = est_i / est_u if est_u > 0.0 else 0.0
    small_overlap = (est_u > 0.0) and (overlap_ratio < 0.05)
    msg = (
        "Small overlap (<5% of union): inclusion-exclusion error may be large"
        if small_overlap
        else ""
    )
    return SetOpResult(
        estimate=est_i,
        error_estimate=err,
        warning=small_overlap,
        message=msg,
    )


def hll_jaccard(a: HyperLogLog, b: HyperLogLog) -> SetOpResult:
    """Estimate Jaccard similarity ``|A ∩ B| / |A ∪ B|`` in ``[0, 1]``."""
    est_u = union(a, b).count()
    if est_u == 0.0:
        return SetOpResult(estimate=0.0, error_estimate=0.0, warning=False)

    inter = hll_intersection(a, b)
    jaccard = min(1.0, max(0.0, inter.estimate / est_u))
    err = inter.error_estimate / est_u
    return SetOpResult(
        estimate=jaccard,
        error_estimate=err,
        warning=inter.warning,
        message=inter.message,
    )
