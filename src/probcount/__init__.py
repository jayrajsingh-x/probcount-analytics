"""probcount — Probabilistic Counting of Unique Visitors.

A web-analytics engine using HyperLogLog and Count-Min Sketch
for memory-efficient approximate counting of unique visitors
and page-view frequencies.
"""

from probcount.cms import CountMinSketch
from probcount.engine import AnalyticsEngine, EngineConfig
from probcount.exact_engine import ExactAnalyticsEngine
from probcount.hll import HyperLogLog, union
from probcount.hll_sparse import SparseHyperLogLog
from probcount.space_saving import SpaceSaving
from probcount.topk import TopKTracker
from probcount.windows import TimeBucketedCMS, TimeBucketedHLL

__all__ = [
    "AnalyticsEngine",
    "CountMinSketch",
    "EngineConfig",
    "ExactAnalyticsEngine",
    "HyperLogLog",
    "SpaceSaving",
    "SparseHyperLogLog",
    "TimeBucketedCMS",
    "TimeBucketedHLL",
    "TopKTracker",
    "union",
]
__version__ = "0.1.0"
