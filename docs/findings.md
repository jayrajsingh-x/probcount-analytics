# Key Research Findings

1. **HLL Accuracy Matches Theory Across Scales (`s1_hll_stats.png`, `e1_error_vs_cardinality.png`)**:
   At $p = 14$ ($m = 16{,}384$), HyperLogLog remains unbiased with MAPE near `0.6%–0.8%` from $10^2$ to $10^6+$ unique visitors, matching $1.04/\sqrt{m} = 0.8125\%$.
2. **Optimal Precision Sweet Spot (`s2_param_grid.png`)**:
   Increasing $p$ by $+2$ halves standard error at $4\times$ memory; $p = 14$ (12 KB packed, 0.81% SE) is the best accuracy-vs-memory trade-off for web analytics.
3. **Massive Sub-Linear Memory Savings (`s5_scalability.png`, `e3_memory.png`)**:
   HyperLogLog stays flat at 12 KB from $10^3$ to $5 \times 10^6+$ events while `ExactUniqueCounter` grows linearly to tens/hundreds of megabytes (>99% memory saved).
4. **Weak Hash Functions Break Guarantees (`s4_hash_ablation.png`)**:
   Sum-of-bytes hashing suffers >90% error on sequential IDs (`user_000001`, …) due to bucket clustering, whereas MurmurHash3 and BLAKE2b stay below 2%.
5. **Conservative Update Halves CMS Over-Estimation (`s3_cms_ablation.png`)**:
   Across Zipf skew $s \in [0.8, 1.5]$, Conservative Update cuts mean over-estimation by ~45%–52% while preserving the zero-undercount guarantee.
6. **CMS Excels on Heavy Hitters, Struggles on Long Tail (`c4_heavy_vs_tail.png`)**:
   Additive error $\varepsilon N$ is negligible for top pages (<1% relative error) but large relative to rare 1-view URLs.
7. **Sparse HLL++ Saves Memory at Low Cardinality (`s6_sparse_hll.png`)**:
   For $n \le 500$, `SparseHyperLogLog` uses significantly less memory than dense 12 KB registers before seamlessly converting at threshold.
8. **Windowed Merge Beats Raw Scanning (`s7_sliding_window.png`, `a2_window_size.png`)**:
   Merging pre-aggregated hourly HLL buckets answers 7-day queries in <1 ms (>100× faster than raw event scans) with zero error accumulation.
9. **Inclusion-Exclusion Warning at Low Overlap (`a3_intersection.png`)**:
   When true set overlap is below 5% of the union, HLL inclusion-exclusion error rises sharply (>19%–80%), justifying our automatic warning flag.
10. **Resilience to Adversarial Streams (`s8_adversarial.png`)**:
    Empty, tiny, all-identical, and all-unique streams all stay within theoretical error bounds without crashes or memory spikes.
