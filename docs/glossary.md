# Glossary of Terms & Symbols

- **$\alpha_m$**: HyperLogLog bias-correction constant (`0.7213 / (1 + 1.079/m)` for $m \ge 128$).
- **$\delta$ (Delta)**: Failure probability in Count-Min Sketch; confidence is $1 - \delta$.
- **$\varepsilon$ (Epsilon)**: Additive error factor in Count-Min Sketch ($\hat{f}(x) - f(x) \le \varepsilon N$).
- **$\rho(w)$ (Rho)**: Position of the leftmost `1`-bit in the $(64-p)$-bit tail $w$ (`leading_zeros + 1`).
- **Cardinality ($F_0$)**: Number of distinct items in a stream.
- **Conservative Update**: CMS update rule setting counters to $\max(\text{counter}, \hat{f}(x) + c)$ to reduce over-estimation.
- **Count-Min Sketch (CMS)**: $d \times w$ counter matrix estimating item frequencies with zero under-counting.
- **Hot Set**: Bounded top-$N$ pages tracked with per-page HLLs in `AnalyticsEngine`.
- **HyperLogLog (HLL)**: $m = 2^p$ register sketch estimating cardinality with standard error $1.04/\sqrt{m}$.
- **Inclusion-Exclusion**: Estimating $|A \cap B| = |A| + |B| - |A \cup B|$.
- **Jaccard Similarity**: $|A \cap B| / |A \cup B| \in [0, 1]$.
- **Kirsch-Mitzenmacher**: Deriving $d$ hash functions from two base hashes $(h_1 + i \cdot h_2) \bmod w$.
- **Linear Counting**: Small-range cardinality estimator $m \ln(m / V)$ using empty register count $V$.
- **MurmurHash3**: Non-cryptographic 128-bit hash function with strong avalanche properties.
- **Roll-up**: Lossless hierarchical merging of `60m → 1h → 1d` time buckets.
- **Space-Saving**: Deterministic $k$-slot heavy-hitter counter (Metwally et al.).
- **Watermark**: Maximum lateness tolerated before events are dropped (`dropped_late`).
- **Zipf's Law**: Power-law rank-frequency distribution $P(r) \propto 1/r^s$.
