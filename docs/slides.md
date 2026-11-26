# Presentation Slides Outline (16 Slides)

**Pacing Guides**:
- **10-Minute Talk**: Slides 1, 2, 4, 5, 6, 8, 9, 11, 13, 15, 16 (~55s/slide)
- **20-Minute Talk**: All 16 slides + 3-minute live dashboard walkthrough on Slide 10

---

1. **Title**: Probabilistic Counting of Unique Visitors (HyperLogLog & Count-Min Sketch) — *Speaker notes: Introduce problem & 99% memory savings.*
2. **Motivation**: Why exact `set` & `Counter` fail at web scale — *Notes: 200k visitors × 168 hourly buckets × pages = hundreds of MB.*
3. **Streaming Model & Hashing**: Single-pass constraints; why MurmurHash3 matters — *Notes: Reference S4 hash ablation (99.99% error for weak hash).*
4. **HyperLogLog Intuition**: Leading-zero coin-flip analogy & stochastic averaging — *Notes: $m = 2^p$ buckets, rank $\rho$, harmonic mean.*
5. **HyperLogLog Corrections & Merge**: Linear Counting ($E \le 2.5m$), 64-bit hash, element-wise `max` — *Notes: Lossless distributed merge.*
6. **Count-Min Sketch**: $d \times w$ matrix, $\min$ estimator, $(\varepsilon, \delta)$ Markov proof — *Notes: Zero undercount guarantee.*
7. **Conservative Update & Top-K**: Estan-Varghese rule + bounded min-heap — *Notes: ~50% error reduction (S3) & 52 KB heap compaction.*
8. **System Architecture**: `AnalyticsEngine`, `TimeBucketedHLL`/`CMS`, hot-page bounding — *Notes: Walk through Mermaid diagram.*
9. **API & Privacy**: FastAPI, edge BLAKE2b hashing, atomic `PCS1` SHA-256 snapshots — *Notes: 7,738.9 ev/s batch throughput.*
10. **Live Demo**: `make demo` (500k events, Streamlit Overview → Sketch Inspector → Merge Demo) — *Notes: Switch to `:8501`.*
11. **Results — HLL Accuracy & Memory (`S1`, `S5`)**: 0.49%–0.62% MAPE; 12 KB vs 96.22 MB at 5M events — *Notes: Show `s1_hll_stats.png` and `s5_scalability.png`.*
12. **Results — CMS & Hash Ablation (`S3`, `S4`)**: Conservative cuts error ~50%; weak hashes fail — *Notes: Show `s3_cms_ablation.png` and `s4_hash_ablation.png`.*
13. **Results — Sparse HLL++ & Windows (`S6`, `S7`)**: <1 ms bucketed queries vs raw scans — *Notes: Show `s6_sparse_hll.png`.*
14. **Limitations**: Inclusion-exclusion `<5%` overlap warning; cold-page sentinel; Python hash speed — *Notes: Reference `a3_intersection.png`.*
15. **Future Work**: HLL++ bias tables, Count Sketch deletions, Rust/Cython SIMD — *Notes: Production roadmap.*
16. **Conclusion & Q&A**: Summary of deliverables, 170+ tests, and open questions.
