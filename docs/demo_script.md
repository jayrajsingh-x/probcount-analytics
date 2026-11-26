# Live Demo Script (Minute-by-Minute)

## Pre-Flight (30 seconds before presentation)
```bash
make demo
# Wait until "Seeded: {'events_generated': 500000, ...}" and open http://127.0.0.1:8501
```

## Minute-by-Minute Plan
- **0:00–1:00 — Overview (`http://127.0.0.1:8501`)**: Show 500k events, HLL unique visitors, **Memory Saved (>92%–96%)**, and toggle **Overlay Exact Line & ±SE Band** to show the sketch hugging ground truth.
- **1:00–2:00 — Top Pages & Accuracy Lab**: Adjust $K$ slider on **Top Pages** (`<1%` relative error), then open **Accuracy Lab** to show the `y = x` scatter and error histogram inside the `1.04/sqrt(m)` line.
- **2:00–3:00 — Sketch Inspector**: Show the HLL 64-register heat strip and CMS heatmap; type `u_demo_123` in **Step-by-Step Trace** to reveal the 64-bit MurmurHash3 split.
- **3:00–4:00 — Parameter Playground & Merge Demo**: Lower `p` from 14 to 8 to show the memory-vs-error trade-off curve; on **Merge Demo**, set overlap to `2%` to trigger the inclusion-exclusion warning.
- **4:00–5:00 — Simulator & Admin**: Trigger a `5x` spike in **Simulator**, then open **Admin** to save a SHA-256 snapshot and download the 12 KB `.bin` HLL.

## Offline Fallback Plan
If the live server cannot run:
1. Show pre-generated publication plots in `docs/figures/` (`s1_hll_stats.png`, `s5_scalability.png`, `a1_hourly_series.png`).
2. Run `python -m pytest tests/test_e2e_integration.py -v` to demonstrate live API + snapshot parity in terminal.
