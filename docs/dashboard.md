# ProbCount Dashboard Guide & 5-Minute Demo Script

Launch the full demo (API + 500k seeded events + Streamlit UI):

```bash
make demo
# or: python scripts/run_demo.py
```

---

## Page-by-Page Guide

1. **Overview (`app.py`)**: Displays headline KPIs (total events, HLL unique visitors, CMS views, views/visitor, % memory saved) and an hourly unique-visitor time series with an Exact Ground-Truth overlay and $\pm 2\cdot\text{SE}$ confidence band.
2. **Top Pages**: Horizontal bar chart of top-$K$ pages with CMS estimates, time-range selector, and an exact-comparison error table.
3. **Accuracy Lab**: Scatter plot (`y = x`), relative error histogram, and hourly error vs. theoretical $1.04/\sqrt{m}$ bound.
4. **Parameter Playground**: Live sliders for HLL precision $p \in [4, 18]$ and CMS $(\varepsilon, \delta)$ on a synthetic Zipf stream, with a memory-vs-error trade-off curve.
5. **Sketch Inspector**: Teaching view showing the HLL $\rho$ distribution, first 64 registers as an $8 \times 8$ heat strip, CMS counter heatmap, and a single-event hash/index/$\rho$ tracer.
6. **Merge Demo**: Interactive two-set HLL merge showing exact union accuracy vs. inclusion-exclusion intersection (and its `<5%` overlap warning).
7. **Simulator**: Controls for event count, unique visitors, Zipf skew, bot fraction, and traffic spike calling `POST /admin/simulate`.
8. **Admin**: Save snapshots, reset engine, and download metrics CSV or serialized `.bin` site HLL.

---

## 5-Minute Live Examiner Demo Script

1. **[0:00–1:00] Overview**: Point to the 500k events seeded into the engine. Highlight **Memory Saved (>95%)** — kilobytes of sketches vs. tens of megabytes for exact Python sets/Counters — and toggle the **Exact Overlay** to show the HLL curve hugging ground truth inside the shaded band.
2. **[1:00–2:00] Top Pages & Accuracy Lab**: Open **Top Pages**, adjust $K$, and show the `rel_error` column is typically `< 1%`. Switch to **Accuracy Lab** to show points hugging `y = x` and errors staying within the theoretical `3·SE` line.
3. **[2:00–3:00] Sketch Inspector**: Show the 64-register HLL heat strip and CMS heatmap. Type a `visitor_id` into the **Step-by-Step Trace** to show the 64-bit MurmurHash3 split into bucket index and leading-zero rank $\rho$.
4. **[3:00–4:00] Parameter Playground & Merge Demo**: Drag `p` from `14` down to `8` to show memory dropping while error rises on the trade-off curve. Open **Merge Demo**, set overlap to `2%`, and point out the inclusion-exclusion warning banner.
5. **[4:00–5:00] Simulator & Admin**: Trigger a `5x` traffic spike in **Simulator**, then open **Admin** to save a durable SHA-256 snapshot and download the 12 KB serialized site HLL.
