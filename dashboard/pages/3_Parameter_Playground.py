"""Page 3: Parameter Playground — Interactive Memory vs Error Trade-Off."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from dashboard.app import render_sidebar

from probcount.cms import CountMinSketch
from probcount.hll import HyperLogLog


def main() -> None:
    """Render Parameter Playground page."""
    render_sidebar()
    st.title("🎛️ Parameter Playground")
    st.caption("Experiment with local HLL precision `p` and CMS `(epsilon, delta)` on a live synthetic stream.")

    col_ctrl, col_viz = st.columns([1, 2])
    with col_ctrl:
        p = st.slider("HLL Precision (p)", min_value=4, max_value=18, value=14)
        eps = st.select_slider("CMS Epsilon", options=[0.05, 0.02, 0.01, 0.005, 0.002, 0.001], value=0.005)
        delta = st.select_slider("CMS Delta", options=[0.1, 0.05, 0.01, 0.001], value=0.01)
        n_stream = st.slider("Generated Stream Size", min_value=1_000, max_value=20_000, value=5_000, step=1_000)

    rng = np.random.default_rng(42)
    items = [f"u_{int(x)}" for x in rng.zipf(1.2, size=n_stream)]
    true_unique = len(set(items))

    hll = HyperLogLog(p=p, seed=42)
    hll.add_many(items)
    est = hll.count()
    rel_err = abs(est - true_unique) / true_unique * 100.0

    cms = CountMinSketch.from_error(epsilon=eps, delta=delta, seed=42, conservative=True)
    cms.add_many(items)

    with col_viz:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("True Unique", f"{true_unique:,}")
        m2.metric("HLL Estimate", f"{est:,.0f}")
        m3.metric("Observed Error", f"{rel_err:.2f}%")
        m4.metric("HLL Packed Size", f"{hll.memory_bytes() / 1024:.2f} KB")

        st.caption(
            f"CMS dimensions: **{cms.depth} × {cms.width}** "
            f"({cms.memory_bytes() / 1024:.1f} KB, error bound ±{cms.error_bound():.0f} views)"
        )

        ps = list(range(4, 19))
        mem_kb = [math.ceil((1 << px) * 6 / 8) / 1024.0 for px in ps]
        se_pct = [1.04 / math.sqrt(1 << px) * 100.0 for px in ps]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=mem_kb, y=se_pct, mode="lines+markers", text=[f"p={px}" for px in ps], name="HLL Curve")
        )
        fig.add_trace(
            go.Scatter(
                x=[hll.memory_bytes() / 1024.0],
                y=[hll.standard_error() * 100.0],
                mode="markers",
                marker={"size": 14, "color": "red"},
                name=f"Selected p={p}",
            )
        )
        fig.update_layout(
            title="HLL Trade-Off Curve: Memory (KB) vs. Standard Error (%)",
            xaxis_type="log",
            yaxis_type="log",
            xaxis_title="Packed Memory (KB)",
            yaxis_title="Standard Error (%)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
