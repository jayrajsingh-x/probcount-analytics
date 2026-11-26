"""Page 5: Merge Demo — Union & Inclusion-Exclusion Intersection."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from dashboard.app import render_sidebar

from probcount.hll import HyperLogLog
from probcount.setops import hll_intersection, hll_jaccard, hll_union


def main() -> None:
    """Render Merge Demo page."""
    render_sidebar()
    st.title("🔗 HLL Merge & Set Operations Demo")

    c1, c2, c3 = st.columns(3)
    size_a = c1.slider("Set A Size", min_value=1_000, max_value=20_000, value=10_000, step=1_000)
    size_b = c2.slider("Set B Size", min_value=1_000, max_value=20_000, value=10_000, step=1_000)
    overlap_pct = c3.slider("True Overlap (% of min(|A|,|B|))", min_value=0, max_value=100, value=25)

    n_shared = int(min(size_a, size_b) * overlap_pct / 100.0)
    shared = [f"shared_{i}" for i in range(n_shared)]
    set_a = shared + [f"a_{i}" for i in range(size_a - n_shared)]
    set_b = shared + [f"b_{i}" for i in range(size_b - n_shared)]

    hll_a = HyperLogLog(p=14, seed=42)
    hll_b = HyperLogLog(p=14, seed=42)
    hll_a.add_many(set_a)
    hll_b.add_many(set_b)

    true_union = size_a + size_b - n_shared
    u_res = hll_union(hll_a, hll_b)
    i_res = hll_intersection(hll_a, hll_b)
    j_res = hll_jaccard(hll_a, hll_b)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("True Union", f"{true_union:,}")
    m2.metric("HLL Union Estimate", f"{u_res.estimate:,.0f}")
    m3.metric("True Intersection", f"{n_shared:,}")
    m4.metric("Inclusion-Exclusion Est", f"{i_res.estimate:,.0f}", f"Jaccard {j_res.estimate:.3f}")

    if i_res.warning:
        st.warning(i_res.message)

    fig = go.Figure(
        data=[
            go.Bar(name="True", x=["Union", "Intersection"], y=[true_union, n_shared]),
            go.Bar(name="HLL Estimate", x=["Union", "Intersection"], y=[u_res.estimate, i_res.estimate]),
        ]
    )
    fig.update_layout(barmode="group", title="True vs. HLL Set Operations", yaxis_title="Visitors", height=360)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
