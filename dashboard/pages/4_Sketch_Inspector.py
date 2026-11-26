"""Page 4: Sketch Inspector — Teaching View of HLL Registers & CMS Table."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import streamlit as st
from dashboard.app import render_sidebar
from dashboard.data import fetch_sketch_inspect

from probcount.hashing import derive_seeds, hash64, split_hash


def main() -> None:
    """Render Sketch Inspector page."""
    api_url, _ = render_sidebar()
    st.title("🔬 Sketch Inspector (Teaching View)")

    res = fetch_sketch_inspect(api_url)
    if res.ok and res.data.get("hll_estimate", 0) > 0:
        data = res.data
        st.caption(
            f"Inspecting live site HLL (`p={data['hll_p']}`, `m={data['hll_m']:,}`) "
            f"and CMS (`{data['cms_depth']}×{data['cms_width']}`)"
        )
        hist = data["register_histogram"]
        first_64 = np.array(data["first_64_registers"]).reshape(8, 8)
        cms_slice = np.array(data["cms_slice"])

        c1, c2 = st.columns(2)
        with c1:
            fig_h = px.bar(
                x=[int(k) for k in hist],
                y=list(hist.values()),
                title="HLL Register Value Distribution (ρ)",
                labels={"x": "Register Value (ρ)", "y": "Count"},
            )
            st.plotly_chart(fig_h, use_container_width=True)

        with c2:
            fig_strip = px.imshow(
                first_64,
                text_auto=True,
                title="First 64 HLL Registers (8×8 Heat Strip)",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_strip, use_container_width=True)

        fig_cms = px.imshow(
            cms_slice,
            title="Count-Min Sketch Counter Table (First 32 Columns)",
            labels={"x": "Column Bucket", "y": "Hash Row", "color": "Count"},
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_cms, use_container_width=True)
    else:
        st.info("No live sketch data yet — run a simulation or use the step-by-step tracer below.")

    st.subheader("Step-by-Step Single-Event Trace")
    col_a, col_b, col_c = st.columns(3)
    vid = col_a.text_input("visitor_id", value="u_a3f8c1d9e2b0")
    pid = col_b.text_input("page_id", value="/products/42")
    p_trace = col_c.slider("HLL p", min_value=4, max_value=14, value=14, key="trace_p")

    h = hash64(vid, 0)
    idx, w = split_hash(h, p_trace)
    rho = (64 - p_trace) - w.bit_length() + 1
    cms_cols = [hash64(pid, s) % 1024 for s in derive_seeds(4, 0)]

    st.code(
        f"visitor_id = {vid!r}\n"
        f"  hash64 = 0x{h:016x} ({h:064b})\n"
        f"  register index (top {p_trace} bits) = {idx}\n"
        f"  rho (leading zeros in remaining {64 - p_trace} bits + 1) = {rho}\n\n"
        f"page_id = {pid!r}\n"
        f"  CMS (width=1024, depth=4) columns touched = {cms_cols}",
        language="text",
    )


if __name__ == "__main__":
    main()
