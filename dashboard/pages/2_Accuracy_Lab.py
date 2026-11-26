"""Page 2: Accuracy Lab — Sketch vs Exact Error Analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dashboard.app import render_sidebar
from dashboard.data import fetch_metrics, fetch_series


def main() -> None:
    """Render Accuracy Lab page."""
    api_url, _ = render_sidebar()
    st.title("🎯 Accuracy Lab — Sketch vs. Exact")

    mt = fetch_metrics(api_url)
    res = fetch_series(api_url, granularity="hour", compare=True)
    if not res.ok:
        st.warning(res.error)
        return

    pts = res.data.get("points", [])
    df = pd.DataFrame(pts)
    if df.empty or "exact_unique_visitors" not in df.columns or df["exact_unique_visitors"].isna().all():
        st.info("Enable `PROBCOUNT_ENABLE_EXACT=true` and simulate events to inspect exact error.")
        return

    df = df[df["exact_unique_visitors"] > 0].copy()
    df["rel_err_pct"] = (
        np.abs(df["unique_visitors"] - df["exact_unique_visitors"])
        / df["exact_unique_visitors"]
        * 100.0
    )
    hll_p = int(mt.data.get("sketch_params", {}).get("hll_p", 14)) if mt.ok else 14
    se_pct = (1.04 / (1 << hll_p) ** 0.5) * 100.0

    c1, c2 = st.columns(2)
    with c1:
        max_v = float(max(df["exact_unique_visitors"].max(), df["unique_visitors"].max()))
        fig_sc = go.Figure()
        fig_sc.add_trace(
            go.Scatter(
                x=df["exact_unique_visitors"],
                y=df["unique_visitors"],
                mode="markers",
                name="Hourly Buckets",
            )
        )
        fig_sc.add_trace(
            go.Scatter(
                x=[0, max_v],
                y=[0, max_v],
                mode="lines",
                name="y = x (Perfect)",
                line={"color": "red", "dash": "dash"},
            )
        )
        fig_sc.update_layout(
            title="Sketch Estimate vs. Exact Ground Truth",
            xaxis_title="Exact Unique Visitors",
            yaxis_title="HLL Estimate",
            height=360,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with c2:
        fig_hist = px.histogram(
            df,
            x="rel_err_pct",
            nbins=20,
            title="Relative Error Distribution (%)",
            labels={"rel_err_pct": "Relative Error (%)"},
        )
        fig_hist.add_vline(
            x=se_pct,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Theory SE ({se_pct:.2f}%)",
        )
        fig_hist.update_layout(height=360)
        st.plotly_chart(fig_hist, use_container_width=True)

    df["dt"] = pd.to_datetime(df["timestamp"], unit="s")
    fig_run = go.Figure()
    fig_run.add_trace(
        go.Scatter(x=df["dt"], y=df["rel_err_pct"], mode="lines+markers", name="Observed Rel Error (%)")
    )
    fig_run.add_hline(y=se_pct, line_dash="dash", line_color="red", annotation_text=f"Theory SE ({se_pct:.2f}%)")
    fig_run.add_hline(y=3 * se_pct, line_dash="dot", line_color="orange", annotation_text="3·SE Bound")
    fig_run.update_layout(
        title="Hourly Relative Error vs. Theoretical Bound",
        xaxis_title="Time (UTC)",
        yaxis_title="Relative Error (%)",
        height=320,
    )
    st.plotly_chart(fig_run, use_container_width=True)


if __name__ == "__main__":
    main()
