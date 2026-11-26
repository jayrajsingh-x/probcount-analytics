"""ProbCount Interactive Analytics Dashboard — Overview & Shared Sidebar."""

from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import (
    compute_memory_saved_pct,
    fetch_health,
    fetch_metrics,
    fetch_overview,
    fetch_series,
)

st.set_page_config(
    page_title="ProbCount Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar() -> tuple[str, str]:
    """Render shared sidebar with API URL, connection status, and auto-refresh."""
    default_url = os.environ.get("PROBCOUNT_API_URL", "http://127.0.0.1:8000")
    api_url = st.sidebar.text_input("API URL", value=default_url, key="api_url")
    health = fetch_health(api_url)
    if health.ok:
        uptime = health.data.get("uptime_seconds", 0.0)
        st.sidebar.success(f"● Connected (uptime {uptime:.0f}s)")
    else:
        st.sidebar.error("● Offline — start uvicorn probcount.api:app")

    refresh_choice = st.sidebar.selectbox(
        "Auto-refresh",
        options=["Off", "5s", "15s", "30s"],
        index=0,
        key="auto_refresh",
    )
    if refresh_choice != "Off":
        secs = int(refresh_choice.rstrip("s"))
        st.sidebar.caption(f"Refreshing every {secs}s")
        time.sleep(0.01)

    return api_url, refresh_choice


@st.cache_data(ttl=5)
def _cached_overview(api_url: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    ov = fetch_overview(api_url)
    mt = fetch_metrics(api_url)
    err = ov.error or mt.error
    return ov.data, mt.data, err


@st.cache_data(ttl=5)
def _cached_series(api_url: str, compare: bool) -> tuple[list[dict[str, Any]], str]:
    res = fetch_series(api_url, granularity="hour", compare=compare)
    return list(res.data.get("points", [])), res.error


def main() -> None:
    """Render the Overview page."""
    api_url, _ = render_sidebar()
    st.title("📊 ProbCount — Overview")
    st.caption("Real-time probabilistic web analytics powered by HyperLogLog & Count-Min Sketch")

    ov_data, mt_data, err = _cached_overview(api_url)
    if err:
        st.warning(f"Unable to load overview data: {err}")
        return

    total_events = int(ov_data.get("total_events", 0) or 0)
    if total_events == 0:
        st.info("No events ingested yet. Visit **Simulator** or run `make demo` to seed traffic.")
        return

    unique_uv = float(ov_data.get("unique_visitors", 0.0) or 0.0)
    views_per_visitor = total_events / unique_uv if unique_uv > 0 else 0.0
    mem = mt_data.get("memory", {}) if isinstance(mt_data.get("memory"), dict) else {}
    saved_pct = compute_memory_saved_pct(mem)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Events", f"{total_events:,}")
    c2.metric("Unique Visitors (HLL)", f"{unique_uv:,.0f}")
    c3.metric("Page Views (CMS)", f"{total_events:,}")
    c4.metric("Views / Visitor", f"{views_per_visitor:.2f}")
    c5.metric("Memory Saved", f"{saved_pct:.2f}%")

    sketch_kb = float(mem.get("total_bytes", 0)) / 1024.0
    exact_mb = float(mem.get("exact_bytes", 0)) / (1024.0 * 1024.0)
    st.caption(
        f"Sketch memory: **{sketch_kb:,.1f} KB** vs Exact engine: **{exact_mb:,.2f} MB**"
    )

    overlay_exact = st.toggle("Overlay Exact Line & ±SE Band", value=True)
    points, s_err = _cached_series(api_url, compare=overlay_exact)
    if s_err:
        st.warning(s_err)
        return
    if not points:
        st.info("No time-series points available yet.")
        return

    df = pd.DataFrame(points)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s")
    sketch_params = mt_data.get("sketch_params") or {}
    hll_p = int(sketch_params.get("hll_p", 14)) if isinstance(sketch_params, dict) else 14
    se = 1.04 / (1 << hll_p) ** 0.5
    df["upper"] = df["unique_visitors"] * (1.0 + 2 * se)
    df["lower"] = (df["unique_visitors"] * (1.0 - 2 * se)).clip(lower=0.0)

    fig = go.Figure()
    if overlay_exact:
        fig.add_trace(
            go.Scatter(
                x=df["dt"],
                y=df["upper"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["dt"],
                y=df["lower"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(31, 119, 180, 0.18)",
                name="±2·SE Band",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=df["dt"],
            y=df["unique_visitors"],
            mode="lines+markers",
            name="Sketch (HLL)",
            line={"color": "#1f77b4", "width": 2},
        )
    )
    if overlay_exact and "exact_unique_visitors" in df.columns and df["exact_unique_visitors"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["dt"],
                y=df["exact_unique_visitors"],
                mode="lines",
                name="Exact (Ground Truth)",
                line={"color": "#d62728", "dash": "dash", "width": 2},
            )
        )

    fig.update_layout(
        title="Hourly Unique Visitors — HyperLogLog vs. Exact",
        xaxis_title="Time (UTC)",
        yaxis_title="Unique Visitors",
        height=400,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
