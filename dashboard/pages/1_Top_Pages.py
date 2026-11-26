"""Page 1: Top-K Heavy-Hitter Pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard.app import render_sidebar
from dashboard.data import fetch_metrics, fetch_top_pages


def main() -> None:
    """Render Top Pages view."""
    api_url, _ = render_sidebar()
    st.title("🏆 Top Pages (Count-Min Sketch + Min-Heap)")

    c1, c2, c3 = st.columns([2, 2, 1])
    k = c1.slider("Top K", min_value=5, max_value=50, value=15, step=5)
    window_label = c2.selectbox("Time Range", ["All Data", "Last 24 Hours", "Last Hour"])
    compare = c3.checkbox("Compare Exact", value=True)

    start: float | None = None
    end: float | None = None
    if window_label != "All Data":
        mt = fetch_metrics(api_url)
        if mt.ok:
            import time

            now = time.time()
            span = 3600.0 if window_label == "Last Hour" else 86_400.0
            start, end = now - span, now + 1.0

    res = fetch_top_pages(api_url, k=k, start=start, end=end, compare=compare)
    if not res.ok:
        st.warning(res.error)
        return

    items = res.data.get("items", [])
    if not items:
        st.info("No page views found in the selected range.")
        return

    df = pd.DataFrame(items)
    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        fig = px.bar(
            df.iloc[::-1],
            x="views",
            y="page",
            orientation="h",
            title=f"Top {len(df)} Pages by CMS Estimated Views",
            labels={"views": "Estimated Views", "page": "Page Path"},
        )
        fig.update_layout(height=450, margin={"l": 20, "r": 20, "t": 50, "b": 40})
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.subheader("Rankings & Error")
        st.dataframe(df, use_container_width=True, height=420)


if __name__ == "__main__":
    main()
