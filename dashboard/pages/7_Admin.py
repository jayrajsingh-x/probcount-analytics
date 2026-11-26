"""Page 7: Admin — Snapshots, Reset, and Exports."""

from __future__ import annotations

import streamlit as st
from dashboard.app import render_sidebar
from dashboard.data import (
    fetch_metrics,
    fetch_site_hll_bytes,
    metrics_to_csv,
    trigger_reset,
    trigger_snapshot,
)


def main() -> None:
    """Render Admin page."""
    api_url, _ = render_sidebar()
    st.title("🛠️ Admin & Exports")

    c1, c2 = st.columns(2)
    if c1.button("💾 Save Snapshot Now", use_container_width=True):
        res = trigger_snapshot(api_url)
        if res.ok:
            st.success(f"Saved snapshot: `{res.data.get('snapshot_id')}`")
        else:
            st.error(res.error)

    if c2.button("🔄 Reset Engine", use_container_width=True):
        res = trigger_reset(api_url)
        if res.ok:
            st.success("Engine state reset.")
            st.cache_data.clear()
        else:
            st.error(res.error)

    st.divider()
    st.subheader("Downloads")

    mt = fetch_metrics(api_url)
    if mt.ok:
        csv_text = metrics_to_csv(mt.data)
        st.download_button(
            "📥 Download Metrics CSV",
            data=csv_text,
            file_name="probcount_metrics.csv",
            mime="text/csv",
        )
    else:
        st.warning(mt.error)

    hll_bytes = fetch_site_hll_bytes(api_url)
    if hll_bytes is not None:
        st.download_button(
            "📥 Download Serialized Site HLL (.bin)",
            data=hll_bytes,
            file_name="site_hll.bin",
            mime="application/octet-stream",
        )


if __name__ == "__main__":
    main()
