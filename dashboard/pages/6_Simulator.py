"""Page 6: Traffic Simulator."""

from __future__ import annotations

import streamlit as st
from dashboard.app import render_sidebar
from dashboard.data import trigger_simulation


def main() -> None:
    """Render Simulator page."""
    api_url, _ = render_sidebar()
    st.title("🚀 Synthetic Traffic Simulator")

    with st.form("sim_form"):
        c1, c2, c3 = st.columns(3)
        n_events = c1.number_input("Events", min_value=1_000, max_value=500_000, value=50_000, step=5_000)
        n_visitors = c2.number_input("Unique Visitors", min_value=100, max_value=200_000, value=10_000, step=1_000)
        n_pages = c3.number_input("Distinct Pages", min_value=5, max_value=1_000, value=50, step=5)

        c4, c5, c6 = st.columns(3)
        zipf_s = c4.slider("Page Zipf Skew (s)", min_value=0.5, max_value=2.0, value=1.1, step=0.1)
        bot_frac = c5.slider("Bot Fraction", min_value=0.0, max_value=0.5, value=0.05, step=0.05)
        spike_mult = c6.slider("Traffic Spike Multiplier", min_value=1.0, max_value=10.0, value=1.0, step=0.5)

        submitted = st.form_submit_button("Run Simulation")

    if submitted:
        bar = st.progress(10, text="Running traffic generator on API...")
        res = trigger_simulation(
            api_url,
            n_events=int(n_events),
            n_visitors=int(n_visitors),
            n_pages=int(n_pages),
            page_zipf_s=float(zipf_s),
            bot_fraction=float(bot_frac),
            spike_multiplier=float(spike_mult),
        )
        bar.progress(100, text="Complete")
        if res.ok:
            st.success(
                f"Generated **{res.data.get('events_generated', 0):,}** events in "
                f"**{res.data.get('elapsed_seconds', 0):.2f}s**."
            )
            st.cache_data.clear()
        else:
            st.error(res.error)


if __name__ == "__main__":
    main()
