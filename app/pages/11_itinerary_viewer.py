"""
pages/6_itinerary_viewer.py
============================
Itinerary Viewer — day-by-day display of the agent's final itinerary.

Renders EpisodeLog.final_itinerary using read-only card components styled
after the my-travel-world frontend (travel_world.frontend.components.flight_card
and hotel_card).  No interactive buttons — this is a post-hoc inspector, not a
booking UI.

Layout
------
- Summary row: days, total cost, flights/hotels/activities counts + costs
- Cost breakdown bar chart (flights / hotels / activities)
- Day-by-day timeline, each day in an expander:
    ✈  Transport segments (flight cards)
    🏨 Accommodation (hotel card)
    🎯 Activities and events (activity cards)
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import streamlit as st

st.set_page_config(page_title="Itinerary Viewer", layout="wide")
st.title("Itinerary Viewer")

from app.utils.data_loader import load_episode  # noqa: E402

# ── Episode selection ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Episode")
    default_id = st.session_state.get("selected_episode_id", "")
    ep_id_input = st.text_input("Episode ID or file path", value=default_id)

if not ep_id_input:
    st.info("Enter an episode ID in the sidebar, or select one from **Episode Browser**.")
    st.stop()

try:
    ep = load_episode(ep_id_input)
except FileNotFoundError:
    st.error(f"Episode not found: `{ep_id_input}`")
    st.stop()

it = ep.final_itinerary

if it is None:
    st.warning(
        f"Episode `{ep.episode_id}` produced no final itinerary.  \n"
        f"Termination reason: `{ep.termination_reason or 'unknown'}`."
    )
    st.stop()

if not it.days:
    st.info("The itinerary object exists but contains no booked days yet.")
    st.stop()

# ── Summary header ────────────────────────────────────────────────────────────

sorted_days = sorted(it.days, key=lambda d: d.date)
cities = list(dict.fromkeys(d.city for d in sorted_days if d.city))
n_flights = sum(len(d.transport_segments) for d in sorted_days)
n_hotels = sum(1 for d in sorted_days if d.accommodation)
n_activities = sum(len(d.activities) for d in sorted_days)

total_flights_cost = sum(
    seg.cost_usd for d in sorted_days for seg in d.transport_segments
)
total_hotels_cost = sum(
    d.accommodation.total_cost_usd for d in sorted_days if d.accommodation
)
total_activities_cost = sum(
    act.cost_usd for d in sorted_days for act in d.activities
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Days", len(sorted_days))
c2.metric("Total cost", f"${it.total_cost_usd:,.2f}")
c3.metric("✈️ Flights", f"${total_flights_cost:,.2f}")
c4.metric("🏨 Hotels", f"${total_hotels_cost:,.2f}")
c5.metric("🎯 Activities", f"${total_activities_cost:,.2f}")

if cities:
    st.caption("Route: " + " → ".join(cities))

complete_icon = "✅" if getattr(it, "is_complete", False) else "⚠️ partial"
st.caption(
    f"Itinerary status: {complete_icon}  ·  "
    f"Mode: `{ep.agent_mode}`  ·  "
    f"Steps: {ep.total_steps}  ·  "
    f"Termination: `{ep.termination_reason or 'unknown'}`"
)

# Cost breakdown chart
if any([total_flights_cost, total_hotels_cost, total_activities_cost]):
    import pandas as pd

    cost_df = pd.DataFrame(
        {
            "Category": ["Flights", "Hotels", "Activities"],
            "Cost (USD)": [total_flights_cost, total_hotels_cost, total_activities_cost],
        }
    )
    st.bar_chart(cost_df.set_index("Category"), height=160, use_container_width=True)

st.divider()

# ── Card renderers (read-only, styled after my-travel-world) ─────────────────


def _render_flight(seg) -> None:
    """
    Read-only flight card.

    Visual structure mirrors travel_world.frontend.components.flight_card
    but omits the interactive Select button — this is a completed booking
    display, not a search result.
    """
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 4, 2])
        with col1:
            st.markdown("**✈️ Flight**")
            mode_label = (seg.mode or "flight").upper()
            st.caption(mode_label)
            if seg.booking_ref:
                st.caption(f"Ref: `{seg.booking_ref}`")
        with col2:
            dep = (seg.departure_datetime or "")[:16].replace("T", " ")
            arr = (seg.arrival_datetime or "")[:16].replace("T", " ")
            st.markdown(f"🛫 **{seg.from_location}** → 🛬 **{seg.to_location}**")
            if dep:
                st.caption(f"Dep: `{dep}`  ·  Arr: `{arr}`")
        with col3:
            st.markdown(f"**${seg.cost_usd:,.2f}**")


def _render_hotel(acc) -> None:
    """
    Read-only hotel card.

    Visual structure mirrors travel_world.frontend.components.hotel_card
    but omits the bed selector and Book button.
    """
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.markdown(f"**🏨 {acc.hotel_name}**")
            if acc.city:
                st.caption(f"📍 {acc.city}")
            if acc.booking_ref:
                st.caption(f"Ref: `{acc.booking_ref}`")
        with col2:
            nights = 0
            try:
                from datetime import date as _date
                nights = (_date.fromisoformat(acc.check_out) - _date.fromisoformat(acc.check_in)).days
            except Exception:
                pass
            st.caption(f"Check-in: `{acc.check_in}`")
            st.caption(f"Check-out: `{acc.check_out}`")
            if nights:
                st.caption(f"{nights} night{'s' if nights != 1 else ''}")
        with col3:
            st.markdown(f"**${acc.total_cost_usd:,.2f}**")
            if acc.cost_per_night_usd:
                st.caption(f"${acc.cost_per_night_usd:,.2f}/night")


def _render_activity(act) -> None:
    """Read-only activity / event card."""
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            icon = "🎭" if getattr(act, "category", "") == "event" else "🎯"
            st.markdown(f"**{icon} {act.activity_name}**")
            if getattr(act, "location", ""):
                st.caption(f"📍 {act.location}")
            if act.booking_ref:
                st.caption(f"Ref: `{act.booking_ref}`")
        with col2:
            if getattr(act, "start_datetime", ""):
                st.caption(f"Start: `{act.start_datetime[:16].replace('T', ' ')}`")
            if getattr(act, "duration_hours", 0):
                st.caption(f"Duration: {act.duration_hours:.1f}h")
            if getattr(act, "category", ""):
                st.caption(f"Category: {act.category}")
        with col3:
            st.markdown(f"**${act.cost_usd:,.2f}**")


# ── Day-by-day timeline ───────────────────────────────────────────────────────

for day in sorted_days:
    day_label = f"{day.date} — {day.city}" if day.city else day.date
    has_content = day.transport_segments or day.accommodation or day.activities
    day_cost_label = f"  ·  ${day.total_cost_usd:,.2f}" if day.total_cost_usd else ""

    with st.expander(f"📅 {day_label}{day_cost_label}", expanded=True):
        if not has_content:
            st.caption("No bookings recorded for this day.")
            continue

        for seg in day.transport_segments:
            _render_flight(seg)

        if day.accommodation:
            _render_hotel(day.accommodation)

        for act in day.activities:
            _render_activity(act)
