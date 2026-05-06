"""
app/pages/5_request_browser.py
================================
Request Browser — inspect the user request population.

Panels
------
1. Distribution charts  — complexity tier, budget tier, archetype, group type.
2. Filterable table     — all requests with key attributes.
3. Request detail       — formatted request card + full constraint breakdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.utils.ui_style import PALETTE, PLOTLY_LAYOUT, PRIMARY, SUCCESS, WARNING, inject_css, sidebar_header
from optimized_llm_planning_memory.evaluation.eval_store import EvalStore

BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = BASE_DIR / "outputs" / "eval_results"
EPISODES_DIR = BASE_DIR / "outputs" / "episodes"
REQUESTS_DIR = BASE_DIR / "data" / "user_requests"

st.set_page_config(page_title="Request Browser", layout="wide", page_icon="📂")
inject_css()

store = EvalStore(EVAL_DIR, EPISODES_DIR, REQUESTS_DIR)


@st.cache_data(ttl=60)
def _load_requests(requests_dir: str) -> pd.DataFrame:
    _store = EvalStore(EVAL_DIR, EPISODES_DIR, Path(requests_dir))
    requests = _store.list_requests()
    rows = []
    for req in requests:
        meta = req.metadata or {}
        cb = meta.get("complexity_breakdown", {})
        rows.append({
            "request_id": req.request_id,
            "archetype": meta.get("archetype", "?"),
            "tone": meta.get("tone_variant", "?"),
            "complexity_tier": cb.get("complexity_tier", "?"),
            "complexity_score": cb.get("complexity_score", None),
            "n_hard": cb.get("n_hard_constraints", len(req.hard_constraints)),
            "n_soft": cb.get("n_soft_constraints", len(req.soft_constraints)),
            "n_destinations": cb.get("n_destinations", len(req.destination_cities)),
            "duration_days": cb.get("trip_duration_days", "?"),
            "budget_usd": req.budget_usd,
            "budget_ppd": cb.get("budget_per_traveler_per_day", None),
            "group_type": cb.get("group_type", "?"),
            "geographic_complexity": cb.get("geographic_complexity", "?"),
            "has_special_needs": cb.get("has_special_needs", False),
            "world_id": req.world_id or "?",
            "_req": req,
        })
    return pd.DataFrame(rows)


all_req_df = _load_requests(str(REQUESTS_DIR))

# ── Sidebar filters ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Filters")
    sidebar_header("Request Attributes")
    if not all_req_df.empty:
        all_archetypes = sorted(all_req_df["archetype"].dropna().unique().tolist())
        sel_archetypes = st.multiselect("Archetype", ["(all)"] + all_archetypes, default=["(all)"])
        sel_tiers = st.multiselect("Complexity tier", ["(all)", "low", "medium", "high"], default=["(all)"])
        all_groups = sorted(all_req_df["group_type"].dropna().unique().tolist())
        sel_groups = st.multiselect("Group type", ["(all)"] + all_groups, default=["(all)"])
        sel_geo = st.multiselect("Geographic complexity", ["(all)", "single_city", "multi_city"], default=["(all)"])
        budget_range = st.slider("Budget (USD)", 0, 25000, (0, 25000), step=500)


if all_req_df.empty:
    st.info(
        "No user requests found under `data/user_requests/`. "
        "Generate them with `python scripts/generate_user_requests.py`."
    )
    st.stop()

# ── Apply filters ─────────────────────────────────────────────────────────────

df = all_req_df.copy()
if "(all)" not in sel_archetypes:
    df = df[df["archetype"].isin(sel_archetypes)]
if "(all)" not in sel_tiers:
    df = df[df["complexity_tier"].isin(sel_tiers)]
if "(all)" not in sel_groups:
    df = df[df["group_type"].isin(sel_groups)]
if "(all)" not in sel_geo:
    df = df[df["geographic_complexity"].isin(sel_geo)]
df = df[(df["budget_usd"] >= budget_range[0]) & (df["budget_usd"] <= budget_range[1])]
df = df.reset_index(drop=True)

st.markdown(f"## 📂 Request Browser  <span style='color:#9090A8;font-size:14px'>({len(df)} requests)</span>",
            unsafe_allow_html=True)
st.caption(
    "Explore the user request population: complexity distribution charts, "
    "filterable request table, and per-request constraint breakdown."
)

# ── Distribution charts ───────────────────────────────────────────────────────

try:
    import plotly.express as px
    import plotly.graph_objects as go

    chart_cols = st.columns(4)

    with chart_cols[0]:
        tier_counts = df["complexity_tier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        fig = px.bar(tier_counts, x="tier", y="count", title="Complexity Tier",
                     color="tier",
                     color_discrete_map={"low": "#4CAF50", "medium": "#FFC107", "high": "#E84545"},
                     height=240)
        fig.update_layout(showlegend=False, **{**PLOTLY_LAYOUT, "margin": {"t": 30, "b": 10}})
        st.plotly_chart(fig, use_container_width=True)

    with chart_cols[1]:
        arch_counts = df["archetype"].value_counts().reset_index()
        arch_counts.columns = ["archetype", "count"]
        fig2 = px.pie(arch_counts, names="archetype", values="count", title="Archetype",
                      color_discrete_sequence=PALETTE, height=240)
        fig2.update_layout(**{**PLOTLY_LAYOUT, "margin": {"t": 30, "b": 10}})
        st.plotly_chart(fig2, use_container_width=True)

    with chart_cols[2]:
        grp_counts = df["group_type"].value_counts().reset_index()
        grp_counts.columns = ["group_type", "count"]
        fig3 = px.bar(grp_counts, x="group_type", y="count", title="Group Type",
                      color_discrete_sequence=PALETTE, height=240)
        fig3.update_layout(showlegend=False, **{**PLOTLY_LAYOUT, "margin": {"t": 30, "b": 10}})
        st.plotly_chart(fig3, use_container_width=True)

    with chart_cols[3]:
        geo_counts = df["geographic_complexity"].value_counts().reset_index()
        geo_counts.columns = ["geo", "count"]
        fig4 = px.pie(geo_counts, names="geo", values="count", title="Geographic",
                      color_discrete_sequence=PALETTE, height=240)
        fig4.update_layout(**{**PLOTLY_LAYOUT, "margin": {"t": 30, "b": 10}})
        st.plotly_chart(fig4, use_container_width=True)

except ImportError:
    pass

# ── Request table ─────────────────────────────────────────────────────────────

st.markdown("### All Requests")

display_df = df[[
    "request_id", "archetype", "complexity_tier", "n_hard", "n_soft",
    "n_destinations", "duration_days", "budget_usd", "group_type",
    "geographic_complexity", "has_special_needs",
]].copy()
display_df["request_id"] = display_df["request_id"].str[:28]
display_df["budget_usd"] = display_df["budget_usd"].apply(lambda v: f"${v:,.0f}")

with st.container(height=400):
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Request detail ────────────────────────────────────────────────────────────

st.markdown("### Request Detail")
req_options = df["request_id"].str[:28].tolist()
if not req_options:
    st.stop()

sel_req_short = st.selectbox("Select a request", req_options)
matching = df[df["request_id"].str[:28] == sel_req_short]
if matching.empty:
    st.stop()

req_obj = matching.iloc[0]["_req"]
cb = (req_obj.metadata or {}).get("complexity_breakdown", {})
tier = cb.get("complexity_tier", "?")
tier_color = {"low": SUCCESS, "medium": WARNING, "high": PRIMARY}.get(tier, "#9090A8")

st.markdown(
    f"**{req_obj.request_id}**  &nbsp; "
    f"<span style='background:{tier_color};color:#fff;padding:2px 7px;border-radius:4px;font-size:12px'>"
    f"{tier.upper()}</span>",
    unsafe_allow_html=True,
)
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"**Archetype:** {req_obj.metadata.get('archetype','?')}")
    st.markdown(f"**Trip:** {req_obj.start_date} → {req_obj.end_date}  ({cb.get('trip_duration_days','?')} days)")
    st.markdown(f"**Budget:** ${req_obj.budget_usd:,.0f}")
    st.markdown(f"**Group:** {req_obj.traveler_profile.num_adults} adult(s), {req_obj.traveler_profile.num_children} child(ren)")
    if cb:
        st.markdown(
            f"**Complexity score:** {cb.get('complexity_score','?')}  \n"
            f"Constraint density: {cb.get('constraint_density','?')} | "
            f"Geographic: {cb.get('geographic_complexity','?')} | "
            f"Budget tightness: {cb.get('budget_tightness_score','?'):.3f}"
        )

with col_b:
    st.markdown("**Raw request text:**")
    with st.container(height=160):
        st.write(req_obj.raw_text)

with st.expander(f"Hard constraints ({len(req_obj.hard_constraints)})"):
    for c in req_obj.hard_constraints:
        st.markdown(f"- **[{c.category}]** {c.description} *(value: {c.value} {c.unit or ''})*")

with st.expander(f"Soft constraints ({len(req_obj.soft_constraints)})"):
    for c in req_obj.soft_constraints:
        st.markdown(f"- **[{c.category}]** {c.description} *(value: {c.value} {c.unit or ''})*")
