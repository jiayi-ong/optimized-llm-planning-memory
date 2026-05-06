"""
app/pages/2_eval_episode_log.py
================================
Historical Evaluation Log — filterable table of all evaluated episodes.

Each row is a single EvalResult. Clicking a row copies the episode_id so the
user can navigate to Page 3 (Episode Detail) for a full drill-down.

Filters (sidebar)
-----------------
run_id              multiselect
agent_mode          multiselect
augmentation_id     multiselect
metric_version      multiselect
date range          date_input pair
archetype           multiselect (from UserRequest.metadata)
complexity_tier     multiselect (low / medium / high)
overall_score range slider
latest-per-config   toggle (deduplicate by eval_key)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.utils.ui_style import inject_css, score_color
from optimized_llm_planning_memory.evaluation.eval_store import EvalStore

BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = BASE_DIR / "outputs" / "eval_results"
EPISODES_DIR = BASE_DIR / "outputs" / "episodes"
REQUESTS_DIR = BASE_DIR / "data" / "user_requests"

st.set_page_config(page_title="Eval Episode Log", layout="wide", page_icon="📝")
inject_css()

store = EvalStore(EVAL_DIR, EPISODES_DIR, REQUESTS_DIR)


@st.cache_data(ttl=20)
def _load_flat(eval_dir: str) -> pd.DataFrame:
    """Load all eval results as a flat DataFrame with request metadata joined."""
    _store = EvalStore(Path(eval_dir), EPISODES_DIR, REQUESTS_DIR)
    df = _store.flat_results()
    if df.empty:
        return df

    # Attempt to join request metadata (archetype, complexity_tier)
    req_meta: dict[str, dict] = {}
    try:
        for req in _store.list_requests():
            meta = req.metadata or {}
            cb = meta.get("complexity_breakdown", {})
            req_meta[req.request_id] = {
                "archetype": meta.get("archetype", "unknown"),
                "complexity_tier": cb.get("complexity_tier", "?"),
                "n_hard": cb.get("n_hard_constraints", "?"),
                "n_soft": cb.get("n_soft_constraints", "?"),
                "n_destinations": cb.get("n_destinations", "?"),
                "duration_days": cb.get("trip_duration_days", "?"),
                "group_type": cb.get("group_type", "?"),
            }
    except Exception:
        pass

    if req_meta:
        meta_df = pd.DataFrame.from_dict(req_meta, orient="index").reset_index()
        meta_df.rename(columns={"index": "request_id"}, inplace=True)
        df = df.merge(meta_df, on="request_id", how="left")

    return df


raw_df = _load_flat(str(EVAL_DIR))

# ── Sidebar filters ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Filters")

    if raw_df.empty:
        st.info("No results loaded yet.")
    else:
        all_runs = sorted(raw_df["run_id"].dropna().unique().tolist(), reverse=True)
        sel_runs = st.multiselect("Run ID", ["(all)"] + all_runs, default=["(all)"])

        all_modes = sorted(raw_df["agent_mode"].dropna().unique().tolist())
        sel_modes = st.multiselect("Agent mode", all_modes, default=all_modes)

        all_augs = sorted(raw_df["augmentation_id"].dropna().unique().tolist())
        if all_augs:
            sel_augs = st.multiselect("Augmentation ID", ["(all)"] + all_augs, default=["(all)"])
        else:
            sel_augs = ["(all)"]

        all_versions = sorted(raw_df["metric_version"].dropna().unique().tolist())
        sel_versions = st.multiselect("Metric version", all_versions, default=all_versions)

        if "archetype" in raw_df.columns:
            all_archetypes = sorted(raw_df["archetype"].dropna().unique().tolist())
            sel_archetypes = st.multiselect("Archetype", ["(all)"] + all_archetypes, default=["(all)"])
        else:
            sel_archetypes = ["(all)"]

        if "complexity_tier" in raw_df.columns:
            sel_tiers = st.multiselect("Complexity tier", ["(all)", "low", "medium", "high"], default=["(all)"])
        else:
            sel_tiers = ["(all)"]

        score_range = st.slider("Overall score range", 0.0, 1.0, (0.0, 1.0), step=0.05)
        latest_only = st.checkbox("Latest per config only", value=True,
                                  help="Keep only the newest result per eval_key.")

        sort_by = st.selectbox("Sort by",
                               ["newest first", "overall_score ↓", "hard_constraint_ratio ↓"])

# ── Apply filters ─────────────────────────────────────────────────────────────

if raw_df.empty:
    st.info(
        "No evaluation results found. Run `python scripts/run_eval.py --all --deterministic_only` "
        "or use the **Eval Dashboard** control panel."
    )
    st.stop()

df = raw_df.copy()

if "(all)" not in sel_runs:
    df = df[df["run_id"].isin(sel_runs)]
if sel_modes:
    df = df[df["agent_mode"].isin(sel_modes)]
if "(all)" not in sel_augs and "augmentation_id" in df.columns:
    df = df[df["augmentation_id"].isin(sel_augs)]
if sel_versions:
    df = df[df["metric_version"].isin(sel_versions)]
if "(all)" not in sel_archetypes and "archetype" in df.columns:
    df = df[df["archetype"].isin(sel_archetypes)]
if "(all)" not in sel_tiers and "complexity_tier" in df.columns:
    df = df[df["complexity_tier"].isin(sel_tiers)]

df = df[
    (df["overall_score"] >= score_range[0]) &
    (df["overall_score"] <= score_range[1])
]

if latest_only and "eval_key" in df.columns and not df.empty:
    df = df.sort_values("created_at", ascending=False).drop_duplicates("eval_key", keep="first")

if sort_by == "newest first":
    df = df.sort_values("created_at", ascending=False)
elif sort_by == "overall_score ↓":
    df = df.sort_values("overall_score", ascending=False)
elif "hard_constraint_ratio ↓" in sort_by and "det_hard_constraint_ratio" in df.columns:
    df = df.sort_values("det_hard_constraint_ratio", ascending=False)

df = df.reset_index(drop=True)

# ── Main table ────────────────────────────────────────────────────────────────

st.markdown(f"## 📝 Episode Evaluation Log  <span style='color:#9090A8;font-size:14px'>({len(df)} results)</span>",
            unsafe_allow_html=True)

if len(raw_df["metric_version"].unique()) > 1 and "(all)" in sel_versions:
    st.warning(
        f"Mixed metric versions in selection: **{', '.join(sorted(raw_df['metric_version'].unique()))}**. "
        "Scores may not be directly comparable."
    )

display_cols = ["episode_id", "request_id", "agent_mode", "augmentation_id"]
if "archetype" in df.columns:
    display_cols.append("archetype")
if "complexity_tier" in df.columns:
    display_cols.append("complexity_tier")
if "det_hard_constraint_ratio" in df.columns:
    display_cols.append("det_hard_constraint_ratio")
display_cols += ["overall_score", "metric_version", "created_at"]

show_df = df[display_cols].copy()
show_df["created_at"] = show_df["created_at"].str[:19].str.replace("T", " ")
show_df["episode_id"] = show_df["episode_id"].str[:24]
show_df["request_id"] = show_df["request_id"].str[:28]

# Scrollable container for the table
with st.container(height=460):
    st.dataframe(
        show_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "overall_score": st.column_config.ProgressColumn(
                "Overall ↑", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "det_hard_constraint_ratio": st.column_config.ProgressColumn(
                "Hard Constr ↑", min_value=0.0, max_value=1.0, format="%.3f"
            ),
        },
    )

# ── Episode drill-down navigator ──────────────────────────────────────────────

st.markdown("---")
st.markdown("### 🔎 Open Episode Detail")
st.caption("Select an episode below to see full trajectory, itinerary, and eval scores.")

episode_options = df["episode_id"].str[:24].tolist()
if episode_options:
    sel_ep = st.selectbox("Episode ID", episode_options)
    if sel_ep:
        full_ep_id = df.loc[df["episode_id"].str[:24] == sel_ep, "episode_id"].values
        if len(full_ep_id):
            st.info(
                f"Navigate to **Eval Episode Detail** page and enter episode ID:  \n"
                f"`{full_ep_id[0]}`"
            )
