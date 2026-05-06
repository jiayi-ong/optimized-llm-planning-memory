"""
app/pages/2_eval_episode_log.py
================================
Historical Evaluation Log — filterable table of all evaluated episodes.

Sidebar filters are organised into five hierarchical sections that match the
natural experiment hierarchy:

  RUN LEVEL       run_name, run_id, date range
  CONDITION LEVEL agent_mode, augmentation_id, prompt_id, metric_version
  WORLD LEVEL     world_id (when present in results)
  REQUEST LEVEL   archetype, complexity_tier, group_type
  EPISODE FILTERS overall_score range, deduplication, sort order
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.utils.ui_style import inject_css, score_color, sidebar_header
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
        # ── RUN LEVEL ────────────────────────────────────────────────────────
        sidebar_header("Run Level")

        if "run_name" in raw_df.columns:
            all_run_names = sorted(raw_df["run_name"].dropna().unique().tolist())
            sel_run_names = st.multiselect(
                "Run name", ["(all)"] + all_run_names, default=["(all)"]
            )
        else:
            sel_run_names = ["(all)"]

        all_runs = sorted(raw_df["run_id"].dropna().unique().tolist(), reverse=True)
        sel_runs = st.multiselect("Run ID", ["(all)"] + all_runs, default=["(all)"])

        date_col1, date_col2 = st.columns(2)
        import datetime as _dt
        date_min = date_col1.date_input("From", value=_dt.date(2025, 1, 1))
        date_max = date_col2.date_input("To", value=_dt.date(2030, 12, 31))

        # ── CONDITION LEVEL ──────────────────────────────────────────────────
        sidebar_header("Condition Level")

        all_modes = sorted(raw_df["agent_mode"].dropna().unique().tolist())
        sel_modes = st.multiselect("Agent mode", all_modes, default=all_modes)

        all_augs = sorted(raw_df["augmentation_id"].dropna().unique().tolist())
        if all_augs:
            sel_augs = st.multiselect("Augmentation ID", ["(all)"] + all_augs, default=["(all)"])
        else:
            sel_augs = ["(all)"]

        if "prompt_id" in raw_df.columns:
            all_prompts = sorted(raw_df["prompt_id"].dropna().unique().tolist())
            sel_prompts = st.multiselect("Prompt ID", ["(all)"] + all_prompts, default=["(all)"])
        else:
            sel_prompts = ["(all)"]

        all_versions = sorted(raw_df["metric_version"].dropna().unique().tolist())
        sel_versions = st.multiselect("Metric version", all_versions, default=all_versions)

        # ── WORLD LEVEL ──────────────────────────────────────────────────────
        if "world_seed" in raw_df.columns:
            sidebar_header("World Level")
            all_seeds = sorted(
                [str(s) for s in raw_df["world_seed"].dropna().unique().tolist()]
            )
            if all_seeds:
                sel_seeds = st.multiselect("World seed", ["(all)"] + all_seeds, default=["(all)"])
            else:
                sel_seeds = ["(all)"]
        else:
            sel_seeds = ["(all)"]

        # ── REQUEST LEVEL ────────────────────────────────────────────────────
        sidebar_header("Request Level")

        if "archetype" in raw_df.columns:
            all_archetypes = sorted(raw_df["archetype"].dropna().unique().tolist())
            sel_archetypes = st.multiselect(
                "Archetype", ["(all)"] + all_archetypes, default=["(all)"]
            )
        else:
            sel_archetypes = ["(all)"]

        if "complexity_tier" in raw_df.columns:
            sel_tiers = st.multiselect(
                "Complexity tier", ["(all)", "low", "medium", "high"], default=["(all)"]
            )
        else:
            sel_tiers = ["(all)"]

        if "group_type" in raw_df.columns:
            all_groups = sorted(raw_df["group_type"].dropna().unique().tolist())
            sel_groups = st.multiselect("Group type", ["(all)"] + all_groups, default=["(all)"])
        else:
            sel_groups = ["(all)"]

        # ── EPISODE FILTERS ──────────────────────────────────────────────────
        sidebar_header("Episode Filters")

        score_range = st.slider("Overall score range", 0.0, 1.0, (0.0, 1.0), step=0.05)
        latest_only = st.checkbox(
            "Latest per config only", value=True,
            help="Keep only the newest result per eval_key (deduplicates re-runs).",
        )
        sort_by = st.selectbox(
            "Sort by", ["newest first", "overall_score ↓", "hard_constraint_ratio ↓"]
        )


# ── Apply filters ─────────────────────────────────────────────────────────────

if raw_df.empty:
    st.info(
        "No evaluation results found. Run `python scripts/run_eval.py --all --deterministic_only` "
        "or use the **Eval Dashboard** control panel."
    )
    st.stop()

df = raw_df.copy()

if "run_name" in df.columns and "(all)" not in sel_run_names:
    df = df[df["run_name"].isin(sel_run_names)]
if "(all)" not in sel_runs:
    df = df[df["run_id"].isin(sel_runs)]
if sel_modes:
    df = df[df["agent_mode"].isin(sel_modes)]
if "(all)" not in sel_augs and "augmentation_id" in df.columns:
    df = df[df["augmentation_id"].isin(sel_augs)]
if "(all)" not in sel_prompts and "prompt_id" in df.columns:
    df = df[df["prompt_id"].isin(sel_prompts)]
if sel_versions:
    df = df[df["metric_version"].isin(sel_versions)]
if "(all)" not in sel_seeds and "world_seed" in df.columns:
    df = df[df["world_seed"].astype(str).isin(sel_seeds)]
if "(all)" not in sel_archetypes and "archetype" in df.columns:
    df = df[df["archetype"].isin(sel_archetypes)]
if "(all)" not in sel_tiers and "complexity_tier" in df.columns:
    df = df[df["complexity_tier"].isin(sel_tiers)]
if "(all)" not in sel_groups and "group_type" in df.columns:
    df = df[df["group_type"].isin(sel_groups)]

df = df[
    (df["overall_score"] >= score_range[0]) &
    (df["overall_score"] <= score_range[1])
]

# Date filter on created_at
if "created_at" in df.columns:
    try:
        df = df[df["created_at"].str[:10] >= str(date_min)]
        df = df[df["created_at"].str[:10] <= str(date_max)]
    except Exception:
        pass

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

st.markdown(
    f"## 📝 Episode Evaluation Log  "
    f"<span style='color:#9090A8;font-size:14px'>({len(df)} results)</span>",
    unsafe_allow_html=True,
)

if len(raw_df["metric_version"].unique()) > 1 and all(v in sel_versions for v in raw_df["metric_version"].unique()):
    st.warning(
        f"Mixed metric versions in selection: **{', '.join(sorted(raw_df['metric_version'].unique()))}**. "
        "Scores may not be directly comparable."
    )

display_cols = ["episode_id", "request_id", "agent_mode", "augmentation_id"]
if "run_name" in df.columns:
    display_cols.insert(0, "run_name")
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

col_cfg: dict = {
    "overall_score": st.column_config.ProgressColumn(
        "Overall ↑", min_value=0.0, max_value=1.0, format="%.3f"
    ),
}
if "det_hard_constraint_ratio" in show_df.columns:
    col_cfg["det_hard_constraint_ratio"] = st.column_config.ProgressColumn(
        "Hard Constr ↑", min_value=0.0, max_value=1.0, format="%.3f"
    )

with st.container(height=460):
    st.dataframe(show_df, use_container_width=True, hide_index=True, column_config=col_cfg)

# ── Episode drill-down navigator ──────────────────────────────────────────────

st.markdown("---")
st.markdown("### 🔎 Open Episode Detail")
st.caption("Select an episode below, then navigate to the **Eval Episode Detail** page.")

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
