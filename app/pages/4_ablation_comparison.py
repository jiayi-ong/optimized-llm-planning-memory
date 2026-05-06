"""
app/pages/4_ablation_comparison.py
====================================
Ablation Study — side-by-side metric comparison across conditions.

The three core ablation axes for this project:
  1. Agent mode: raw vs llm_summary vs compressor vs mcts_compressor
  2. Compressor checkpoint: init vs trained (by augmentation_id)
  3. Performance vs complexity tier or archetype

Panels
------
1. Dimension selector    — prominent dropdown on the main page to choose what
                           to compare across (agent_mode, augmentation_id, …).
2. Condition selection   — sidebar multiselect, dynamically filtered to the
                           chosen dimension.
3. Metric table          — mean ± std per metric per condition.
4. Bar chart             — overall_score per condition with error bars.
5. Radar chart           — 6 key metrics per condition.
6. Complexity breakdown  — overall_score by complexity_tier for each condition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.utils.ui_style import PALETTE, PLOTLY_LAYOUT, inject_css, sidebar_header
from optimized_llm_planning_memory.evaluation.eval_store import EvalStore

BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = BASE_DIR / "outputs" / "eval_results"
EPISODES_DIR = BASE_DIR / "outputs" / "episodes"
REQUESTS_DIR = BASE_DIR / "data" / "user_requests"

st.set_page_config(page_title="Ablation Comparison", layout="wide", page_icon="📊")
inject_css()

store = EvalStore(EVAL_DIR, EPISODES_DIR, REQUESTS_DIR)

KEY_METRICS = [
    "hard_constraint_ratio", "soft_constraint_score", "budget_adherence",
    "logical_consistency", "tool_efficiency", "destination_coverage_ratio",
]

# Human-readable labels for each axis the user can compare across
AXIS_LABELS: dict[str, str] = {
    "agent_mode":       "Agent Mode  (raw vs llm_summary vs compressor vs mcts)",
    "augmentation_id":  "Augmentation / Checkpoint  (init vs trained)",
    "run_name":         "Experiment Name  (run_name grouping)",
    "run_id":           "Run ID  (exact eval invocation)",
    "prompt_id":        "Prompt Variant  (prompt_id)",
    "world_seed":       "World Seed  (cross-seed robustness)",
    "complexity_tier":  "Complexity Tier  (low vs medium vs high)",
    "archetype":        "Request Archetype  (solo_adventurer vs family_trip vs …)",
}

# ── Load all results ──────────────────────────────────────────────────────────

@st.cache_data(ttl=20)
def _all_results(eval_dir: str) -> pd.DataFrame:
    _store = EvalStore(Path(eval_dir), EPISODES_DIR, REQUESTS_DIR)
    df = _store.flat_results()
    if df.empty:
        return df
    req_meta: dict[str, dict] = {}
    try:
        for req in _store.list_requests():
            cb = (req.metadata or {}).get("complexity_breakdown", {})
            req_meta[req.request_id] = {
                "archetype": req.metadata.get("archetype", "?"),
                "complexity_tier": cb.get("complexity_tier", "?"),
            }
    except Exception:
        pass
    if req_meta:
        meta_df = pd.DataFrame.from_dict(req_meta, orient="index").reset_index()
        meta_df.rename(columns={"index": "request_id"}, inplace=True)
        df = df.merge(meta_df, on="request_id", how="left")
    return df


raw_df = _all_results(str(EVAL_DIR))

if raw_df.empty:
    st.info("No evaluation results found. Run evaluations first via the Eval Dashboard.")
    st.stop()

# ── Main page: dimension selector ─────────────────────────────────────────────

st.markdown("## 📊 Ablation Comparison")
st.caption(
    "Side-by-side metric comparison across agent modes, augmentation checkpoints, or prompt variants. "
    "Select a comparison dimension below, then choose which values to compare in the sidebar."
)

# Only offer axes that are actually present as columns in the data
available_axes = [ax for ax in AXIS_LABELS if ax in raw_df.columns]

comparison_axis = st.selectbox(
    "**Comparison Dimension** — what to compare across:",
    options=available_axes,
    format_func=lambda ax: AXIS_LABELS.get(ax, ax),
    index=0,
    help=(
        "Changing this dimension updates the condition selector in the sidebar "
        "and recomputes all panels below."
    ),
)

# ── Sidebar — conditions + filters (reactive to comparison_axis) ──────────────

with st.sidebar:
    st.markdown("## Condition Selection")

    sidebar_header("Conditions")

    unique_vals = sorted(raw_df[comparison_axis].dropna().unique().tolist())
    selected_conditions = st.multiselect(
        f"Select {comparison_axis} values to compare (2–4)",
        unique_vals,
        default=unique_vals[:min(3, len(unique_vals))],
    )

    sidebar_header("Filters")

    metric_version_filter = st.selectbox(
        "Metric version",
        ["(all)"] + sorted(raw_df["metric_version"].dropna().unique().tolist()),
    )

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

    if "run_name" in raw_df.columns:
        all_run_names = sorted(raw_df["run_name"].dropna().unique().tolist())
        sel_run_names = st.multiselect(
            "Run name (experiment)", ["(all)"] + all_run_names, default=["(all)"]
        )
    else:
        sel_run_names = ["(all)"]


if len(selected_conditions) < 2:
    st.warning("Select at least 2 conditions in the sidebar to enable comparison.")
    st.stop()

# ── Filter ────────────────────────────────────────────────────────────────────

df = raw_df[raw_df[comparison_axis].isin(selected_conditions)].copy()
if metric_version_filter != "(all)":
    df = df[df["metric_version"] == metric_version_filter]
if "(all)" not in sel_archetypes and "archetype" in df.columns:
    df = df[df["archetype"].isin(sel_archetypes)]
if "(all)" not in sel_tiers and "complexity_tier" in df.columns:
    df = df[df["complexity_tier"].isin(sel_tiers)]
if "(all)" not in sel_run_names and "run_name" in df.columns:
    df = df[df["run_name"].isin(sel_run_names)]

if df.empty:
    st.warning("No results match the selected conditions and filters.")
    st.stop()

st.caption(
    f"Comparing **{len(selected_conditions)} {comparison_axis}** values · "
    f"{len(df)} episode results after filters"
)

# ── Panel 1: Metric comparison table ─────────────────────────────────────────

st.markdown("### Metric Comparison (mean ± std)")

all_det_metrics = sorted(
    {c.replace("det_", "") for c in df.columns if c.startswith("det_")}
)
table_rows = []
for metric in all_det_metrics:
    col = f"det_{metric}"
    if col not in df.columns:
        continue
    row: dict = {"Metric": metric}
    for cond in selected_conditions:
        sub = df[df[comparison_axis] == cond][col].dropna()
        row[str(cond)[:24]] = (
            f"{sub.mean():.3f} ± {sub.std():.3f} (n={len(sub)})" if len(sub) > 0 else "—"
        )
    table_rows.append(row)

with st.container(height=360):
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# ── Panel 2: Bar chart — overall score per condition ─────────────────────────

st.markdown("### Overall Score by Condition")

try:
    import plotly.graph_objects as go

    means, stds, labels = [], [], []
    for cond in selected_conditions:
        sub = df[df[comparison_axis] == cond]["overall_score"].dropna()
        means.append(sub.mean() if len(sub) > 0 else 0.0)
        stds.append(sub.std() if len(sub) > 1 else 0.0)
        labels.append(str(cond)[:28])

    fig_bar = go.Figure()
    fig_bar.add_bar(
        x=labels, y=means,
        error_y={"type": "data", "array": stds, "visible": True},
        marker_color=PALETTE[: len(labels)],
        text=[f"{m:.3f}" for m in means],
        textposition="outside",
    )
    fig_bar.update_layout(
        yaxis={"range": [0, 1.05], "title": "Overall score (mean ± std)"},
        height=340,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
except ImportError:
    st.info("Install plotly for charts: `pip install plotly`")

# ── Panel 3: Radar chart ──────────────────────────────────────────────────────

st.markdown("### Radar Chart (Key Metrics)")

available_key = [m for m in KEY_METRICS if f"det_{m}" in df.columns]

if available_key:
    try:
        fig_radar = go.Figure()
        for i, cond in enumerate(selected_conditions):
            sub = df[df[comparison_axis] == cond]
            values = [sub[f"det_{m}"].mean() for m in available_key]
            values_closed = values + [values[0]]
            cats_closed = available_key + [available_key[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=cats_closed,
                fill="toself",
                name=str(cond)[:28],
                line_color=PALETTE[i % len(PALETTE)],
                fillcolor=PALETTE[i % len(PALETTE)],
                opacity=0.3,
            ))
        fig_radar.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            height=400,
            legend={"orientation": "h", "y": -0.1},
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    except (ImportError, NameError):
        pass

# ── Panel 4: Complexity stratification ────────────────────────────────────────

if "complexity_tier" in df.columns and comparison_axis != "complexity_tier":
    st.markdown("### Performance by Complexity Tier")
    tiers = ["low", "medium", "high"]
    try:
        fig_tier = go.Figure()
        for i, cond in enumerate(selected_conditions):
            sub_cond = df[df[comparison_axis] == cond]
            tier_means = [
                sub_cond[sub_cond["complexity_tier"] == t]["overall_score"].mean()
                for t in tiers
            ]
            fig_tier.add_bar(
                x=tiers, y=tier_means,
                name=str(cond)[:28],
                marker_color=PALETTE[i % len(PALETTE)],
            )
        fig_tier.update_layout(
            barmode="group",
            yaxis={"range": [0, 1], "title": "Mean overall score"},
            height=320,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_tier, use_container_width=True)
    except Exception:
        pass
