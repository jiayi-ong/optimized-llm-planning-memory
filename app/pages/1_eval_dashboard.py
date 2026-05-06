"""
app/pages/1_eval_dashboard.py
==============================
Eval Run Dashboard — first landing page for the evaluation UI.

Panels
------
1. Active jobs monitor   — running eval jobs with live progress bars.
2. All Runs table        — every eval run, newest first, with aggregate stats
                           loaded from aggregate.json (O(1) per run).
3. Run detail expander   — aggregate mean/std per metric per agent_mode when
                           a run row is clicked.
4. Control panel         — sidebar expander to configure and launch a new run,
                           or re-run sub-components of an existing one.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.utils.ui_style import PALETTE, inject_css, badge, score_color
from optimized_llm_planning_memory.evaluation.eval_store import EvalStore
from optimized_llm_planning_memory.evaluation.job_manager import EvalJobConfig, EvalJobManager

BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = BASE_DIR / "outputs" / "eval_results"
EPISODES_DIR = BASE_DIR / "outputs" / "episodes"
REQUESTS_DIR = BASE_DIR / "data" / "user_requests"

st.set_page_config(page_title="Eval Dashboard", layout="wide", page_icon="📊")
inject_css()

store = EvalStore(EVAL_DIR, EPISODES_DIR, REQUESTS_DIR)
job_mgr = EvalJobManager(BASE_DIR / "outputs" / ".eval_jobs")


# ── Sidebar — Control Panel ───────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎛️ Control Panel")

    with st.expander("▶ New Eval Run", expanded=False):
        aug_id = st.text_input("Augmentation ID", placeholder="tgad-trained-001 (optional)")
        prompt_id_in = st.text_input("Prompt ID", placeholder="sweep_D (optional)")
        eval_mode = st.selectbox("Eval mode", ["deterministic", "full", "llm_judge"],
                                 help="deterministic = no LLM calls (fast); "
                                      "full = deterministic + LLM judge; "
                                      "llm_judge = rubric only")
        judge_model = st.text_input("Judge model", value="openai/gpt-4o-mini",
                                    disabled=(eval_mode == "deterministic"))

        ep_sel_mode = st.radio("Episode selection",
                               ["All episodes", "Filter by agent_mode", "Manual episode IDs"],
                               horizontal=True)
        episode_ids_in: list[str] = []
        agent_mode_filter = None
        if ep_sel_mode == "Manual episode IDs":
            raw_ids = st.text_area("Episode IDs (one per line)")
            episode_ids_in = [x.strip() for x in raw_ids.splitlines() if x.strip()]
        elif ep_sel_mode == "Filter by agent_mode":
            agent_mode_filter = st.text_input("Agent mode", placeholder="raw")

        runs_for_rerun = ["(none)"] + [m.run_id[:24] for m in store.list_runs()[:20]]
        parent_run_raw = st.selectbox("Parent run (re-run)", runs_for_rerun)
        parent_run_id = None if parent_run_raw == "(none)" else parent_run_raw
        note_in = st.text_input("Notes", placeholder="optional free-text note")

        if st.button("▶ Start Eval Run"):
            cfg = EvalJobConfig(
                eval_mode=eval_mode,
                episode_ids=episode_ids_in,
                agent_mode=agent_mode_filter,
                augmentation_id=aug_id or None,
                prompt_id=prompt_id_in or None,
                parent_run_id=parent_run_id,
                judge_model=judge_model,
                notes=note_in or None,
                episodes_dir=str(EPISODES_DIR),
                eval_dir=str(EVAL_DIR),
            )
            job_id = job_mgr.submit(cfg)
            st.success(f"Job submitted: `{job_id}`")
            st.session_state["active_job_id"] = job_id

    st.markdown("---")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()


# ── Active Jobs ───────────────────────────────────────────────────────────────

active_jobs = job_mgr.list_jobs(active_only=True)
if active_jobs:
    st.markdown("## ⚡ Active Jobs")
    for job in active_jobs:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(
                f"**{job.job_id}** &nbsp; {badge(job.status)}",
                unsafe_allow_html=True,
            )
            st.progress(job.progress_pct / 100.0,
                        text=f"{job.n_completed}/{job.n_total} episodes")
        with col2:
            st.caption(f"Started: {job.started_at[:19].replace('T', ' ')}")
            if job.run_id:
                st.caption(f"Run: `{job.run_id[:20]}`")
        with col3:
            if st.button("■ Cancel", key=f"cancel_{job.job_id}"):
                job_mgr.cancel(job.job_id)
                st.rerun()
    st.markdown("---")


# ── All Runs Table ────────────────────────────────────────────────────────────

st.markdown("## 📋 All Evaluation Runs")

@st.cache_data(ttl=15)
def _load_run_summaries(eval_dir: str) -> pd.DataFrame:
    _store = EvalStore(Path(eval_dir), EPISODES_DIR, REQUESTS_DIR)
    manifests = _store.list_runs()
    rows = []
    for m in manifests:
        agg = _store.load_aggregate(m.run_id)
        overall_mean = (
            agg["overall"]["overall_score"].get("mean", float("nan"))
            if agg and "overall" in agg else float("nan")
        )
        hard_mean = float("nan")
        if agg and "by_agent_mode" in agg:
            for mode_stats in agg["by_agent_mode"].values():
                hcr = mode_stats.get("hard_constraint_ratio", {})
                if hcr:
                    hard_mean = hcr.get("mean", float("nan"))
                    break
        rows.append({
            "run_id": m.run_id,
            "created_at": m.created_at[:19].replace("T", " "),
            "status": m.status,
            "agent_mode": m.agent_mode,
            "augmentation_id": m.augmentation_id or "—",
            "prompt_id": m.prompt_id or "—",
            "metric_ver": m.metric_version,
            "n_episodes": m.n_episodes,
            "overall_mean": overall_mean,
            "hard_constr_mean": hard_mean,
            "notes": (m.notes or "")[:40],
        })
    return pd.DataFrame(rows)


summary_df = _load_run_summaries(str(EVAL_DIR))

if summary_df.empty:
    st.info(
        "No evaluation runs found. Use **New Eval Run** in the sidebar to start one, "
        "or run `python scripts/run_eval.py --all --deterministic_only` from the terminal."
    )
    st.stop()

# Colour the status column
def _status_badge_html(s: str) -> str:
    colors = {"completed": "#4CAF50", "running": "#FF6B35",
              "failed": "#E84545", "cancelled": "#9090A8"}
    c = colors.get(s, "#9090A8")
    return f'<span style="background:{c};color:#fff;padding:2px 7px;border-radius:4px;font-size:11px">{s}</span>'

styled = summary_df.copy()
styled["overall_mean"] = styled["overall_mean"].apply(
    lambda v: f"{v:.3f}" if not pd.isna(v) else "—"
)
styled["hard_constr_mean"] = styled["hard_constr_mean"].apply(
    lambda v: f"{v:.3f}" if not pd.isna(v) else "—"
)

st.dataframe(
    styled[["run_id", "created_at", "status", "agent_mode", "augmentation_id",
            "metric_ver", "n_episodes", "overall_mean", "hard_constr_mean", "notes"]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "run_id": st.column_config.TextColumn("Run ID", width=200),
        "created_at": st.column_config.TextColumn("Created", width=140),
        "overall_mean": st.column_config.TextColumn("Overall ↑", width=90),
        "hard_constr_mean": st.column_config.TextColumn("Hard Constr ↑", width=110),
    },
)


# ── Run Detail ────────────────────────────────────────────────────────────────

st.markdown("## 🔍 Run Detail")

available_run_ids = summary_df["run_id"].tolist()
if not available_run_ids:
    st.stop()

selected_run_id = st.selectbox(
    "Select a run to inspect",
    available_run_ids,
    format_func=lambda r: f"{r[:24]}  ({summary_df.loc[summary_df['run_id']==r, 'created_at'].values[0]})",
)

agg = store.load_aggregate(selected_run_id)

if agg is None:
    st.info("No aggregate stats file found for this run. Re-save it with the updated code to generate one.")
else:
    st.caption(
        f"**{agg['n_results']} results** · computed at {agg.get('computed_at', '?')[:19].replace('T', ' ')}"
    )

    by_mode = agg.get("by_agent_mode", {})
    if not by_mode:
        st.info("No per-mode stats available.")
    else:
        modes = sorted(by_mode.keys())
        all_metrics = sorted(
            {m for mode_data in by_mode.values() for m in mode_data}
        )
        # Show mean ± std table per metric per mode
        table_rows = []
        for metric in all_metrics:
            row: dict = {"metric": metric}
            for mode in modes:
                stats = by_mode[mode].get(metric, {})
                if stats:
                    row[mode] = f"{stats['mean']:.3f} ± {stats['std']:.3f}"
                else:
                    row[mode] = "—"
            table_rows.append(row)

        detail_df = pd.DataFrame(table_rows)
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

        # Bar chart: overall_score mean per mode
        try:
            import plotly.graph_objects as go

            means = []
            stds = []
            for mode in modes:
                s = by_mode[mode].get("overall_score", {})
                means.append(s.get("mean", 0.0))
                stds.append(s.get("std", 0.0))

            fig = go.Figure()
            fig.add_bar(
                x=modes, y=means,
                error_y={"type": "data", "array": stds, "visible": True},
                marker_color=PALETTE[: len(modes)],
                name="Overall score",
            )
            fig.update_layout(
                title="Overall score by agent mode",
                yaxis={"range": [0, 1], "title": "Mean ± std"},
                plot_bgcolor="#1E1E2E",
                paper_bgcolor="#1E1E2E",
                font={"color": "#E0E0E0"},
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass  # plotly not installed — table is enough

st.markdown(
    f"[→ View episodes in this run](2_eval_episode_log?run_id={selected_run_id})",
    unsafe_allow_html=False,
)
