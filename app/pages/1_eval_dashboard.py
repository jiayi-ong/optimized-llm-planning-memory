"""
app/pages/1_eval_dashboard.py
==============================
Eval Run Dashboard — first landing page for the evaluation UI.

Panels
------
1. Active jobs monitor   — running eval jobs with live progress bars.
2. All Runs table        — every eval run, newest first, grouped by run_name
                           when present; aggregate stats from aggregate.json.
3. Run detail            — aggregate mean/std per metric per agent_mode for a
                           selected run, plus an overall-score bar chart.
4. Control panel (sidebar)  — configure and launch a new eval run, or re-run
                           sub-components of an existing one.  The sidebar
                           also exposes a "Generate Requests" panel for
                           multi-world dataset generation without leaving the UI.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.utils.ui_style import PALETTE, PLOTLY_LAYOUT, inject_css, badge, sidebar_header
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
    st.markdown("## Control Panel")

    # ── EVAL RUN CONFIG ───────────────────────────────────────────────────────
    sidebar_header("Eval Run Config")

    run_name_in = st.text_input("Run name", placeholder="sweep_D-ablation (optional)")
    aug_id = st.text_input("Augmentation ID", placeholder="tgad-trained-001 (optional)")
    prompt_id_in = st.text_input("Prompt ID", placeholder="sweep_D (optional)")
    eval_mode = st.selectbox(
        "Eval mode",
        ["deterministic", "full", "llm_judge"],
        help="deterministic = no LLM calls (fast); full = det + LLM judge; llm_judge = rubric only",
    )
    judge_model = st.text_input(
        "Judge model",
        value="openai/gpt-4o-mini",
        disabled=(eval_mode == "deterministic"),
    )

    # ── EPISODE SOURCE ────────────────────────────────────────────────────────
    sidebar_header("Episode Source")

    ep_source = st.radio(
        "Source",
        ["All saved episodes", "Re-run existing request set", "Generate new requests"],
        help="Choose which episodes to evaluate.",
    )

    episode_ids_in: list[str] = []
    agent_mode_filter: str | None = None
    request_ids_for_rerun: list[str] | None = None

    if ep_source == "All saved episodes":
        agent_mode_filter_str = st.text_input("Filter by agent_mode", placeholder="raw (optional)")
        agent_mode_filter = agent_mode_filter_str.strip() or None

    elif ep_source == "Re-run existing request set":
        all_run_ids = ["(none)"] + [m.run_id[:28] for m in store.list_runs()[:30]]
        prior_run_raw = st.selectbox("Source run ID", all_run_ids)
        if prior_run_raw != "(none)":
            try:
                prior_manifest, _ = store.load_run(prior_run_raw)
                request_ids_for_rerun = list(prior_manifest.request_ids)
                st.caption(f"{len(request_ids_for_rerun)} requests from this run")
            except Exception:
                st.warning("Could not load selected run.")

    else:  # Generate new requests
        st.info("Configure request generation below, then click Generate & Eval.")

    runs_for_parent = ["(none)"] + [m.run_id[:28] for m in store.list_runs()[:20]]
    parent_run_raw = st.selectbox("Parent run (re-run lineage)", runs_for_parent)
    parent_run_id = None if parent_run_raw == "(none)" else parent_run_raw
    note_in = st.text_input("Notes", placeholder="free-text note (optional)")

    if ep_source != "Generate new requests" and st.button("▶ Start Eval Run"):
        cfg = EvalJobConfig(
            eval_mode=eval_mode,
            episode_ids=episode_ids_in,
            request_ids=request_ids_for_rerun,
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

    # ── GENERATE REQUESTS (only shown when source = Generate) ─────────────────
    if ep_source == "Generate new requests":
        sidebar_header("Generate Requests")

        worlds_dir_in = st.text_input("Worlds directory", value="worlds")
        n_total_in = st.number_input("N total requests", min_value=1, value=20, step=5)
        gen_split = st.selectbox("Split", ["train", "val", "test"])
        gen_seed = st.number_input("Seed", value=42)

        st.caption("Complexity mix (will be normalised to 100 %)")
        c1, c2, c3 = st.columns(3)
        pct_low = c1.number_input("Low %", 0, 100, 33, key="pct_low")
        pct_med = c2.number_input("Med %", 0, 100, 34, key="pct_med")
        pct_high = c3.number_input("High %", 0, 100, 33, key="pct_high")

        # ── Generation status (persists across reruns via session_state) ──────
        gen_job = st.session_state.get("gen_job")
        if gen_job is not None:
            proc: subprocess.Popen = gen_job["proc"]
            rc = proc.poll()  # None = still running
            if rc is None:
                st.info(f"⏳ Running… (PID {proc.pid}, started {gen_job['started_at']})")
                if st.button("■ Cancel generation"):
                    proc.terminate()
                    st.session_state.pop("gen_job", None)
                    st.rerun()
            elif rc == 0:
                st.success(f"✅ Done — {gen_job['n_total']} requests into `{gen_job['split']}`")
                st.caption(gen_job["cmd"])
                if st.button("Clear status"):
                    st.session_state.pop("gen_job", None)
                    st.rerun()
            else:
                output = ""
                try:
                    output = proc.stdout.read() if proc.stdout else ""  # type: ignore[union-attr]
                except Exception:
                    pass
                st.error(f"❌ Failed (exit {rc})")
                if output:
                    st.code(output[-800:], language="text")
                if st.button("Clear error"):
                    st.session_state.pop("gen_job", None)
                    st.rerun()

        if st.button("⚙ Generate & Eval"):
            cmd = [
                sys.executable,
                str(BASE_DIR / "scripts" / "generate_eval_dataset.py"),
                "--world_dirs", worlds_dir_in,
                "--n_total", str(int(n_total_in)),
                "--split", gen_split,
                "--seed", str(int(gen_seed)),
                "--output_dir", str(REQUESTS_DIR),
                "--complexity_weights", str(pct_low), str(pct_med), str(pct_high),
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(BASE_DIR),
            )
            st.session_state["gen_job"] = {
                "proc": proc,
                "cmd": " ".join(cmd),
                "started_at": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                "split": gen_split,
                "n_total": int(n_total_in),
            }
            st.rerun()

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
st.caption("One row per eval run, newest first. Runs sharing a Run Name belong to the same experiment.")


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
            "run_name": m.run_name or "—",
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
        "No evaluation runs found. Use the **Control Panel** in the sidebar to start one, "
        "or run `python scripts/run_eval.py --all --deterministic_only` from the terminal."
    )
    st.stop()

styled = summary_df.copy()
styled["overall_mean"] = styled["overall_mean"].apply(
    lambda v: f"{v:.3f}" if not pd.isna(v) else "—"
)
styled["hard_constr_mean"] = styled["hard_constr_mean"].apply(
    lambda v: f"{v:.3f}" if not pd.isna(v) else "—"
)

with st.container(height=340):
    st.dataframe(
        styled[["run_name", "run_id", "created_at", "status", "agent_mode",
                "augmentation_id", "metric_ver", "n_episodes", "overall_mean",
                "hard_constr_mean", "notes"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "run_name": st.column_config.TextColumn("Run Name", width=160),
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
    format_func=lambda r: (
        f"{summary_df.loc[summary_df['run_id']==r, 'run_name'].values[0]}  ·  "
        f"{r[:24]}  ({summary_df.loc[summary_df['run_id']==r, 'created_at'].values[0]})"
    ),
)

agg = store.load_aggregate(selected_run_id)

if agg is None:
    st.info("No aggregate stats file for this run. Re-save it with the updated code to generate one.")
else:
    st.caption(
        f"**{agg['n_results']} results** · computed at "
        f"{agg.get('computed_at', '?')[:19].replace('T', ' ')}"
    )

    by_mode = agg.get("by_agent_mode", {})
    if not by_mode:
        st.info("No per-mode stats available.")
    else:
        modes = sorted(by_mode.keys())
        all_metrics = sorted({m for mode_data in by_mode.values() for m in mode_data})
        table_rows = []
        for metric in all_metrics:
            row: dict = {"metric": metric}
            for mode in modes:
                stats = by_mode[mode].get(metric, {})
                row[mode] = f"{stats['mean']:.3f} ± {stats['std']:.3f}" if stats else "—"
            table_rows.append(row)

        detail_df = pd.DataFrame(table_rows)
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

        try:
            import plotly.graph_objects as go

            means = [by_mode[m].get("overall_score", {}).get("mean", 0.0) for m in modes]
            stds = [by_mode[m].get("overall_score", {}).get("std", 0.0) for m in modes]

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
                height=320,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

st.markdown(
    f"[→ View episodes in this run](2_eval_episode_log?run_id={selected_run_id})",
    unsafe_allow_html=False,
)
