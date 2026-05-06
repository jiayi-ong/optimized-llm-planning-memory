"""
app/pages/1_eval_dashboard.py
==============================
Eval Run Dashboard — primary landing page for the evaluation UI.

Main content
------------
1. Jobs in progress  — live progress bars for world gen, request gen, and eval
                        scoring jobs; one bar per active job, labelled by stage.
2. All Runs table    — every eval run, newest first, grouped by run_name.
3. Run detail        — aggregate mean/std per metric + overall-score bar chart
                        for the selected run.

Sidebar control panel
---------------------
EVAL RUN CONFIG   — run_name, IDs, eval mode, judge model
EPISODE SOURCE    — all / re-run existing request set
GENERATE WORLDS   — create a new batch of worlds into a named set folder
GENERATE REQUESTS — pick a world set and generate user requests
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.utils.ui_style import PALETTE, PLOTLY_LAYOUT, badge, inject_css, sidebar_header
from optimized_llm_planning_memory.evaluation.eval_store import EvalStore
from optimized_llm_planning_memory.evaluation.job_manager import EvalJobConfig, EvalJobManager

BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = BASE_DIR / "outputs" / "eval_results"
EPISODES_DIR = BASE_DIR / "outputs" / "episodes"
REQUESTS_DIR = BASE_DIR / "data" / "user_requests"
WORLDS_ROOT = BASE_DIR / "worlds"

st.set_page_config(page_title="Eval Dashboard", layout="wide", page_icon="📊")
inject_css()

store = EvalStore(EVAL_DIR, EPISODES_DIR, REQUESTS_DIR)
job_mgr = EvalJobManager(BASE_DIR / "outputs" / ".eval_jobs")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _world_sets() -> list[str]:
    """Sub-folder names inside worlds/ that contain at least one world_* dir."""
    if not WORLDS_ROOT.exists():
        return []
    sets = []
    for d in sorted(WORLDS_ROOT.iterdir()):
        if d.is_dir() and any(d.glob("world_*")):
            sets.append(d.name)
    # Also include the root worlds/ itself if it has world_* directly
    if any(WORLDS_ROOT.glob("world_*")):
        sets.insert(0, "(root)")
    return sets


def _world_count(set_name: str) -> int:
    folder = WORLDS_ROOT if set_name == "(root)" else WORLDS_ROOT / set_name
    return len(list(folder.glob("world_*")))


def _count_files(directory: str | Path, pattern: str) -> int:
    p = Path(directory)
    return len(list(p.glob(pattern))) if p.exists() else 0


def _launch_subprocess(cmd: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(BASE_DIR),
    )


def _poll_proc(job: dict) -> tuple[str, int | None]:
    """Return (status, return_code). status: 'running' | 'done' | 'failed'."""
    rc = job["proc"].poll()
    if rc is None:
        return "running", None
    return ("done" if rc == 0 else "failed"), rc


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Control Panel")

    # ── EVAL RUN CONFIG ───────────────────────────────────────────────────────
    sidebar_header("Eval Run Config")

    run_name_in = st.text_input("Run name", placeholder="sweep_D-ablation (optional)")
    aug_id = st.text_input("Augmentation ID", placeholder="tgad-trained-001 (optional)")
    prompt_id_in = st.text_input("Prompt ID", placeholder="sweep_D (optional)")
    eval_mode = st.selectbox(
        "Eval mode", ["deterministic", "full", "llm_judge"],
        help="deterministic = no LLM calls; full = det + LLM judge; llm_judge = rubric only",
    )
    judge_model = st.text_input(
        "Judge model", value="openai/gpt-4o-mini",
        disabled=(eval_mode == "deterministic"),
    )

    # ── EPISODE SOURCE ────────────────────────────────────────────────────────
    sidebar_header("Episode Source")

    ep_source = st.radio(
        "Source", ["All saved episodes", "Re-run existing request set"],
    )
    agent_mode_filter: str | None = None
    request_ids_for_rerun: list[str] | None = None

    if ep_source == "All saved episodes":
        af = st.text_input("Filter by agent_mode", placeholder="raw (optional)")
        agent_mode_filter = af.strip() or None
    else:
        run_opts = ["(none)"] + [m.run_id[:28] for m in store.list_runs()[:30]]
        prior_run_raw = st.selectbox("Source run ID", run_opts)
        if prior_run_raw != "(none)":
            try:
                prior_manifest, _ = store.load_run(prior_run_raw)
                request_ids_for_rerun = list(prior_manifest.request_ids)
                st.caption(f"{len(request_ids_for_rerun)} requests from this run")
            except Exception:
                st.warning("Could not load selected run.")

    parent_run_raw = st.selectbox(
        "Parent run (re-run lineage)",
        ["(none)"] + [m.run_id[:28] for m in store.list_runs()[:20]],
    )
    parent_run_id = None if parent_run_raw == "(none)" else parent_run_raw
    note_in = st.text_input("Notes", placeholder="free-text note (optional)")

    if st.button("▶ Start Eval Run"):
        cfg = EvalJobConfig(
            eval_mode=eval_mode,
            episode_ids=[],
            request_ids=request_ids_for_rerun or [],
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

    # ── GENERATE WORLDS ───────────────────────────────────────────────────────
    sidebar_header("Generate Worlds")

    existing_sets = _world_sets()
    set_name_in = st.text_input(
        "World set name",
        placeholder="batch_1",
        help="New or existing sub-folder under worlds/. Worlds are appended if it already exists.",
    )
    n_worlds_in = st.number_input("N worlds", min_value=1, value=3, step=1)
    base_seed_in = st.number_input("Base seed", value=42)

    if st.button("🌍 Generate Worlds"):
        if not set_name_in.strip():
            st.error("Enter a world set name.")
        else:
            cmd = [
                sys.executable,
                str(BASE_DIR / "scripts" / "generate_worlds.py"),
                "--set_name", set_name_in.strip(),
                "--n_worlds", str(int(n_worlds_in)),
                "--base_seed", str(int(base_seed_in)),
                "--worlds_dir", str(WORLDS_ROOT),
            ]
            proc = _launch_subprocess(cmd)
            st.session_state["gen_world_job"] = {
                "proc": proc,
                "cmd": " ".join(cmd),
                "started_at": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                "set_name": set_name_in.strip(),
                "set_dir": str(WORLDS_ROOT / set_name_in.strip()),
                "n_worlds": int(n_worlds_in),
            }
            st.rerun()

    if existing_sets:
        st.caption(f"Existing sets: {', '.join(existing_sets)}")

    # ── GENERATE REQUESTS ─────────────────────────────────────────────────────
    sidebar_header("Generate Requests")

    all_sets = _world_sets()
    if all_sets:
        gen_world_set = st.selectbox("World set", all_sets,
                                     help="Select which world set to generate requests from.")
        worlds_in_set = _world_count(gen_world_set)
        st.caption(f"{worlds_in_set} world(s) in this set")
    else:
        gen_world_set = None
        st.warning("No world sets found. Generate worlds first.")

    n_total_in = st.number_input("N total requests", min_value=1, value=20, step=5)
    gen_split = st.selectbox("Split", ["train", "val", "test"])
    gen_seed = st.number_input("Seed", value=42)

    st.caption("Complexity mix (auto-normalised)")
    c1, c2, c3 = st.columns(3)
    pct_low = c1.number_input("Low %", 0, 100, 33, key="pct_low")
    pct_med = c2.number_input("Med %", 0, 100, 34, key="pct_med")
    pct_high = c3.number_input("High %", 0, 100, 33, key="pct_high")

    if st.button("⚙ Generate Requests"):
        if gen_world_set is None:
            st.error("No world set selected.")
        else:
            world_set_path = WORLDS_ROOT if gen_world_set == "(root)" else WORLDS_ROOT / gen_world_set
            cmd = [
                sys.executable,
                str(BASE_DIR / "scripts" / "generate_eval_dataset.py"),
                "--world_set", str(world_set_path),
                "--n_total", str(int(n_total_in)),
                "--split", gen_split,
                "--seed", str(int(gen_seed)),
                "--output_dir", str(REQUESTS_DIR),
                "--complexity_weights", str(pct_low), str(pct_med), str(pct_high),
            ]
            proc = _launch_subprocess(cmd)
            st.session_state["gen_req_job"] = {
                "proc": proc,
                "cmd": " ".join(cmd),
                "started_at": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                "split": gen_split,
                "n_total": int(n_total_in),
                "output_dir": str(REQUESTS_DIR / gen_split),
            }
            st.rerun()

    st.markdown("---")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ═════════════════════════════════════════════════════════════════════════════

# ── Jobs in Progress ──────────────────────────────────────────────────────────

gen_world_job = st.session_state.get("gen_world_job")
gen_req_job = st.session_state.get("gen_req_job")
active_eval_jobs = job_mgr.list_jobs(active_only=True)
has_active = gen_world_job or gen_req_job or active_eval_jobs

if has_active:
    st.markdown("## ⏳ Jobs in Progress")
    st.caption("Press **Refresh** in the sidebar to update progress.")

    # ── World generation job ─────────────────────────────────────────────────
    if gen_world_job:
        status, rc = _poll_proc(gen_world_job)
        n_created = _count_files(gen_world_job["set_dir"], "world_*")
        n_total_w = gen_world_job["n_worlds"]
        pct_w = n_created / n_total_w if n_total_w > 0 else 0.0

        with st.container(border=True):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(
                    f"**🌍 World Generation — `{gen_world_job['set_name']}`** &nbsp; "
                    f"{'🟡 Running' if status == 'running' else ('✅ Done' if status == 'done' else '❌ Failed')}  \n"
                    f"<span style='font-size:12px;color:#9090A8'>Started {gen_world_job['started_at']} · "
                    f"PID {gen_world_job['proc'].pid}</span>",
                    unsafe_allow_html=True,
                )
                st.progress(pct_w, text=f"Worlds created: {n_created} / {n_total_w}")
            with col_b:
                if status == "running" and st.button("■ Cancel", key="cancel_world"):
                    gen_world_job["proc"].terminate()
                    st.session_state.pop("gen_world_job", None)
                    st.rerun()
                if status != "running" and st.button("Clear", key="clear_world"):
                    st.session_state.pop("gen_world_job", None)
                    st.rerun()
            if status == "failed":
                try:
                    out = gen_world_job["proc"].stdout.read()
                    if out:
                        st.code(out[-600:], language="text")
                except Exception:
                    pass

    # ── Request generation job ───────────────────────────────────────────────
    if gen_req_job:
        status, rc = _poll_proc(gen_req_job)
        n_created_r = _count_files(gen_req_job["output_dir"], "request_*.json")
        n_total_r = gen_req_job["n_total"]
        pct_r = min(1.0, n_created_r / n_total_r) if n_total_r > 0 else 0.0

        with st.container(border=True):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(
                    f"**📋 Request Generation — `{gen_req_job['split']}` split** &nbsp; "
                    f"{'🟡 Running' if status == 'running' else ('✅ Done' if status == 'done' else '❌ Failed')}  \n"
                    f"<span style='font-size:12px;color:#9090A8'>Started {gen_req_job['started_at']} · "
                    f"PID {gen_req_job['proc'].pid}</span>",
                    unsafe_allow_html=True,
                )
                st.progress(pct_r, text=f"Requests saved: {n_created_r} / {n_total_r}")
            with col_b:
                if status == "running" and st.button("■ Cancel", key="cancel_req"):
                    gen_req_job["proc"].terminate()
                    st.session_state.pop("gen_req_job", None)
                    st.rerun()
                if status != "running" and st.button("Clear", key="clear_req"):
                    st.session_state.pop("gen_req_job", None)
                    st.rerun()
            if status == "failed":
                try:
                    out = gen_req_job["proc"].stdout.read()
                    if out:
                        st.code(out[-600:], language="text")
                except Exception:
                    pass

    # ── Active eval scoring jobs ─────────────────────────────────────────────
    for job in active_eval_jobs:
        with st.container(border=True):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(
                    f"**🔬 Eval Scoring** &nbsp; "
                    f"{badge(job.status)}  \n"
                    f"<span style='font-size:12px;color:#9090A8'>"
                    f"Job `{job.job_id}` · started {job.started_at[:19].replace('T', ' ')}</span>",
                    unsafe_allow_html=True,
                )
                # Stage breakdown
                st.progress(
                    job.progress_pct / 100.0,
                    text=f"Scoring episodes: {job.n_completed} / {job.n_total}",
                )
                if job.run_id:
                    st.caption(f"Run: `{job.run_id[:28]}`")
            with col_b:
                if st.button("■ Cancel", key=f"cancel_eval_{job.job_id}"):
                    job_mgr.cancel(job.job_id)
                    st.rerun()

    st.markdown("---")


# ── All Runs Table ────────────────────────────────────────────────────────────

st.markdown("## 📋 All Evaluation Runs")
st.caption("Newest first. Runs sharing a **Run Name** belong to the same logical experiment.")


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

with st.container(height=320):
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
    st.info("No aggregate stats file for this run. Re-save it with updated code to generate one.")
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
        all_metrics = sorted({m for md in by_mode.values() for m in md})
        table_rows = []
        for metric in all_metrics:
            row: dict = {"metric": metric}
            for mode in modes:
                s = by_mode[mode].get(metric, {})
                row[mode] = f"{s['mean']:.3f} ± {s['std']:.3f}" if s else "—"
            table_rows.append(row)

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

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
)
