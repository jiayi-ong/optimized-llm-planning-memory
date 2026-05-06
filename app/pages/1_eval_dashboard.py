"""
app/pages/1_eval_dashboard.py
==============================
Eval Run Dashboard — primary landing page for the evaluation UI.

Pipeline (four sequential stages)
-----------------------------------
  Stage 1 — Worlds        : pick an existing world set or generate a new one
  Stage 2 — Requests      : pick an existing request set or generate new requests
  Stage 3 — Agent Run     : run the ReAct agent to produce episode files
  Stage 4 — Score         : evaluate episodes (deterministic + optional LLM judge)

Each stage is a collapsible expander on the main page. They are independent —
skip early stages when the required inputs already exist:

  Scenario A (full pipeline from scratch)    → expand all 4 stages
  Scenario B (new agent on existing requests) → Stage 3 + Stage 4
  Scenario C (re-score existing episodes)    → Stage 4 only
  Scenario D (add LLM judge to det-only run) → Stage 4 (llm_judge mode)

Below the pipeline the page shows active jobs, the full run table, a detail
view for the selected run, and a delete-run confirmation flow.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.utils.ui_style import PALETTE, PLOTLY_LAYOUT, badge, inject_css
from optimized_llm_planning_memory.evaluation.eval_store import EvalStore
from optimized_llm_planning_memory.evaluation.job_manager import EvalJobConfig, EvalJobManager

BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = BASE_DIR / "outputs" / "eval_results"
EPISODES_DIR = BASE_DIR / "outputs" / "episodes"
REQUESTS_DIR = BASE_DIR / "data" / "user_requests"
WORLDS_ROOT = BASE_DIR / "worlds"
LOG_DIR = BASE_DIR / "outputs" / ".ui_logs"

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
    if any(WORLDS_ROOT.glob("world_*")):
        sets.insert(0, "(root)")
    return sets


def _world_count(set_name: str) -> int:
    folder = WORLDS_ROOT if set_name == "(root)" else WORLDS_ROOT / set_name
    return len(list(folder.glob("world_*")))


def _count_files(directory: str | Path, pattern: str) -> int:
    p = Path(directory)
    return len(list(p.glob(pattern))) if p.exists() else 0


def _list_collections() -> list[str]:
    """All immediate sub-folders of REQUESTS_DIR that contain request files."""
    if not REQUESTS_DIR.exists():
        return []
    return sorted(
        d.name for d in REQUESTS_DIR.iterdir()
        if d.is_dir() and any(d.glob("request_*.json"))
    )


def _count_requests(collection: str) -> int:
    return _count_files(REQUESTS_DIR / collection, "request_*.json")


def _count_new_files(directory: str | Path, pattern: str, since_ts: float) -> int:
    """Count files matching pattern whose mtime is >= since_ts (unix float)."""
    p = Path(directory)
    if not p.exists():
        return 0
    return sum(1 for f in p.glob(pattern) if f.stat().st_mtime >= since_ts)


def _count_new_episodes(output_dir: Path, since_ts: float) -> int:
    """Count ep_*.json files whose mtime is at or after since_ts (unix float)."""
    if not output_dir.exists():
        return 0
    return sum(1 for f in output_dir.glob("ep_*.json") if f.stat().st_mtime >= since_ts)


def _get_new_episode_ids(output_dir: Path, since_ts: float) -> list[str]:
    """Return episode IDs for ep_*.json files created at or after since_ts."""
    ids = []
    for f in output_dir.glob("ep_*.json"):
        if f.stat().st_mtime >= since_ts:
            stem = f.stem  # "ep_<uuid>"
            ids.append(stem[3:] if stem.startswith("ep_") else stem)
    return ids


def _launch_subprocess(cmd: list[str], log_key: str = "job") -> tuple[subprocess.Popen, Path]:
    """Launch a subprocess that writes all output to a persistent log file.

    Returns (proc, log_path). The log file survives session state clears and
    page reloads — the debug panel reads it on every refresh.
    PYTHONUNBUFFERED=1 ensures each print() line appears in the file immediately.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{log_key}_{ts}.log"
    child_env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",  # line-flush every print() immediately
    }
    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    log_fh.write(f"CMD : {' '.join(str(c) for c in cmd)}\n")
    log_fh.write(f"CWD : {BASE_DIR}\n")
    log_fh.write(f"TIME: {ts}\n\n")
    log_fh.flush()
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=log_fh,
        cwd=str(BASE_DIR),
        env=child_env,
    )
    # Keep file handle alive until proc exits (GC would close it otherwise)
    proc._ui_log_fh = log_fh  # type: ignore[attr-defined]
    return proc, log_path


def _read_log(log_path: Path | None) -> str:
    """Read the current contents of a subprocess log file."""
    if log_path is None or not Path(log_path).exists():
        return ""
    try:
        return Path(log_path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"(could not read log: {exc})"


def _show_job_output(job: dict, label: str = "Output") -> None:
    """Show subprocess log file content and the command used."""
    out = _read_log(job.get("log_path"))
    status, _ = _poll_proc(job)
    expanded = status == "failed"
    n_lines = len(out.splitlines()) if out else 0
    with st.expander(f"{label} ({n_lines} lines, status={status})", expanded=expanded):
        if out:
            st.code(out[-3000:], language="text")
        else:
            st.caption("No output yet — press Refresh to update.")
    with st.expander("Command (run in terminal to debug)", expanded=False):
        st.code(job.get("cmd", "—"), language="bash")


def _poll_proc(job: dict) -> tuple[str, int | None]:
    """Return (status, return_code). status: 'running' | 'done' | 'failed'."""
    rc = job["proc"].poll()
    if rc is None:
        return "running", None
    return ("done" if rc == 0 else "failed"), rc


def _delete_run(run_id: str, eval_dir: Path, episodes_dir: Path) -> tuple[int, int]:
    """Delete all episode files in a run, then remove the run directory.

    Returns (n_episodes_deleted, n_episodes_total).
    """
    run_dir = eval_dir / run_id
    episode_ids: list[str] = []
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            episode_ids = data.get("episode_ids", [])
        except Exception:
            pass
    n_deleted = 0
    for ep_id in episode_ids:
        ep_path = episodes_dir / f"ep_{ep_id}.json"
        if ep_path.exists():
            ep_path.unlink()
            n_deleted += 1
    if run_dir.exists():
        shutil.rmtree(run_dir)
    return n_deleted, len(episode_ids)


# ── Sidebar (minimal) ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Eval Dashboard")
    st.caption("Streamlit developer UI — optimized-llm-planning-memory")
    st.markdown("---")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    _n_eval = len(job_mgr.list_jobs(active_only=True))
    _running_procs = [
        k for k in ("gen_world_job", "gen_req_job", "run_agent_job")
        if st.session_state.get(k) and _poll_proc(st.session_state[k])[0] == "running"
    ]
    if _n_eval + len(_running_procs) > 0:
        st.info(f"{_n_eval + len(_running_procs)} active job(s) — see Jobs in Progress below")

    # ── Debug panel ───────────────────────────────────────────────────────────
    with st.expander("🔧 Debug / Diagnostics", expanded=False):
        st.markdown("**Filesystem state**")
        st.caption(f"BASE_DIR: `{BASE_DIR}`")
        # World sets
        _ws = _world_sets()
        if _ws:
            st.success(f"World sets found: {_ws}")
            for _sn in _ws:
                _sd = WORLDS_ROOT if _sn == "(root)" else WORLDS_ROOT / _sn
                _wc = [d.name for d in sorted(_sd.glob("world_*")) if d.is_dir()]
                st.caption(f"  {_sn}: {len(_wc)} worlds — {', '.join(_wc[:5])}" + (" …" if len(_wc) > 5 else ""))
        else:
            st.warning(f"No world sets — `{WORLDS_ROOT}` exists: {WORLDS_ROOT.exists()}")

        # Collections
        _cols = _list_collections()
        if _cols:
            st.success(f"Request collections: {_cols}")
            for _cn in _cols:
                _nc = _count_requests(_cn)
                st.caption(f"  {_cn}: {_nc} requests")
        else:
            st.warning(f"No request collections — `{REQUESTS_DIR}` exists: {REQUESTS_DIR.exists()}")

        # Episodes
        _ne = _count_files(EPISODES_DIR, "ep_*.json")
        st.caption(f"Episodes: {_ne} in `{EPISODES_DIR}`")

        st.markdown("**Active subprocess jobs (session state)**")
        _any_job = False
        for _jk in ("gen_world_job", "gen_req_job", "run_agent_job"):
            _j = st.session_state.get(_jk)
            if _j:
                _any_job = True
                _s, _rc = _poll_proc(_j)
                _safe = {k: v for k, v in _j.items() if k != "proc"}
                st.markdown(f"**{_jk}** — `{_s}` (rc={_rc})")
                st.json(_safe)
        if not _any_job:
            st.caption("No subprocess jobs in session state.")

        st.markdown("**Recent subprocess log files**")
        _log_files = sorted(LOG_DIR.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)[:5] \
            if LOG_DIR.exists() else []
        if _log_files:
            for _lf in _log_files:
                with st.expander(_lf.name, expanded=False):
                    st.code(_read_log(_lf)[-3000:], language="text")
        else:
            st.caption(f"No log files yet in `{LOG_DIR}`.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ═════════════════════════════════════════════════════════════════════════════

# Show one-shot action banner (set in button handlers; cleared here after display)
if "_banner" in st.session_state:
    st.info(st.session_state.pop("_banner"))

st.markdown("## 📊 Eval Dashboard")
st.caption(
    "Run the full evaluation pipeline in four stages, or jump directly to any stage "
    "if earlier inputs already exist. Expand only the stages you need."
)

# ── Stage 1: Worlds ───────────────────────────────────────────────────────────

with st.expander("🌍  **Stage 1 — Worlds**   ·   pick or generate a world set", expanded=False):
    existing_sets = _world_sets()

    s1_option = st.radio(
        "World source",
        ["Use existing world set", "Generate new worlds"],
        key="s1_option", horizontal=True,
    )

    if s1_option == "Use existing world set":
        if existing_sets:
            s1_sel = st.selectbox("World set", existing_sets, key="s1_world_set_select")
            n_w = _world_count(s1_sel)
            st.success(f"✓  **{n_w}** world(s) in `{s1_sel}`")
        else:
            st.warning("No world sets found under `worlds/`. Switch to 'Generate new worlds' below.")

    else:
        if existing_sets:
            st.caption(f"Existing sets: `{'`, `'.join(existing_sets)}`")
        c1, c2, c3 = st.columns([3, 1, 1])
        s1_set_name = c1.text_input("World set name", placeholder="batch_1", key="s1_set_name")
        s1_n_worlds = c2.number_input("N worlds", min_value=1, value=3, step=1, key="s1_n_worlds")
        s1_base_seed = c3.number_input("Base seed", value=42, key="s1_base_seed")

        col_gen, col_tip = st.columns([2, 5])
        with col_gen:
            if st.button("🌍 Generate Worlds", key="btn_gen_worlds"):
                if not s1_set_name.strip():
                    st.error("Enter a world set name.")
                else:
                    try:
                        cmd = [
                            sys.executable, str(BASE_DIR / "scripts" / "generate_worlds.py"),
                            "--set_name", s1_set_name.strip(),
                            "--n_worlds", str(int(s1_n_worlds)),
                            "--base_seed", str(int(s1_base_seed)),
                            "--worlds_dir", str(WORLDS_ROOT),
                        ]
                        proc, log_path = _launch_subprocess(cmd, "gen_worlds")
                        started_ts = time.time()
                        st.session_state["gen_world_job"] = {
                            "proc": proc,
                            "log_path": str(log_path),
                            "cmd": " ".join(str(c) for c in cmd),
                            "started_at": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                            "started_at_ts": started_ts,
                            "set_name": s1_set_name.strip(),
                            "set_dir": str(WORLDS_ROOT / s1_set_name.strip()),
                            "n_worlds": int(s1_n_worlds),
                        }
                        st.session_state["_banner"] = f"World generation started (PID {proc.pid}) — scroll to Jobs in Progress."
                        st.rerun()
                    except Exception as _exc:
                        st.error(f"Failed to launch subprocess: {_exc}")
        with col_tip:
            st.caption("Worlds are saved to `worlds/{set_name}/world_{seed}_{ts}/`. "
                       "Each world is a self-contained travel graph used by the simulator.")

# ── Stage 2: Requests ─────────────────────────────────────────────────────────

with st.expander("📋  **Stage 2 — Requests**   ·   pick or generate user requests", expanded=False):
    s2_option = st.radio(
        "Request source",
        ["Use existing request set", "Generate new requests"],
        key="s2_option", horizontal=True,
    )

    if s2_option == "Use existing request set":
        existing_collections = _list_collections()
        if existing_collections:
            s2_collection = st.selectbox("Collection", existing_collections, key="s2_collection_pick")
            n_r = _count_requests(s2_collection)
            st.success(f"✓  **{n_r}** request(s) in `data/user_requests/{s2_collection}/`")
        else:
            st.warning("No request collections found. Generate requests below.")
            s2_collection = None

    else:
        all_sets = _world_sets()
        if not all_sets:
            st.warning("No world sets found. Complete Stage 1 first to create worlds.")
        else:
            c1, c2 = st.columns([2, 1])
            s2_world_set = c1.selectbox("World set", all_sets, key="s2_world_set")
            s2_n_worlds_in_set = _world_count(s2_world_set)
            c1.caption(f"{s2_n_worlds_in_set} world(s) · requests distributed evenly across worlds")

            s2_collection_name = c2.text_input(
                "Collection name",
                placeholder="(timestamp if blank)",
                key="s2_collection_name",
                help="Sub-folder name under data/user_requests/. Leave blank for auto timestamp.",
            )

            c3, c4 = st.columns(2)
            s2_n_total = c3.number_input("N total requests", min_value=1, value=20, step=5, key="s2_n_total")
            s2_seed = c4.number_input("Seed", value=42, key="s2_seed")

            st.caption("Complexity mix (proportions, auto-normalised to 100 %)")
            cc1, cc2, cc3 = st.columns(3)
            s2_pct_low = cc1.number_input("Low %", 0, 100, 33, key="s2_pct_low")
            s2_pct_med = cc2.number_input("Med %", 0, 100, 34, key="s2_pct_med")
            s2_pct_high = cc3.number_input("High %", 0, 100, 33, key="s2_pct_high")

            if st.button("⚙ Generate Requests", key="btn_gen_reqs"):
                try:
                    world_set_path = WORLDS_ROOT if s2_world_set == "(root)" else WORLDS_ROOT / s2_world_set
                    cmd = [
                        sys.executable, str(BASE_DIR / "scripts" / "generate_eval_dataset.py"),
                        "--world_set", str(world_set_path),
                        "--n_total", str(int(s2_n_total)),
                        "--seed", str(int(s2_seed)),
                        "--output_dir", str(REQUESTS_DIR),
                        "--complexity_weights", str(s2_pct_low), str(s2_pct_med), str(s2_pct_high),
                    ]
                    if s2_collection_name.strip():
                        cmd += ["--collection", s2_collection_name.strip()]
                    proc, log_path = _launch_subprocess(cmd, "gen_reqs")
                    started_ts = time.time()
                    st.session_state["gen_req_job"] = {
                        "proc": proc,
                        "log_path": str(log_path),
                        "cmd": " ".join(str(c) for c in cmd),
                        "started_at": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                        "started_at_ts": started_ts,
                        "collection": s2_collection_name.strip() or "(timestamp)",
                        "n_total": int(s2_n_total),
                        "requests_dir": str(REQUESTS_DIR),
                    }
                    st.session_state["_banner"] = f"Request generation started (PID {proc.pid}) — scroll to Jobs in Progress."
                    st.rerun()
                except Exception as _exc:
                    st.error(f"Failed to launch subprocess: {_exc}")

# ── Stage 3: Agent Run ────────────────────────────────────────────────────────

with st.expander("🤖  **Stage 3 — Agent Run**   ·   generate episodes from requests", expanded=False):
    st.caption(
        "Run the ReAct agent on a request set. Each request is automatically routed "
        "to the world it was generated from (via `world_id` embedded in the request file), "
        "so the agent's tool calls hit the correct simulator instance."
    )

    s3_option = st.radio(
        "Episode source",
        ["Use existing episodes  (skip agent run)", "Run agent on requests"],
        key="s3_option", horizontal=True,
    )

    if s3_option == "Use existing episodes  (skip agent run)":
        n_ep = _count_files(EPISODES_DIR, "ep_*.json")
        if n_ep:
            st.success(f"✓  **{n_ep}** episode file(s) in `outputs/episodes/` — proceed to Stage 4.")
        else:
            st.warning("No episode files found. Run the agent or place episode files in `outputs/episodes/`.")

    else:
        # ── Request source for this agent run ─────────────────────────────────
        st.markdown("**Which requests to run?**")
        s3_req_src = st.radio(
            "Request source",
            ["Collection (all requests in collection)", "Re-run request set from eval run", "Specific request IDs"],
            key="s3_req_src", horizontal=True,
        )

        agent_batch_req_args: list[str] = []
        s3_n_requests_estimate = 0

        if s3_req_src == "Collection (all requests in collection)":
            s3_collections = _list_collections()
            if s3_collections:
                s3_collection = st.selectbox("Collection", s3_collections, key="s3_collection")
                s3_n_requests_estimate = _count_requests(s3_collection)
                st.caption(f"{s3_n_requests_estimate} request(s) in `data/user_requests/{s3_collection}/`")
            else:
                st.warning("No request collections found — generate them in Stage 2 first.")
                s3_collection = None
            agent_batch_req_args = ["--collection", s3_collection] if s3_collections else []

        elif s3_req_src == "Re-run request set from eval run":
            st.caption("The agent will run on the exact same requests as the chosen prior eval run, "
                       "each routed to its original world — enabling a controlled comparison.")
            run_opts = ["(none)"] + [m.run_id for m in store.list_runs()[:30]]
            s3_prior_run = st.selectbox("Source eval run", run_opts, key="s3_prior_run")
            if s3_prior_run != "(none)":
                try:
                    prior_m, _ = store.load_run(s3_prior_run)
                    s3_n_requests_estimate = len(prior_m.request_ids)
                    st.caption(f"{s3_n_requests_estimate} request(s) from run `{s3_prior_run[:28]}`")
                    agent_batch_req_args = ["--from_eval_run", str(EVAL_DIR / s3_prior_run)]
                except Exception:
                    st.warning("Could not load selected run.")
            else:
                st.info("Select an eval run above.")

        else:
            s3_ids_raw = st.text_area(
                "Request IDs (one per line)",
                placeholder="request_abc123\nrequest_def456",
                key="s3_req_ids",
            )
            ids_list = [x.strip() for x in s3_ids_raw.splitlines() if x.strip()]
            if ids_list:
                s3_n_requests_estimate = len(ids_list)
                agent_batch_req_args = ["--request_ids"] + ids_list
                st.caption(f"{len(ids_list)} ID(s) entered")

        # ── Agent config ──────────────────────────────────────────────────────
        st.markdown("**Agent configuration**")
        c1, c2, c3 = st.columns(3)
        s3_agent_mode = c1.selectbox(
            "Agent mode",
            ["raw", "llm_summary", "compressor", "mcts_compressor"],
            key="s3_agent_mode",
            help=(
                "raw = no compression (baseline); "
                "llm_summary = prompt-based summary; "
                "compressor = TGAD trained compressor; "
                "mcts_compressor = MCTS + LLM compressor"
            ),
        )
        s3_aug_id = c2.text_input(
            "Augmentation ID",
            placeholder="tgad-trained-001",
            key="s3_aug_id",
            help="Registry ID for the compressor checkpoint. Required for 'compressor' mode.",
        )
        s3_prompt_id = c3.text_input(
            "Prompt ID",
            placeholder="sweep_D (optional)",
            key="s3_prompt_id",
        )

        c4, c5, c6 = st.columns(3)
        s3_llm_model = c4.text_input("LLM model", value="openai/gpt-4o-mini", key="s3_llm_model")
        s3_max_steps = c5.number_input("Max steps", min_value=5, value=30, step=5, key="s3_max_steps")
        s3_compress_n = c6.number_input("Compress every N steps", min_value=1, value=5, key="s3_compress_n")

        can_run = bool(agent_batch_req_args)
        col_run, col_hint = st.columns([2, 5])
        with col_run:
            if st.button("▶ Run Agent", type="primary", disabled=not can_run, key="btn_run_agent"):
                cmd = [
                    sys.executable, str(BASE_DIR / "scripts" / "run_agent_batch.py"),
                    "--agent_mode", s3_agent_mode,
                    "--llm_model_id", s3_llm_model,
                    "--max_steps", str(int(s3_max_steps)),
                    "--compress_every_n_steps", str(int(s3_compress_n)),
                    "--worlds_root", str(WORLDS_ROOT),
                    "--requests_dir", str(REQUESTS_DIR),
                    "--output_dir", str(EPISODES_DIR),
                ] + agent_batch_req_args
                if s3_aug_id.strip():
                    cmd += ["--augmentation_id", s3_aug_id.strip()]
                if s3_prompt_id.strip():
                    cmd += ["--prompt_id", s3_prompt_id.strip()]

                try:
                    proc, log_path = _launch_subprocess(cmd, "agent_run")
                    started_ts = time.time()
                    st.session_state["run_agent_job"] = {
                        "proc": proc,
                        "log_path": str(log_path),
                        "cmd": " ".join(str(c) for c in cmd),
                        "started_at": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                        "started_at_ts": started_ts,
                        "agent_mode": s3_agent_mode,
                        "aug_id": s3_aug_id.strip() or None,
                        "n_total": s3_n_requests_estimate,
                        "output_dir": str(EPISODES_DIR),
                    }
                    st.session_state["_banner"] = f"Agent run started (PID {proc.pid}) — scroll to Jobs in Progress."
                    st.rerun()
                except Exception as _exc:
                    st.error(f"Failed to launch agent subprocess: {_exc}")
        with col_hint:
            if not can_run:
                st.caption("⬆ Select a request source above first.")

# ── Stage 4: Score Episodes ───────────────────────────────────────────────────

with st.expander("🔬  **Stage 4 — Score Episodes**   ·   evaluate with deterministic + LLM metrics",
                 expanded=True):
    # Surface new episode count from a just-completed agent run
    _agent_job = st.session_state.get("run_agent_job")
    if _agent_job:
        _aj_status, _ = _poll_proc(_agent_job)
        if _aj_status == "done":
            _new_ep_ids = _get_new_episode_ids(EPISODES_DIR, _agent_job["started_at_ts"])
            if _new_ep_ids:
                st.info(
                    f"Agent run completed — **{len(_new_ep_ids)}** new episode(s) ready. "
                    f"Set a **Run name** and click **Score Episodes** to evaluate them."
                )

    c1, c2, c3 = st.columns(3)
    s4_run_name = c1.text_input(
        "Run name", placeholder="sweep_D-ablation",
        key="s4_run_name",
        help="Groups related eval runs in the run table and ablation page.",
    )
    s4_aug_id = c2.text_input(
        "Augmentation ID", placeholder="tgad-trained-001 (optional)",
        key="s4_aug_id",
    )
    s4_prompt_id = c3.text_input(
        "Prompt ID", placeholder="sweep_D (optional)",
        key="s4_prompt_id",
    )

    c4, c5, c6 = st.columns(3)
    s4_eval_mode = c4.selectbox(
        "Eval mode",
        ["deterministic", "full", "llm_judge"],
        key="s4_eval_mode",
        help="deterministic = free & fast; full = det + LLM judge; llm_judge = rubric only",
    )
    s4_judge_model = c5.text_input(
        "Judge model", value="openai/gpt-4o-mini",
        disabled=(s4_eval_mode == "deterministic"),
        key="s4_judge_model",
    )
    s4_agent_mode_filter = c6.text_input(
        "Filter by agent_mode",
        placeholder="raw  (blank = all modes)",
        key="s4_agent_mode_filter",
    )

    c7, c8 = st.columns(2)
    _parent_opts = ["(none)"] + [m.run_id[:36] for m in store.list_runs()[:20]]
    s4_parent_run = c7.selectbox("Parent run (re-run lineage)", _parent_opts, key="s4_parent_run")
    s4_notes = c8.text_input("Notes", placeholder="free-text (optional)", key="s4_notes")

    col_score, col_hint4 = st.columns([2, 5])
    with col_score:
        if st.button("🔬 Score Episodes", type="primary", key="btn_score"):
            # Pass specific new episode IDs when a fresh agent run just completed
            ep_ids: list[str] = []
            if _agent_job and _poll_proc(_agent_job)[0] == "done":
                ep_ids = _get_new_episode_ids(EPISODES_DIR, _agent_job["started_at_ts"])

            cfg = EvalJobConfig(
                eval_mode=s4_eval_mode,
                episode_ids=ep_ids,
                request_ids=[],
                agent_mode=s4_agent_mode_filter.strip() or None,
                augmentation_id=s4_aug_id.strip() or None,
                prompt_id=s4_prompt_id.strip() or None,
                run_name=s4_run_name.strip() or None,
                parent_run_id=None if s4_parent_run == "(none)" else s4_parent_run,
                judge_model=s4_judge_model,
                notes=s4_notes.strip() or None,
                episodes_dir=str(EPISODES_DIR),
                eval_dir=str(EVAL_DIR),
            )
            job_id = job_mgr.submit(cfg)
            st.success(f"Scoring job submitted: `{job_id}`")
            st.session_state["active_job_id"] = job_id
            st.rerun()
    with col_hint4:
        if s4_eval_mode == "deterministic":
            st.caption("Deterministic mode — no API key required.")
        else:
            st.caption("LLM judge mode — requires `OPENAI_API_KEY` in `.env`.")

st.markdown("---")

# ── Jobs in Progress ──────────────────────────────────────────────────────────

gen_world_job = st.session_state.get("gen_world_job")
gen_req_job = st.session_state.get("gen_req_job")
run_agent_job = st.session_state.get("run_agent_job")
active_eval_jobs = job_mgr.list_jobs(active_only=True)
has_active = gen_world_job or gen_req_job or run_agent_job or active_eval_jobs

if has_active:
    st.markdown("## ⏳ Jobs in Progress")
    st.caption("Press **Refresh** in the sidebar to update progress.")

    # ── World generation job ─────────────────────────────────────────────────
    if gen_world_job:
        status, _ = _poll_proc(gen_world_job)
        n_created = _count_new_files(
            gen_world_job["set_dir"], "world_*", gen_world_job.get("started_at_ts", 0.0)
        )
        n_total_w = gen_world_job["n_worlds"]
        pct_w = n_created / n_total_w if n_total_w > 0 else 0.0
        icon = {"running": "🟡", "done": "✅", "failed": "❌"}.get(status, "⏳")

        with st.container(border=True):
            ca, cb = st.columns([4, 1])
            with ca:
                st.markdown(
                    f"**🌍 World Generation — `{gen_world_job['set_name']}`** &nbsp; {icon}  \n"
                    f"<span style='font-size:12px;color:#9090A8'>Started {gen_world_job['started_at']} · "
                    f"PID {gen_world_job['proc'].pid}</span>",
                    unsafe_allow_html=True,
                )
                st.progress(pct_w, text=f"Worlds created: {n_created} / {n_total_w}")
            with cb:
                if status == "running" and st.button("■ Cancel", key="cancel_world"):
                    gen_world_job["proc"].terminate()
                    st.session_state.pop("gen_world_job", None)
                    st.rerun()
                if status != "running" and st.button("Clear", key="clear_world"):
                    st.session_state.pop("gen_world_job", None)
                    st.rerun()
            _show_job_output(gen_world_job, "World generation log")

    # ── Request generation job ───────────────────────────────────────────────
    if gen_req_job:
        status, _ = _poll_proc(gen_req_job)
        since_r = gen_req_job.get("started_at_ts", 0.0)
        requests_dir = Path(gen_req_job.get("requests_dir", str(REQUESTS_DIR)))
        # Count request_*.json files created after this job started, across all collection subdirs
        n_created_r = sum(
            1 for f in requests_dir.rglob("request_*.json")
            if f.stat().st_mtime >= since_r
        ) if requests_dir.exists() else 0
        n_total_r = gen_req_job["n_total"]
        pct_r = min(1.0, n_created_r / n_total_r) if n_total_r > 0 else 0.0
        icon = {"running": "🟡", "done": "✅", "failed": "❌"}.get(status, "⏳")
        collection_label = gen_req_job.get("collection", "(timestamp)")

        with st.container(border=True):
            ca, cb = st.columns([4, 1])
            with ca:
                st.markdown(
                    f"**📋 Request Generation — collection `{collection_label}`** &nbsp; {icon}  \n"
                    f"<span style='font-size:12px;color:#9090A8'>Started {gen_req_job['started_at']} · "
                    f"PID {gen_req_job['proc'].pid}</span>",
                    unsafe_allow_html=True,
                )
                st.progress(pct_r, text=f"Requests saved: {n_created_r} / {n_total_r}")
            with cb:
                if status == "running" and st.button("■ Cancel", key="cancel_req"):
                    gen_req_job["proc"].terminate()
                    st.session_state.pop("gen_req_job", None)
                    st.rerun()
                if status != "running" and st.button("Clear", key="clear_req"):
                    st.session_state.pop("gen_req_job", None)
                    st.rerun()
            _show_job_output(gen_req_job, "Request generation log")

    # ── Agent run job ────────────────────────────────────────────────────────
    if run_agent_job:
        status, _ = _poll_proc(run_agent_job)
        since_ts = run_agent_job.get("started_at_ts", 0.0)
        n_done_ep = _count_new_episodes(EPISODES_DIR, since_ts)
        n_total_ep = run_agent_job.get("n_total", 0)
        pct_ep = (n_done_ep / n_total_ep) if n_total_ep > 0 else 1.0 if status == "done" else 0.0
        icon = {"running": "🟡", "done": "✅", "failed": "❌"}.get(status, "⏳")

        with st.container(border=True):
            ca, cb = st.columns([4, 1])
            with ca:
                mode_label = run_agent_job.get("agent_mode", "?")
                aug_label = run_agent_job.get("aug_id") or "—"
                st.markdown(
                    f"**🤖 Agent Run — `{mode_label}` / aug `{aug_label}`** &nbsp; {icon}  \n"
                    f"<span style='font-size:12px;color:#9090A8'>Started {run_agent_job['started_at']} · "
                    f"PID {run_agent_job['proc'].pid}</span>",
                    unsafe_allow_html=True,
                )
                ep_text = (
                    f"Episodes completed: {n_done_ep} / {n_total_ep}"
                    if n_total_ep > 0 else
                    f"Episodes completed: {n_done_ep}"
                )
                st.progress(min(1.0, pct_ep), text=ep_text)
                if status == "done" and n_done_ep > 0:
                    st.caption(f"→ Proceed to Stage 4 to score these {n_done_ep} episode(s).")
            with cb:
                if status == "running" and st.button("■ Cancel", key="cancel_agent"):
                    run_agent_job["proc"].terminate()
                    st.session_state.pop("run_agent_job", None)
                    st.rerun()
                if status != "running" and st.button("Clear", key="clear_agent"):
                    st.session_state.pop("run_agent_job", None)
                    st.rerun()
            _show_job_output(run_agent_job, "Agent run log")

    # ── Active eval scoring jobs ─────────────────────────────────────────────
    for job in active_eval_jobs:
        with st.container(border=True):
            ca, cb = st.columns([4, 1])
            with ca:
                st.markdown(
                    f"**🔬 Eval Scoring** &nbsp; {badge(job.status)}  \n"
                    f"<span style='font-size:12px;color:#9090A8'>"
                    f"Job `{job.job_id}` · started {job.started_at[:19].replace('T', ' ')}</span>",
                    unsafe_allow_html=True,
                )
                st.progress(
                    job.progress_pct / 100.0,
                    text=f"Scoring episodes: {job.n_completed} / {job.n_total}",
                )
                if job.run_id:
                    st.caption(f"Run: `{job.run_id[:36]}`")
            with cb:
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
    rows = []
    for m in _store.list_runs():
        agg = _store.load_aggregate(m.run_id)
        overall_mean = float("nan")
        hard_mean = float("nan")
        if agg:
            overall_mean = agg.get("overall", {}).get("overall_score", {}).get("mean", float("nan"))
            for mode_stats in agg.get("by_agent_mode", {}).values():
                hcr = mode_stats.get("hard_constraint_ratio", {})
                if hcr:
                    hard_mean = hcr.get("mean", float("nan"))
                    break
        rows.append({
            "run_name": m.run_name or "—",
            "run_id": m.run_id,
            "created_at": m.created_at[:19].replace("T", " "),
            "status": m.status,
            "agent_mode": m.agent_mode or "—",
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
        "No evaluation runs found. Use **Stage 4 — Score Episodes** above to start one, "
        "or run `python scripts/run_eval.py --all --deterministic_only` from the terminal."
    )
    st.stop()

styled = summary_df.copy()
styled["overall_mean"] = styled["overall_mean"].apply(lambda v: f"{v:.3f}" if not pd.isna(v) else "—")
styled["hard_constr_mean"] = styled["hard_constr_mean"].apply(lambda v: f"{v:.3f}" if not pd.isna(v) else "—")

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

selected_run_id = st.selectbox(
    "Select a run to inspect",
    available_run_ids,
    format_func=lambda r: (
        f"{summary_df.loc[summary_df['run_id']==r, 'run_name'].values[0]}  ·  "
        f"{r[:28]}  ({summary_df.loc[summary_df['run_id']==r, 'created_at'].values[0]})"
    ),
)

agg = store.load_aggregate(selected_run_id)

if agg is None:
    st.info("No aggregate stats for this run. Re-save it via updated code to generate one.")
else:
    st.caption(
        f"**{agg['n_results']} results** · computed at "
        f"{agg.get('computed_at', '?')[:19].replace('T', ' ')}"
    )
    by_mode = agg.get("by_agent_mode", {})
    if by_mode:
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

# ── Delete Run ────────────────────────────────────────────────────────────────

st.markdown("---")

if st.session_state.get("pending_delete") != selected_run_id:
    st.session_state.pop("pending_delete", None)

if "pending_delete" not in st.session_state:
    if st.button("🗑 Delete this run…", type="secondary"):
        st.session_state["pending_delete"] = selected_run_id
        st.rerun()
else:
    _manifest_path = EVAL_DIR / selected_run_id / "manifest.json"
    _n_eps = 0
    if _manifest_path.exists():
        try:
            _data = json.loads(_manifest_path.read_text(encoding="utf-8"))
            _n_eps = len(_data.get("episode_ids", []))
        except Exception:
            pass

    st.warning(
        f"**This will permanently delete:**\n"
        f"- Run directory: `outputs/eval_results/{selected_run_id}/`\n"
        f"- Up to **{_n_eps} episode file(s)** from `outputs/episodes/`\n\n"
        f"This cannot be undone.",
        icon="⚠️",
    )
    confirm_input = st.text_input(
        "Type the run ID to confirm deletion:",
        placeholder=selected_run_id,
        key="delete_confirm_input",
    )
    col_del, col_cancel, _ = st.columns([1, 1, 6])
    with col_del:
        delete_ok = confirm_input.strip() == selected_run_id
        if st.button("✓ Delete", type="primary", disabled=not delete_ok):
            n_ep_del, n_ep_total = _delete_run(selected_run_id, EVAL_DIR, EPISODES_DIR)
            st.session_state.pop("pending_delete", None)
            st.cache_data.clear()
            st.success(
                f"Run `{selected_run_id}` deleted. "
                f"Removed {n_ep_del} / {n_ep_total} episode file(s)."
            )
            st.rerun()
    with col_cancel:
        if st.button("✕ Cancel"):
            st.session_state.pop("pending_delete", None)
            st.rerun()
