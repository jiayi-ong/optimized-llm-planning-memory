"""
pages/2_trajectory_viewer.py
=============================
Trajectory Viewer — step-by-step ReAct trail inspector.

Features
--------
- Load any saved episode by ID or file path.
- Pre-populates from ``st.session_state["selected_episode_id"]`` if set by
  the Episode Browser.
- Live mode: polls ``outputs/episodes/live/<id>.jsonl`` every second and
  updates the display as new steps arrive.
- Each step is rendered in an expander: Thought / Action / Observation /
  Itinerary snapshot.
- Sidebar shows compressed states with per-section expandable content.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import time

import streamlit as st

st.set_page_config(page_title="Trajectory Viewer", layout="wide")
st.title("Trajectory Viewer")

from app.utils.data_loader import load_episode, load_live_events  # noqa: E402

# ── Episode selection ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Episode")
    default_id = st.session_state.get("selected_episode_id", "")
    ep_id_input = st.text_input("Episode ID or file path", value=default_id)
    live_mode = st.checkbox("Live mode (poll for updates)", value=False)
    refresh_interval = st.slider("Refresh interval (s)", 1, 10, 2) if live_mode else 2

if not ep_id_input:
    st.info("Enter an episode ID in the sidebar, or select one from the **Episode Browser**.")
    st.stop()

# ── Load episode ──────────────────────────────────────────────────────────────

placeholder = st.empty()

def render_episode(ep_id: str, live: bool) -> None:
    """Render the full trajectory for the given episode."""
    with placeholder.container():
        if live:
            events = load_live_events(ep_id)
            _render_live_events(ep_id, events)
        else:
            try:
                ep = load_episode(ep_id)
            except FileNotFoundError:
                st.error(f"Episode not found: `{ep_id}`")
                return
            _render_full_episode(ep)


def _render_full_episode(ep) -> None:
    """Render a completed EpisodeLog."""
    # Header
    st.caption(
        f"Mode: `{ep.agent_mode}` · Steps: {ep.total_steps} · "
        f"Success: {ep.success} · Reward: {ep.reward_components.total_reward:.3f}"
    )

    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.subheader("ReAct Trajectory")
        for step in ep.trajectory.steps:
            _render_step(step)

    with col_side:
        _render_sidebar_content(ep)


def _render_live_events(ep_id: str, events: list[dict]) -> None:
    """Render in-progress episode events from JSONL."""
    react_events = [e for e in events if e.get("type") == "react_step"]
    compression_events = [e for e in events if e.get("type") == "compression"]
    done = any(e.get("type") == "episode_complete" for e in events)

    status = "COMPLETE" if done else f"LIVE — {len(react_events)} steps so far"
    st.caption(f"Episode `{ep_id}` | {status}")

    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.subheader("ReAct Trajectory (Live)")
        for ev in react_events:
            _render_live_step(ev)

    with col_side:
        if compression_events:
            st.subheader("Compressions")
            for i, ev in enumerate(compression_events):
                with st.expander(f"Compression @ step {ev.get('step_index', i)}"):
                    st.write(f"**Method:** {ev.get('compression_method', '?')}")
                    st.write(f"**Tokens:** {ev.get('token_count', '?')}")
                    if ev.get("decisions_made"):
                        st.write("**Decisions:**")
                        for d in ev["decisions_made"]:
                            st.markdown(f"- {d}")
                    if ev.get("open_questions"):
                        st.write("**Open questions:**")
                        for q in ev["open_questions"]:
                            st.markdown(f"- {q}")
                    if ev.get("current_itinerary_sketch"):
                        st.text_area("Sketch", ev["current_itinerary_sketch"], height=80, key=f"live_sketch_{i}")


def _render_step(step) -> None:
    """Render a single ReActStep from a completed EpisodeLog."""
    action_label = step.action.tool_name if step.action else "DONE"
    success_icon = ""
    if step.observation:
        success_icon = "✅" if step.observation.success else "❌"

    label = f"Step {step.step_index} — {action_label} {success_icon}"
    with st.expander(label, expanded=step.step_index == 0):
        if step.thought:
            st.markdown("**Thought**")
            st.text(step.thought)

        if step.action:
            st.markdown("**Action**")
            st.code(
                f"{step.action.tool_name}({_fmt_args(step.action.arguments)})",
                language="python",
            )

        if step.observation:
            obs = step.observation
            st.markdown(f"**Observation** ({'OK' if obs.success else 'FAIL'} · {obs.latency_ms:.0f} ms)")
            if obs.success and obs.result:
                st.json(obs.result if isinstance(obs.result, dict) else {"result": str(obs.result)})
            elif obs.error_message:
                st.error(obs.error_message)

        if step.itinerary_snapshot:
            it = step.itinerary_snapshot
            st.caption(
                f"Itinerary snapshot: {len(it.days)} days · ${it.total_cost_usd:.2f} · "
                f"Complete: {it.is_complete}"
            )


def _render_live_step(ev: dict) -> None:
    """Render a step from a live JSONL event."""
    action = ev.get("action")
    obs = ev.get("observation")
    action_label = action["tool_name"] if action else "DONE"
    success_icon = ""
    if obs:
        success_icon = "✅" if obs.get("success") else "❌"

    label = f"Step {ev.get('step_index', '?')} — {action_label} {success_icon}"
    with st.expander(label):
        if ev.get("thought"):
            st.markdown("**Thought**")
            st.text(ev["thought"])
        if action:
            st.markdown("**Action**")
            st.code(f"{action['tool_name']}({action.get('arguments', {})})", language="python")
        if obs:
            st.markdown(f"**Observation** ({'OK' if obs.get('success') else 'FAIL'} · {obs.get('latency_ms', 0):.0f} ms)")
            if obs.get("result"):
                st.json(obs["result"] if isinstance(obs["result"], dict) else {"result": str(obs["result"])})
            elif obs.get("error_message"):
                st.error(obs["error_message"])


def _render_sidebar_content(ep) -> None:
    """Render compressed states and reward summary in the side column."""
    if ep.compressed_states:
        st.subheader(f"Compressed States ({len(ep.compressed_states)})")
        for i, cs in enumerate(ep.compressed_states):
            with st.expander(f"@ step {cs.step_index} ({cs.compression_method})"):
                st.write(f"**Tokens:** {cs.token_count}")
                hcl = cs.hard_constraint_ledger
                st.write(
                    f"Constraints: ✅ {len(hcl.satisfied_ids)} · "
                    f"❌ {len(hcl.violated_ids)} · "
                    f"? {len(hcl.unknown_ids)}"
                )
                if cs.decisions_made:
                    st.write("**Decisions:**")
                    for d in cs.decisions_made:
                        st.markdown(f"- {d}")
                if cs.open_questions:
                    st.write("**Open questions:**")
                    for q in cs.open_questions:
                        st.markdown(f"- {q}")
                if cs.key_discoveries:
                    st.write("**Key discoveries:**")
                    for k in cs.key_discoveries:
                        st.markdown(f"- {k}")
                if cs.current_itinerary_sketch:
                    st.text_area("Sketch", cs.current_itinerary_sketch, height=80, key=f"sketch_{i}")

    st.subheader("Reward")
    rc = ep.reward_components
    _bar_metric("Hard constraints", rc.hard_constraint_score)
    _bar_metric("Soft constraints", rc.soft_constraint_score)
    _bar_metric("Tool efficiency", rc.tool_efficiency_score)
    st.metric("Total", f"{rc.total_reward:.3f}")

    if ep.tool_stats:
        st.subheader("Tool Stats")
        tool_rows = [
            {
                "tool": ts.tool_name,
                "calls": ts.call_count,
                "ok": ts.success_count,
                "fail": ts.failure_count,
                "latency_ms": round(ts.avg_latency_ms, 1),
            }
            for ts in ep.tool_stats
        ]
        import pandas as pd
        st.dataframe(pd.DataFrame(tool_rows), use_container_width=True, hide_index=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_args(args: dict) -> str:
    parts = [f"{k}={repr(v)}" for k, v in list(args.items())[:4]]
    return ", ".join(parts)


def _bar_metric(label: str, value: float) -> None:
    pct = int(value * 100)
    st.progress(max(0.0, min(1.0, value)), text=f"{label}: {pct}%")


# ── Render loop ───────────────────────────────────────────────────────────────

if live_mode:
    while True:
        render_episode(ep_id_input, live=True)
        events = load_live_events(ep_id_input)
        if any(e.get("type") == "episode_complete" for e in events):
            st.success("Episode complete! Switch to static mode to inspect the full log.")
            break
        time.sleep(refresh_interval)
        st.rerun()
else:
    render_episode(ep_id_input, live=False)
