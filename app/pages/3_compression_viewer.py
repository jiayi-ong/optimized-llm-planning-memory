"""
pages/3_compression_viewer.py
==============================
Compression Viewer — side-by-side trajectory text vs CompressedState.

For each compression event in an episode this page shows:
- Left column: the raw trajectory steps that were fed to the compressor.
- Right column: the resulting CompressedState, section by section.
- Token delta between input and output.

Useful for understanding what information the compressor preserves or discards,
and for debugging compressor quality.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import streamlit as st

st.set_page_config(page_title="Compression Viewer", layout="wide")
st.title("Compression Viewer")

from app.utils.data_loader import load_episode  # noqa: E402

# ── Episode selection ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Episode")
    default_id = st.session_state.get("selected_episode_id", "")
    ep_id_input = st.text_input("Episode ID or file path", value=default_id)

if not ep_id_input:
    st.info("Enter an episode ID, or select one from **Episode Browser**.")
    st.stop()

try:
    ep = load_episode(ep_id_input)
except FileNotFoundError:
    st.error(f"Episode not found: `{ep_id_input}`")
    st.stop()

if not ep.compressed_states:
    st.warning(
        f"Episode `{ep.episode_id}` has no compression events (mode: `{ep.agent_mode}`). "
        "Compression is only active in `compressor` and `mcts_compressor` modes."
    )
    st.stop()

# ── Compression selector ──────────────────────────────────────────────────────

cs_options = [
    f"Compression {i+1} @ step {cs.step_index} ({cs.compression_method})"
    for i, cs in enumerate(ep.compressed_states)
]
sel_label = st.selectbox("Select compression event", cs_options)
sel_idx = cs_options.index(sel_label)
cs = ep.compressed_states[sel_idx]

# Identify which steps were compressed (steps up to and including cs.step_index)
steps = ep.trajectory.steps
compressed_steps = [s for s in steps if s.step_index <= cs.step_index]

# Token counts
input_chars = sum(len(s.thought or "") + len(str(s.action or "")) for s in compressed_steps)
output_tokens = cs.token_count or "?"

col_a, col_b, col_c = st.columns(3)
col_a.metric("Steps compressed", len(compressed_steps))
col_b.metric("Input chars (approx)", input_chars)
col_c.metric("Output tokens", output_tokens)

st.divider()

# ── Side-by-side view ─────────────────────────────────────────────────────────

left, right = st.columns(2)

with left:
    st.subheader("Trajectory input")
    for step in compressed_steps:
        with st.expander(
            f"Step {step.step_index}"
            + (f" — {step.action.tool_name}" if step.action else " — DONE"),
            expanded=False,
        ):
            if step.thought:
                st.markdown("**Thought:**")
                st.text(step.thought)
            if step.action:
                st.markdown("**Action:**")
                st.code(
                    f"{step.action.tool_name}({step.action.arguments})",
                    language="python",
                )
            if step.observation:
                obs = step.observation
                status = "✅" if obs.success else "❌"
                st.markdown(f"**Observation** {status}")
                if obs.result:
                    result_str = (
                        str(obs.result)[:300] + "…"
                        if len(str(obs.result)) > 300
                        else str(obs.result)
                    )
                    st.text(result_str)
                elif obs.error_message:
                    st.error(obs.error_message)

with right:
    st.subheader("CompressedState output")

    # Hard constraint ledger
    with st.container(border=True):
        st.markdown("**Hard Constraint Ledger**")
        hcl = cs.hard_constraint_ledger
        c1, c2, c3 = st.columns(3)
        c1.metric("Satisfied ✅", len(hcl.satisfied_ids))
        c2.metric("Violated ❌", len(hcl.violated_ids))
        c3.metric("Unknown ❓", len(hcl.unknown_ids))
        for constraint in hcl.constraints:
            icon = (
                "✅" if getattr(constraint, "constraint_id", None) in hcl.satisfied_ids
                else "❌" if getattr(constraint, "constraint_id", None) in hcl.violated_ids
                else "❓"
            )
            st.caption(f"{icon} {getattr(constraint, 'description', str(constraint))}")

    # Soft constraints
    with st.container(border=True):
        st.markdown("**Soft Constraints Summary**")
        st.text(cs.soft_constraints_summary or "(empty)")

    # Decisions made
    with st.container(border=True):
        st.markdown("**Decisions Made**")
        if cs.decisions_made:
            for d in cs.decisions_made:
                st.markdown(f"- {d}")
        else:
            st.caption("(none)")

    # Open questions
    with st.container(border=True):
        st.markdown("**Open Questions**")
        if cs.open_questions:
            for q in cs.open_questions:
                st.markdown(f"- {q}")
        else:
            st.caption("(none)")

    # Key discoveries
    with st.container(border=True):
        st.markdown("**Key Discoveries**")
        if cs.key_discoveries:
            for k in cs.key_discoveries:
                st.markdown(f"- {k}")
        else:
            st.caption("(none)")

    # Itinerary sketch
    with st.container(border=True):
        st.markdown("**Current Itinerary Sketch**")
        st.text(cs.current_itinerary_sketch or "(empty)")

    # MCTS fields (only for MCTS compressor)
    if cs.top_candidates:
        with st.container(border=True):
            st.markdown("**Top Candidate Plans (MCTS)**")
            for i, cand in enumerate(cs.top_candidates):
                st.markdown(f"{i+1}. {cand}")

    if cs.tradeoffs:
        with st.container(border=True):
            st.markdown("**Tradeoffs (MCTS)**")
            st.text(cs.tradeoffs)
