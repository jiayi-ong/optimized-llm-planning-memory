"""
pages/4_mcts_viewer.py
=======================
MCTS Viewer — search statistics, top candidate plans, and tradeoffs.

Shows the ``MCTSStats`` and MCTS-specific CompressedState fields from an
episode that used ``mcts_compressor`` mode.

Deep tree visualisation (graph rendering) is intentionally deferred: the
full ``MCTSTreeRepresentation`` is not stored in ``EpisodeLog`` (only the
aggregate ``MCTSStats`` is).  A future enhancement would add an optional
``mcts_tree_repr`` field to ``EpisodeLog`` and render it here with networkx /
graphviz.  For now, the top-K candidate paths and tradeoff summaries captured
in the CompressedState are displayed.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="MCTS Viewer", layout="wide")
st.title("MCTS Viewer")

from app.utils.data_loader import load_episode  # noqa: E402

# ── Episode selection ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Episode")
    default_id = st.session_state.get("selected_episode_id", "")
    ep_id_input = st.text_input("Episode ID or file path", value=default_id)

if not ep_id_input:
    st.info("Enter an episode ID in the sidebar, or select one from **Episode Browser**.")
    st.stop()

try:
    ep = load_episode(ep_id_input)
except FileNotFoundError:
    st.error(f"Episode not found: `{ep_id_input}`")
    st.stop()

if ep.agent_mode != "mcts_compressor":
    st.warning(
        f"Episode `{ep.episode_id}` used mode `{ep.agent_mode}`. "
        "MCTS data is only present for `mcts_compressor` mode episodes."
    )

# ── MCTSStats ─────────────────────────────────────────────────────────────────

st.subheader("Search Statistics")

if ep.mcts_stats is None:
    st.info("No ``MCTSStats`` in this episode (MCTS may not have run).")
else:
    stats = ep.mcts_stats
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Nodes explored", stats.nodes_explored)
    c2.metric("Max depth reached", stats.max_depth_reached)
    c3.metric("Simulations run", stats.num_simulations)
    c4.metric("Root value", f"{stats.root_value:.3f}")
    c5.metric("Avg branching factor", f"{stats.avg_branching_factor:.2f}")
    if hasattr(stats, "best_path_length"):
        st.caption(f"Best path length: {stats.best_path_length}")

st.divider()

# ── MCTS fields from CompressedStates ─────────────────────────────────────────

mcts_states = [cs for cs in ep.compressed_states if cs.top_candidates or cs.tradeoffs]

if not mcts_states:
    st.info(
        "No CompressedState with MCTS fields (top_candidates / tradeoffs) found. "
        "These are only populated by ``MCTSAwareCompressor`` implementations."
    )
else:
    options = [
        f"Compression @ step {cs.step_index} ({cs.compression_method})"
        for cs in mcts_states
    ]
    sel = st.selectbox("Select compression event", options)
    cs = mcts_states[options.index(sel)]

    left, right = st.columns(2)

    with left:
        st.subheader("Top Candidate Plans")
        if cs.top_candidates:
            for i, cand in enumerate(cs.top_candidates):
                with st.container(border=True):
                    st.markdown(f"**Candidate {i+1}**")
                    st.text(cand)
        else:
            st.caption("(no candidates recorded)")

    with right:
        st.subheader("Tradeoff Analysis")
        if cs.tradeoffs:
            st.text_area(
                label="Tradeoffs",
                value=cs.tradeoffs,
                height=300,
                disabled=True,
            )
        else:
            st.caption("(no tradeoff summary recorded)")
