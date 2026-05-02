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

import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

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
        f"Episode `{ep.episode_id}` used mode `{ep.agent_mode}` — "
        "MCTS data is only present for `mcts_compressor` mode episodes.  \n"
        "Correct command: `python scripts/run_episode.py agent=react_mcts compressor=llm_mcts`"
    )

# ── MCTSStats ─────────────────────────────────────────────────────────────────

st.subheader("Search Statistics")

if ep.mcts_stats is None:
    if ep.agent_mode == "mcts_compressor" and ep.total_steps < 6:
        st.warning(
            f"Episode ran only **{ep.total_steps}** step(s) — MCTS fires at compression "
            f"events, which require at least **5 steps** by default (first fires at step 5).  \n"
            f"Termination reason: `{ep.termination_reason or 'unknown'}`.  \n"
            "Run a longer episode to see MCTS statistics."
        )
    elif ep.agent_mode == "mcts_compressor":
        st.warning(
            "This episode ran in `mcts_compressor` mode but MCTSStats were not recorded.  \n"
            "Verify that both `agent=react_mcts` **and** `compressor=llm_mcts` were specified."
        )
    else:
        st.info("MCTSStats are only present for `mcts_compressor` mode episodes.")
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
    if ep.agent_mode == "mcts_compressor" and not ep.compressed_states:
        st.info(
            "No compression events occurred — MCTS fields are populated at each "
            "compression event. See the note above about episode length."
        )
    else:
        st.info(
            "No CompressedState with MCTS fields (top_candidates / tradeoffs) found. "
            "These are populated by `LLMMCTSCompressor.compress_with_tree()` — "
            "ensure `compressor=llm_mcts` was used alongside `agent=react_mcts`."
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
