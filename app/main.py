"""
app/main.py
===========
Developer UI — entry point.

Launch with:
    streamlit run app/main.py

Or use the convenience wrapper:
    python scripts/launch_ui.py

Pages (defined in app/pages/):
    1. Episode Browser      — browse and filter saved EpisodeLogs
    2. Trajectory Viewer    — step-by-step ReAct trail + live feed
    3. Compression Viewer   — side-by-side trajectory vs CompressedState
    4. MCTS Viewer          — MCTS stats, top candidates, tradeoffs
    5. Training Dashboard   — PPO convergence charts from JSONL + TensorBoard
    6. Itinerary Viewer     — day-by-day itinerary cards (flights, hotels, activities)

Architecture
------------
Streamlit's native multipage support (pages/ directory) handles routing.
This file renders the landing/home page and sets shared page config.
"""

import sys
from pathlib import Path

# Streamlit adds app/ (the script dir) to sys.path, not the repo root.
# Insert the repo root so `from app.*` imports resolve correctly.
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import streamlit as st

st.set_page_config(
    page_title="Travel Planner Dev UI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Travel Planner — Developer UI")

st.markdown(
    """
Use the sidebar to navigate between views.

| Page | Purpose |
|---|---|
| **Episode Browser** | Browse and filter all saved episodes |
| **Trajectory Viewer** | Inspect a full ReAct trail step-by-step; live mode for running episodes |
| **Compression Viewer** | Compare trajectory text with the resulting CompressedState |
| **MCTS Viewer** | Explore MCTS search statistics and top candidate plans |
| **Training Dashboard** | PPO convergence curves, reward trends, episode diagnostics |
| **Itinerary Viewer** | Day-by-day view of the agent's final itinerary (flights, hotels, activities) |
"""
)

st.divider()

# Quick-status panel: how many episodes are saved, any live episodes running?
from app.utils.data_loader import (  # noqa: E402
    _episodes_dir,
    _training_dir,
    list_live_episode_ids,
    list_run_ids,
)

col1, col2, col3 = st.columns(3)

with col1:
    ep_dir = _episodes_dir()
    n_episodes = len(list(ep_dir.glob("ep_*.json"))) if ep_dir.exists() else 0
    st.metric("Saved Episodes", n_episodes)

with col2:
    live_ids = list_live_episode_ids()
    st.metric("Live Episodes", len(live_ids))
    if live_ids:
        st.caption("Active: " + ", ".join(live_ids[:3]))

with col3:
    run_ids = list_run_ids()
    st.metric("Training Runs", len(run_ids))
    if run_ids:
        st.caption("Latest: " + run_ids[0])
