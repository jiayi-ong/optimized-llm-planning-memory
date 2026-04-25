"""
pages/1_episode_browser.py
===========================
Episode Browser — list, filter, and select saved EpisodeLogs.

Displays a filterable table of all episodes in ``outputs/episodes/``.
Clicking a row stores the episode_id in ``st.session_state`` so Trajectory
Viewer can open it directly.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Episode Browser", layout="wide")
st.title("Episode Browser")

from app.utils.data_loader import load_episodes  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tool_success_rate(ep) -> float:
    tool_stats = ep.tool_stats
    if not tool_stats:
        return 0.0
    total_calls = sum(getattr(ts, "call_count", 0) for ts in tool_stats)
    total_success = sum(getattr(ts, "success_count", 0) for ts in tool_stats)
    return round(total_success / total_calls, 3) if total_calls > 0 else 0.0


# ── Load episodes ─────────────────────────────────────────────────────────────

with st.spinner("Loading episodes..."):
    episodes = load_episodes()

if not episodes:
    st.info("No episodes found in `outputs/episodes/`. Run `scripts/run_episode.py` to generate one.")
    st.stop()

# ── Build summary table ───────────────────────────────────────────────────────

rows = []
for ep in episodes:
    rows.append(
        {
            "episode_id": ep.episode_id,
            "mode": ep.agent_mode,
            "success": ep.success,
            "steps": ep.total_steps,
            "compressions": len(ep.compressed_states),
            "total_reward": round(ep.reward_components.total_reward, 3),
            "hard_constraint": round(ep.reward_components.hard_constraint_score, 3),
            "tool_success": _tool_success_rate(ep),
            "request_id": ep.request_id,
            "created_at": ep.created_at[:19].replace("T", " "),
        }
    )

import pandas as pd  # noqa: E402

df = pd.DataFrame(rows)

# ── Filters ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters")

    mode_opts = ["All"] + sorted(df["mode"].unique().tolist())
    sel_mode = st.selectbox("Agent mode", mode_opts)

    success_opts = ["All", "Success only", "Failures only"]
    sel_success = st.selectbox("Outcome", success_opts)

    min_reward = float(df["total_reward"].min())
    max_reward = float(df["total_reward"].max())
    if min_reward < max_reward:
        reward_range = st.slider(
            "Total reward range",
            min_value=min_reward,
            max_value=max_reward,
            value=(min_reward, max_reward),
        )
    else:
        reward_range = (min_reward, max_reward)

filt = df.copy()
if sel_mode != "All":
    filt = filt[filt["mode"] == sel_mode]
if sel_success == "Success only":
    filt = filt[filt["success"]]
elif sel_success == "Failures only":
    filt = filt[~filt["success"]]
filt = filt[
    (filt["total_reward"] >= reward_range[0])
    & (filt["total_reward"] <= reward_range[1])
]

st.caption(f"Showing {len(filt)} of {len(df)} episodes")

# ── Display table ─────────────────────────────────────────────────────────────

selected = st.dataframe(
    filt.reset_index(drop=True),
    use_container_width=True,
    selection_mode="single-row",
    on_select="rerun",
    key="episode_table",
)

# ── Selection handling ────────────────────────────────────────────────────────

sel_rows = selected.get("selection", {}).get("rows", [])
if sel_rows:
    row_idx = sel_rows[0]
    ep_id = filt.iloc[row_idx]["episode_id"]
    st.session_state["selected_episode_id"] = ep_id
    st.success(f"Selected episode `{ep_id}` — open **Trajectory Viewer** to inspect it.")

    # Quick summary panel for the selected episode
    sel_ep = next((e for e in episodes if e.episode_id == ep_id), None)
    if sel_ep:
        st.subheader("Quick Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mode", sel_ep.agent_mode)
        c2.metric("Steps", sel_ep.total_steps)
        c3.metric("Total reward", round(sel_ep.reward_components.total_reward, 3))
        c4.metric("Hard constraints", round(sel_ep.reward_components.hard_constraint_score, 3))

        if sel_ep.final_itinerary:
            it = sel_ep.final_itinerary
            st.caption(
                f"Itinerary: {len(it.days)} days · ${it.total_cost_usd:.2f} · "
                f"Complete: {it.is_complete}"
            )
