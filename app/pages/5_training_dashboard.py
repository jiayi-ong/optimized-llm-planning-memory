"""
pages/5_training_dashboard.py
==============================
Training Dashboard — PPO convergence analysis and reward trend charts.

Reads from ``outputs/training/<run_id>/``:
  - ``ppo_metrics.jsonl``     — per-PPO-update diagnostics (loss, KL, clip, etc.)
  - ``episode_metrics.jsonl`` — per-episode summaries (reward components, step counts)

Charts rendered (using Streamlit's built-in line chart):
  - Episode reward (total + rolling mean)
  - Reward component breakdown over episodes
  - PPO losses: policy, value, entropy
  - Convergence diagnostics: approx_kl, clip_fraction, explained_variance
  - Episode length over time
  - Tool success rate over episodes

Convergence interpretation guide is shown in the sidebar.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Training Dashboard", layout="wide")
st.title("Training Dashboard")

from app.utils.data_loader import (  # noqa: E402
    list_run_ids,
    load_episode_metrics,
    load_ppo_metrics,
)

# ── Run selector ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Training Run")
    run_ids = list_run_ids()
    if not run_ids:
        st.warning("No training runs found in `outputs/training/`.")
        st.stop()

    sel_run = st.selectbox("Select run", run_ids)
    st.caption(f"Run ID: `{sel_run}`")

    st.divider()
    st.subheader("Convergence Guide")
    st.markdown(
        """
**approx_kl** should stay below ~0.02. A spike means the policy
moved too far from the previous version in one update — lower lr or increase epochs.

**clip_fraction** > 0.2 = PPO is clipping aggressively.
Reduce learning rate or clip_epsilon.

**explained_variance** near 1.0 = value function fits returns well.
Near 0 = value head is not learning (check vf_coef).

**reward_mean_20** should trend upward.  Flat or declining = compressor
not improving — check entropy_loss and KL.
"""
    )

# ── Load data ─────────────────────────────────────────────────────────────────

with st.spinner("Loading training data..."):
    ppo_records = load_ppo_metrics(sel_run)
    ep_records = load_episode_metrics(sel_run)

if not ppo_records and not ep_records:
    st.info(
        f"No data found for run `{sel_run}`. "
        "Make sure training completed at least one update."
    )
    st.stop()

# ── Build DataFrames ──────────────────────────────────────────────────────────

ppo_df = pd.DataFrame([r.model_dump() for r in ppo_records]) if ppo_records else pd.DataFrame()
ep_df = pd.DataFrame([r.model_dump() for r in ep_records]) if ep_records else pd.DataFrame()

# ── Episode reward section ────────────────────────────────────────────────────

if not ep_df.empty:
    st.subheader("Episode Rewards")

    col1, col2 = st.columns(2)
    with col1:
        # Raw reward + rolling mean
        reward_chart_df = ep_df[["total_reward"]].copy()
        if "reward_mean_20" in ep_df.columns:
            reward_chart_df["reward_mean_20"] = ep_df["reward_mean_20"]
        st.line_chart(reward_chart_df, use_container_width=True)
        st.caption("Total reward per episode (raw + 20-ep rolling mean)")

    with col2:
        # Reward component breakdown
        component_cols = [
            c for c in [
                "hard_constraint_score",
                "soft_constraint_score",
                "tool_efficiency_score",
                "logical_consistency_score",
            ]
            if c in ep_df.columns
        ]
        if component_cols:
            st.line_chart(ep_df[component_cols], use_container_width=True)
            st.caption("Reward components over episodes")

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        if "total_steps" in ep_df.columns:
            st.line_chart(ep_df[["total_steps"]], use_container_width=True)
            st.caption("Episode length (steps) over training")

    with col4:
        if "tool_success_rate" in ep_df.columns:
            st.line_chart(ep_df[["tool_success_rate"]], use_container_width=True)
            st.caption("Tool call success rate over episodes")

    # Summary stats
    st.subheader("Episode Summary Stats")
    summary_cols = [
        c for c in [
            "total_reward", "hard_constraint_score", "soft_constraint_score",
            "tool_success_rate", "total_steps", "num_compressions",
        ]
        if c in ep_df.columns
    ]
    st.dataframe(ep_df[summary_cols].describe().round(3), use_container_width=True)

# ── PPO update section ────────────────────────────────────────────────────────

if not ppo_df.empty:
    st.divider()
    st.subheader("PPO Update Diagnostics")

    col5, col6 = st.columns(2)
    with col5:
        loss_cols = [c for c in ["policy_loss", "value_loss", "entropy_loss"] if c in ppo_df.columns]
        if loss_cols:
            st.line_chart(ppo_df[loss_cols], use_container_width=True)
            st.caption("Policy / value / entropy losses per update")

    with col6:
        conv_cols = [c for c in ["approx_kl", "clip_fraction", "explained_variance"] if c in ppo_df.columns]
        if conv_cols:
            st.line_chart(ppo_df[conv_cols], use_container_width=True)
            st.caption("Convergence diagnostics per update (KL, clip fraction, explained variance)")

    if "learning_rate" in ppo_df.columns:
        st.line_chart(ppo_df[["learning_rate"]], use_container_width=True)
        st.caption("Learning rate schedule")

    if "grad_norm" in ppo_df.columns and not ppo_df["grad_norm"].isna().all():
        st.line_chart(ppo_df[["grad_norm"]], use_container_width=True)
        st.caption("Gradient norm per update")

    # Latest-update metrics at a glance
    latest = ppo_records[-1]
    st.subheader("Latest Update")
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric("approx_kl", f"{latest.approx_kl:.4f}")
    lc2.metric("clip_fraction", f"{latest.clip_fraction:.3f}")
    lc3.metric("explained_variance", f"{latest.explained_variance:.3f}")
    lc4.metric("learning_rate", f"{latest.learning_rate:.2e}")

    st.subheader("PPO Update Stats")
    stat_cols = [c for c in ["policy_loss", "value_loss", "entropy_loss", "approx_kl", "clip_fraction", "explained_variance"] if c in ppo_df.columns]
    st.dataframe(ppo_df[stat_cols].describe().round(4), use_container_width=True)

# ── Raw data download ─────────────────────────────────────────────────────────

with st.expander("Raw data"):
    tab_ep, tab_ppo = st.tabs(["Episode metrics", "PPO metrics"])
    with tab_ep:
        if not ep_df.empty:
            st.dataframe(ep_df, use_container_width=True)
            st.download_button(
                "Download episode_metrics.csv",
                ep_df.to_csv(index=False),
                file_name=f"{sel_run}_episode_metrics.csv",
            )
    with tab_ppo:
        if not ppo_df.empty:
            st.dataframe(ppo_df, use_container_width=True)
            st.download_button(
                "Download ppo_metrics.csv",
                ppo_df.to_csv(index=False),
                file_name=f"{sel_run}_ppo_metrics.csv",
            )
