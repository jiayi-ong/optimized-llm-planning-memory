"""
app/pages/3_eval_episode_detail.py
====================================
Episode Detail — full drill-down for a single evaluated episode.

Tabs
----
1. Overview       — user request card, episode summary, key metric gauges.
2. Itinerary      — day-by-day view of the final itinerary with constraint overlay.
3. Trajectory     — paginated step viewer (thought / action / observation).
4. Evaluation     — all deterministic + LLM judge scores; rubric breakdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.utils.ui_style import PALETTE, PRIMARY, SECONDARY, SUCCESS, WARNING, inject_css, score_color
from optimized_llm_planning_memory.evaluation.eval_store import EvalStore

BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = BASE_DIR / "outputs" / "eval_results"
EPISODES_DIR = BASE_DIR / "outputs" / "episodes"
REQUESTS_DIR = BASE_DIR / "data" / "user_requests"

st.set_page_config(page_title="Episode Detail", layout="wide", page_icon="🔍")
inject_css()

store = EvalStore(EVAL_DIR, EPISODES_DIR, REQUESTS_DIR)

# ── Episode ID input ──────────────────────────────────────────────────────────

st.markdown("## 🔍 Episode Detail")
st.caption(
    "Full drill-down for a single evaluated episode: request card, day-by-day itinerary, "
    "step-by-step trajectory, and all eval scores."
)

episode_id_in = st.text_input(
    "Episode ID",
    placeholder="Paste episode_id from the Episode Log page",
)

if not episode_id_in:
    st.info("Enter an episode ID above to load details.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _load(ep_id: str, eval_dir: str) -> tuple:
    """Return (episode, user_request, eval_result | None)."""
    _store = EvalStore(Path(eval_dir), EPISODES_DIR, REQUESTS_DIR)
    try:
        episode = _store.load_episode(ep_id)
    except FileNotFoundError:
        return None, None, None
    req = _store.load_request(episode.request_id) if episode else None
    # Search for an EvalResult for this episode
    eval_result = None
    for manifest in _store.list_runs():
        try:
            _, results = _store.load_run(manifest.run_id)
            for r in results:
                if r.episode_id == ep_id:
                    eval_result = r
                    break
        except Exception:
            continue
        if eval_result:
            break
    return episode, req, eval_result


episode, req, eval_result = _load(episode_id_in.strip(), str(EVAL_DIR))

if episode is None:
    st.error(f"Episode `{episode_id_in}` not found in `{EPISODES_DIR}`.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Itinerary", "Trajectory", "Evaluation"])

# ── Tab 1: Overview ───────────────────────────────────────────────────────────

with tab1:
    col_req, col_ep = st.columns([1, 1], gap="large")

    with col_req:
        st.markdown("### 📋 User Request")
        if req:
            st.markdown(f"**ID:** `{req.request_id}`")
            st.markdown(f"**Archetype:** {req.metadata.get('archetype', '?')}  &nbsp;|&nbsp;  **World:** {req.world_id or '?'}")
            st.markdown(f"**Trip:** {req.start_date} → {req.end_date} &nbsp;|&nbsp; Budget: **${req.budget_usd:,.0f}**")
            prof = req.traveler_profile
            st.markdown(f"**Party:** {prof.num_adults} adult(s), {prof.num_children} child(ren)")
            if prof.accessibility_needs:
                st.caption(f"Accessibility: {', '.join(prof.accessibility_needs)}")
            if prof.dietary_restrictions:
                st.caption(f"Dietary: {', '.join(prof.dietary_restrictions)}")

            with st.expander("Raw request text"):
                st.write(req.raw_text)

            with st.expander(f"Hard constraints ({len(req.hard_constraints)})"):
                for c in req.hard_constraints:
                    sat = "✅" if c.satisfied else ("❌" if c.satisfied is False else "⬜")
                    st.markdown(f"{sat} **[{c.category}]** {c.description}")

            with st.expander(f"Soft constraints ({len(req.soft_constraints)})"):
                for c in req.soft_constraints:
                    sat = "✅" if c.satisfied else ("⬜")
                    st.markdown(f"{sat} **[{c.category}]** {c.description}")

            cb = req.metadata.get("complexity_breakdown", {})
            if cb:
                tier = cb.get("complexity_tier", "?")
                tier_color = {"low": SUCCESS, "medium": WARNING, "high": PRIMARY}.get(tier, "#9090A8")
                st.markdown(
                    f"**Complexity:** <span style='background:{tier_color};color:#fff;"
                    f"padding:2px 8px;border-radius:4px;font-size:12px'>{tier.upper()}</span>  "
                    f"&nbsp; score: **{cb.get('complexity_score', '?')}**",
                    unsafe_allow_html=True,
                )
        else:
            st.warning("UserRequest not found for this episode.")

    with col_ep:
        st.markdown("### 🎬 Episode Summary")
        ep_cols = st.columns(3)
        with ep_cols[0]:
            st.metric("Steps", episode.total_steps)
        with ep_cols[1]:
            status_str = "✅ Success" if episode.success else "❌ Failed"
            st.metric("Status", status_str)
        with ep_cols[2]:
            st.metric("Tokens", episode.total_tokens_used or "—")

        st.caption(f"Agent mode: **{episode.agent_mode}** &nbsp;|&nbsp; Termination: `{episode.termination_reason or '?'}`")
        st.caption(f"Episode ID: `{episode.episode_id}`")
        st.caption(f"Created: {episode.created_at[:19].replace('T', ' ')}")

        if eval_result:
            st.markdown("---")
            st.markdown("**Key metrics**")
            m1, m2, m3 = st.columns(3)
            hcr = eval_result.deterministic_scores.get("hard_constraint_ratio", 0.0)
            bud = eval_result.deterministic_scores.get("budget_adherence", 0.0)
            eff = eval_result.deterministic_scores.get("tool_efficiency", 0.0)
            m1.metric("Hard constraints", f"{hcr:.2f}")
            m2.metric("Budget adherence", f"{bud:.2f}")
            m3.metric("Tool efficiency", f"{eff:.2f}")
            st.metric("Overall score", f"{eval_result.overall_score:.3f}")


# ── Tab 2: Itinerary ──────────────────────────────────────────────────────────

with tab2:
    itinerary = episode.final_itinerary
    if itinerary is None:
        st.warning("No final itinerary recorded for this episode.")
    else:
        st.markdown(f"**Total cost:** ${itinerary.total_cost_usd:,.2f}  &nbsp;|&nbsp;  **Complete:** {'✅' if itinerary.is_complete else '❌'}  &nbsp;|&nbsp;  **Version:** {itinerary.version}")

        # Build hard constraint satisfaction set from eval_result if available
        sat_map: dict[str, bool] = {}
        if eval_result and req:
            for c in (req.hard_constraints + req.soft_constraints):
                if c.satisfied is not None:
                    sat_map[c.constraint_id] = c.satisfied

        for day in itinerary.days:
            with st.expander(f"📅 {day.date}  —  {day.city}", expanded=True):
                # Transport
                if day.transport_segments:
                    st.markdown("**Transport**")
                    for seg in day.transport_segments:
                        st.markdown(
                            f"  ✈ {seg.mode.upper()}  {seg.from_location} → {seg.to_location}  "
                            f"({seg.departure_datetime[11:16]} → {seg.arrival_datetime[11:16]})  "
                            f"**${seg.cost_usd:.0f}**"
                        )
                # Accommodation
                if day.accommodation:
                    acc = day.accommodation
                    stars = f"{'★' * int(acc.star_rating or 0)}" if acc.star_rating else ""
                    st.markdown(f"**Hotel** 🏨 {acc.hotel_name} {stars} — ${acc.total_cost_usd:.0f}")
                # Activities
                if day.activities:
                    st.markdown("**Activities**")
                    for act in day.activities:
                        st.markdown(
                            f"  🎯 [{act.category}] {act.activity_name}  "
                            f"@ {act.start_datetime[11:16]} ({act.duration_hours:.1f}h)  "
                            f"**${act.cost_usd:.0f}**"
                        )
                day_total = sum(
                    [s.cost_usd for s in day.transport_segments]
                    + ([day.accommodation.total_cost_usd] if day.accommodation else [])
                    + [a.cost_usd for a in day.activities]
                )
                st.caption(f"Day total: ${day_total:.0f}")


# ── Tab 3: Trajectory ─────────────────────────────────────────────────────────

with tab3:
    steps = episode.trajectory.steps if episode.trajectory else []
    if not steps:
        st.info("No trajectory steps recorded.")
    else:
        steps_per_page = 8
        n_pages = max(1, (len(steps) + steps_per_page - 1) // steps_per_page)
        page_num = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1)
        start = (page_num - 1) * steps_per_page
        page_steps = steps[start: start + steps_per_page]

        st.caption(f"Showing steps {start + 1}–{min(start + steps_per_page, len(steps))} of {len(steps)}")

        with st.container(height=560):
            for step in page_steps:
                with st.expander(
                    f"Step {step.step_index + 1}"
                    + (f" → {step.action.tool_name}" if step.action else " [terminal]"),
                    expanded=True,
                ):
                    st.markdown(f"<div style='color:#FF6B35;font-size:13px'><b>Thought</b></div>", unsafe_allow_html=True)
                    st.write(step.thought)
                    if step.action:
                        st.markdown(f"<div style='color:#4A90D9;font-size:13px'><b>Action: {step.action.tool_name}</b></div>", unsafe_allow_html=True)
                        st.json(step.action.arguments)
                    if step.observation:
                        ok = step.observation.success
                        color = SUCCESS if ok else PRIMARY
                        label = "Observation (success)" if ok else "Observation (failure)"
                        st.markdown(f"<div style='color:{color};font-size:13px'><b>{label}</b></div>", unsafe_allow_html=True)
                        obs_val = step.observation.result or step.observation.error_message
                        if isinstance(obs_val, dict):
                            st.json(obs_val)
                        else:
                            st.write(str(obs_val)[:800])


# ── Tab 4: Evaluation ─────────────────────────────────────────────────────────

with tab4:
    if eval_result is None:
        st.info("No evaluation results found for this episode. Run `python scripts/run_eval.py` first.")
    else:
        st.caption(
            f"eval_key: `{eval_result.eval_key}`  &nbsp;|&nbsp;  "
            f"metric_version: **{eval_result.metric_version}**  &nbsp;|&nbsp;  "
            f"judge: {eval_result.judge_model}"
        )
        if eval_result.itinerary_id:
            st.caption(f"itinerary_id: `{eval_result.itinerary_id}`")

        col_det, col_llm = st.columns(2)

        with col_det:
            st.markdown("#### Deterministic Scores")
            det_rows = [
                {"Metric": k, "Score": v}
                for k, v in sorted(eval_result.deterministic_scores.items())
            ]
            import pandas as pd
            det_df = pd.DataFrame(det_rows)
            st.dataframe(
                det_df.style.format({"Score": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

        with col_llm:
            st.markdown("#### LLM Judge Scores")
            if eval_result.llm_judge_scores:
                llm_rows = [
                    {"Dimension": k, "Score": v}
                    for k, v in sorted(eval_result.llm_judge_scores.items())
                ]
                llm_df = pd.DataFrame(llm_rows)
                st.dataframe(
                    llm_df.style.format({"Score": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No LLM judge scores (deterministic-only run).")

        st.markdown(f"**Overall score: {eval_result.overall_score:.4f}**")

        if eval_result.rubric_breakdown:
            st.markdown("#### Rubric Breakdown")
            with st.container(height=380):
                for dim, detail in eval_result.rubric_breakdown.items():
                    if isinstance(detail, dict):
                        score = detail.get("score", detail.get(dim, "?"))
                        reasoning = detail.get("reasoning", detail.get(f"{dim}_reasoning", ""))
                        with st.expander(f"{dim}: {score}", expanded=False):
                            if reasoning:
                                st.write(reasoning)
                    else:
                        st.write(f"{dim}: {detail}")
