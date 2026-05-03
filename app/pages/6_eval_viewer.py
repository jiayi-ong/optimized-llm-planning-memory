"""
app/pages/6_eval_viewer.py
===========================
Evaluation results viewer — multi-dimensional, version-aware.

Panels
------
1. Sidebar filters  — metric_version, agent_mode, world_seed, request_id,
                      "latest per config" toggle, sort order
2. All Runs table   — high-level summary across selected runs
3. Metric Heatmap   — requests × metrics for a single selected run
4. Cross-Mode Comparison — same request × world, different agent_modes
5. Per-Episode Drilldown — full scores, rubric breakdown, eval_key

Metric version mismatch
-----------------------
When results from different metric versions are mixed in the selection,
a st.warning() banner is shown. The eval_key field is used to deduplicate
when "latest per config only" is enabled.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from optimized_llm_planning_memory.utils.episode_io import list_eval_runs, load_eval_run

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = BASE_DIR / "outputs" / "eval_results"

V1_METRICS = [
    "hard_constraint_ratio", "soft_constraint_score", "budget_adherence",
    "logical_consistency", "tool_efficiency", "tool_failure_rate",
    "avg_tool_latency_ms", "steps_per_episode",
]
V2_EXTRA_METRICS = [
    "destination_coverage_ratio", "accommodation_coverage_ratio",
    "activity_density_score", "rest_day_ratio",
    "schedule_overlap_score", "intra_city_feasibility",
]
SCORE_METRICS = [m for m in V1_METRICS + V2_EXTRA_METRICS
                 if m not in ("avg_tool_latency_ms", "steps_per_episode", "tool_failure_rate")]

# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _load_all_results(eval_dir: str) -> pd.DataFrame:
    """Load all eval runs into a single flat DataFrame."""
    rows: list[dict] = []
    manifests = list_eval_runs(Path(eval_dir))
    for manifest in manifests:
        try:
            _, results = load_eval_run(manifest.run_id, Path(eval_dir))
        except Exception:
            continue
        for r in results:
            row: dict = {
                "run_id": manifest.run_id,
                "created_at": r.created_at,
                "agent_mode": r.agent_mode,
                "metric_version": r.metric_version,
                "world_seed": r.world_seed,
                "episode_id": r.episode_id,
                "request_id": r.request_id,
                "overall_score": r.overall_score,
                "eval_key": r.eval_key,
                "judge_model": r.judge_model,
                "_rubric": r.rubric_breakdown,
                "_llm_scores": r.llm_judge_scores,
            }
            row.update({f"det_{k}": v for k, v in r.deterministic_scores.items()})
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ── Page ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Eval Viewer", layout="wide")
st.title("Evaluation Viewer")

raw_df = _load_all_results(str(EVAL_DIR))

if raw_df.empty:
    st.info(
        "No evaluation runs found in `outputs/eval_results/`. "
        "Run `python scripts/run_eval.py --deterministic_only` to generate some."
    )
    st.stop()

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    all_versions = sorted(raw_df["metric_version"].dropna().unique().tolist())
    sel_versions = st.multiselect("Metric version", all_versions, default=all_versions)

    all_modes = sorted(raw_df["agent_mode"].dropna().unique().tolist())
    sel_modes = st.multiselect("Agent mode", all_modes, default=all_modes)

    all_seeds = sorted(raw_df["world_seed"].dropna().unique().tolist())
    seed_options = ["(all)"] + [str(s) for s in all_seeds]
    sel_seeds_raw = st.multiselect("World seed", seed_options, default=["(all)"])
    sel_seeds = None if "(all)" in sel_seeds_raw else {int(s) for s in sel_seeds_raw if s != "(all)"}

    all_requests = sorted(raw_df["request_id"].dropna().unique().tolist())
    req_options = ["(all)"] + all_requests
    sel_requests_raw = st.multiselect("Request ID", req_options, default=["(all)"])
    sel_requests = None if "(all)" in sel_requests_raw else set(sel_requests_raw) - {"(all)"}

    latest_only = st.checkbox("Latest per config only", value=True,
                              help="For each eval_key, keep only the most-recent result.")

    sort_by = st.selectbox("Sort by", ["newest first", "overall_score ↓", "request_id ↑"])

# ── Apply filters ─────────────────────────────────────────────────────────────
df = raw_df.copy()
if sel_versions:
    df = df[df["metric_version"].isin(sel_versions)]
if sel_modes:
    df = df[df["agent_mode"].isin(sel_modes)]
if sel_seeds is not None:
    df = df[df["world_seed"].isin(sel_seeds)]
if sel_requests is not None:
    df = df[df["request_id"].isin(sel_requests)]

if latest_only and "eval_key" in df.columns and not df.empty:
    df = df.sort_values("created_at", ascending=False)
    df = df.drop_duplicates(subset="eval_key", keep="first")

if sort_by == "newest first":
    df = df.sort_values("created_at", ascending=False)
elif sort_by == "overall_score ↓":
    df = df.sort_values("overall_score", ascending=False)
elif sort_by == "request_id ↑":
    df = df.sort_values("request_id", ascending=True)

if df.empty:
    st.warning("No results match the current filters.")
    st.stop()

# ── Metric version mismatch banner ────────────────────────────────────────────
selected_versions = df["metric_version"].unique().tolist()
if len(selected_versions) > 1:
    st.warning(
        f"⚠️ Comparing results from metric versions: **{', '.join(sorted(selected_versions))}**. "
        "Scores may not be directly comparable — v2 results have 6 additional metrics."
    )

# ── Panel 1: All Runs summary table ───────────────────────────────────────────
st.subheader("All Runs")

summary_rows = []
for run_id, grp in df.groupby("run_id"):
    summary_rows.append({
        "run_id": run_id[:20],
        "timestamp": grp["created_at"].min()[:19].replace("T", " "),
        "agent_mode": ", ".join(sorted(grp["agent_mode"].unique())),
        "world_seed": ", ".join(str(s) for s in sorted(grp["world_seed"].dropna().unique())),
        "metric_ver": ", ".join(sorted(grp["metric_version"].unique())),
        "n_episodes": len(grp),
        "overall_mean": grp["overall_score"].mean(),
        "hard_constr_mean": grp["det_hard_constraint_ratio"].mean() if "det_hard_constraint_ratio" in grp else float("nan"),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("timestamp", ascending=False)
st.dataframe(
    summary_df.style.format({
        "overall_mean": "{:.3f}",
        "hard_constr_mean": "{:.3f}",
    }),
    use_container_width=True,
    hide_index=True,
)

# ── Panel 2: Metric Heatmap ────────────────────────────────────────────────────
st.subheader("Metric Heatmap")

run_ids_available = sorted(df["run_id"].unique().tolist(), reverse=True)
selected_run = st.selectbox("Select run for heatmap", run_ids_available)

heatmap_df = df[df["run_id"] == selected_run].copy()
det_cols = [c for c in heatmap_df.columns if c.startswith("det_")
            and c.replace("det_", "") in SCORE_METRICS]

if det_cols and not heatmap_df.empty:
    pivot = heatmap_df.pivot_table(
        index="request_id", values=det_cols, aggfunc="mean"
    )
    pivot.columns = [c.replace("det_", "") for c in pivot.columns]
    pivot = pivot.reindex(columns=[m for m in SCORE_METRICS if m in pivot.columns])

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.1), max(3, len(pivot) * 0.5 + 1)))
        data = pivot.values.astype(float)
        im = ax.imshow(data, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels([rid[:24] for rid in pivot.index], fontsize=8)
        for i in range(len(pivot)):
            for j in range(len(pivot.columns)):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="black" if 0.25 < v < 0.75 else "white")
        plt.colorbar(im, ax=ax, label="Score [0–1]")
        plt.title(f"Metric Heatmap — run {selected_run[:20]}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except ImportError:
        st.dataframe(
            pivot.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1).format("{:.3f}"),
            use_container_width=True,
        )
else:
    st.info("No scoreable deterministic metrics found in the selected run.")

# ── Panel 3: Cross-Mode Comparison ────────────────────────────────────────────
unique_modes_in_selection = df["agent_mode"].unique().tolist()
if len(unique_modes_in_selection) >= 2:
    st.subheader("Cross-Mode Comparison")
    st.caption("Rows where 2+ agent_modes have results for the same (request_id, world_seed).")

    pivot_key = df.groupby(["request_id", "world_seed", "agent_mode"])["overall_score"].mean().unstack("agent_mode")
    # Keep only rows with 2+ modes scored
    pivot_key = pivot_key.dropna(thresh=2)

    if not pivot_key.empty:
        pivot_key = pivot_key.reset_index()
        st.dataframe(
            pivot_key.style.format({c: "{:.3f}" for c in pivot_key.select_dtypes("float").columns}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No (request_id, world_seed) pairs have results under multiple agent_modes yet.")

# ── Panel 4: Per-Episode Drilldown ────────────────────────────────────────────
st.subheader("Per-Episode Drilldown")

for _, row in df.iterrows():
    with st.expander(
        f"{row['request_id'][:28]}  ·  {row['agent_mode']}  ·  overall={row['overall_score']:.3f}  "
        f"·  {row.get('created_at', '')[:19].replace('T', ' ')}"
    ):
        col1, col2 = st.columns(2)

        with col1:
            st.caption(f"**eval_key:** `{row.get('eval_key', 'n/a')}`")
            st.caption(f"**episode_id:** `{row['episode_id']}`")
            st.caption(f"**metric_version:** {row['metric_version']}  |  **world_seed:** {row.get('world_seed', 'n/a')}")

            det_rows = [
                {"metric": k.replace("det_", ""), "score": f"{v:.4f}"}
                for k, v in row.items()
                if isinstance(k, str) and k.startswith("det_")
            ]
            if det_rows:
                st.markdown("**Deterministic scores**")
                st.dataframe(pd.DataFrame(det_rows), use_container_width=True, hide_index=True)

        with col2:
            llm_scores = row.get("_llm_scores", {})
            if llm_scores:
                st.markdown("**LLM judge scores**")
                llm_rows = [{"dimension": k, "score": f"{v:.4f}"} for k, v in llm_scores.items()]
                st.dataframe(pd.DataFrame(llm_rows), use_container_width=True, hide_index=True)

            rubric = row.get("_rubric", {})
            if rubric:
                st.markdown("**Rubric breakdown**")
                for dim, detail in rubric.items():
                    if isinstance(detail, dict):
                        score = detail.get("score", detail.get(dim, "?"))
                        reasoning = detail.get("reasoning", detail.get(f"{dim}_reasoning", ""))
                        with st.expander(f"{dim}: {score}", expanded=False):
                            if reasoning:
                                st.write(reasoning)
                    else:
                        st.write(f"{dim}: {detail}")

        st.caption("→ View trajectory in the **Trajectory Viewer** tab (filter by episode_id above)")
