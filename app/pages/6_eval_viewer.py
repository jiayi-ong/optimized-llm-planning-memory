import streamlit as st
from pathlib import Path
import pandas as pd

from optimized_llm_planning_memory.utils.episode_io import (
    list_eval_runs,
    load_eval_run,
)

BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = BASE_DIR / "outputs" / "eval_results"

st.title("Evaluation Viewer")

manifests = list_eval_runs(EVAL_DIR)

if not manifests:
    st.info("No evaluation runs found.")
    st.stop()

run_ids = [m.run_id for m in manifests]
selected_run = st.selectbox("Select run", run_ids)

_, results = load_eval_run(selected_run, EVAL_DIR)

rows = []
for r in results:
    rows.append({
        "episode_id": r.episode_id,
        "request_id": r.request_id,
        "overall_score": r.overall_score,
        **r.deterministic_scores,
    })

df = pd.DataFrame(rows)

st.subheader("Per-episode results")
st.dataframe(df)

st.subheader("Averages")
st.write(df.mean(numeric_only=True))