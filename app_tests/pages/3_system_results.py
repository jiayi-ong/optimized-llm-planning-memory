"""
app_tests/pages/3_system_results.py
=====================================
Drill-down page for system test results.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app_tests.utils.display import render_test_list_for_level
from app_tests.utils.result_loader import aggregate_results, cached_load_results

REPO_ROOT = Path(__file__).parent.parent.parent

st.set_page_config(page_title="System Test Results", layout="wide")
st.title("System Test Results")

col1, col2 = st.columns(2)
results_dir = col1.text_input("Results directory", value=str(REPO_ROOT / "test_results"))
status_filter = col2.radio("Status filter", ["All", "Passed", "Failed"], horizontal=True)

all_results = cached_load_results(results_dir)
system_results = [r for r in all_results if r.get("level") == "system"]

if status_filter == "Passed":
    system_results = [
        {**r, "tests": [t for t in r.get("tests", []) if t["status"] == "passed"]}
        for r in system_results
    ]
elif status_filter == "Failed":
    system_results = [
        {**r, "tests": [t for t in r.get("tests", []) if t["status"] in ("failed", "error")]}
        for r in system_results
    ]

agg = aggregate_results(system_results)
overall = agg.get("overall", {})

st.info(
    "System tests run cross-component flows with mocked LLM calls. "
    "Failures here indicate integration issues between components."
)

metric_cols = st.columns(4)
metric_cols[0].metric("Total",   overall.get("total", 0))
metric_cols[1].metric("Passed",  overall.get("passed", 0))
metric_cols[2].metric("Failed",  overall.get("failed", 0))
metric_cols[3].metric("Errors",  overall.get("errors", 0))

st.divider()
render_test_list_for_level(system_results, level="system")
