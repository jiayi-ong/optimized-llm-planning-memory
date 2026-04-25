"""
app_tests/pages/2_module_results.py
=====================================
Drill-down page for module test results.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app_tests.utils.display import render_test_list_for_level
from app_tests.utils.result_loader import aggregate_results, cached_load_results

REPO_ROOT = Path(__file__).parent.parent.parent
ALL_COMPONENTS = [
    "all", "core", "simulator", "tools", "agent",
    "compressor", "training", "evaluation", "mcts", "utils",
]

st.set_page_config(page_title="Module Test Results", layout="wide")
st.title("Module Test Results")

col1, col2 = st.columns(2)
results_dir = col1.text_input("Results directory", value=str(REPO_ROOT / "test_results"))
component_filter = col2.selectbox("Component", ALL_COMPONENTS)
status_filter = st.radio("Status filter", ["All", "Passed", "Failed"], horizontal=True)

all_results = cached_load_results(results_dir)
module_results = [r for r in all_results if r.get("level") == "module"]

if status_filter == "Passed":
    module_results = [
        {**r, "tests": [t for t in r.get("tests", []) if t["status"] == "passed"]}
        for r in module_results
    ]
elif status_filter == "Failed":
    module_results = [
        {**r, "tests": [t for t in r.get("tests", []) if t["status"] in ("failed", "error")]}
        for r in module_results
    ]

agg = aggregate_results(module_results)
overall = agg.get("overall", {})

metric_cols = st.columns(4)
metric_cols[0].metric("Total",   overall.get("total", 0))
metric_cols[1].metric("Passed",  overall.get("passed", 0))
metric_cols[2].metric("Failed",  overall.get("failed", 0))
metric_cols[3].metric("Errors",  overall.get("errors", 0))

st.divider()
render_test_list_for_level(module_results, level="module", component_filter=component_filter)
