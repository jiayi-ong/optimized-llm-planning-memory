"""
app_tests/pages/1_unit_results.py
==================================
Drill-down page for unit test results.
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

st.set_page_config(page_title="Unit Test Results", layout="wide")
st.title("Unit Test Results")

# ── Filters ───────────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)
results_dir = col1.text_input("Results directory", value=str(REPO_ROOT / "test_results"))
component_filter = col2.selectbox("Component", ALL_COMPONENTS)
status_filter = st.radio("Status filter", ["All", "Passed", "Failed"], horizontal=True)

# ── Load and display ──────────────────────────────────────────────────────────

all_results = cached_load_results(results_dir)
unit_results = [r for r in all_results if r.get("level") == "unit"]

if status_filter == "Passed":
    unit_results = [
        {**r, "tests": [t for t in r.get("tests", []) if t["status"] == "passed"]}
        for r in unit_results
    ]
elif status_filter == "Failed":
    unit_results = [
        {**r, "tests": [t for t in r.get("tests", []) if t["status"] in ("failed", "error")]}
        for r in unit_results
    ]

agg = aggregate_results(unit_results)
overall = agg.get("overall", {})
total   = overall.get("total", 0)
passed  = overall.get("passed", 0)
failed  = overall.get("failed", 0)
errors  = overall.get("errors", 0)

metric_cols = st.columns(4)
metric_cols[0].metric("Total",   total)
metric_cols[1].metric("Passed",  passed)
metric_cols[2].metric("Failed",  failed)
metric_cols[3].metric("Errors",  errors)

st.divider()
render_test_list_for_level(unit_results, level="unit", component_filter=component_filter)
