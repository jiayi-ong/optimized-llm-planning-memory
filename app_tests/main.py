"""
app_tests/main.py
==================
Test Results Dashboard — Streamlit entry point.

Launch:
    streamlit run app_tests/main.py
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import streamlit as st

from app_tests.utils.display import (
    render_aggregate_metrics,
    render_live_output_section,
    render_matrix,
    render_test_list_for_level,
)
from app_tests.utils.result_loader import (
    aggregate_results,
    cached_load_results,
    list_result_files,
    load_latest,
    load_result_file,
)
from app_tests.utils.runner import start_test_run

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
ALL_LEVELS = ["unit", "module", "system"]
ALL_COMPONENTS = [
    "core", "simulator", "tools", "agent",
    "compressor", "training", "evaluation", "mcts", "utils",
]

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Test Results Dashboard",
    page_icon="🧪",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────────────────────

if "run_thread" not in st.session_state:
    st.session_state.run_thread = None
if "stdout_queue" not in st.session_state:
    st.session_state.stdout_queue = queue.Queue()
if "running" not in st.session_state:
    st.session_state.running = False

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    results_dir = st.text_input(
        "Results directory",
        value=str(REPO_ROOT / "test_results"),
    )

    result_files = list_result_files(results_dir)
    file_options = ["Latest aggregate"] + [f.name for f in result_files]
    selected_file = st.selectbox("Load results file", file_options)

    auto_refresh = st.toggle("Auto-refresh (30s)", value=False)
    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=30_000, key="autorefresh")
        except ImportError:
            st.warning("Install streamlit-autorefresh for auto-refresh.")

    st.divider()
    st.subheader("Run tests")

    selected_levels = st.multiselect(
        "Levels", ALL_LEVELS, default=ALL_LEVELS
    )
    selected_components = st.multiselect(
        "Components", ALL_COMPONENTS, default=["all"]
    )

    col1, col2 = st.columns(2)
    run_selected = col1.button("Run Selected", use_container_width=True)
    run_all = col2.button("Run All", use_container_width=True)

    if run_all:
        selected_levels = ALL_LEVELS
        selected_components = ["all"]
        run_selected = True

    if run_selected and not st.session_state.running:
        st.session_state.stdout_queue = queue.Queue()
        st.session_state.running = True
        st.session_state.run_thread = start_test_run(
            levels=selected_levels or ALL_LEVELS,
            components=selected_components or ["all"],
            output_dir=results_dir,
            stdout_queue=st.session_state.stdout_queue,
        )

# ── Load results ──────────────────────────────────────────────────────────────

if selected_file == "Latest aggregate":
    latest = load_latest(results_dir)
    if latest and "runs" in latest:
        all_results = latest["runs"]
    else:
        all_results = cached_load_results(results_dir)
else:
    file_path = Path(results_dir) / selected_file
    try:
        data = load_result_file(file_path)
        all_results = data.get("runs", [data])
    except Exception:
        all_results = []

agg = aggregate_results(all_results)

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🧪 Test Results Dashboard")

if all_results:
    render_aggregate_metrics(agg["overall"])
else:
    st.info(
        "No test results found. Run tests using the sidebar or:\n"
        "```\npython scripts/run_tests.py\n```"
    )

if agg["matrix"]:
    st.subheader("Component × Level Matrix")
    render_matrix(agg["matrix"])

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_unit, tab_module, tab_system = st.tabs(["Unit Tests", "Module Tests", "System Tests"])

with tab_unit:
    render_test_list_for_level(all_results, level="unit")

with tab_module:
    render_test_list_for_level(all_results, level="module")

with tab_system:
    render_test_list_for_level(all_results, level="system")

# ── Live output ───────────────────────────────────────────────────────────────

if st.session_state.running:
    st.subheader("Live Test Output")
    output_area = st.empty()
    render_live_output_section(st.session_state.stdout_queue, output_area)

    if (
        st.session_state.run_thread is not None
        and not st.session_state.run_thread.is_alive()
    ):
        st.session_state.running = False
        st.success("Test run complete. Reload results to see updated pass/fail status.")
        cached_load_results.clear()
