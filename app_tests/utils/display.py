"""
app_tests/utils/display.py
===========================
Shared Streamlit display components for the test results UI.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


# ── Colour helpers ────────────────────────────────────────────────────────────

def _cell_colour(val: str) -> str:
    if val == "pass":
        return "background-color: #1a7a1a; color: white"
    if val == "fail":
        return "background-color: #a81a1a; color: white"
    return "background-color: #444; color: #aaa"


def render_aggregate_metrics(overall: dict[str, Any]) -> None:
    """Render the 4-metric aggregate row."""
    cols = st.columns(4)
    cols[0].metric("Total",   overall.get("total",   0))
    cols[1].metric("Passed",  overall.get("passed",  0), delta=None)
    cols[2].metric("Failed",  overall.get("failed",  0))
    cols[3].metric("Errors",  overall.get("errors",  0))

    rate = overall.get("pass_rate", 0.0)
    colour = "green" if rate == 1.0 else ("orange" if rate >= 0.8 else "red")
    st.markdown(
        f"**Pass rate:** :{colour}[{rate:.1%}]",
        unsafe_allow_html=False,
    )


def render_matrix(matrix: dict[str, dict[str, str]]) -> None:
    """Render the component × level matrix as a colour-coded DataFrame."""
    if not matrix:
        st.info("No results loaded.")
        return

    levels = ["unit", "module", "system"]
    rows = {comp: {lvl: matrix.get(comp, {}).get(lvl, "none") for lvl in levels}
            for comp in sorted(matrix.keys())}

    df = pd.DataFrame(rows).T
    df.index.name = "component"

    styled = df.style.applymap(_cell_colour)
    st.dataframe(styled, use_container_width=True)


def render_test_list_for_level(results: list[dict], level: str, component_filter: str = "all") -> None:
    """Render per-component expanders for a specific test level."""
    level_results = [r for r in results if r.get("level") == level]
    if component_filter != "all":
        level_results = [r for r in level_results if r.get("component") == component_filter]

    if not level_results:
        st.info(f"No {level} results loaded.")
        return

    # Group by component
    by_comp: dict[str, list[dict]] = {}
    for r in level_results:
        comp = r.get("component", "unknown")
        by_comp.setdefault(comp, []).append(r)

    for comp, runs in sorted(by_comp.items()):
        tests = [t for r in runs for t in r.get("tests", [])]
        passed = sum(1 for t in tests if t["status"] == "passed")
        total  = len(tests)
        label = f"{'✅' if passed == total else '❌'} {comp}  ({passed}/{total} passed)"

        with st.expander(label, expanded=(passed < total)):
            _render_test_table(tests)


def _render_test_table(tests: list[dict]) -> None:
    """Render a table of individual test results, with failure details."""
    if not tests:
        st.write("No tests found.")
        return

    # Summary table
    rows = []
    for t in tests:
        rows.append({
            "test": t["name"].split("::")[-1],
            "status": t["status"],
            "duration_ms": t["duration_ms"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Failure details
    failures = [t for t in tests if t["status"] in ("failed", "error")]
    for f in failures:
        msg = f.get("failure_message") or f.get("error_traceback") or "(no details)"
        with st.expander(f"❌ {f['name'].split('::')[-1]}", expanded=False):
            st.warning(f"Status: {f['status']}")
            st.code(msg, language="python")


def render_live_output_section(stdout_queue, output_placeholder: "st.empty") -> None:  # type: ignore[name-defined]
    """Drain the queue and render new lines into the placeholder."""
    lines: list[str] = []
    import queue as q
    while True:
        try:
            line = stdout_queue.get_nowait()
            if line == "__DONE__":
                break
            lines.append(line)
        except q.Empty:
            break
    if lines:
        output_placeholder.code("\n".join(lines))
