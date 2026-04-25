"""
app_tests/utils/result_loader.py
=================================
Functions to load and aggregate test result JSON files produced by scripts/run_tests.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


def list_result_files(results_dir: str) -> list[Path]:
    """Return all test_results_*.json files sorted newest-first."""
    base = Path(results_dir)
    if not base.exists():
        return []
    files = sorted(
        base.glob("test_results_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [f for f in files if f.name != "test_results_latest.json"]


def load_result_file(path: Path | str) -> dict[str, Any]:
    """Load a single test result JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_latest(results_dir: str) -> dict[str, Any] | None:
    """Load test_results_latest.json if it exists."""
    path = Path(results_dir) / "test_results_latest.json"
    if not path.exists():
        return None
    return load_result_file(path)


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate a list of run result dicts into summary by level and component.

    Returns
    -------
    dict with keys:
        by_level: dict[level -> {"total", "passed", "failed", "errors"}]
        by_component: dict[component -> {"total", "passed", "failed", "errors"}]
        matrix: dict[component -> dict[level -> "pass" | "fail" | "none"]]
        overall: {"total", "passed", "failed", "errors", "pass_rate"}
    """
    by_level: dict[str, dict] = {}
    by_component: dict[str, dict] = {}
    matrix: dict[str, dict[str, str]] = {}

    def _add(d: dict, key: str, s: dict) -> None:
        if key not in d:
            d[key] = {"total": 0, "passed": 0, "failed": 0, "errors": 0}
        d[key]["total"]  += s.get("total", 0)
        d[key]["passed"] += s.get("passed", 0)
        d[key]["failed"] += s.get("failed", 0)
        d[key]["errors"] += s.get("errors", 0)

    for r in results:
        level = r.get("level", "unknown")
        component = r.get("component", "unknown")
        s = r.get("summary", {})

        _add(by_level, level, s)
        _add(by_component, component, s)

        if component not in matrix:
            matrix[component] = {}
        failed = s.get("failed", 0) + s.get("errors", 0)
        matrix[component][level] = "fail" if failed > 0 else "pass"

    total  = sum(v["total"]  for v in by_level.values())
    passed = sum(v["passed"] for v in by_level.values())
    failed = sum(v["failed"] for v in by_level.values())
    errors = sum(v["errors"] for v in by_level.values())

    return {
        "by_level":     by_level,
        "by_component": by_component,
        "matrix":       matrix,
        "overall": {
            "total":     total,
            "passed":    passed,
            "failed":    failed,
            "errors":    errors,
            "pass_rate": round(passed / total, 4) if total > 0 else 0.0,
        },
    }


@st.cache_data(ttl=30)
def cached_load_results(results_dir: str) -> list[dict[str, Any]]:
    """Load all result files from a directory (cached for 30s)."""
    files = list_result_files(results_dir)
    results = []
    for f in files:
        try:
            data = load_result_file(f)
            # Handle both flat result format and aggregate (latest) format
            if "runs" in data:
                results.extend(data["runs"])
            else:
                results.append(data)
        except Exception:
            continue
    return results
