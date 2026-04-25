"""
scripts/run_tests.py
====================
CLI test runner — runs pytest with level/component filtering and produces
structured JSON output consumed by the Streamlit test results UI.

Usage
-----
    python scripts/run_tests.py                              # run all
    python scripts/run_tests.py --level unit                 # unit only
    python scripts/run_tests.py --component tools            # tools component
    python scripts/run_tests.py --level unit --component core
    python scripts/run_tests.py --output-dir results/ --verbose
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ── Path maps ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent

LEVEL_PATHS: dict[str, str] = {
    "unit":   "tests/unit",
    "module": "tests/module",
    "system": "tests/system",
}

COMPONENT_TO_PATHS: dict[str, list[str]] = {
    "core":        ["tests/unit/test_core",        "tests/module/test_core_module.py"],
    "simulator":   ["tests/unit/test_simulator",   "tests/module/test_simulator_module.py"],
    "tools":       ["tests/unit/test_tools",       "tests/module/test_tools_module.py"],
    "agent":       ["tests/unit/test_agent",       "tests/module/test_agent_module.py"],
    "compressor":  ["tests/unit/test_compressor",  "tests/module/test_compressor_module.py"],
    "training":    ["tests/unit/test_training",    "tests/module/test_training_module.py"],
    "evaluation":  ["tests/unit/test_evaluation",  "tests/module/test_evaluation_module.py"],
    "mcts":        ["tests/unit/test_mcts",        "tests/module/test_mcts_module.py"],
    "utils":       ["tests/unit/test_utils",       "tests/module/test_utils_module.py"],
}

ALL_LEVELS = list(LEVEL_PATHS.keys())
ALL_COMPONENTS = list(COMPONENT_TO_PATHS.keys())


# ── Argument resolution ───────────────────────────────────────────────────────

def build_pytest_args(
    level: str,
    component: str,
    report_file: Path,
    verbose: bool,
    fail_fast: bool,
    no_capture: bool,
) -> list[str]:
    """Resolve which paths to pass to pytest based on level/component filters."""
    args = [sys.executable, "-m", "pytest"]

    # Determine test paths
    if level == "all" and component == "all":
        paths = list(LEVEL_PATHS.values())
    elif level != "all" and component == "all":
        paths = [LEVEL_PATHS[level]]
    elif level == "all" and component != "all":
        paths = COMPONENT_TO_PATHS[component] + ["tests/system"]
    else:
        # intersection: component paths filtered to the specified level
        level_dir = LEVEL_PATHS[level]
        component_paths = COMPONENT_TO_PATHS[component]
        paths = [p for p in component_paths if p.startswith(level_dir)]
        if not paths:
            # No intersection (e.g. system level + specific component): run level only
            paths = [level_dir]

    # Only include paths that exist
    paths = [p for p in paths if (REPO_ROOT / p).exists()]

    args.extend(paths)
    args.extend([
        "--tb=short",
        f"--json-report",
        f"--json-report-file={report_file}",
    ])
    if verbose:
        args.append("-v")
    if fail_fast:
        args.append("-x")
    if no_capture:
        args.append("-s")

    return args


# ── JSON result parsing ───────────────────────────────────────────────────────

def parse_json_report(report_path: Path, level: str, component: str) -> dict:
    """Convert pytest-json-report output to our schema."""
    raw = json.loads(report_path.read_text(encoding="utf-8"))

    summary_raw = raw.get("summary", {})
    summary = {
        "total":   summary_raw.get("total", 0),
        "passed":  summary_raw.get("passed", 0),
        "failed":  summary_raw.get("failed", 0),
        "errors":  summary_raw.get("errors", 0),
        "skipped": summary_raw.get("skipped", 0),
    }

    tests = []
    for t in raw.get("tests", []):
        outcome = t.get("outcome", "unknown")
        call_data = t.get("call", {})
        longrepr = call_data.get("longrepr", "") if call_data else ""

        tests.append({
            "name": t.get("nodeid", ""),
            "status": outcome,
            "duration_ms": round((t.get("duration", 0.0)) * 1000, 2),
            "failure_message": longrepr if outcome in ("failed", "error") else None,
            "error_traceback": longrepr if outcome == "error" else None,
        })

    return {
        "level":     level,
        "component": component,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "summary":   summary,
        "tests":     tests,
    }


# ── Output writing ────────────────────────────────────────────────────────────

def write_result_file(result: dict, output_dir: Path) -> Path:
    """Write result to test_results_{level}_{component}_{timestamp}.json"""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"test_results_{result['level']}_{result['component']}_{ts}.json"
    path = output_dir / filename
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def write_latest_aggregate(all_results: list[dict], output_dir: Path) -> Path:
    """Write test_results_latest.json with aggregate summary across all runs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total = sum(r["summary"]["total"] for r in all_results)
    passed = sum(r["summary"]["passed"] for r in all_results)
    failed = sum(r["summary"]["failed"] for r in all_results)
    errors = sum(r["summary"]["errors"] for r in all_results)

    aggregate = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "aggregate": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": round(passed / total, 4) if total > 0 else 0.0,
        },
        "runs": all_results,
    }

    path = output_dir / "test_results_latest.json"
    path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return path


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run project tests with level/component filtering.")
    parser.add_argument(
        "--level",
        choices=ALL_LEVELS + ["all"],
        default="all",
        help="Test level to run (default: all)",
    )
    parser.add_argument(
        "--component",
        choices=ALL_COMPONENTS + ["all"],
        default="all",
        help="Component to run tests for (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="test_results",
        help="Directory to write JSON result files (default: test_results/)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose pytest output")
    parser.add_argument("--fail-fast", "-x", action="store_true", help="Stop after first failure")
    parser.add_argument("--no-capture", "-s", action="store_true", help="Don't capture stdout/stderr")
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir

    print(f"\n{'='*60}")
    print(f"Test Runner — level={args.level}  component={args.component}")
    print(f"Output dir:  {output_dir}")
    print(f"{'='*60}\n")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        report_file = Path(tf.name)

    pytest_args = build_pytest_args(
        level=args.level,
        component=args.component,
        report_file=report_file,
        verbose=args.verbose,
        fail_fast=args.fail_fast,
        no_capture=args.no_capture,
    )

    print("Running:", " ".join(pytest_args[2:]))  # skip python -m

    proc = subprocess.run(pytest_args, cwd=str(REPO_ROOT))
    exit_code = proc.returncode

    if report_file.exists() and report_file.stat().st_size > 0:
        result = parse_json_report(report_file, level=args.level, component=args.component)
        result_path = write_result_file(result, output_dir)
        latest_path = write_latest_aggregate([result], output_dir)

        s = result["summary"]
        print(f"\nResults: {s['passed']}/{s['total']} passed  "
              f"({s['failed']} failed, {s['errors']} errors)")
        print(f"Written to: {result_path}")
        print(f"Latest:     {latest_path}")
    else:
        print("WARNING: No JSON report generated (pytest-json-report not installed or no tests ran).")

    report_file.unlink(missing_ok=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
