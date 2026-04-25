"""
app_tests/utils/runner.py
=========================
Subprocess test runner for the Streamlit UI.

Streams stdout from `scripts/run_tests.py` to a queue.Queue so the UI
can display live output without blocking.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "run_tests.py"


def run_tests_subprocess(
    level: str,
    component: str,
    output_dir: str,
    stdout_queue: queue.Queue,
) -> int:
    """
    Run `scripts/run_tests.py` as a subprocess and stream output to `stdout_queue`.

    Parameters
    ----------
    level        : "unit" | "module" | "system" | "all"
    component    : component name or "all"
    output_dir   : Path string for --output-dir argument
    stdout_queue : queue.Queue receiving (str) lines of output

    Returns
    -------
    int — subprocess exit code (0 = success)
    """
    cmd = [
        sys.executable, str(RUNNER_SCRIPT),
        "--level", level,
        "--component", component,
        "--output-dir", output_dir,
        "--verbose",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:  # type: ignore[union-attr]
        stdout_queue.put(line.rstrip("\n"))

    proc.wait()
    stdout_queue.put(f"\n[Runner exited with code {proc.returncode}]")
    return proc.returncode


def start_test_run(
    levels: list[str],
    components: list[str],
    output_dir: str,
    stdout_queue: queue.Queue,
) -> threading.Thread:
    """
    Start background threads to run each (level, component) combination.

    A single sentinel thread per pair is spawned; all write to the same queue.

    Parameters
    ----------
    levels      : List of levels ("unit", "module", "system", or ["all"]).
    components  : List of components or ["all"].
    output_dir  : --output-dir passed to the runner script.
    stdout_queue: Queue for streaming output.

    Returns
    -------
    threading.Thread — the orchestrating thread (already started).
    """
    def _run_all():
        for level in levels:
            for comp in components:
                stdout_queue.put(f"\n--- Running level={level} component={comp} ---")
                run_tests_subprocess(level, comp, output_dir, stdout_queue)
        stdout_queue.put("__DONE__")

    t = threading.Thread(target=_run_all, daemon=True)
    t.start()
    return t
