"""
scripts/launch_ui.py
=====================
Convenience launcher for the developer Streamlit UI.

Usage
-----
    python scripts/launch_ui.py                  # default: outputs/ as data root
    python scripts/launch_ui.py --port 8502      # custom port
    python scripts/launch_ui.py --data /my/path  # custom data root
    python scripts/launch_ui.py --headless       # no browser auto-open

What it does
------------
1. Sets the DATA_ROOT env var so app/utils/data_loader.py resolves the
   correct outputs directory (useful when running from a different cwd).
2. Launches ``streamlit run app/main.py`` as a subprocess.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_ENTRY = _REPO_ROOT / "app" / "main.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the travel planner developer UI")
    parser.add_argument(
        "--port", type=int, default=8501, help="Streamlit port (default: 8501)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override data root (default: <repo>/outputs)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open browser automatically",
    )
    args = parser.parse_args()

    # Set DATA_ROOT so data_loader.py picks up the right directory
    data_root = args.data or str(_REPO_ROOT / "outputs")
    os.environ["DATA_ROOT"] = data_root
    print(f"Data root: {data_root}")

    # Ensure the repo root is on PYTHONPATH so `from app.*` imports work.
    # Streamlit adds the script's directory (app/) to sys.path, not the repo root.
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    repo_root_str = str(_REPO_ROOT)
    paths = [p for p in existing_pythonpath.split(os.pathsep) if p]
    if repo_root_str not in paths:
        paths.insert(0, repo_root_str)
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(_APP_ENTRY),
        f"--server.port={args.port}",
        "--server.headless=true" if args.headless else "--server.headless=false",
    ]

    print(f"Launching UI on http://localhost:{args.port}")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
