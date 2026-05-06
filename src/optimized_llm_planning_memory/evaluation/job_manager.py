"""
evaluation/job_manager.py
==========================
EvalJobManager — submit, monitor, and cancel eval jobs launched from the UI.

Design
------
Each job is a subprocess running ``scripts/run_eval.py`` with caller-supplied
arguments. Job state is persisted in a small JSON file under
``outputs/.eval_jobs/{job_id}.json`` so the Streamlit UI can poll it across
page re-renders (Streamlit re-runs the script on every interaction).

Progress tracking
-----------------
The runner updates ``EvalRunManifest.n_completed`` after each episode.
``EvalJobManager.status()`` reads the *live* manifest from the run directory
(if known) to surface an accurate progress percentage.

Limitations
-----------
  - PID-based cancellation works on POSIX; on Windows ``os.kill`` is used with
    SIGTERM equivalent (signal 15). If the process has already exited the call
    is a no-op.
  - The UI must poll at a reasonable cadence (≤5 s) so cancelled/failed jobs
    are surfaced quickly.

Usage
-----
    mgr = EvalJobManager()
    job_id = mgr.submit(EvalJobConfig(
        episode_ids=["abc123", "def456"],
        eval_mode="deterministic",
        augmentation_id="tgad-trained-001",
        notes="re-run v2 metrics",
    ))
    status = mgr.status(job_id)
    print(status["progress_pct"])
    mgr.cancel(job_id)
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JOBS_DIR = Path("outputs/.eval_jobs")


@dataclass
class EvalJobConfig:
    """Parameters forwarded to ``run_eval.py`` as CLI arguments."""
    eval_mode: str = "deterministic"             # "deterministic" | "llm_judge" | "full"
    episode_ids: list[str] = field(default_factory=list)
    request_ids: list[str] = field(default_factory=list)
    agent_mode: str | None = None
    augmentation_id: str | None = None
    prompt_id: str | None = None
    parent_run_id: str | None = None
    judge_model: str = "openai/gpt-4o-mini"
    metric_version: str | None = None
    notes: str | None = None
    episodes_dir: str = "outputs/episodes"
    eval_dir: str = "outputs/eval_results"


@dataclass
class EvalJobStatus:
    job_id: str
    pid: int | None
    status: str          # "running" | "completed" | "failed" | "cancelled"
    run_id: str | None
    started_at: str
    finished_at: str | None
    progress_pct: float  # 0.0–100.0
    n_completed: int
    n_total: int
    notes: str | None


class EvalJobManager:
    """Submit and manage eval jobs as subprocesses."""

    def __init__(self, jobs_dir: Path = JOBS_DIR) -> None:
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    # ── Submit ────────────────────────────────────────────────────────────────

    def submit(self, config: EvalJobConfig) -> str:
        """Launch run_eval.py as a subprocess and return the job_id."""
        job_id = uuid.uuid4().hex[:12]
        args = self._build_args(config)

        proc = subprocess.Popen(
            [sys.executable] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        state: dict[str, Any] = {
            "job_id": job_id,
            "pid": proc.pid,
            "status": "running",
            "run_id": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "n_completed": 0,
            "n_total": len(config.episode_ids) or len(config.request_ids) or 0,
            "notes": config.notes,
        }
        self._write(job_id, state)

        # Non-blocking: let the subprocess run; UI polls status()
        return job_id

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self, job_id: str) -> EvalJobStatus:
        """Read current job state, refreshing progress from the live manifest."""
        state = self._read(job_id)
        pid = state.get("pid")

        # Check if process is still alive
        if state["status"] == "running" and pid is not None:
            if not _pid_alive(pid):
                state["status"] = "completed"
                state["finished_at"] = datetime.now(timezone.utc).isoformat()
                self._write(job_id, state)

        # Refresh progress from the run's live manifest if we know the run_id
        n_completed = state.get("n_completed", 0)
        n_total = state.get("n_total", 0)
        run_id = state.get("run_id")
        if run_id:
            n_completed, n_total = _read_manifest_progress(
                run_id, state.get("eval_dir", "outputs/eval_results")
            )

        progress_pct = (n_completed / n_total * 100.0) if n_total > 0 else 0.0
        if state["status"] == "completed":
            progress_pct = 100.0

        return EvalJobStatus(
            job_id=job_id,
            pid=pid,
            status=state["status"],
            run_id=run_id,
            started_at=state["started_at"],
            finished_at=state.get("finished_at"),
            progress_pct=progress_pct,
            n_completed=n_completed,
            n_total=n_total,
            notes=state.get("notes"),
        )

    # ── Cancel ────────────────────────────────────────────────────────────────

    def cancel(self, job_id: str) -> None:
        """Send SIGTERM to the job process."""
        state = self._read(job_id)
        pid = state.get("pid")
        if pid and _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        state["status"] = "cancelled"
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._write(job_id, state)

    # ── List ──────────────────────────────────────────────────────────────────

    def list_jobs(self, active_only: bool = False) -> list[EvalJobStatus]:
        """Return all known jobs, newest first."""
        statuses = []
        for json_file in sorted(self.jobs_dir.glob("*.json"), reverse=True):
            job_id = json_file.stem
            try:
                statuses.append(self.status(job_id))
            except Exception:
                continue
        if active_only:
            statuses = [s for s in statuses if s.status == "running"]
        return statuses

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_args(self, cfg: EvalJobConfig) -> list[str]:
        args = ["scripts/run_eval.py"]
        args += ["--eval_mode", cfg.eval_mode]
        if cfg.episode_ids:
            args += ["--episode_ids"] + cfg.episode_ids
        elif cfg.request_ids:
            args += ["--request_ids"] + cfg.request_ids
        else:
            args += ["--all"]
        if cfg.agent_mode:
            args += ["--agent_mode", cfg.agent_mode]
        if cfg.augmentation_id:
            args += ["--augmentation_id", cfg.augmentation_id]
        if cfg.prompt_id:
            args += ["--prompt_id", cfg.prompt_id]
        if cfg.parent_run_id:
            args += ["--parent_run_id", cfg.parent_run_id]
        if cfg.eval_mode != "deterministic":
            args += ["--judge_model", cfg.judge_model]
        if cfg.notes:
            args += ["--note", cfg.notes]
        args += ["--episodes_dir", cfg.episodes_dir, "--eval_dir", cfg.eval_dir]
        return args

    def _write(self, job_id: str, state: dict) -> None:
        (self.jobs_dir / f"{job_id}.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )

    def _read(self, job_id: str) -> dict:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Job not found: {job_id}")
        return json.loads(path.read_text(encoding="utf-8"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pid_alive(pid: int) -> bool:
    """Return True if the process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _read_manifest_progress(run_id: str, eval_dir: str) -> tuple[int, int]:
    """Read n_completed and n_episodes from the live manifest, if available."""
    manifest_path = Path(eval_dir) / run_id / "manifest.json"
    if not manifest_path.exists():
        return 0, 0
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data.get("n_completed", 0), data.get("n_episodes", 0)
    except Exception:
        return 0, 0
