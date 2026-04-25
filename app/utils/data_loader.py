"""
app/utils/data_loader.py
========================
Shared data-loading helpers for the Streamlit developer UI.

All functions are thin wrappers that resolve default output paths and return
typed objects.  They can also be imported directly in Jupyter notebooks for
offline analysis, keeping the data-access layer consistent.

Usage
-----
    from app.utils.data_loader import (
        load_episodes, load_live_events,
        load_ppo_metrics, load_episode_metrics, list_run_ids,
    )

    episodes = load_episodes()           # all saved episodes
    events   = load_live_events(ep_id)   # live JSONL for one in-progress episode
    ppo_df   = load_ppo_metrics(run_id)  # per-update PPO diagnostics as list
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ── Default paths (can be overridden via DATA_ROOT env var) ───────────────────

import os

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_OUTPUTS = _REPO_ROOT / "outputs"


def _outputs_root() -> Path:
    env = os.environ.get("DATA_ROOT")
    return Path(env) if env else _DEFAULT_OUTPUTS


def _episodes_dir() -> Path:
    return _outputs_root() / "episodes"


def _training_dir() -> Path:
    return _outputs_root() / "training"


# ── Episode helpers ───────────────────────────────────────────────────────────


def load_episodes(directory: str | Path | None = None):
    """
    Load all saved ``EpisodeLog`` objects from the episodes directory.

    Returns an empty list (not an error) if the directory does not exist.
    """
    from optimized_llm_planning_memory.utils.episode_io import list_episodes

    dir_path = Path(directory) if directory else _episodes_dir()
    return list_episodes(dir_path)


def load_episode(episode_id_or_path: str | Path):
    """
    Load a single ``EpisodeLog`` by episode ID (looks in default dir) or full path.
    """
    from optimized_llm_planning_memory.utils.episode_io import load_episode as _load

    p = Path(episode_id_or_path)
    if p.suffix == ".json" and p.exists():
        return _load(p)
    # Treat as episode_id — look in default episodes dir
    candidate = _episodes_dir() / f"ep_{episode_id_or_path}.json"
    if candidate.exists():
        return _load(candidate)
    raise FileNotFoundError(f"Episode not found: {episode_id_or_path}")


def load_live_events(episode_id: str, directory: str | Path | None = None) -> list[dict]:
    """
    Read all events emitted so far by ``LiveEpisodeWriter`` for an in-progress
    (or recently completed) episode.

    Returns a list of event dicts, in the order they were written.
    Returns an empty list if the file does not yet exist.
    """
    live_dir = (Path(directory) if directory else _episodes_dir()) / "live"
    path = live_dir / f"{episode_id}.jsonl"
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def list_live_episode_ids(directory: str | Path | None = None) -> list[str]:
    """Return episode IDs for all active/recent live JSONL files."""
    live_dir = (Path(directory) if directory else _episodes_dir()) / "live"
    if not live_dir.exists():
        return []
    return [p.stem for p in sorted(live_dir.glob("*.jsonl"))]


# ── Training log helpers ──────────────────────────────────────────────────────


def load_ppo_metrics(run_id: str, training_dir: str | Path | None = None):
    """
    Load ``PPOUpdateMetrics`` records for a training run.

    Returns an empty list if no file exists.
    """
    from optimized_llm_planning_memory.training.run_logger import load_ppo_metrics as _load

    return _load(run_id, Path(training_dir) if training_dir else _training_dir())


def load_episode_metrics(run_id: str, training_dir: str | Path | None = None):
    """
    Load ``EpisodeMetricsSummary`` records for a training run.

    Returns an empty list if no file exists.
    """
    from optimized_llm_planning_memory.training.run_logger import load_episode_metrics as _load

    return _load(run_id, Path(training_dir) if training_dir else _training_dir())


def list_run_ids(training_dir: str | Path | None = None) -> list[str]:
    """
    Return all training run IDs found in the training directory, newest first.
    """
    from optimized_llm_planning_memory.training.run_logger import list_run_ids as _list

    return _list(Path(training_dir) if training_dir else _training_dir())
