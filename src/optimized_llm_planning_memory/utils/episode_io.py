"""
utils/episode_io.py
====================
EpisodeLog ↔ JSON file utilities.

Usage
-----
    from optimized_llm_planning_memory.utils.episode_io import save_episode, load_episode, list_episodes

    save_episode(episode_log, "outputs/episodes/")
    log = load_episode("outputs/episodes/ep_abc123.json")
    all_logs = list_episodes("outputs/episodes/")
"""

from __future__ import annotations

import json
from pathlib import Path

from optimized_llm_planning_memory.core.models import EpisodeLog


def save_episode(episode_log: EpisodeLog, directory: str | Path) -> Path:
    """
    Serialise ``EpisodeLog`` to a JSON file.

    File is named ``ep_{episode_id}.json`` inside ``directory``.

    Parameters
    ----------
    episode_log : Completed episode log to save.
    directory   : Output directory (created if it does not exist).

    Returns
    -------
    Path to the written file.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_path = out_dir / f"ep_{episode_log.episode_id}.json"
    file_path.write_text(
        episode_log.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return file_path


def load_episode(path: str | Path) -> EpisodeLog:
    """
    Deserialise ``EpisodeLog`` from a JSON file.

    Parameters
    ----------
    path : Path to the JSON file written by ``save_episode()``.

    Returns
    -------
    EpisodeLog
    """
    data = Path(path).read_text(encoding="utf-8")
    return EpisodeLog.model_validate_json(data)


def list_episodes(directory: str | Path) -> list[EpisodeLog]:
    """
    Load all episode JSON files from a directory.

    Files are loaded in alphabetical order (by filename). Non-JSON files and
    subdirectories are silently skipped.

    Parameters
    ----------
    directory : Directory containing ``ep_*.json`` files.

    Returns
    -------
    List of EpisodeLog objects.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    episodes = []
    for json_file in sorted(dir_path.glob("ep_*.json")):
        try:
            episodes.append(load_episode(json_file))
        except Exception:
            # Skip corrupted files without crashing the whole load
            continue
    return episodes
