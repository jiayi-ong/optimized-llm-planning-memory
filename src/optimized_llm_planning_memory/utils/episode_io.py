"""
utils/episode_io.py
====================
EpisodeLog and EvalRun ↔ JSON file utilities.

Usage
-----
    from optimized_llm_planning_memory.utils.episode_io import (
        save_episode, load_episode, list_episodes,
        save_eval_run, load_eval_run, list_eval_runs,
    )

    save_episode(episode_log, "outputs/episodes/")
    log = load_episode("outputs/episodes/ep_abc123.json")
    all_logs = list_episodes("outputs/episodes/")

    save_eval_run(manifest, results, "outputs/eval_results/")
    manifest, results = load_eval_run("abc12345", "outputs/eval_results/")
    all_manifests = list_eval_runs("outputs/eval_results/")
"""

from __future__ import annotations

import json
from pathlib import Path

from optimized_llm_planning_memory.core.models import EpisodeLog, EvalResult
from optimized_llm_planning_memory.evaluation.manifest import EvalRunManifest


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


def list_episodes_by_request(
    directory: str | Path,
    request_ids: set[str],
) -> list[EpisodeLog]:
    """Load only episodes whose ``request_id`` is in the given set.

    More efficient than ``list_episodes()`` when evaluating a targeted subset
    of requests, because it skips loading and parsing unrelated episode files.

    Parameters
    ----------
    directory   : Directory containing ``ep_*.json`` files.
    request_ids : Set of request IDs to filter by.

    Returns
    -------
    List of matching EpisodeLog objects, in filename order.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    matches: list[EpisodeLog] = []
    for json_file in sorted(dir_path.glob("ep_*.json")):
        try:
            ep = load_episode(json_file)
            if ep.request_id in request_ids:
                matches.append(ep)
        except Exception:
            continue
    return matches


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


# ── Eval run utilities ────────────────────────────────────────────────────────

def save_eval_run(
    manifest: EvalRunManifest,
    results: list[EvalResult],
    base_directory: str | Path,
) -> Path:
    """
    Persist an evaluation run to disk.

    Creates ``{base_directory}/{manifest.run_id}/manifest.json`` and
    ``{base_directory}/{manifest.run_id}/results.jsonl`` (one EvalResult per line).

    Parameters
    ----------
    manifest        : Run-level metadata.
    results         : Per-episode evaluation results.
    base_directory  : Root output directory (e.g. ``outputs/eval_results``).

    Returns
    -------
    Path to the run directory that was created.
    """
    run_dir = Path(base_directory) / manifest.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")

    return run_dir


def load_eval_run(
    run_id: str,
    base_directory: str | Path,
) -> tuple[EvalRunManifest, list[EvalResult]]:
    """
    Load a previously saved evaluation run.

    Parameters
    ----------
    run_id          : The run identifier (matches the directory name).
    base_directory  : Root output directory.

    Returns
    -------
    (EvalRunManifest, list[EvalResult])

    Raises
    ------
    FileNotFoundError if the run directory or manifest does not exist.
    """
    run_dir = Path(base_directory) / run_id
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Eval run manifest not found: {manifest_path}")

    manifest = EvalRunManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )

    results: list[EvalResult] = []
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    results.append(EvalResult.model_validate_json(line))
                except Exception:
                    continue

    return manifest, results


def list_eval_runs(
    base_directory: str | Path,
) -> list[EvalRunManifest]:
    """
    Return all evaluation run manifests sorted by ``created_at`` descending.

    Directories without a ``manifest.json`` are silently skipped.

    Parameters
    ----------
    base_directory : Root output directory.

    Returns
    -------
    List of EvalRunManifest, newest first.
    """
    base = Path(base_directory)
    if not base.exists():
        return []

    manifests: list[EvalRunManifest] = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifests.append(
                EvalRunManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
            )
        except Exception:
            continue

    return sorted(manifests, key=lambda m: m.created_at, reverse=True)
