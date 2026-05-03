"""
training/run_manifest.py
========================
TrainingRunManifest — captures the full resolved config + metadata for one
training run, written alongside the JSONL artifacts in
``outputs/training/<run_id>/``.

Why a manifest?
---------------
JSONL files record what happened during training (episode rewards, PPO losses).
The manifest records *why* — which compressor, which reward weights, which PPO
hyperparams, which code revision.  Without a manifest, comparing two runs
requires reconstructing the config from memory or CLI history.

With ``resolve_checkpoint(run_id)``, post-processing scripts (run_evaluation.py,
the Streamlit eval viewer) can locate a run's final checkpoint automatically
from just the run_id, without the user having to manually pass a path.

Usage
-----
    # In RLTrainer.train():
    manifest = TrainingRunManifest.create(run_id=run_id, config=self._config, ...)
    save_manifest(manifest, run_dir)

    # In run_evaluation.py --run-id mode:
    ckpt_path = resolve_checkpoint(run_id, output_dir="outputs")
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _git_sha() -> str:
    """Return the current git commit SHA (short), or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class TrainingRunManifest(BaseModel):
    """
    Immutable record of a training run's configuration and environment.

    Written once at the start of ``RLTrainer.train()`` to
    ``outputs/training/<run_id>/manifest.json``.

    Fields
    ------
    run_id           : Timestamp-based unique identifier (``YYYYMMDD_HHMMSS``).
    run_name         : Human-readable name from ``project.run_name`` config.
    git_sha          : Short git commit SHA at run time.
    compressor_type  : Compressor class name (e.g., ``"IdentityCompressor"``).
    agent_mode       : Agent mode string (``"raw"`` | ``"compressor"`` | …).
    reward_weights   : Dict of weight name → float value.
    ppo_hyperparams  : Dict of PPO hyperparameter name → value.
    n_envs           : Number of parallel training environments.
    num_timesteps    : Total training steps requested.
    n_train_requests : Number of training UserRequest files loaded.
    checkpoint_dir   : Absolute path to the checkpoint directory for this run.
    created_at       : ISO 8601 UTC timestamp.
    extra            : Catch-all dict for any additional metadata.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    run_name: str = ""
    git_sha: str = Field(default_factory=_git_sha)
    compressor_type: str = ""
    agent_mode: str = ""
    reward_weights: dict[str, float] = Field(default_factory=dict)
    ppo_hyperparams: dict[str, Any] = Field(default_factory=dict)
    n_envs: int = 1
    num_timesteps: int = 0
    n_train_requests: int = 0
    checkpoint_dir: str = ""
    created_at: str = Field(default_factory=_now)
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        run_id: str,
        compressor_type: str,
        n_train_requests: int,
        checkpoint_dir: str | Path,
        run_name: str = "",
        agent_mode: str = "",
        reward_weights: dict[str, float] | None = None,
        ppo_hyperparams: dict[str, Any] | None = None,
        n_envs: int = 1,
        num_timesteps: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> "TrainingRunManifest":
        """Build a manifest from explicit values (avoids tight coupling to config types)."""
        return cls(
            run_id=run_id,
            run_name=run_name,
            compressor_type=compressor_type,
            agent_mode=agent_mode,
            reward_weights=reward_weights or {},
            ppo_hyperparams=ppo_hyperparams or {},
            n_envs=n_envs,
            num_timesteps=num_timesteps,
            n_train_requests=n_train_requests,
            checkpoint_dir=str(checkpoint_dir),
            extra=extra or {},
        )


def save_manifest(manifest: TrainingRunManifest, run_dir: str | Path) -> Path:
    """
    Write ``manifest.json`` into ``run_dir``.

    Parameters
    ----------
    manifest : The manifest to serialise.
    run_dir  : Directory where ``manifest.json`` will be written
               (typically ``outputs/training/<run_id>/``).

    Returns
    -------
    Path to the written file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_manifest(run_id: str, training_dir: str | Path = "outputs/training") -> TrainingRunManifest | None:
    """
    Load the manifest for a given run_id.

    Returns None if no manifest file exists (e.g., old runs created before
    this feature was added).
    """
    path = Path(training_dir) / run_id / "manifest.json"
    if not path.exists():
        return None
    try:
        return TrainingRunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def resolve_checkpoint(
    run_id: str,
    output_dir: str | Path = "outputs",
    prefer_final: bool = True,
) -> Path | None:
    """
    Resolve the best checkpoint path for a given run_id.

    Search order:
    1. ``<output_dir>/checkpoints/<run_id>/final/ppo_model.zip``  (if prefer_final)
    2. ``<output_dir>/checkpoints/final/ppo_model.zip``           (fallback for legacy layout)
    3. Latest ``ppo_compressor_*_steps.zip`` sorted by step count

    Also checks the manifest's ``checkpoint_dir`` if available.

    Returns None if no checkpoint can be found.
    """
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"

    # 1. Check manifest for recorded checkpoint_dir
    manifest = load_manifest(run_id, training_dir=output_dir / "training")
    if manifest and manifest.checkpoint_dir:
        mf_dir = Path(manifest.checkpoint_dir)
        final_zip = mf_dir / "final" / "ppo_model.zip"
        if final_zip.exists():
            return final_zip

    # 2. Standard final checkpoint location
    if prefer_final:
        final_zip = checkpoint_dir / "final" / "ppo_model.zip"
        if final_zip.exists():
            return final_zip

    # 3. Latest numbered checkpoint
    zips = sorted(
        checkpoint_dir.glob("ppo_compressor_*_steps.zip"),
        key=lambda p: int(p.stem.split("_")[-2]) if p.stem.split("_")[-2].isdigit() else 0,
    )
    if zips:
        return zips[-1]

    return None


def list_manifests(
    training_dir: str | Path = "outputs/training",
) -> list[TrainingRunManifest]:
    """
    Return manifests for all runs in ``training_dir``, newest first.

    Runs without a manifest.json are skipped silently.
    """
    base = Path(training_dir)
    if not base.exists():
        return []
    manifests: list[TrainingRunManifest] = []
    for run_dir in sorted(base.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        m = load_manifest(run_dir.name, training_dir=base)
        if m is not None:
            manifests.append(m)
    return manifests
