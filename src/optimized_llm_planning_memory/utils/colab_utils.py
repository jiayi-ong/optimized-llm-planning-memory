"""
utils/colab_utils.py
====================
Utilities for managing training artifacts in Google Colab + Google Drive.

Purpose
-------
Each developer runs their own Colab instance for a different training session
(different compressor, reward weights, or PPO hyperparams).  These functions
standardise how runs are packaged and shared, making downstream comparison
straightforward.

Functions
---------
bundle_run(run_id)        : Zip a complete run (manifest + JSONL + final checkpoint)
                            into a single ``.tar.gz`` file for download or Drive upload.
upload_to_drive(...)      : Copy a bundle to a Google Drive mount path.
download_bundle(run_id)   : Trigger a Colab ``files.download()`` call (Colab only).
list_drive_runs(drive_dir): List available run bundles in a Drive directory.
estimate_run_size(run_id) : Estimate total size of a run's artifacts in MB.

Design
------
All functions work whether or not running in Colab.  Colab-specific functionality
(``files.download()``, Drive detection) is guarded with ``_is_colab()`` checks.
This lets the same code run in unit tests and local scripts without importing
``google.colab``.
"""

from __future__ import annotations

import os
import shutil
import tarfile
from pathlib import Path
from typing import Any


def _is_colab() -> bool:
    """Return True when executing inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def bundle_run(
    run_id: str,
    output_dir: str | Path = "outputs",
    bundle_dir: str | Path = "outputs/bundles",
) -> Path:
    """
    Package a complete training run into a single ``.tar.gz`` archive.

    Contents
    --------
    - ``training/<run_id>/manifest.json``
    - ``training/<run_id>/ppo_metrics.jsonl``
    - ``training/<run_id>/episode_metrics.jsonl``
    - ``checkpoints/final/``           (final SB3 zip + compressor weights)
    - ``logs/``                        (TensorBoard event files, if present)

    Parameters
    ----------
    run_id     : The run identifier (timestamped subdirectory name).
    output_dir : Root of the outputs directory tree.
    bundle_dir : Directory where the resulting ``.tar.gz`` is written.

    Returns
    -------
    Path to the created archive.

    Raises
    ------
    FileNotFoundError
        If the run directory does not exist.
    """
    output_dir = Path(output_dir)
    run_dir = output_dir / "training" / run_id

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Training run directory not found: {run_dir}. "
            f"Check that run_id='{run_id}' is correct."
        )

    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    archive_path = bundle_dir / f"{run_id}.tar.gz"

    with tarfile.open(archive_path, "w:gz") as tar:
        # Training JSONL and manifest
        if run_dir.exists():
            tar.add(run_dir, arcname=f"training/{run_id}")

        # Final checkpoint (if present)
        final_ckpt = output_dir / "checkpoints" / "final"
        if final_ckpt.exists():
            tar.add(final_ckpt, arcname="checkpoints/final")

        # TensorBoard logs (if present, can be large — include only if small)
        tb_log_dir = output_dir / "logs"
        if tb_log_dir.exists():
            total_bytes = sum(
                f.stat().st_size for f in tb_log_dir.rglob("*") if f.is_file()
            )
            if total_bytes < 50 * 1024 * 1024:  # < 50 MB
                tar.add(tb_log_dir, arcname="logs")

    size_mb = archive_path.stat().st_size / 1e6
    print(f"Bundle created: {archive_path}  ({size_mb:.1f} MB)")
    return archive_path


def upload_to_drive(
    bundle_path: str | Path,
    drive_dir: str | Path = "/content/drive/MyDrive/optllm_training",
) -> Path:
    """
    Copy a bundle archive to a Google Drive mount path.

    Parameters
    ----------
    bundle_path : Path to the ``.tar.gz`` bundle file.
    drive_dir   : Target directory on the Drive mount.

    Returns
    -------
    Path to the uploaded file on Drive.
    """
    bundle_path = Path(bundle_path)
    drive_dir = Path(drive_dir)
    drive_dir.mkdir(parents=True, exist_ok=True)

    dest = drive_dir / bundle_path.name
    shutil.copy2(bundle_path, dest)
    print(f"Uploaded to Drive: {dest}")
    return dest


def download_bundle(
    run_id: str,
    bundle_dir: str | Path = "outputs/bundles",
) -> None:
    """
    Trigger a Colab file download for the run bundle.

    Only works inside Google Colab.  Silently prints the bundle path if not in Colab.

    Parameters
    ----------
    run_id     : The run identifier.
    bundle_dir : Directory containing the bundle archive.
    """
    bundle_path = Path(bundle_dir) / f"{run_id}.tar.gz"
    if not bundle_path.exists():
        print(
            f"Bundle not found at {bundle_path}. "
            f"Call bundle_run('{run_id}') first."
        )
        return

    if _is_colab():
        from google.colab import files
        files.download(str(bundle_path))
    else:
        print(f"Bundle ready for download: {bundle_path}")


def list_drive_runs(
    drive_dir: str | Path = "/content/drive/MyDrive/optllm_training",
) -> list[dict[str, Any]]:
    """
    List available run bundles in a Google Drive directory.

    Returns
    -------
    List of dicts, each with: ``run_id``, ``size_mb``, ``path``.
    Sorted by run_id (newest first, since run_ids are timestamps).
    """
    drive_dir = Path(drive_dir)
    if not drive_dir.exists():
        print(f"Drive directory not found: {drive_dir}")
        return []

    bundles = sorted(drive_dir.glob("*.tar.gz"), reverse=True)
    results = []
    for b in bundles:
        run_id = b.stem  # strip .tar.gz
        size_mb = b.stat().st_size / 1e6
        results.append({"run_id": run_id, "size_mb": round(size_mb, 1), "path": str(b)})

    return results


def estimate_run_size(
    run_id: str,
    output_dir: str | Path = "outputs",
) -> dict[str, float]:
    """
    Estimate the disk usage of a training run's artifacts in MB.

    Returns
    -------
    Dict with keys: ``training_jsonl_mb``, ``checkpoints_mb``, ``episodes_mb``,
    ``tensorboard_mb``, ``total_mb``.
    """

    def _dir_size_mb(path: Path) -> float:
        if not path.exists():
            return 0.0
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6

    output_dir = Path(output_dir)

    training_mb = _dir_size_mb(output_dir / "training" / run_id)
    ckpt_mb = _dir_size_mb(output_dir / "checkpoints")
    episodes_mb = _dir_size_mb(output_dir / "episodes")
    tb_mb = _dir_size_mb(output_dir / "logs")

    total = training_mb + ckpt_mb + episodes_mb + tb_mb
    return {
        "training_jsonl_mb": round(training_mb, 2),
        "checkpoints_mb": round(ckpt_mb, 2),
        "episodes_mb": round(episodes_mb, 2),
        "tensorboard_mb": round(tb_mb, 2),
        "total_mb": round(total, 2),
    }
