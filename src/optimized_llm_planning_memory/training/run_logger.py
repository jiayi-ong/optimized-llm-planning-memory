"""
training/run_logger.py
=======================
RLRunLogger — persists training-only diagnostics to JSONL.

Separation of concerns
-----------------------
``EpisodeLog`` is the main system artifact: always produced, both during
training and inference, and stored in ``outputs/episodes/``.

Training-only data — PPO update diagnostics, advantage statistics, clip
fractions — has a different lifecycle.  It is only meaningful during a
training run, should not pollute the main episode store, and needs to be
queryable offline without TensorBoard.  ``RLRunLogger`` handles this by
writing two separate JSONL files inside ``outputs/training/<run_id>/``:

  ppo_metrics.jsonl     — one line per PPO update cycle
  episode_metrics.jsonl — one line per completed episode (step count, tool stats)

These files are loaded by the Training Dashboard page of the developer UI
(``app/pages/5_training_dashboard.py``) and can also be analysed in notebooks.

Usage (from RLTrainer.train())
------------------------------
    from optimized_llm_planning_memory.training.run_logger import RLRunLogger, PPOUpdateMetrics

    logger = RLRunLogger(run_id="20260424_130000", training_dir=Path("outputs/training"))
    try:
        # ...training loop...
        logger.log_ppo_update(PPOUpdateMetrics(update_step=1, policy_loss=0.04, ...))
        logger.log_episode_summary({"episode_id": "...", "total_steps": 12, ...})
    finally:
        logger.close()
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ── Data models ────────────────────────────────────────────────────────────────


class PPOUpdateMetrics(BaseModel):
    """
    Diagnostics captured after a single PPO update cycle.

    These are training-only artifacts: they describe how the compressor policy
    is being optimised and are not meaningful outside of a training context.

    Fields map directly to SB3's internal ``logger.name_to_value`` keys.

    convergence indicators
    ----------------------
    ``approx_kl``        : Should stay below ~0.02.  A spike signals the policy
                           moved too far from the old policy in one update.
    ``clip_fraction``    : Fraction of clipped PPO gradient updates.  Values
                           above 0.2 suggest the clip range is too tight or the
                           learning rate is too high.
    ``explained_variance``: How well the value function fits the returns.
                            Values near 1.0 = good fit; near 0 = value head is
                            not learning.
    """

    model_config = ConfigDict(frozen=True)

    update_step: int = Field(description="Monotonically increasing PPO update counter.")
    policy_loss: float = Field(description="PPO policy gradient loss (negative = normal).")
    value_loss: float = Field(description="Value function MSE loss.")
    entropy_loss: float = Field(description="Entropy bonus (negative = more entropy).")
    total_loss: float = Field(description="Weighted sum of policy + value + entropy losses.")
    clip_fraction: float = Field(
        description="Fraction of probability ratio clipped by epsilon.",
        ge=0.0, le=1.0,
    )
    approx_kl: float = Field(description="Approximate KL divergence between old and new policy.")
    explained_variance: float = Field(
        description="Proportion of variance in returns explained by the value function.",
    )
    learning_rate: float = Field(description="Current learning rate (may decay over time).")
    grad_norm: float | None = Field(
        default=None,
        description="Global gradient norm before clipping. None if not available.",
    )
    advantages_mean: float | None = Field(
        default=None,
        description="Mean of the GAE advantage estimates in this rollout batch.",
    )
    advantages_std: float | None = Field(
        default=None,
        description="Std dev of GAE advantage estimates — signals reward scale.",
    )
    num_timesteps: int | None = Field(
        default=None,
        description="Total environment steps collected so far.",
    )
    timestamp: str = Field(default_factory=_now, description="ISO 8601 UTC timestamp.")


class EpisodeMetricsSummary(BaseModel):
    """
    Lightweight per-episode summary written during training.

    Complements the full ``EpisodeLog`` (which is written to
    ``outputs/episodes/``) with a compact record optimised for time-series
    analysis of training progress.
    """

    model_config = ConfigDict(frozen=True)

    episode_id: str
    request_id: str
    agent_mode: str
    total_steps: int
    success: bool
    total_reward: float
    hard_constraint_score: float
    soft_constraint_score: float
    tool_efficiency_score: float
    tool_failure_penalty: float
    logical_consistency_score: float
    terminal_itinerary_score: float | None = None
    tool_calls_total: int = 0
    tool_success_rate: float = 0.0
    num_compressions: int = 0
    reward_mean_20: float | None = Field(
        default=None,
        description="Rolling mean of total_reward over the last 20 episodes.",
    )
    timestamp: str = Field(default_factory=_now)


# ── Logger ─────────────────────────────────────────────────────────────────────


class RLRunLogger:
    """
    Writes per-update PPO diagnostics and per-episode summaries to JSONL files.

    This is the training-only counterpart to ``EpisodeLog``.  It writes to:
      ``<training_dir>/<run_id>/ppo_metrics.jsonl``
      ``<training_dir>/<run_id>/episode_metrics.jsonl``

    Both files are append-only JSONL: one Pydantic model serialised to JSON per
    line.  This makes them easy to load with ``pandas.read_json(..., lines=True)``
    or iterate with ``json.loads(line)`` in notebooks.

    Parameters
    ----------
    run_id       : Unique run identifier, typically ``YYYYMMDD_HHMMSS``.
    training_dir : Parent directory for all training run output.
    """

    def __init__(self, run_id: str, training_dir: str | Path = "outputs/training") -> None:
        self._run_id = run_id
        run_dir = Path(training_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        self._ppo_path = run_dir / "ppo_metrics.jsonl"
        self._episode_path = run_dir / "episode_metrics.jsonl"

        self._ppo_file = self._ppo_path.open("a", encoding="utf-8")
        self._episode_file = self._episode_path.open("a", encoding="utf-8")

        # Write a header comment as the first line so files are self-describing
        _write_comment(self._ppo_file, f"PPO update metrics | run_id={run_id}")
        _write_comment(self._episode_file, f"Episode metrics | run_id={run_id}")

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def ppo_metrics_path(self) -> Path:
        return self._ppo_path

    @property
    def episode_metrics_path(self) -> Path:
        return self._episode_path

    # ── Write methods ──────────────────────────────────────────────────────────

    def log_ppo_update(self, metrics: PPOUpdateMetrics) -> None:
        """Append one ``PPOUpdateMetrics`` record to ``ppo_metrics.jsonl``."""
        self._ppo_file.write(metrics.model_dump_json() + "\n")
        self._ppo_file.flush()

    def log_episode_summary(self, summary: EpisodeMetricsSummary) -> None:
        """Append one ``EpisodeMetricsSummary`` record to ``episode_metrics.jsonl``."""
        self._episode_file.write(summary.model_dump_json() + "\n")
        self._episode_file.flush()

    def close(self) -> None:
        """Flush and close both JSONL files."""
        for f in (self._ppo_file, self._episode_file):
            if f and not f.closed:
                f.flush()
                f.close()

    def __enter__(self) -> "RLRunLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ── Load helpers ───────────────────────────────────────────────────────────────


def load_ppo_metrics(run_id: str, training_dir: str | Path = "outputs/training") -> list[PPOUpdateMetrics]:
    """
    Load all PPO update metrics for a training run from JSONL.

    Parameters
    ----------
    run_id       : The run identifier (subdirectory name).
    training_dir : Parent directory for all training run output.

    Returns
    -------
    List of ``PPOUpdateMetrics``, in the order they were written.
    Blank lines and comment lines (starting with ``#``) are skipped silently.
    """
    path = Path(training_dir) / run_id / "ppo_metrics.jsonl"
    if not path.exists():
        return []
    records: list[PPOUpdateMetrics] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            records.append(PPOUpdateMetrics.model_validate_json(line))
        except Exception:
            continue
    return records


def load_episode_metrics(
    run_id: str, training_dir: str | Path = "outputs/training"
) -> list[EpisodeMetricsSummary]:
    """
    Load all episode metric summaries for a training run from JSONL.

    Parameters
    ----------
    run_id       : The run identifier.
    training_dir : Parent directory for all training run output.

    Returns
    -------
    List of ``EpisodeMetricsSummary``, in the order they were written.
    """
    path = Path(training_dir) / run_id / "episode_metrics.jsonl"
    if not path.exists():
        return []
    records: list[EpisodeMetricsSummary] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            records.append(EpisodeMetricsSummary.model_validate_json(line))
        except Exception:
            continue
    return records


def list_run_ids(training_dir: str | Path = "outputs/training") -> list[str]:
    """
    Return all run IDs found in ``training_dir``, newest first.

    A run ID is any subdirectory that contains at least one of the expected
    JSONL files.
    """
    base = Path(training_dir)
    if not base.exists():
        return []
    run_ids = [
        d.name
        for d in sorted(base.iterdir(), reverse=True)
        if d.is_dir() and (
            (d / "ppo_metrics.jsonl").exists()
            or (d / "episode_metrics.jsonl").exists()
        )
    ]
    return run_ids


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_comment(file: Any, text: str) -> None:
    """Write a JSONL comment line (starts with ``#``, skipped on load)."""
    file.write(f"# {text}\n")
    file.flush()
