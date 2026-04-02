"""
utils/tensorboard.py
=====================
Typed TensorBoard writer helpers.

Wraps ``torch.utils.tensorboard.SummaryWriter`` with project-specific
convenience methods so that callsites do not scatter raw SummaryWriter
boilerplate across the codebase.

Usage
-----
    from optimized_llm_planning_memory.utils.tensorboard import TBLogger

    tb = TBLogger("outputs/logs/run_001")
    tb.log_reward(reward_components, step=100)
    tb.log_eval(eval_result, step=100)
    tb.log_scalar("ppo/policy_loss", 0.032, step=100)
    tb.close()

Or use as a context manager:
    with TBLogger("outputs/logs/run_001") as tb:
        tb.log_scalar("train/reward", 0.5, step=1)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from optimized_llm_planning_memory.core.models import EvalResult, RewardComponents


class TBLogger:
    """
    Thin typed wrapper over ``torch.utils.tensorboard.SummaryWriter``.

    If TensorBoard (or PyTorch) is not installed, all methods are no-ops so
    that code depending on TBLogger does not crash in minimal environments.
    """

    def __init__(self, log_dir: str | Path) -> None:
        self._log_dir = Path(log_dir)
        self._writer: Any = None
        self._available = False
        self._init_writer()

    def _init_writer(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(self._log_dir))
            self._available = True
        except ImportError:
            pass

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "TBLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        if self._writer is not None:
            self._writer.close()

    # ── Logging helpers ───────────────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Write a single scalar to TensorBoard."""
        if self._available and self._writer is not None:
            self._writer.add_scalar(tag, value, global_step=step)

    def log_reward(self, rc: RewardComponents, step: int) -> None:
        """Write all reward components from an episode."""
        self.log_scalar("reward/hard_constraint", rc.hard_constraint_score, step)
        self.log_scalar("reward/soft_constraint", rc.soft_constraint_score, step)
        self.log_scalar("reward/tool_efficiency", rc.tool_efficiency_score, step)
        self.log_scalar("reward/tool_failure_penalty", rc.tool_failure_penalty, step)
        self.log_scalar("reward/logical_consistency", rc.logical_consistency_score, step)
        self.log_scalar("reward/total", rc.total_reward, step)
        if rc.terminal_itinerary_score is not None:
            self.log_scalar("reward/terminal_itinerary", rc.terminal_itinerary_score, step)

    def log_eval(self, result: EvalResult, step: int) -> None:
        """Write all scores from an EvalResult."""
        for key, val in result.deterministic_scores.items():
            self.log_scalar(f"eval/det/{key}", val, step)
        for key, val in result.llm_judge_scores.items():
            self.log_scalar(f"eval/judge/{key}", val, step)
        self.log_scalar("eval/overall", result.overall_score, step)

    def log_scalars(self, tag_prefix: str, scalars: dict[str, float], step: int) -> None:
        """Write multiple scalars with a shared tag prefix."""
        for key, val in scalars.items():
            self.log_scalar(f"{tag_prefix}/{key}", val, step)
