"""
compressor/identity_compressor.py
===================================
IdentityCompressor — returns the full trajectory with no distillation.

Design rationale
----------------
This compressor is used as the baseline condition where NO compression is
applied. The full trajectory text is passed through to the agent context
unchanged (wrapped in a CompressedState to satisfy the shared interface).

Trainability for the RL pipeline
---------------------------------
TrainableCompressorBase requires get_log_probs() and get_trainable_parameters()
so that PPO can compute the importance-sampling ratio and backpropagate.

IdentityCompressor satisfies this contract with a single scalar nn.Parameter
(_dummy_param). PPO can update _dummy_param via gradient descent, keeping the
RL infrastructure fully operational. _dummy_param does NOT affect compress()
output — the identity behaviour is preserved regardless of its value.

Reward predictor integration
-----------------------------
An optional RewardPredictorComponent can be attached at construction time. The
RL trainer's RewardPredictorCallback populates and fits this component after each
batch of episodes. Like _dummy_param, it does not affect compress() output;
it is a diagnostic artefact that tracks which episode features predict reward.

Usage
-----
    from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
    from optimized_llm_planning_memory.compressor.reward_predictor import RewardPredictorComponent

    rp = RewardPredictorComponent()
    compressor = IdentityCompressor(reward_predictor=rp)
    state = compressor.compress(trajectory_model)
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

from optimized_llm_planning_memory.compressor.trainable_base import TrainableCompressorBase
from optimized_llm_planning_memory.core.exceptions import CompressorCheckpointError
from optimized_llm_planning_memory.core.models import (
    CompressedState,
    HardConstraintLedger,
    TrajectoryModel,
)


class IdentityCompressor(TrainableCompressorBase):
    """
    Identity compressor: passes the full trajectory through unchanged.

    compress() always returns the complete trajectory text as current_itinerary_sketch,
    so the agent's context window contains the full planning history at each step.
    This matches the RAW baseline condition but implements the CompressorBase
    interface, allowing it to be swapped for a trained compressor without
    changing the agent or training code.

    Parameters
    ----------
    reward_predictor : Optional RewardPredictorComponent for RL reward prediction.
                       Not used in compress(); updated by RewardPredictorCallback.
    """

    def __init__(self, reward_predictor: object | None = None) -> None:
        # Single scalar trainable parameter — satisfies PPO's need for a
        # differentiable action distribution without changing compress() output.
        self._dummy_param = nn.Parameter(torch.zeros(1))
        self._reward_predictor = reward_predictor  # reference only

    # ── CompressorBase interface ───────────────────────────────────────────────

    def compress(
        self,
        trajectory: TrajectoryModel,
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        """
        Return the full trajectory text as a CompressedState (identity).

        The entire trajectory text is placed in current_itinerary_sketch.
        When ContextBuilder renders this state via the template, the agent
        receives the complete planning history — equivalent to RAW mode.

        Parameters
        ----------
        trajectory     : Frozen trajectory snapshot.
        previous_state : Ignored (identity has no incremental state).

        Returns
        -------
        CompressedState with current_itinerary_sketch = full trajectory text.
        """
        full_text = trajectory.to_text()

        # Build decisions_made from the last few steps for template completeness
        decisions: list[str] = []
        for step in trajectory.steps:
            tool_name = step.action.tool_name if step.action else "no action"
            decisions.append(f"step {step.step_index}: {tool_name}")

        ledger = HardConstraintLedger(
            constraints=(),
            satisfied_ids=(),
            violated_ids=(),
            unknown_ids=(),
        )

        return CompressedState(
            state_id=str(uuid.uuid4()),
            trajectory_id=trajectory.trajectory_id,
            step_index=trajectory.total_steps,
            hard_constraint_ledger=ledger,
            soft_constraints_summary=(
                "[Identity] Full trajectory preserved without compression."
            ),
            decisions_made=decisions,
            open_questions=[],
            key_discoveries=[],
            current_itinerary_sketch=full_text,
            compression_method="identity",
            token_count=len(full_text.split()),
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )

    # ── TrainableCompressorBase interface ─────────────────────────────────────

    def get_log_probs(
        self,
        trajectory_text: str,
        compressed_text: str,
    ) -> torch.Tensor:
        """
        Return uniform log-probabilities over the compressed text tokens.

        For an identity compressor the "action" is always the same (copy
        the trajectory), so all positions have equal probability. The tiny
        _dummy_param term (scaled by 1e-8) ensures gradient flow to the
        PPO optimizer while keeping the log_prob values effectively constant.

        Parameters
        ----------
        trajectory_text : Source text (not used; identity ignores it).
        compressed_text : The compressed state text produced by compress().

        Returns
        -------
        (T,) float tensor where T = number of whitespace-split tokens.
        """
        n = max(len(compressed_text.split()), 1)
        # Uniform: each token has log_prob = -log(n)
        # _dummy_param * 1e-8 ≈ 0, so values are essentially constant
        # but autograd can still flow gradients into _dummy_param.
        uniform = torch.full(
            (n,), -math.log(n), dtype=torch.float32
        )
        return uniform + self._dummy_param.squeeze() * 1e-8

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return the single dummy parameter for PPO to optimise."""
        return [self._dummy_param]

    def save_checkpoint(self, path: str) -> None:
        """
        Save _dummy_param to {path}/identity_compressor.pt.

        Also delegates to reward_predictor.save() if one is attached.
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"dummy_param": self._dummy_param.data},
            save_dir / "identity_compressor.pt",
        )
        if self._reward_predictor is not None and hasattr(self._reward_predictor, "save"):
            self._reward_predictor.save(str(save_dir / "reward_predictor"))

    def load_checkpoint(self, path: str) -> None:
        """
        Load _dummy_param from {path}/identity_compressor.pt.

        Raises
        ------
        CompressorCheckpointError
            If the file does not exist or is incompatible.
        """
        p = Path(path)
        if p.is_dir():
            p = p / "identity_compressor.pt"
        if not p.exists():
            raise CompressorCheckpointError(
                f"IdentityCompressor checkpoint not found at {p}."
            )
        try:
            ckpt = torch.load(p, map_location="cpu")
            self._dummy_param.data = ckpt["dummy_param"]
        except Exception as exc:
            raise CompressorCheckpointError(
                f"Failed to load IdentityCompressor checkpoint from {p}: {exc}"
            ) from exc

        rp_path = Path(path) / "reward_predictor"
        if (
            rp_path.exists()
            and self._reward_predictor is not None
            and hasattr(self._reward_predictor, "load")
        ):
            self._reward_predictor.load(str(rp_path))

    def get_metadata(self) -> dict:
        return {
            "type": "identity",
            "param_count": 1,
            "trainable": True,
            "has_reward_predictor": self._reward_predictor is not None,
        }
