"""
compressor/base.py
==================
CompressorBase — the universal interface for all compression strategies.

Design pattern: ABC (Abstract Base Class)
------------------------------------------
ABC is chosen over Protocol here because:
  - All concrete compressors are part of THIS codebase (unlike the simulator).
  - We want Python to enforce ``compress()`` at class-definition time.
  - The default ``get_log_probs()`` and ``get_trainable_parameters()``
    implementations are meaningful (raise / return []) rather than just ``...``.

Extensibility
-------------
To add a new compression strategy (e.g., a rule-based compressor that never
calls an LLM):
  1. Subclass ``CompressorBase``.
  2. Implement ``compress()``.
  3. If trainable, subclass ``TrainableCompressorBase`` instead.

The agent and environment accept ``CompressorBase``; they never import concrete
compressor classes directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from optimized_llm_planning_memory.core.exceptions import LogProbsNotSupportedError
from optimized_llm_planning_memory.core.models import CompressedState, TrajectoryModel


class CompressorBase(ABC):
    """
    Abstract base class for all context compression strategies.

    Subclasses must implement ``compress()``. All other methods have default
    implementations that express "this compressor is not RL-trainable."

    Usage in the RL loop
    --------------------
    During RL training, the ``CompressionEnv`` calls ``compress()`` on each
    compression event step. The result's text is the "action" for PPO.
    Log-probs for PPO clipping are computed via ``get_log_probs()`` —
    which requires a ``TrainableCompressorBase`` subclass.
    """

    @abstractmethod
    def compress(
        self,
        trajectory: TrajectoryModel,
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        """
        Distill a ReAct trajectory into a structured CompressedState.

        Parameters
        ----------
        trajectory      : Frozen TrajectoryModel produced by Trajectory.to_model().
        previous_state  : The last CompressedState, if any. The compressor may
                          use this for continuity (e.g., carry over known decisions).
                          None on the first compression of an episode.

        Returns
        -------
        CompressedState
            All template sections must be populated.
            See ``CompressedStateTemplate.SECTIONS`` for the required fields.
        """

    def get_log_probs(
        self,
        trajectory_text: str,
        compressed_text: str,
    ) -> "torch.Tensor":  # type: ignore[name-defined]  # noqa: F821
        """
        Compute token-level log-probabilities for the compressed output.

        Required by the PPO training loop for computing the clipping ratio.
        Non-trainable compressors raise ``LogProbsNotSupportedError``.

        Parameters
        ----------
        trajectory_text  : The input that was fed to the compressor.
        compressed_text  : The output the compressor produced.

        Returns
        -------
        torch.Tensor of shape (sequence_length,)
            Per-token log p(token | context).
        """
        raise LogProbsNotSupportedError(
            f"{type(self).__name__} does not support log-probability computation. "
            f"Use a TrainableCompressorBase subclass for RL training."
        )

    def get_trainable_parameters(self) -> list:
        """
        Return the list of ``torch.nn.Parameter`` objects to be optimised by PPO.

        Non-trainable compressors return an empty list. This allows the RL
        trainer to check ``len(compressor.get_trainable_parameters()) == 0``
        and skip gradient updates.
        """
        return []

    def is_trainable(self) -> bool:
        """Return True if this compressor supports RL training (has trainable parameters)."""
        return len(self.get_trainable_parameters()) > 0

    def get_metadata(self) -> dict[str, Any]:
        """
        Return a JSON-serialisable dict describing this compressor instance.

        Used to populate EvalRunManifest.compressor_type and checkpoint logs.
        Subclasses should override to add type-specific fields (model_id, etc.).

        Minimum keys returned by the base implementation:
            {"type": str, "param_count": int, "trainable": bool}
        """
        return {
            "type": type(self).__name__.lower().replace("compressor", ""),
            "param_count": len(self.get_trainable_parameters()),
            "trainable": self.is_trainable(),
        }
