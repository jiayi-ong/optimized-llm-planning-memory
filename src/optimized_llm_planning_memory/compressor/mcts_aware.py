"""
compressor/mcts_aware.py
========================
MCTSAwareCompressor — ABC for compressors that consume an MCTS search tree.

Design
------
This ABC extends ``CompressorBase`` with a single additional abstract method:
``compress_with_tree()``. Subclasses must implement BOTH:

  - ``compress()``            — standard linear compression (non-MCTS fallback).
  - ``compress_with_tree()``  — tree-aware distillation of an MCTSTreeRepresentation.

The two-method contract preserves the ``CompressorBase.compress()`` interface
unchanged, so all existing code that type-checks against ``CompressorBase``
continues to work. The MCTS path is taken only when:

  1. ``ReActAgent._mode == AgentMode.MCTS_COMPRESSOR``, AND
  2. ``isinstance(self._compressor, MCTSAwareCompressor)``, AND
  3. ``self._mcts_controller is not None``.

If any of these conditions is not met, the agent silently falls back to
the standard ``compress()`` path.

Invariant: ``compress_with_tree()`` MUST produce a CompressedState that passes
``CompressedStateTemplate.validate()`` — all 6 standard template sections must
be populated. The two optional MCTS fields (``top_candidates``, ``tradeoffs``)
are populated additionally and injected into the agent context by
``ContextBuilder._history_mcts_compressor()``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from optimized_llm_planning_memory.compressor.base import CompressorBase
from optimized_llm_planning_memory.core.models import CompressedState, TrajectoryModel

if TYPE_CHECKING:
    from optimized_llm_planning_memory.mcts.node import MCTSTreeRepresentation


class MCTSAwareCompressor(CompressorBase):
    """
    Abstract base class for compressors that distill an MCTS search tree.

    Extends ``CompressorBase`` with ``compress_with_tree()``.
    Subclasses must implement both abstract methods.

    Why not a separate ABC hierarchy?
    ----------------------------------
    Keeping ``MCTSAwareCompressor`` as a subclass of ``CompressorBase`` means
    it satisfies all existing ``isinstance(compressor, CompressorBase)`` checks.
    The agent checks ``isinstance(compressor, MCTSAwareCompressor)`` specifically
    to enable the MCTS path — a clean opt-in without modifying existing code.
    """

    @abstractmethod
    def compress(
        self,
        trajectory: TrajectoryModel,
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        """
        Standard compression fallback (no MCTS tree).

        Used when: the agent is in COMPRESSOR mode (not MCTS_COMPRESSOR), or
        when no MCTS tree is available for the first compression of an episode.

        All 6 template sections must be populated.
        """

    @abstractmethod
    def compress_with_tree(
        self,
        tree_repr: "MCTSTreeRepresentation",
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        """
        Distill an MCTS search tree into a structured CompressedState.

        Parameters
        ----------
        tree_repr      : Full MCTSTreeRepresentation from MCTSController.search().
                         Contains the best-path trajectory, alternative paths,
                         top_candidates, tradeoffs, and MCTSStats.
        previous_state : Last CompressedState from the current episode, if any.

        Returns
        -------
        CompressedState
            All 6 standard template sections must be populated (required by
            CompressedStateTemplate.validate()).

            Additionally, the two optional MCTS fields should be populated:
            - ``top_candidates`` : list[str] with human-readable branch descriptions.
            - ``tradeoffs``      : str with a tradeoff summary.

        Raises
        ------
        CompressedStateRenderError
            If any required template section is missing from the output.
        """
