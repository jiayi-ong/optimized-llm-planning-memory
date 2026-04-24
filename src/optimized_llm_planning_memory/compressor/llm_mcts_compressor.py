"""
compressor/llm_mcts_compressor.py
==================================
LLMMCTSCompressor — non-trainable LLM-based MCTS-aware compressor.

Role
----
This is the first concrete ``MCTSAwareCompressor``:

  - ``compress()``           → delegates to an internal ``LLMCompressor`` instance
                               for full backward-compatibility with non-MCTS runs.
  - ``compress_with_tree()`` → sends the MCTSTreeRepresentation to an LLM prompt
                               that produces structured output including all 6
                               standard template sections PLUS top_candidates and
                               tradeoffs.

Not trainable: does not implement ``get_log_probs()`` or override
``get_trainable_parameters()``.

Used for:
  - Evaluation under ``AgentMode.MCTS_COMPRESSOR``.
  - Establishing a supervised training signal for a trainable MCTS distiller.
  - Prototyping in the compressor dev notebook (Section 9).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import instructor
import litellm
from pydantic import BaseModel

from optimized_llm_planning_memory.compressor.llm_compressor import LLMCompressor
from optimized_llm_planning_memory.compressor.mcts_aware import MCTSAwareCompressor
from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate
from optimized_llm_planning_memory.core.models import (
    CompressedState,
    HardConstraintLedger,
    TrajectoryModel,
)

if TYPE_CHECKING:
    from optimized_llm_planning_memory.mcts.node import MCTSTreeRepresentation


# ── Instructor response schema ────────────────────────────────────────────────

class _MCTSCompressorLLMResponse(BaseModel):
    """
    Structured output schema for the MCTS compression prompt.
    Includes the 6 standard compression fields plus 2 MCTS-specific fields.
    """
    satisfied_constraint_ids: list[str]
    violated_constraint_ids: list[str]
    unknown_constraint_ids: list[str]
    soft_constraints_summary: str
    decisions_made: list[str]
    open_questions: list[str]
    key_discoveries: list[str]
    current_itinerary_sketch: str
    # MCTS-specific
    top_candidates: list[str]
    tradeoffs: str


# ── LLMMCTSCompressor ─────────────────────────────────────────────────────────

class LLMMCTSCompressor(MCTSAwareCompressor):
    """
    LLM-based MCTS-aware compressor.

    Parameters
    ----------
    llm_model_id     : litellm model string, e.g. ``"openai/gpt-4o-mini"``.
    temperature      : LLM temperature (0.0 recommended for determinism).
    max_output_tokens: Max tokens for the LLM compression response.
    """

    def __init__(
        self,
        llm_model_id: str = "openai/gpt-4o-mini",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> None:
        self._model_id = llm_model_id
        self._temperature = temperature
        self._max_tokens = max_output_tokens
        self._template = CompressedStateTemplate()
        self._client = instructor.from_litellm(litellm.completion)
        # Non-MCTS fallback: delegate to LLMCompressor
        self._fallback = LLMCompressor(
            model_id=llm_model_id,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )

    # ── CompressorBase contract ───────────────────────────────────────────────

    def compress(
        self,
        trajectory: TrajectoryModel,
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        """Non-MCTS path: delegate to LLMCompressor."""
        return self._fallback.compress(trajectory, previous_state)

    # ── MCTSAwareCompressor contract ──────────────────────────────────────────

    def compress_with_tree(
        self,
        tree_repr: "MCTSTreeRepresentation",
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        """
        Distill the MCTS tree into a CompressedState using an LLM call.

        The prompt includes:
        - Best-path trajectory text (linearised via TrajectoryModel.to_text()).
        - Text of up to 2 alternative branch trajectories (first 20 lines each).
        - top_candidates list from the tree representation.
        - tradeoffs string from the tree.
        - Previous CompressedState if available.

        Output is parsed by instructor into ``_MCTSCompressorLLMResponse``
        and then assembled into a ``CompressedState``.
        """
        prompt = self._build_tree_prompt(tree_repr, previous_state)

        response: _MCTSCompressorLLMResponse = self._client.chat.completions.create(
            model=self._model_id,
            response_model=_MCTSCompressorLLMResponse,
            messages=[
                {"role": "system", "content": _MCTS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        # Build HardConstraintLedger from response + previous state
        prior_constraints = (
            tuple(previous_state.hard_constraint_ledger.constraints)
            if previous_state else ()
        )
        ledger = HardConstraintLedger(
            constraints=prior_constraints,
            satisfied_ids=tuple(response.satisfied_constraint_ids),
            violated_ids=tuple(response.violated_constraint_ids),
            unknown_ids=tuple(response.unknown_constraint_ids),
        )

        best_traj = tree_repr.best_path_trajectory
        state = CompressedState(
            state_id=str(uuid.uuid4()),
            trajectory_id=best_traj.trajectory_id,
            step_index=best_traj.total_steps,
            hard_constraint_ledger=ledger,
            soft_constraints_summary=response.soft_constraints_summary,
            decisions_made=response.decisions_made,
            open_questions=response.open_questions,
            key_discoveries=response.key_discoveries,
            current_itinerary_sketch=response.current_itinerary_sketch,
            compression_method="llm_mcts",
            token_count=None,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            top_candidates=response.top_candidates or None,
            tradeoffs=response.tradeoffs or None,
        )

        self._template.validate(state)
        return state

    # ── Metadata ──────────────────────────────────────────────────────────────

    def get_metadata(self) -> dict:
        return {
            "type": "llm_mcts",
            "model_id": self._model_id,
            "param_count": 0,
            "trainable": False,
        }

    # ── Prompt construction ───────────────────────────────────────────────────

    def _build_tree_prompt(
        self,
        tree_repr: "MCTSTreeRepresentation",
        previous_state: CompressedState | None,
    ) -> str:
        parts: list[str] = []

        if previous_state is not None:
            parts.append("== PREVIOUS COMPRESSED STATE ==")
            parts.append(self._template.render(previous_state))
            parts.append("")

        # Best path
        parts.append("== BEST CANDIDATE TRAJECTORY (highest MCTS value) ==")
        parts.append(tree_repr.best_path_trajectory.to_text())
        parts.append("")

        # Alternative paths (first 20 lines each to keep prompt size bounded)
        for i, alt_traj in enumerate(tree_repr.alternative_paths[:2]):
            alt_text = "\n".join(alt_traj.to_text().splitlines()[:20])
            parts.append(f"== ALTERNATIVE BRANCH {i + 1} (partial) ==")
            parts.append(alt_text)
            parts.append("")

        # MCTS metadata
        if tree_repr.top_candidates:
            parts.append("== MCTS TOP CANDIDATES ==")
            for cand in tree_repr.top_candidates:
                parts.append(f"  - {cand}")
            parts.append("")

        if tree_repr.tradeoffs:
            parts.append("== MCTS TRADEOFFS ANALYSIS ==")
            parts.append(tree_repr.tradeoffs)
            parts.append("")

        stats = tree_repr.stats
        parts.append(
            f"== SEARCH STATISTICS ==\n"
            f"Simulations: {stats.num_simulations}, "
            f"Nodes: {stats.nodes_explored}, "
            f"Max depth: {stats.max_depth_reached}, "
            f"Root Q: {stats.root_value:.3f}"
        )

        return "\n".join(parts)


# ── System prompt ─────────────────────────────────────────────────────────────

_MCTS_SYSTEM_PROMPT = """\
You are a travel planning memory compressor with access to MCTS search results.

You will receive:
1. The best candidate planning trajectory (highest MCTS value).
2. Up to 2 alternative branch trajectories.
3. Top candidate descriptions from the tree search.
4. A tradeoffs analysis from the MCTS.

Your task is to produce a structured JSON object that captures:
- The current constraint satisfaction status (satisfied/violated/unknown IDs).
- A concise soft constraints summary.
- Committed decisions from the best trajectory.
- Open planning questions, especially unresolved branch choices.
- Key discoveries about availability, prices, and feasibility.
- A current itinerary sketch based on the best trajectory.
- top_candidates: A list of the top 3 candidate plans with their trade-offs.
- tradeoffs: A paragraph explaining the key tradeoffs between the branches.

Be specific and decision-critical. Do not lose hard constraints.
Prefer the best trajectory's decisions in decisions_made and itinerary_sketch,
but surface alternative options in top_candidates and tradeoffs.
"""
