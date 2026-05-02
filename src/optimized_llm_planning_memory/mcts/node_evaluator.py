"""
mcts/node_evaluator.py
======================
NodeEvaluator — scores an MCTSNode during the SIMULATE phase.

Design
------
The evaluator uses a two-tier strategy to balance speed and accuracy:

  Tier 1 (always): Fast structural heuristics that require no LLM calls.
    - Hard constraint coverage: fraction of constraint IDs mentioned in the
      trajectory text (proxy for progress toward constraint satisfaction).
    - Tool success rate: fraction of tool calls that succeeded.
    - Booking depth: number of successful booking actions (hotel, flight, event).

  Tier 2 (conditional): A single LLM call that rates the trajectory 0–10.
    Called only when the heuristic score falls in the ambiguous band [0.3, 0.7],
    saving API budget while providing signal where the heuristic is uncertain.

The final score is a weighted combination:
    score = (1 - llm_weight) * heuristic + llm_weight * llm_score
    where llm_weight = 0.0 if Tier 2 was not called.

Evaluation caching
------------------
When ``MCTSConfig.use_cached_evaluations=True``, results are cached keyed by
a short hash of the trajectory text. This prevents duplicate LLM calls for
sibling nodes that share the same trajectory prefix.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import litellm

from optimized_llm_planning_memory.mcts.node import MCTSNode

if TYPE_CHECKING:
    from optimized_llm_planning_memory.core.models import TrajectoryModel, UserRequest
    from optimized_llm_planning_memory.mcts.config import MCTSConfig


# Constants
_HEURISTIC_AMBIGUOUS_LOW = 0.3
_HEURISTIC_AMBIGUOUS_HIGH = 0.7
_LLM_BLEND_WEIGHT = 0.4   # weight given to LLM score in the blend
_BOOKING_TOOLS = {"book_hotel", "book_event", "select_flight"}


class NodeEvaluator:
    """
    Scores MCTSNodes for the SIMULATE phase.

    Parameters
    ----------
    model_id : litellm model string for LLM-based evaluation.
    config   : MCTSConfig; provides ``use_cached_evaluations``.
    request  : The original UserRequest — used in LLM scoring prompt for context.
               May be None; if None, the LLM prompt omits the request.
    """

    def __init__(
        self,
        model_id: str,
        config: "MCTSConfig",
        request: "UserRequest | None" = None,
    ) -> None:
        self._model_id = model_id
        self._config = config
        self._request = request
        self._cache: dict[str, float] = {}  # trajectory_hash → score

    def set_request(self, request: "UserRequest") -> None:
        """Update the request used for LLM scoring. Called per-search by MCTSController."""
        self._request = request

    def evaluate(self, node: MCTSNode) -> float:
        """
        Return a scalar value in [0.0, 1.0] for the node's trajectory.

        Checks the cache first; computes and caches on miss.
        """
        trajectory = node.trajectory_snapshot
        cache_key = _trajectory_hash(trajectory)

        if self._config.use_cached_evaluations and cache_key in self._cache:
            return self._cache[cache_key]

        score = self._compute(trajectory)

        if self._config.use_cached_evaluations:
            self._cache[cache_key] = score

        return score

    def _compute(self, trajectory: "TrajectoryModel") -> float:
        h_score = self._heuristic_score(trajectory)

        if _HEURISTIC_AMBIGUOUS_LOW <= h_score <= _HEURISTIC_AMBIGUOUS_HIGH:
            try:
                llm_score = self._llm_score(trajectory)
                return (1.0 - _LLM_BLEND_WEIGHT) * h_score + _LLM_BLEND_WEIGHT * llm_score
            except Exception:
                # LLM call failure → fall back to heuristic only
                pass

        return h_score

    def _heuristic_score(self, trajectory: "TrajectoryModel") -> float:
        """
        Fast structural score (no LLM calls).

        Three sub-scores, equally weighted:
          1. Tool success rate — fraction of tool calls that succeeded.
          2. Booking depth    — normalised count of booking actions (cap at 5).
          3. Constraint coverage — fraction of constraint keywords present in
                                  trajectory text (rough proxy for progress).
        """
        steps = trajectory.steps

        # Sub-score 1: tool success rate
        tool_calls = [s for s in steps if s.observation is not None]
        if tool_calls:
            success_rate = sum(1 for s in tool_calls if s.observation.success) / len(tool_calls)
        else:
            success_rate = 0.5  # neutral — no data yet

        # Sub-score 2: booking depth (normalised)
        booking_count = sum(
            1 for s in steps
            if s.action is not None and s.action.tool_name.lower() in _BOOKING_TOOLS
            and s.observation is not None and s.observation.success
        )
        booking_score = min(booking_count / 5.0, 1.0)

        # Sub-score 3: constraint coverage (heuristic — checks if constraint
        # keywords from the request appear in the trajectory text)
        coverage = self._constraint_coverage(trajectory)

        return (success_rate + booking_score + coverage) / 3.0

    def _constraint_coverage(self, trajectory: "TrajectoryModel") -> float:
        """
        Rough proxy for how many hard constraints have been addressed.
        Returns 0.5 if no request is set (neutral).
        """
        if self._request is None:
            return 0.5

        constraints = list(self._request.hard_constraints) + list(self._request.soft_constraints)
        if not constraints:
            return 0.5

        traj_text = trajectory.to_text().lower()
        satisfied = sum(
            1 for c in constraints
            if c.description and c.description.lower()[:30] in traj_text
        )
        return satisfied / len(constraints)

    def _llm_score(self, trajectory: "TrajectoryModel") -> float:
        """
        Single LLM call that rates the trajectory progress 0–10.
        Normalised to [0.0, 1.0].
        """
        request_snippet = ""
        if self._request is not None:
            raw = getattr(self._request, "raw_text", "")
            request_snippet = f"\nUser request: {raw[:300]}"

        prompt = (
            f"You are evaluating a travel planning agent's progress.{request_snippet}\n\n"
            f"Trajectory (last {min(len(trajectory.steps), 6)} steps):\n"
            f"{trajectory.to_text()}\n\n"
            "Rate this trajectory's progress toward a complete, constraint-satisfying "
            "itinerary on a scale of 0 to 10. "
            "Output ONLY an integer from 0 to 10, nothing else."
        )

        response = litellm.completion(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8,
        )
        raw = (response.choices[0].message.content or "5").strip()
        try:
            score_10 = float(raw.split()[0])
        except (ValueError, IndexError):
            score_10 = 5.0
        return max(0.0, min(score_10 / 10.0, 1.0))


# ── Private helpers ───────────────────────────────────────────────────────────

def _trajectory_hash(trajectory: "TrajectoryModel") -> str:
    """Short MD5 hash of the trajectory text for cache keying."""
    text = trajectory.to_text()
    return hashlib.md5(text.encode()).hexdigest()[:16]
