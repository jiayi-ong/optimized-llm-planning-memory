"""
training/reward.py
==================
RewardFunction — multi-component shaped reward for planning episodes.

CRITICAL DESIGN INVARIANT
--------------------------
This module imports ``ConstraintSatisfactionEngine`` from ``core/constraints.py``.
The same import is used by ``evaluation/deterministic.py``. This means:

    training reward signal ≡ evaluation metric

If they ever diverge, the compressor will optimise a proxy and fail at
evaluation. Guard this invariant carefully.

Reward design rationale
------------------------
Multi-component rewards help with credit assignment in long-horizon episodes:
- Hard constraint score provides a dense, step-level signal.
- Tool efficiency discourages redundant calls.
- Terminal bonus gives a large signal for completing all hard constraints.
- Step penalty encourages efficiency without over-penalising exploration.

The ``normalise=True`` option clips the total to [-1, 1] for stable PPO training.

Design pattern: Composition
-----------------------------
``RewardFunction`` holds a ``ConstraintSatisfactionEngine`` by composition,
not inheritance. Both this class and ``DeterministicEvaluator`` hold their
own engine instances — this is intentional for thread safety in parallel workers.
"""

from __future__ import annotations

from optimized_llm_planning_memory.core.constraints import ConstraintSatisfactionEngine
from optimized_llm_planning_memory.core.config import RewardConfig
from optimized_llm_planning_memory.core.models import (
    EpisodeLog,
    Itinerary,
    RewardComponents,
    UserRequest,
)
from optimized_llm_planning_memory.tools.tracker import ToolCallTracker


class RewardFunction:
    """
    Computes multi-component shaped reward for a planning episode.

    Parameters
    ----------
    constraint_engine : Shared ``ConstraintSatisfactionEngine`` instance.
                        MUST be the same implementation used by evaluation.
    config            : Reward weights and penalty coefficients.
    """

    def __init__(
        self,
        constraint_engine: ConstraintSatisfactionEngine | None = None,
        config: RewardConfig | None = None,
    ) -> None:
        self._engine = constraint_engine or ConstraintSatisfactionEngine()
        self._config = config or RewardConfig()

    def compute(
        self,
        episode_log: EpisodeLog,
        user_request: UserRequest,
        is_terminal: bool = False,
    ) -> RewardComponents:
        """
        Compute all reward components for the current episode state.

        Parameters
        ----------
        episode_log  : The current episode log (may be partial during training).
        user_request : The original user request (for constraint evaluation).
        is_terminal  : True on the final step; activates terminal bonus and
                       ``terminal_itinerary_score``.

        Returns
        -------
        RewardComponents
        """
        itinerary = episode_log.final_itinerary
        tracker_stats = episode_log.tool_stats

        hard_score = self._hard_constraint_score(itinerary, user_request)
        soft_score = self._soft_constraint_score(itinerary, user_request)
        efficiency_score = self._tool_efficiency_score(tracker_stats)
        failure_penalty = self._tool_failure_penalty(tracker_stats)
        consistency_score = self._logical_consistency_score(itinerary)
        terminal_score = self._terminal_itinerary_score(itinerary, user_request) if is_terminal else None

        w = self._config.weights
        total = (
            w.hard_constraint * hard_score
            + w.soft_constraint * soft_score
            + w.tool_efficiency * efficiency_score
            + w.tool_failure_penalty * failure_penalty  # already ≤ 0
            + w.logical_consistency * consistency_score
            + self._config.step_penalty  # per-step cost
        )
        if is_terminal and terminal_score is not None:
            total += w.terminal_itinerary * terminal_score
            # Terminal bonus if all hard constraints satisfied
            if hard_score >= 1.0:
                total += self._config.terminal_bonus

        if self._config.normalize:
            max_possible = (
                w.hard_constraint + w.soft_constraint + w.tool_efficiency
                + w.logical_consistency + w.terminal_itinerary + self._config.terminal_bonus
            )
            total = total / max_possible if max_possible > 0 else 0.0

        return RewardComponents(
            hard_constraint_score=hard_score,
            soft_constraint_score=soft_score,
            tool_efficiency_score=efficiency_score,
            tool_failure_penalty=failure_penalty,
            logical_consistency_score=consistency_score,
            terminal_itinerary_score=terminal_score,
            total_reward=total,
        )

    # ── Component methods ─────────────────────────────────────────────────────

    def _hard_constraint_score(
        self, itinerary: Itinerary | None, request: UserRequest
    ) -> float:
        """Fraction of hard constraints satisfied. [0.0, 1.0]"""
        if itinerary is None or not request.hard_constraints:
            return 0.0
        results = self._engine.evaluate(itinerary, list(request.hard_constraints))
        return self._engine.hard_satisfaction_ratio(results, list(request.hard_constraints))

    def _soft_constraint_score(
        self, itinerary: Itinerary | None, request: UserRequest
    ) -> float:
        """Weighted average soft constraint satisfaction. [0.0, 1.0]"""
        if itinerary is None or not request.soft_constraints:
            return 0.5  # neutral if no soft constraints
        results = self._engine.evaluate(itinerary, list(request.soft_constraints))
        return self._engine.soft_satisfaction_score(results, list(request.soft_constraints))

    def _tool_efficiency_score(self, stats: tuple) -> float:
        """Penalises redundant tool calls. [0.0, 1.0]"""
        if not stats:
            return 1.0
        total_calls = sum(s.call_count for s in stats)
        redundant = sum(s.redundant_call_count for s in stats)
        if total_calls == 0:
            return 1.0
        return max(0.0, 1.0 - redundant / total_calls)

    def _tool_failure_penalty(self, stats: tuple) -> float:
        """Penalty proportional to failure rate. ≤ 0.0"""
        if not stats:
            return 0.0
        total_calls = sum(s.call_count for s in stats)
        failures = sum(s.failure_count for s in stats)
        if total_calls == 0:
            return 0.0
        failure_rate = failures / total_calls
        return -failure_rate  # maps to [-1.0, 0.0]; config weight scales it

    def _logical_consistency_score(self, itinerary: Itinerary | None) -> float:
        """Check date ordering and no double-bookings. [0.0, 1.0]"""
        if itinerary is None or not itinerary.days:
            return 0.0

        issues = 0
        total_checks = 0

        # Check 1: dates are in order
        dates = [d.date for d in itinerary.days]
        total_checks += 1
        if dates != sorted(dates):
            issues += 1

        # Check 2: no duplicate hotel bookings
        hotel_ids = [
            d.accommodation.hotel_id
            for d in itinerary.days
            if d.accommodation is not None
        ]
        total_checks += 1
        if len(hotel_ids) != len(set(hotel_ids)):
            issues += 1

        # Check 3: budget not exceeded
        if itinerary.total_cost_usd > 0:
            total_checks += 1
            # (budget check is in hard constraint; here we check internal consistency)

        if total_checks == 0:
            return 1.0
        return max(0.0, 1.0 - issues / total_checks)

    def _terminal_itinerary_score(
        self, itinerary: Itinerary | None, request: UserRequest
    ) -> float:
        """Composite terminal score: hard constraint ratio + completeness. [0.0, 1.0]"""
        if itinerary is None:
            return 0.0
        hard_score = self._hard_constraint_score(itinerary, request)
        completeness = 1.0 if itinerary.is_complete else 0.5
        return (hard_score + completeness) / 2.0
