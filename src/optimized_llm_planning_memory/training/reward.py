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

from typing import Callable

from optimized_llm_planning_memory.core.constraints import ConstraintSatisfactionEngine
from optimized_llm_planning_memory.core.config import RewardConfig
from optimized_llm_planning_memory.core.models import (
    EpisodeLog,
    Itinerary,
    RewardComponents,
    UserRequest,
)
from optimized_llm_planning_memory.tools.tracker import ToolCallTracker

# Type alias for a pluggable reward component function.
# Receives (episode_log, user_request, is_terminal) and returns a scalar in [0, 1].
RewardComponentFn = Callable[[EpisodeLog, UserRequest, bool], float]


class RewardFunction:
    """
    Computes multi-component shaped reward for a planning episode.

    Parameters
    ----------
    constraint_engine : Shared ``ConstraintSatisfactionEngine`` instance.
                        MUST be the same implementation used by evaluation.
    config            : Reward weights and penalty coefficients.
    extra_components  : Optional dict of {name: (fn, weight)} for injecting
                        additional reward components without modifying this class.
                        Each function receives (episode_log, user_request, is_terminal)
                        and must return a float in [0, 1].

    Example — adding a custom component
    ------------------------------------
        def my_component(ep, req, is_terminal):
            return 1.0 if ep.total_steps < 15 else 0.5

        rf = RewardFunction(
            extra_components={"brevity": (my_component, 0.2)}
        )
    """

    def __init__(
        self,
        constraint_engine: ConstraintSatisfactionEngine | None = None,
        config: RewardConfig | None = None,
        extra_components: dict[str, tuple[RewardComponentFn, float]] | None = None,
    ) -> None:
        self._engine = constraint_engine or ConstraintSatisfactionEngine()
        self._config = config or RewardConfig()
        self._extra_components: dict[str, tuple[RewardComponentFn, float]] = extra_components or {}

        # Register optional v2 components from config if enabled
        self._register_optional_components()

    def _register_optional_components(self) -> None:
        """Read config.optional and register enabled v2 metrics as extra components."""
        opt = self._config.optional
        if opt.destination_coverage.enabled:
            self._extra_components.setdefault(
                "destination_coverage",
                (self._destination_coverage_score, opt.destination_coverage.weight),
            )
        if opt.activity_density.enabled:
            self._extra_components.setdefault(
                "activity_density",
                (self._activity_density_score, opt.activity_density.weight),
            )
        if opt.budget_adherence.enabled:
            self._extra_components.setdefault(
                "budget_adherence",
                (self._budget_adherence_score, opt.budget_adherence.weight),
            )

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

        # Sum extra/optional components
        extra_total_weight = 0.0
        for _name, (fn, weight) in self._extra_components.items():
            score = fn(episode_log, user_request, is_terminal)
            total += weight * score
            extra_total_weight += weight

        if is_terminal and terminal_score is not None:
            total += w.terminal_itinerary * terminal_score
            # Terminal bonus if all hard constraints satisfied
            if hard_score >= 1.0:
                total += self._config.terminal_bonus

        if self._config.normalize:
            # Only include terminal components in the denominator when the terminal
            # bonus actually applies — including them unconditionally would under-scale
            # intermediate-step rewards (H6 fix).
            max_possible = (
                w.hard_constraint + w.soft_constraint + w.tool_efficiency
                + w.logical_consistency + extra_total_weight
            )
            if is_terminal:
                max_possible += w.terminal_itinerary + self._config.terminal_bonus
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
            # 1.0 matches constraints.py::soft_satisfaction_score() for the no-constraint case,
            # preserving the training-reward ≡ evaluation-metric invariant (H5 fix).
            return 1.0
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
        """
        Date ordering, no duplicate hotel bookings, temporal feasibility. [0.0, 1.0]

        Mirrors DeterministicEvaluator._logical_consistency() to maintain the
        training-reward ≡ evaluation-metric invariant (M3 fix).
        """
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

        # Check 3: no overlapping activities within a day
        for day in itinerary.days:
            if not day.activities:
                continue
            slots: list[tuple[str, str]] = []
            for act in day.activities:
                start = getattr(act, "start_datetime", None)
                dur = getattr(act, "duration_hours", None)
                if start and dur:
                    try:
                        from datetime import datetime, timedelta
                        start_dt = datetime.fromisoformat(start)
                        end_dt = start_dt + timedelta(hours=float(dur))
                        slots.append((start_dt.isoformat(), end_dt.isoformat()))
                    except (ValueError, TypeError):
                        pass
            if len(slots) > 1:
                total_checks += 1
                sorted_slots = sorted(slots)
                overlap = any(
                    sorted_slots[i][1] > sorted_slots[i + 1][0]
                    for i in range(len(sorted_slots) - 1)
                )
                if overlap:
                    issues += 1

        # Check 4: flight arrival ≤ hotel check-in date
        for day in itinerary.days:
            if day.accommodation is None:
                continue
            hotel_checkin = getattr(day.accommodation, "check_in", None)
            for seg in day.transport_segments:
                arrival = getattr(seg, "arrival_datetime", None)
                if hotel_checkin and arrival:
                    try:
                        arrival_date = arrival[:10]
                        total_checks += 1
                        if arrival_date > hotel_checkin:
                            issues += 1
                    except (IndexError, TypeError):
                        pass

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

    # ── Optional v2 components (delegates to DeterministicEvaluator logic) ────
    # These are not called during training by default — only when enabled via
    # config.optional.* or passed as extra_components.

    def _destination_coverage_score(
        self, episode_log: EpisodeLog, request: UserRequest, _is_terminal: bool
    ) -> float:
        """
        Fraction of requested destination cities that appear in the itinerary.
        Delegates to the same logic as DeterministicEvaluator._destination_coverage_ratio().
        [0.0, 1.0]
        """
        itinerary = episode_log.final_itinerary
        if itinerary is None or not request.destination_cities:
            return 0.0
        covered = {day.city for day in itinerary.days if day.city}
        requested = set(request.destination_cities)
        return len(covered & requested) / len(requested)

    def _activity_density_score(
        self, episode_log: EpisodeLog, _request: UserRequest, _is_terminal: bool
    ) -> float:
        """
        Per-day step-function score averaged across all days.
        Mirrors DeterministicEvaluator._activity_density_score() exactly. [0.0, 1.0]
        """
        itinerary = episode_log.final_itinerary
        if itinerary is None or not itinerary.days:
            return 0.0

        def _day_score(n: int) -> float:
            if n == 0:
                return 0.0
            if n == 1:
                return 0.5
            if n <= 4:
                return 1.0
            return max(0.0, 1.0 - 0.2 * (n - 4))

        scores = [_day_score(len(day.activities)) for day in itinerary.days]
        return sum(scores) / len(scores)

    def _budget_adherence_score(
        self, episode_log: EpisodeLog, request: UserRequest, _is_terminal: bool
    ) -> float:
        """
        1.0 if itinerary stays within budget; penalty proportional to overshoot.
        Delegates to DeterministicEvaluator._budget_adherence() logic. [0.0, 1.0]
        """
        itinerary = episode_log.final_itinerary
        if itinerary is None or request.budget_usd <= 0:
            return 0.0
        if itinerary.total_cost_usd <= request.budget_usd:
            return 1.0
        overshoot = itinerary.total_cost_usd - request.budget_usd
        return max(0.0, 1.0 - overshoot / request.budget_usd)
