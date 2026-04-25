"""
evaluation/deterministic.py
============================
DeterministicEvaluator — constraint-based scoring without LLM calls.

CRITICAL DESIGN INVARIANT
--------------------------
This module uses ``ConstraintSatisfactionEngine`` from ``core/constraints.py``.
``training/reward.py`` uses the **same class**. This is intentional and must
be preserved. If the imports ever diverge (e.g., one module monkey-patches the
engine), the training reward and evaluation metric will silently diverge, and
the compressor will optimise a proxy.

Metrics produced
----------------
hard_constraint_ratio   — fraction of hard constraints satisfied [0, 1]
soft_constraint_score   — weighted soft constraint score [0, 1]
tool_efficiency         — 1 - redundant_calls / total_calls [0, 1]
tool_failure_rate       — failures / total_calls [0, 1]  (lower is better)
avg_tool_latency_ms     — mean latency per tool call
steps_per_episode       — total ReAct steps taken
budget_adherence        — 1 if within budget, fraction remaining otherwise
logical_consistency     — date ordering + no double-bookings [0, 1]
"""

from __future__ import annotations

from optimized_llm_planning_memory.core.constraints import ConstraintSatisfactionEngine
from optimized_llm_planning_memory.core.models import EpisodeLog, Itinerary, UserRequest

METRIC_VERSION = "v1"

METRIC_CHANGELOG: dict[str, str] = {
    "v1": (
        "Initial 8 metrics: hard_constraint_ratio, soft_constraint_score, "
        "tool_efficiency, tool_failure_rate, avg_tool_latency_ms, "
        "steps_per_episode, budget_adherence, logical_consistency. "
        "Constraint evaluation via ConstraintSatisfactionEngine."
    ),
}


class DeterministicEvaluator:
    """
    Computes deterministic (rule-based) scores for a completed episode.

    Parameters
    ----------
    constraint_engine : Shared ``ConstraintSatisfactionEngine`` instance.
                        MUST be the same implementation used by RewardFunction.
    """

    def __init__(
        self,
        constraint_engine: ConstraintSatisfactionEngine | None = None,
    ) -> None:
        self._engine = constraint_engine or ConstraintSatisfactionEngine()

    def score(
        self,
        episode_log: EpisodeLog,
        user_request: UserRequest,
    ) -> dict[str, float]:
        """
        Compute all deterministic metrics for the episode.

        Parameters
        ----------
        episode_log  : Completed episode log.
        user_request : Original user request (for constraint evaluation).

        Returns
        -------
        dict[str, float] with keys matching the metric names in the module docstring.
        """
        itinerary = episode_log.final_itinerary
        stats = episode_log.tool_stats

        return {
            "hard_constraint_ratio": self._hard_constraint_ratio(itinerary, user_request),
            "soft_constraint_score": self._soft_constraint_score(itinerary, user_request),
            "tool_efficiency": self._tool_efficiency(stats),
            "tool_failure_rate": self._tool_failure_rate(stats),
            "avg_tool_latency_ms": self._avg_tool_latency_ms(stats),
            "steps_per_episode": float(episode_log.total_steps),
            "budget_adherence": self._budget_adherence(itinerary, user_request),
            "logical_consistency": self._logical_consistency(itinerary),
        }

    # ── Component scorers ─────────────────────────────────────────────────────

    def _hard_constraint_ratio(
        self, itinerary: Itinerary | None, request: UserRequest
    ) -> float:
        if itinerary is None or not request.hard_constraints:
            return 0.0
        results = self._engine.evaluate(itinerary, list(request.hard_constraints))
        return self._engine.hard_satisfaction_ratio(results, list(request.hard_constraints))

    def _soft_constraint_score(
        self, itinerary: Itinerary | None, request: UserRequest
    ) -> float:
        if itinerary is None or not request.soft_constraints:
            # 1.0 matches constraints.py and reward.py: no constraints = fully satisfied (H5 fix).
            return 1.0
        results = self._engine.evaluate(itinerary, list(request.soft_constraints))
        return self._engine.soft_satisfaction_score(results, list(request.soft_constraints))

    def _tool_efficiency(self, stats: tuple) -> float:
        """1 - redundant_call_ratio. Higher is better."""
        if not stats:
            return 1.0
        total = sum(s.call_count for s in stats)
        redundant = sum(s.redundant_call_count for s in stats)
        if total == 0:
            return 1.0
        return max(0.0, 1.0 - redundant / total)

    def _tool_failure_rate(self, stats: tuple) -> float:
        """Fraction of tool calls that failed. Lower is better."""
        if not stats:
            return 0.0
        total = sum(s.call_count for s in stats)
        failures = sum(s.failure_count for s in stats)
        if total == 0:
            return 0.0
        return failures / total

    def _avg_tool_latency_ms(self, stats: tuple) -> float:
        """Mean latency per tool call in milliseconds."""
        if not stats:
            return 0.0
        total_calls = sum(s.call_count for s in stats)
        total_latency = sum(s.total_latency_ms for s in stats)
        if total_calls == 0:
            return 0.0
        return total_latency / total_calls

    def _budget_adherence(
        self, itinerary: Itinerary | None, request: UserRequest
    ) -> float:
        """
        1.0 if itinerary is within budget.
        Fraction of budget used if over budget (clamped to [0, 1]).
        """
        if itinerary is None or request.budget_usd <= 0:
            return 0.0
        if itinerary.total_cost_usd <= request.budget_usd:
            return 1.0
        # Penalise overspend proportionally
        overshoot = itinerary.total_cost_usd - request.budget_usd
        return max(0.0, 1.0 - overshoot / request.budget_usd)

    def _logical_consistency(self, itinerary: Itinerary | None) -> float:
        """
        Date ordering, no duplicate hotel bookings, and temporal feasibility. [0, 1]

        Checks (M3 fix — added temporal feasibility):
        1. Itinerary days are in chronological order.
        2. No hotel_id appears on more than one day.
        3. No two activities on the same day have overlapping time windows
           (requires start_datetime + duration_hours on ActivityBooking).
        4. Flight arrival datetime is on or before the hotel check-in date.
        """
        if itinerary is None or not itinerary.days:
            return 0.0
        issues = 0
        checks = 0

        # Check 1: days are sorted
        dates = [d.date for d in itinerary.days]
        checks += 1
        if dates != sorted(dates):
            issues += 1

        # Check 2: no duplicate hotel bookings
        hotel_ids = [
            d.accommodation.hotel_id
            for d in itinerary.days
            if d.accommodation is not None
        ]
        checks += 1
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
                checks += 1
                sorted_slots = sorted(slots)
                overlap = any(
                    sorted_slots[i][1] > sorted_slots[i + 1][0]
                    for i in range(len(sorted_slots) - 1)
                )
                if overlap:
                    issues += 1

        # Check 4: flight arrival ≤ hotel check-in date on the same day
        for day in itinerary.days:
            if day.accommodation is None:
                continue
            hotel_checkin = getattr(day.accommodation, "check_in", None)
            for seg in day.transport_segments:
                arrival = getattr(seg, "arrival_datetime", None)
                if hotel_checkin and arrival:
                    try:
                        arrival_date = arrival[:10]
                        checks += 1
                        if arrival_date > hotel_checkin:
                            issues += 1
                    except (IndexError, TypeError):
                        pass

        return max(0.0, 1.0 - issues / checks) if checks > 0 else 1.0
