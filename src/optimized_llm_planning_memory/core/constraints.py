"""
core/constraints.py
===================
ConstraintSatisfactionEngine — the single implementation of constraint scoring.

CRITICAL DESIGN INVARIANT
--------------------------
This module is imported by BOTH ``training/reward.py`` AND
``evaluation/deterministic.py``. This guarantees that the RL training signal
and the evaluation metric use identical logic. If they diverged, the compressor
would optimise a proxy metric and fail at evaluation — a common research pitfall.

Design pattern: Strategy (evaluation strategies per ConstraintCategory)
------------------------------------------------------------------------
Each ConstraintCategory gets its own private ``_evaluate_*`` method. The
``evaluate()`` dispatcher selects the right strategy based on category,
so new constraint types can be added without modifying existing logic.

Usage
-----
    engine = ConstraintSatisfactionEngine()
    results = engine.evaluate(itinerary, constraints)
    hard_ratio = engine.hard_satisfaction_ratio(results, constraints)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optimized_llm_planning_memory.core.models import (
    Constraint,
    ConstraintCategory,
    ConstraintSatisfactionResult,
    ConstraintType,
    Itinerary,
)

if TYPE_CHECKING:
    pass


class ConstraintSatisfactionEngine:
    """
    Evaluates a list of constraints against a completed or partial itinerary.

    This class is stateless between calls; instantiate once and reuse across
    episodes and workers.

    Methods
    -------
    evaluate(itinerary, constraints)
        Core evaluation; returns one ConstraintSatisfactionResult per constraint.

    hard_satisfaction_ratio(results, constraints)
        Fraction of HARD constraints that are satisfied. [0.0, 1.0]

    soft_satisfaction_score(results, constraints)
        Weighted average satisfaction score for SOFT constraints. [0.0, 1.0]
    """

    def evaluate(
        self,
        itinerary: Itinerary,
        constraints: list[Constraint],
    ) -> list[ConstraintSatisfactionResult]:
        """
        Evaluate every constraint in ``constraints`` against ``itinerary``.

        Parameters
        ----------
        itinerary   : The itinerary (complete or partial) to evaluate.
        constraints : List of Constraint objects to check.

        Returns
        -------
        list[ConstraintSatisfactionResult]
            One result per input constraint, in the same order.
        """
        return [self._evaluate_single(itinerary, c) for c in constraints]

    def hard_satisfaction_ratio(
        self,
        results: list[ConstraintSatisfactionResult],
        constraints: list[Constraint],
    ) -> float:
        """
        Fraction of HARD constraints satisfied. Returns 1.0 if there are none.
        """
        hard_ids = {c.constraint_id for c in constraints if c.constraint_type == ConstraintType.HARD}
        if not hard_ids:
            return 1.0
        hard_results = [r for r in results if r.constraint_id in hard_ids]
        satisfied = sum(1 for r in hard_results if r.satisfied)
        return satisfied / len(hard_results)

    def soft_satisfaction_score(
        self,
        results: list[ConstraintSatisfactionResult],
        constraints: list[Constraint],
    ) -> float:
        """
        Weighted average score for SOFT constraints. Returns 1.0 if there are none.
        """
        soft_ids = {c.constraint_id for c in constraints if c.constraint_type == ConstraintType.SOFT}
        if not soft_ids:
            return 1.0
        soft_results = [r for r in results if r.constraint_id in soft_ids]
        return sum(r.score for r in soft_results) / len(soft_results)

    # ── Private dispatch ──────────────────────────────────────────────────────

    def _evaluate_single(
        self, itinerary: Itinerary, constraint: Constraint
    ) -> ConstraintSatisfactionResult:
        """Dispatch to the appropriate evaluation strategy based on category."""
        dispatch = {
            ConstraintCategory.BUDGET: self._evaluate_budget,
            ConstraintCategory.DATE: self._evaluate_date,
            ConstraintCategory.DURATION: self._evaluate_duration,
            ConstraintCategory.CITY: self._evaluate_city,
            ConstraintCategory.ACCOMMODATION: self._evaluate_accommodation,
            ConstraintCategory.ACTIVITY: self._evaluate_activity,
            ConstraintCategory.TRANSPORT: self._evaluate_transport,
            ConstraintCategory.GROUP: self._evaluate_group,
            ConstraintCategory.ACCESSIBILITY: self._evaluate_accessibility,
            ConstraintCategory.PREFERENCE: self._evaluate_preference,
        }
        handler = dispatch.get(constraint.category, self._evaluate_unknown)
        return handler(itinerary, constraint)

    # ── Evaluation strategies (one per ConstraintCategory) ───────────────────

    def _evaluate_budget(
        self, itinerary: Itinerary, constraint: Constraint
    ) -> ConstraintSatisfactionResult:
        """Constraint.value = max budget in USD (float)."""
        budget_limit = float(constraint.value)
        actual = itinerary.total_cost_usd
        satisfied = actual <= budget_limit
        # Soft score: 1.0 if under budget, decays linearly to 0.0 at 2× budget.
        score = max(0.0, min(1.0, (2 * budget_limit - actual) / budget_limit)) if budget_limit > 0 else 0.0
        return ConstraintSatisfactionResult(
            constraint_id=constraint.constraint_id,
            satisfied=satisfied,
            score=score if not satisfied and constraint.constraint_type == ConstraintType.SOFT
            else (1.0 if satisfied else 0.0),
            explanation=f"Total cost ${actual:.2f} vs budget ${budget_limit:.2f}.",
        )

    def _evaluate_date(
        self, itinerary: Itinerary, constraint: Constraint
    ) -> ConstraintSatisfactionResult:
        """Constraint.value = expected start date string (ISO 8601)."""
        if not itinerary.days:
            return ConstraintSatisfactionResult(
                constraint_id=constraint.constraint_id,
                satisfied=False,
                score=0.0,
                explanation="Itinerary has no days yet.",
            )
        actual_start = itinerary.days[0].date
        expected = str(constraint.value)
        satisfied = actual_start == expected
        return ConstraintSatisfactionResult(
            constraint_id=constraint.constraint_id,
            satisfied=satisfied,
            score=1.0 if satisfied else 0.0,
            explanation=f"Start date {actual_start!r} vs required {expected!r}.",
        )

    def _evaluate_duration(
        self, itinerary: Itinerary, constraint: Constraint
    ) -> ConstraintSatisfactionResult:
        """Constraint.value = required number of days (int)."""
        required_days = int(constraint.value)
        actual_days = len(itinerary.days)
        satisfied = actual_days == required_days
        score = 1.0 - abs(actual_days - required_days) / max(required_days, 1)
        score = max(0.0, score)
        return ConstraintSatisfactionResult(
            constraint_id=constraint.constraint_id,
            satisfied=satisfied,
            score=score,
            explanation=f"Duration {actual_days} days vs required {required_days} days.",
        )

    def _evaluate_city(
        self, itinerary: Itinerary, constraint: Constraint
    ) -> ConstraintSatisfactionResult:
        """Constraint.value = city name or list of city names that must appear."""
        required: list[str]
        if isinstance(constraint.value, list):
            required = [str(c).lower() for c in constraint.value]
        else:
            required = [str(constraint.value).lower()]
        visited = [d.city.lower() for d in itinerary.days]
        missing = [c for c in required if c not in visited]
        satisfied = len(missing) == 0
        score = (len(required) - len(missing)) / len(required) if required else 1.0
        return ConstraintSatisfactionResult(
            constraint_id=constraint.constraint_id,
            satisfied=satisfied,
            score=score,
            explanation=(
                f"Required cities: {required}. Missing: {missing}."
                if missing else f"All required cities visited: {required}."
            ),
        )

    def _evaluate_accommodation(
        self, itinerary: Itinerary, constraint: Constraint
    ) -> ConstraintSatisfactionResult:
        """Constraint.value = required accommodation type or property (e.g., 'hotel', 'hostel')."""
        # Check that every night has some accommodation booked.
        nights_with_hotel = sum(1 for d in itinerary.days if d.accommodation is not None)
        total_nights = len(itinerary.days)
        satisfied = total_nights > 0 and nights_with_hotel == total_nights
        score = nights_with_hotel / total_nights if total_nights > 0 else 0.0
        return ConstraintSatisfactionResult(
            constraint_id=constraint.constraint_id,
            satisfied=satisfied,
            score=score,
            explanation=f"{nights_with_hotel}/{total_nights} nights have accommodation.",
        )

    def _evaluate_activity(
        self, itinerary: Itinerary, constraint: Constraint
    ) -> ConstraintSatisfactionResult:
        """Constraint.value = required activity category or name."""
        required = str(constraint.value).lower()
        all_activities = [
            a for day in itinerary.days for a in day.activities
        ]
        found = any(
            required in a.category.lower() or required in a.activity_name.lower()
            for a in all_activities
        )
        return ConstraintSatisfactionResult(
            constraint_id=constraint.constraint_id,
            satisfied=found,
            score=1.0 if found else 0.0,
            explanation=f"Activity matching '{required}' {'found' if found else 'not found'}.",
        )

    def _evaluate_transport(
        self, itinerary: Itinerary, constraint: Constraint
    ) -> ConstraintSatisfactionResult:
        """Constraint.value = required transport mode (e.g., 'flight')."""
        required_mode = str(constraint.value).lower()
        all_segments = [s for day in itinerary.days for s in day.transport_segments]
        found = any(s.mode.lower() == required_mode for s in all_segments)
        return ConstraintSatisfactionResult(
            constraint_id=constraint.constraint_id,
            satisfied=found,
            score=1.0 if found else 0.0,
            explanation=f"Transport mode '{required_mode}' {'found' if found else 'not found'}.",
        )

    def _evaluate_group(
        self, itinerary: Itinerary, constraint: Constraint
    ) -> ConstraintSatisfactionResult:
        """Constraint.value = required group size (int). Always satisfied for now."""
        # TODO: verify hotel/activity capacity against group size
        return ConstraintSatisfactionResult(
            constraint_id=constraint.constraint_id,
            satisfied=True,
            score=1.0,
            explanation="Group size check not yet implemented; defaulting to satisfied.",
        )

    def _evaluate_accessibility(
        self, itinerary: Itinerary, constraint: Constraint
    ) -> ConstraintSatisfactionResult:
        """Constraint.value = accessibility requirement string."""
        # TODO: require simulator to tag venues with accessibility attributes
        return ConstraintSatisfactionResult(
            constraint_id=constraint.constraint_id,
            satisfied=True,
            score=0.5,
            explanation="Accessibility check not yet implemented; score 0.5.",
        )

    def _evaluate_preference(
        self, itinerary: Itinerary, constraint: Constraint
    ) -> ConstraintSatisfactionResult:
        """
        Soft preference check. Only used for SOFT constraints.
        Always returns 0.5 (unknown) unless overridden by a more specific evaluator.
        """
        return ConstraintSatisfactionResult(
            constraint_id=constraint.constraint_id,
            satisfied=False,
            score=0.5,
            explanation="Preference evaluation requires LLM judge; score 0.5 by default.",
        )

    def _evaluate_unknown(
        self, itinerary: Itinerary, constraint: Constraint
    ) -> ConstraintSatisfactionResult:
        return ConstraintSatisfactionResult(
            constraint_id=constraint.constraint_id,
            satisfied=False,
            score=0.0,
            explanation=f"Unknown constraint category '{constraint.category}'; score 0.0.",
        )
