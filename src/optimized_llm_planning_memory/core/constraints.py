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
        """Constraint.value = 'YYYY-MM-DD to YYYY-MM-DD' date range or single ISO date."""
        if not itinerary.days:
            return ConstraintSatisfactionResult(
                constraint_id=constraint.constraint_id,
                satisfied=False,
                score=0.0,
                explanation="Itinerary has no days yet.",
            )
        value = str(constraint.value)
        if " to " in value:
            expected_start, expected_end = [p.strip() for p in value.split(" to ", 1)]
            actual_start = itinerary.days[0].date
            actual_end = itinerary.days[-1].date
            start_ok = actual_start == expected_start
            end_ok = actual_end == expected_end
            satisfied = start_ok and end_ok
            score = (int(start_ok) + int(end_ok)) / 2.0
            explanation = (
                f"Start {actual_start!r}=={expected_start!r}: {start_ok}. "
                f"End {actual_end!r}=={expected_end!r}: {end_ok}."
            )
        else:
            actual_start = itinerary.days[0].date
            satisfied = actual_start == value
            score = 1.0 if satisfied else 0.0
            explanation = f"Start date {actual_start!r} vs required {value!r}."
        return ConstraintSatisfactionResult(
            constraint_id=constraint.constraint_id,
            satisfied=satisfied,
            score=score,
            explanation=explanation,
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
        # Hard constraints must be binary — partial credit only meaningful for soft
        if not satisfied and constraint.constraint_type == ConstraintType.HARD:
            score = 0.0
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
        """Constraint.value = min_stars (numeric) when unit='min_stars', else checks booking exists."""
        all_bookings = [d.accommodation for d in itinerary.days if d.accommodation is not None]
        total_nights = len(itinerary.days)
        if not all_bookings:
            return ConstraintSatisfactionResult(
                constraint_id=constraint.constraint_id,
                satisfied=False,
                score=0.0,
                explanation="No accommodation booked.",
            )
        if constraint.unit == "min_stars":
            min_stars = float(constraint.value)
            # Check ALL bookings — a luxury trip requires every hotel to meet the standard.
            # Use the worst-rated hotel so a single sub-standard property fails the constraint.
            rated_bookings = [b for b in all_bookings if b.star_rating is not None]
            if rated_bookings:
                min_actual = min(b.star_rating for b in rated_bookings)
                satisfied = min_actual >= min_stars
                score = 1.0 if satisfied else max(0.0, min_actual / min_stars)
                explanation = (
                    f"Lowest hotel star rating {min_actual} across {len(rated_bookings)} "
                    f"booking(s) vs required >={min_stars}."
                )
            else:
                satisfied, score = True, 1.0
                explanation = "Hotels booked; star ratings not recorded, assuming satisfied."
        else:
            nights_with_hotel = len(all_bookings)
            satisfied = nights_with_hotel == total_nights
            score = nights_with_hotel / total_nights if total_nights > 0 else 0.0
            explanation = f"{nights_with_hotel}/{total_nights} nights have accommodation."
        return ConstraintSatisfactionResult(
            constraint_id=constraint.constraint_id,
            satisfied=satisfied,
            score=score,
            explanation=explanation,
        )

    def _evaluate_activity(
        self, itinerary: Itinerary, constraint: Constraint
    ) -> ConstraintSatisfactionResult:
        """Constraint.value and unit determine evaluation strategy:
        - unit='min_count': count non-event activities >= value
        - unit='max_price_usd': check that at least one event was booked (price enforced at booking)
        - otherwise: text search on category/name (for preference-style activity constraints)
        """
        all_activities = [a for day in itinerary.days for a in day.activities]
        if constraint.unit == "min_count":
            required_count = int(constraint.value)
            count = sum(1 for a in all_activities if a.category.lower() != "event")
            satisfied = count >= required_count
            score = min(1.0, count / required_count) if required_count > 0 else 1.0
            explanation = f"{count} activities found vs required >={required_count}."
        elif constraint.unit == "max_price_usd":
            # Events are pre-filtered at booking time; presence confirms price was met.
            event_acts = [a for a in all_activities if a.category.lower() == "event"]
            satisfied = len(event_acts) > 0
            score = 1.0 if satisfied else 0.0
            explanation = (
                f"Found {len(event_acts)} event(s) within price limit."
                if satisfied else "No events booked."
            )
        else:
            required = str(constraint.value).lower()
            found = any(
                required in a.category.lower() or required in a.activity_name.lower()
                for a in all_activities
            )
            satisfied, score = found, 1.0 if found else 0.0
            explanation = f"Activity matching '{required}' {'found' if found else 'not found'}."
        return ConstraintSatisfactionResult(
            constraint_id=constraint.constraint_id,
            satisfied=satisfied,
            score=score,
            explanation=explanation,
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
        """Soft preference check against activity categories in the itinerary.
        Dining/food preferences match on category keywords; others use text search.
        """
        _DINING_KEYWORDS = {"restaurant", "dining", "food", "cuisine", "cafe", "food_market", "bistro"}
        all_activities = [a for day in itinerary.days for a in day.activities]
        value = str(constraint.value).lower()
        if any(kw in value for kw in ("cuisine", "dining", "restaurant", "food")):
            matched = any(
                any(kw in a.category.lower() for kw in _DINING_KEYWORDS)
                for a in all_activities
            )
            satisfied, score = matched, 1.0 if matched else 0.0
            explanation = (
                "Dining/food activity found in itinerary."
                if matched else "No dining/food activities found in itinerary."
            )
        else:
            matched = any(
                value in a.category.lower() or value in a.activity_name.lower()
                for a in all_activities
            )
            satisfied, score = matched, 0.7 if matched else 0.3
            explanation = (
                f"Preference '{value}' matched in itinerary."
                if matched else f"Preference '{value}' not matched; partial credit."
            )
        return ConstraintSatisfactionResult(
            constraint_id=constraint.constraint_id,
            satisfied=satisfied,
            score=score,
            explanation=explanation,
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
