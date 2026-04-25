"""
tests/test_core/test_constraint_engine_unit.py
===============================================
Unit tests for ConstraintSatisfactionEngine._evaluate_* methods.

These are the core scoring functions shared by training reward and
deterministic evaluation.  Edge cases matter because an off-by-one or
division-by-zero here flows into both training signal and published results.
"""

from __future__ import annotations

import uuid

import pytest

from optimized_llm_planning_memory.core.constraints import ConstraintSatisfactionEngine
from optimized_llm_planning_memory.core.models import (
    AccommodationBooking,
    ActivityBooking,
    Constraint,
    ConstraintCategory,
    ConstraintType,
    Itinerary,
    ItineraryDay,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_itinerary(
    total_cost: float = 500.0,
    days: list[ItineraryDay] | None = None,
    request_id: str = "req-001",
) -> Itinerary:
    if days is not None:
        actual_days = days
    else:
        # Add an activity so Itinerary.total_cost_usd (auto-computed from day sums) equals total_cost.
        actual_days = [_make_day("2025-06-01", activities=[
            ActivityBooking(
                activity_id=str(uuid.uuid4()),
                activity_name="Test Activity",
                location="Paris",
                city="Paris",
                start_datetime="2025-06-01T10:00:00",
                duration_hours=2.0,
                cost_usd=total_cost,
                category="sightseeing",
                booking_ref="SIM-ACT-TEST",
            )
        ])]
    return Itinerary(
        itinerary_id=str(uuid.uuid4()),
        request_id=request_id,
        days=actual_days,
    )


def _make_day(date: str, activities: list | None = None) -> ItineraryDay:
    return ItineraryDay(
        date=date,
        city="Paris",
        activities=activities or [],
    )


def _make_constraint(
    category: ConstraintCategory,
    value,
    constraint_type: ConstraintType = ConstraintType.HARD,
) -> Constraint:
    return Constraint(
        constraint_id=str(uuid.uuid4()),
        constraint_type=constraint_type,
        category=category,
        description=f"Test {category.value}",
        value=value,
    )


engine = ConstraintSatisfactionEngine()


# ── Budget ────────────────────────────────────────────────────────────────────

class TestBudgetConstraint:
    def test_within_budget_satisfies(self):
        itinerary = _make_itinerary(total_cost=400.0)
        c = _make_constraint(ConstraintCategory.BUDGET, 500.0)
        result = engine._evaluate_budget(itinerary, c)
        assert result.satisfied is True
        assert result.score == 1.0

    def test_exactly_at_limit_satisfies(self):
        itinerary = _make_itinerary(total_cost=500.0)
        c = _make_constraint(ConstraintCategory.BUDGET, 500.0)
        result = engine._evaluate_budget(itinerary, c)
        assert result.satisfied is True

    def test_over_budget_not_satisfied(self):
        itinerary = _make_itinerary(total_cost=600.0)
        c = _make_constraint(ConstraintCategory.BUDGET, 500.0)
        result = engine._evaluate_budget(itinerary, c)
        assert result.satisfied is False

    def test_double_budget_score_is_zero(self):
        """At 2× the budget limit the score should be 0."""
        itinerary = _make_itinerary(total_cost=1000.0)
        c = _make_constraint(ConstraintCategory.BUDGET, 500.0)
        result = engine._evaluate_budget(itinerary, c)
        assert result.score == 0.0

    def test_zero_budget_does_not_divide_by_zero(self):
        itinerary = _make_itinerary(total_cost=100.0)
        c = _make_constraint(ConstraintCategory.BUDGET, 0.0)
        result = engine._evaluate_budget(itinerary, c)
        assert result.score >= 0.0  # should not raise


# ── Date ──────────────────────────────────────────────────────────────────────

class TestDateConstraint:
    def test_correct_start_date_satisfies(self):
        itinerary = _make_itinerary(days=[_make_day("2025-06-01")])
        c = _make_constraint(ConstraintCategory.DATE, "2025-06-01")
        result = engine._evaluate_date(itinerary, c)
        assert result.satisfied is True
        assert result.score == 1.0

    def test_wrong_start_date_fails(self):
        itinerary = _make_itinerary(days=[_make_day("2025-06-02")])
        c = _make_constraint(ConstraintCategory.DATE, "2025-06-01")
        result = engine._evaluate_date(itinerary, c)
        assert result.satisfied is False
        assert result.score == 0.0

    def test_no_days_itinerary_fails(self):
        itinerary = _make_itinerary(days=[], total_cost=0.0)
        c = _make_constraint(ConstraintCategory.DATE, "2025-06-01")
        result = engine._evaluate_date(itinerary, c)
        assert result.satisfied is False


# ── Duration ──────────────────────────────────────────────────────────────────

class TestDurationConstraint:
    def test_exact_duration_satisfies(self):
        days = [_make_day(f"2025-06-0{i+1}") for i in range(3)]
        itinerary = _make_itinerary(days=days)
        c = _make_constraint(ConstraintCategory.DURATION, 3)
        result = engine._evaluate_duration(itinerary, c)
        assert result.satisfied is True
        assert result.score == 1.0

    def test_short_duration_partial_score(self):
        """1 day instead of 3 → score should be < 1 but ≥ 0."""
        itinerary = _make_itinerary(days=[_make_day("2025-06-01")])
        c = _make_constraint(ConstraintCategory.DURATION, 3)
        result = engine._evaluate_duration(itinerary, c)
        assert result.satisfied is False
        assert 0.0 <= result.score < 1.0

    def test_one_day_required_zero_days_score_is_zero(self):
        itinerary = _make_itinerary(days=[], total_cost=0.0)
        c = _make_constraint(ConstraintCategory.DURATION, 1)
        result = engine._evaluate_duration(itinerary, c)
        assert result.satisfied is False
        assert result.score == 0.0


# ── Hard satisfaction ratio ───────────────────────────────────────────────────

class TestHardSatisfactionRatio:
    def test_all_hard_satisfied_gives_one(self):
        itinerary = _make_itinerary(total_cost=400.0)
        constraints = [
            _make_constraint(ConstraintCategory.BUDGET, 500.0),
        ]
        ratio = engine.hard_satisfaction_ratio(
            engine.evaluate(itinerary, constraints), constraints
        )
        assert ratio == 1.0

    def test_no_hard_constraints_gives_one(self):
        itinerary = _make_itinerary()
        ratio = engine.hard_satisfaction_ratio([], [])
        assert ratio == 1.0

    def test_half_satisfied_gives_half(self):
        days = [_make_day("2025-06-01")]
        itinerary = _make_itinerary(days=days, total_cost=400.0)
        constraints = [
            _make_constraint(ConstraintCategory.BUDGET, 500.0),     # satisfied
            _make_constraint(ConstraintCategory.DURATION, 5),       # not satisfied (1 day)
        ]
        results = engine.evaluate(itinerary, constraints)
        ratio = engine.hard_satisfaction_ratio(results, constraints)
        assert abs(ratio - 0.5) < 1e-9


# ── Soft satisfaction score ───────────────────────────────────────────────────

class TestSoftSatisfactionScore:
    def test_no_soft_constraints_returns_one(self):
        score = engine.soft_satisfaction_score([], [])
        assert score == 1.0

    def test_all_soft_satisfied(self):
        itinerary = _make_itinerary(total_cost=400.0)
        constraints = [
            _make_constraint(ConstraintCategory.BUDGET, 500.0, ConstraintType.SOFT),
        ]
        results = engine.evaluate(itinerary, constraints)
        score = engine.soft_satisfaction_score(results, constraints)
        assert 0.0 <= score <= 1.0
