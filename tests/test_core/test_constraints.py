"""
tests/test_core/test_constraints.py
=====================================
Unit tests for ConstraintSatisfactionEngine.

Tests cover: budget, date, city, logical consistency on a known itinerary.
"""

from __future__ import annotations

import pytest

from optimized_llm_planning_memory.core.constraints import ConstraintSatisfactionEngine
from optimized_llm_planning_memory.core.models import Constraint, ConstraintType, ConstraintCategory


@pytest.fixture
def engine() -> ConstraintSatisfactionEngine:
    return ConstraintSatisfactionEngine()


def test_budget_constraint_satisfied(engine, sample_itinerary, sample_user_request):
    """Itinerary cost $170 is within budget $2000 → satisfied."""
    results = engine.evaluate(sample_itinerary, list(sample_user_request.hard_constraints))
    assert len(results) == 1
    assert results[0].satisfied is True
    assert results[0].score == pytest.approx(1.0)


def test_hard_satisfaction_ratio_perfect(engine, sample_itinerary, sample_user_request):
    """All hard constraints satisfied → ratio = 1.0."""
    results = engine.evaluate(sample_itinerary, list(sample_user_request.hard_constraints))
    ratio = engine.hard_satisfaction_ratio(results, list(sample_user_request.hard_constraints))
    assert ratio == pytest.approx(1.0)


def test_hard_satisfaction_ratio_zero(engine, sample_itinerary, sample_user_request):
    """Itinerary cost exceeds budget → ratio = 0.0."""
    # Artificially inflate cost
    expensive_itinerary = sample_itinerary.model_copy(update={"total_cost_usd": 9999.0})
    results = engine.evaluate(expensive_itinerary, list(sample_user_request.hard_constraints))
    ratio = engine.hard_satisfaction_ratio(results, list(sample_user_request.hard_constraints))
    assert ratio == pytest.approx(0.0)


def test_soft_satisfaction_score_returns_float(engine, sample_itinerary, sample_user_request):
    results = engine.evaluate(sample_itinerary, list(sample_user_request.soft_constraints))
    score = engine.soft_satisfaction_score(results, list(sample_user_request.soft_constraints))
    assert 0.0 <= score <= 1.0


def test_empty_constraints(engine, sample_itinerary):
    results = engine.evaluate(sample_itinerary, [])
    assert results == []
