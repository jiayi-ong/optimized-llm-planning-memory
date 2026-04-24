"""Module tests for core — ConstraintSatisfactionEngine + Itinerary integration."""

from __future__ import annotations

import pytest

from optimized_llm_planning_memory.core.constraints import ConstraintSatisfactionEngine
from optimized_llm_planning_memory.core.models import (
    Constraint,
    ConstraintCategory,
    ConstraintType,
    ItineraryDay,
    AccommodationBooking,
    ActivityBooking,
)


def _budget_constraint(limit: float) -> Constraint:
    return Constraint(
        constraint_id="hc_budget",
        constraint_type=ConstraintType.HARD,
        category=ConstraintCategory.BUDGET,
        description=f"Total cost <= ${limit}",
        value=limit,
        unit="USD",
    )


def _city_constraint(city: str) -> Constraint:
    return Constraint(
        constraint_id=f"hc_city_{city}",
        constraint_type=ConstraintType.HARD,
        category=ConstraintCategory.CITY,
        description=f"Must visit {city}",
        value=city,
    )


def _make_day(city: str, hotel_cost: float = 100.0, activity_cost: float = 20.0) -> ItineraryDay:
    hotel = AccommodationBooking(
        hotel_id="H1",
        hotel_name="Hotel",
        city=city,
        check_in="2025-06-01",
        check_out="2025-06-02",
        cost_per_night_usd=hotel_cost,
        total_cost_usd=hotel_cost,
    )
    activity = ActivityBooking(
        activity_id="A1",
        activity_name="Tour",
        location=city,
        city=city,
        start_datetime="2025-06-01T10:00:00",
        duration_hours=2.0,
        cost_usd=activity_cost,
        category="tour",
    )
    return ItineraryDay(date="2025-06-01", city=city, accommodation=hotel, activities=[activity])


@pytest.mark.module_test
class TestConstraintSatisfactionEngineIntegration:
    def setup_method(self):
        self.engine = ConstraintSatisfactionEngine()

    def test_budget_constraint_satisfied(self, sample_itinerary, sample_user_request):
        results = self.engine.evaluate(
            itinerary=sample_itinerary,
            constraints=sample_user_request.hard_constraints,
        )
        assert len(results) > 0
        budget_result = next((r for r in results if "budget" in r.constraint_id.lower()), None)
        assert budget_result is not None
        assert budget_result.satisfied is True

    def test_city_constraint_satisfied_when_city_visited(self, sample_itinerary, sample_user_request):
        constraints = [_city_constraint("Paris")]
        results = self.engine.evaluate(
            itinerary=sample_itinerary,
            constraints=constraints,
        )
        city_result = results[0]
        assert city_result.satisfied is True

    def test_city_constraint_violated_when_city_not_visited(self, sample_itinerary):
        constraints = [_city_constraint("Tokyo")]
        results = self.engine.evaluate(
            itinerary=sample_itinerary,
            constraints=constraints,
        )
        city_result = results[0]
        assert city_result.satisfied is False

    def test_no_constraints_returns_empty(self, sample_itinerary):
        results = self.engine.evaluate(itinerary=sample_itinerary, constraints=[])
        assert results == []

    def test_satisfaction_ratio_all_satisfied(self, sample_itinerary, sample_user_request):
        results = self.engine.evaluate(
            itinerary=sample_itinerary,
            constraints=sample_user_request.hard_constraints,
        )
        satisfied = sum(1 for r in results if r.satisfied)
        ratio = satisfied / len(results) if results else 1.0
        assert ratio == 1.0

    def test_mixed_hard_and_soft_constraints(self, sample_itinerary):
        constraints = [
            _budget_constraint(10000.0),  # satisfied — very high limit
            Constraint(
                constraint_id="sc_pref",
                constraint_type=ConstraintType.SOFT,
                category=ConstraintCategory.ACCOMMODATION,
                description="Prefer boutique hotels",
                value="boutique",
            ),
        ]
        results = self.engine.evaluate(itinerary=sample_itinerary, constraints=constraints)
        assert len(results) == 2
