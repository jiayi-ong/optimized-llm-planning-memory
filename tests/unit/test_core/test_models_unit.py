"""Unit tests for core/models.py — Pydantic model structure and validation."""

from __future__ import annotations

import uuid
import pytest
from pydantic import ValidationError

from optimized_llm_planning_memory.core.models import (
    AccommodationBooking,
    ActivityBooking,
    Constraint,
    ConstraintCategory,
    ConstraintType,
    HardConstraintLedger,
    Itinerary,
    ItineraryDay,
    ReActStep,
    RewardComponents,
    ToolCall,
    ToolResult,
    TrajectoryModel,
    TravelerProfile,
    UserRequest,
)


@pytest.mark.unit
class TestConstraint:
    def test_constraint_frozen(self):
        c = Constraint(
            constraint_id="c1",
            constraint_type=ConstraintType.HARD,
            category=ConstraintCategory.BUDGET,
            description="Budget",
            value=1000.0,
        )
        with pytest.raises(Exception):
            c.constraint_id = "new"  # type: ignore[misc]

    def test_constraint_satisfied_defaults_none(self):
        c = Constraint(
            constraint_id="c1",
            constraint_type=ConstraintType.SOFT,
            category=ConstraintCategory.PREFERENCE,
            description="Pref",
            value="vegetarian",
        )
        assert c.satisfied is None
        assert c.score is None

    def test_constraint_type_hard_soft(self):
        assert ConstraintType.HARD.value == "hard"
        assert ConstraintType.SOFT.value == "soft"

    def test_all_constraint_categories_defined(self):
        expected = {
            "budget", "date", "duration", "city", "accommodation",
            "activity", "transport", "group", "accessibility", "preference",
        }
        actual = {c.value for c in ConstraintCategory}
        assert expected == actual


@pytest.mark.unit
class TestTravelerProfile:
    def test_total_travelers_property(self):
        p = TravelerProfile(num_adults=2, num_children=1)
        assert p.total_travelers == 3

    def test_defaults_one_adult(self):
        p = TravelerProfile()
        assert p.num_adults == 1
        assert p.num_children == 0
        assert p.total_travelers == 1


@pytest.mark.unit
class TestUserRequest:
    def test_frozen(self, sample_user_request):
        with pytest.raises(Exception):
            sample_user_request.budget_usd = 99.0  # type: ignore[misc]

    def test_destination_cities_min_length(self):
        with pytest.raises(ValidationError):
            UserRequest(
                request_id="r1",
                raw_text="Test",
                origin_city="NYC",
                destination_cities=[],  # must have at least 1
                start_date="2025-06-01",
                end_date="2025-06-03",
                budget_usd=1000.0,
            )

    def test_budget_gt_zero(self):
        with pytest.raises(ValidationError):
            UserRequest(
                request_id="r1",
                raw_text="Test",
                origin_city="NYC",
                destination_cities=["Paris"],
                start_date="2025-06-01",
                end_date="2025-06-03",
                budget_usd=0.0,  # must be > 0
            )


@pytest.mark.unit
class TestItineraryDay:
    def test_total_cost_sums_transport_hotel_activities(self):
        from optimized_llm_planning_memory.core.models import TransportSegment
        transport = TransportSegment(
            mode="flight", from_location="NYC", to_location="Paris",
            departure_datetime="2025-06-01T08:00:00",
            arrival_datetime="2025-06-01T20:00:00",
            cost_usd=300.0,
        )
        hotel = AccommodationBooking(
            hotel_id="H1", hotel_name="Hotel A", city="Paris",
            check_in="2025-06-01", check_out="2025-06-02",
            cost_per_night_usd=150.0, total_cost_usd=150.0,
        )
        activity = ActivityBooking(
            activity_id="A1", activity_name="Museum", location="Paris",
            city="Paris", start_datetime="2025-06-01T10:00:00",
            duration_hours=2.0, cost_usd=20.0, category="culture",
        )
        day = ItineraryDay(
            date="2025-06-01", city="Paris",
            transport_segments=[transport],
            accommodation=hotel,
            activities=[activity],
        )
        assert day.total_cost_usd == pytest.approx(470.0)

    def test_total_cost_no_accommodation(self):
        day = ItineraryDay(date="2025-06-01", city="Paris")
        assert day.total_cost_usd == 0.0


@pytest.mark.unit
class TestItinerary:
    def test_recompute_total_cost(self, sample_itinerary):
        result = sample_itinerary.recompute_total_cost()
        assert result >= 0.0
        assert result == sample_itinerary.total_cost_usd

    def test_cities_visited_unique_ordered(self):
        day1 = ItineraryDay(date="2025-06-01", city="Paris")
        day2 = ItineraryDay(date="2025-06-02", city="Paris")
        day3 = ItineraryDay(date="2025-06-03", city="Rome")
        itin = Itinerary(
            itinerary_id=str(uuid.uuid4()),
            request_id="r1",
            days=[day1, day2, day3],
        )
        assert itin.cities_visited() == ["Paris", "Rome"]


@pytest.mark.unit
class TestTrajectoryModel:
    def test_total_steps_validator_mismatch_raises(self):
        with pytest.raises(ValidationError):
            TrajectoryModel(
                trajectory_id=str(uuid.uuid4()),
                request_id="r1",
                steps=(),
                total_steps=3,  # mismatches len(steps)=0
            )

    def test_to_text_empty_steps(self):
        traj = TrajectoryModel(
            trajectory_id=str(uuid.uuid4()),
            request_id="r1",
            steps=(),
            total_steps=0,
        )
        assert traj.to_text() == ""

    def test_slice_since_filters_by_step_index(self):
        from datetime import datetime, timezone

        def _make_step(idx):
            return ReActStep(
                step_index=idx,
                thought=f"Thought {idx}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        steps = tuple(_make_step(i) for i in range(5))
        traj = TrajectoryModel(
            trajectory_id=str(uuid.uuid4()),
            request_id="r1",
            steps=steps,
            total_steps=5,
        )
        sliced = traj.slice_since(3)
        assert sliced.total_steps == 2
        assert all(s.step_index >= 3 for s in sliced.steps)


@pytest.mark.unit
class TestHardConstraintLedger:
    def test_satisfaction_ratio_empty(self):
        ledger = HardConstraintLedger()
        assert ledger.satisfaction_ratio == 0.0

    def test_satisfaction_ratio_partial(self):
        c1 = Constraint(
            constraint_id="c1", constraint_type=ConstraintType.HARD,
            category=ConstraintCategory.BUDGET, description="B", value=1000,
        )
        c2 = Constraint(
            constraint_id="c2", constraint_type=ConstraintType.HARD,
            category=ConstraintCategory.CITY, description="C", value="Paris",
        )
        ledger = HardConstraintLedger(
            constraints=(c1, c2),
            satisfied_ids=("c1",),
            violated_ids=("c2",),
        )
        assert ledger.satisfaction_ratio == pytest.approx(0.5)


@pytest.mark.unit
class TestRewardComponents:
    def test_tool_failure_penalty_must_be_le_zero(self):
        with pytest.raises(ValidationError):
            RewardComponents(
                hard_constraint_score=1.0,
                soft_constraint_score=1.0,
                tool_efficiency_score=1.0,
                tool_failure_penalty=0.1,  # must be <= 0
                logical_consistency_score=1.0,
                total_reward=1.0,
            )

    def test_hard_constraint_score_must_be_in_range(self):
        with pytest.raises(ValidationError):
            RewardComponents(
                hard_constraint_score=1.5,  # must be <= 1.0
                soft_constraint_score=1.0,
                tool_efficiency_score=1.0,
                tool_failure_penalty=0.0,
                logical_consistency_score=1.0,
                total_reward=1.0,
            )
