"""
tests/conftest.py
==================
Shared pytest fixtures for the test suite.

Fixtures
--------
sample_user_request   — A valid UserRequest for use across all test modules.
sample_itinerary      — A minimal valid Itinerary (2 days, 1 hotel, 1 activity).
sample_episode_log    — A minimal EpisodeLog wrapping the above.
mock_simulator        — A MagicMock satisfying SimulatorProtocol.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from optimized_llm_planning_memory.core.models import (
    AccommodationBooking,
    ActivityBooking,
    Constraint,
    ConstraintType,
    ConstraintCategory,
    EpisodeLog,
    HardConstraintLedger,
    Itinerary,
    ItineraryDay,
    RewardComponents,
    ToolCallStats,
    TrajectoryModel,
    TravelerProfile,
    UserRequest,
)


@pytest.fixture
def sample_user_request() -> UserRequest:
    return UserRequest(
        request_id="test-req-001",
        raw_text="Plan a 2-day trip from New York to Paris for 1 adult with a $2000 budget.",
        origin_city="New York",
        destination_cities=["Paris"],
        start_date="2025-06-01",
        end_date="2025-06-02",
        budget_usd=2000.0,
        traveler_profile=TravelerProfile(num_adults=1),
        hard_constraints=[
            Constraint(
                constraint_id="hc_budget",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.BUDGET,
                description="Total cost <= $2000",
                value=2000.0,
                unit="USD",
            )
        ],
        soft_constraints=[
            Constraint(
                constraint_id="sc_hotel",
                constraint_type=ConstraintType.SOFT,
                category=ConstraintCategory.ACCOMMODATION,
                description="Prefer boutique hotels",
                value="boutique",
            )
        ],
        preferences=["art museums"],
    )


@pytest.fixture
def sample_itinerary() -> Itinerary:
    hotel = AccommodationBooking(
        hotel_id="HTL001",
        hotel_name="Test Hotel",
        city="Paris",
        check_in="2025-06-01",
        check_out="2025-06-02",
        cost_per_night_usd=150.0,
        total_cost_usd=150.0,
        booking_ref="REF001",
    )
    activity = ActivityBooking(
        activity_id="ACT001",
        activity_name="Louvre Museum",
        location="Louvre, Paris",
        city="Paris",
        start_datetime="2025-06-01T10:00:00",
        duration_hours=3.0,
        cost_usd=20.0,
        category="culture",
        booking_ref="REF002",
    )
    day = ItineraryDay(
        date="2025-06-01",
        city="Paris",
        transport_segments=[],
        accommodation=hotel,
        activities=[activity],
    )
    return Itinerary(
        itinerary_id=str(uuid.uuid4()),
        request_id="test-req-001",
        days=[day],
        total_cost_usd=170.0,
        is_complete=True,
    )


@pytest.fixture
def sample_episode_log(sample_itinerary: Itinerary) -> EpisodeLog:
    traj = TrajectoryModel(
        trajectory_id=str(uuid.uuid4()),
        request_id="test-req-001",
        steps=(),
        total_steps=0,
    )
    reward = RewardComponents(
        hard_constraint_score=1.0,
        soft_constraint_score=0.8,
        tool_efficiency_score=0.9,
        tool_failure_penalty=0.0,
        logical_consistency_score=1.0,
        terminal_itinerary_score=1.0,
        total_reward=0.85,
    )
    return EpisodeLog(
        episode_id=str(uuid.uuid4()),
        request_id="test-req-001",
        agent_mode="compressor",
        trajectory=traj,
        compressed_states=[],
        final_itinerary=sample_itinerary,
        reward_components=reward,
        tool_stats=(),
        total_steps=5,
        success=True,
        config_hash="test",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def mock_simulator() -> MagicMock:
    """A MagicMock that satisfies SimulatorProtocol."""
    sim = MagicMock()
    sim.search_flights.return_value = [
        {"flight_id": "FL001", "airline": "TestAir", "price_per_person": 300.0, "stops": 0}
    ]
    sim.search_hotels.return_value = [
        {"hotel_id": "HTL001", "name": "Test Hotel", "price_per_night": 150.0, "stars": 3}
    ]
    sim.search_activities.return_value = [
        {"activity_id": "ACT001", "name": "City Tour", "cost": 20.0, "category": "tour"}
    ]
    sim.book_flight.return_value = {"booking_ref": "FL-REF-001", "status": "confirmed"}
    sim.book_hotel.return_value = {"booking_ref": "HTL-REF-001", "status": "confirmed"}
    sim.book_activity.return_value = {"booking_ref": "ACT-REF-001", "status": "confirmed"}
    sim.get_city_info.return_value = {"city": "Paris", "country": "France", "timezone": "Europe/Paris"}
    sim.get_world_seed.return_value = 42
    return sim
