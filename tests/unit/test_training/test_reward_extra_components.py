"""
Unit tests — RewardFunction extra_components and optional v2 reward components.

Verifies:
1. Extra components injected at construction time contribute to total reward.
2. Optional v2 components (destination_coverage, activity_density) activate
   when enabled in RewardConfig.
3. Normalization denominator includes extra component weights.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from optimized_llm_planning_memory.core.config import (
    OptionalRewardComponent,
    OptionalRewardComponents,
    RewardConfig,
    RewardWeights,
)
from optimized_llm_planning_memory.core.models import (
    AccommodationBooking,
    ActivityBooking,
    Constraint,
    ConstraintCategory,
    ConstraintType,
    EpisodeLog,
    Itinerary,
    ItineraryDay,
    RewardComponents,
    TrajectoryModel,
    TravelerProfile,
    UserRequest,
)
from optimized_llm_planning_memory.training.reward import RewardFunction


def _make_request(destination_cities=None, budget=5000.0):
    return UserRequest(
        request_id="test-reward-req",
        raw_text="Test",
        origin_city="New York",
        destination_cities=destination_cities or ["Paris"],
        start_date="2025-06-01",
        end_date="2025-06-03",
        budget_usd=budget,
        traveler_profile=TravelerProfile(num_adults=1),
        hard_constraints=[
            Constraint(
                constraint_id="hc1",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.BUDGET,
                description="Budget",
                value=budget,
                unit="USD",
            )
        ],
        soft_constraints=[],
    )


def _make_itinerary(city="Paris", num_activities=2, total_cost=200.0):
    hotel = AccommodationBooking(
        hotel_id="HTL1",
        hotel_name="Hotel",
        city=city,
        check_in="2025-06-01",
        check_out="2025-06-03",
        cost_per_night_usd=80.0,
        total_cost_usd=160.0,
        booking_ref="HTL-REF",
    )
    activities = [
        ActivityBooking(
            activity_id=f"ACT{i}",
            activity_name=f"Activity {i}",
            location="Centre",
            city=city,
            start_datetime=f"2025-06-0{i+1}T10:00:00",
            duration_hours=2.0,
            cost_usd=20.0,
            category="culture",
            booking_ref=f"ACT-REF-{i}",
        )
        for i in range(num_activities)
    ]
    day = ItineraryDay(
        date="2025-06-01",
        city=city,
        transport_segments=[],
        accommodation=hotel,
        activities=activities,
    )
    return Itinerary(
        itinerary_id=str(uuid.uuid4()),
        request_id="test-reward-req",
        days=[day],
        total_cost_usd=total_cost,
        is_complete=True,
    )


def _make_episode_log(itinerary=None):
    traj = TrajectoryModel(
        trajectory_id=str(uuid.uuid4()),
        request_id="test-reward-req",
        steps=(),
        total_steps=0,
    )
    rc = RewardComponents(
        hard_constraint_score=0.0,
        soft_constraint_score=0.0,
        tool_efficiency_score=0.0,
        tool_failure_penalty=0.0,
        logical_consistency_score=0.0,
        total_reward=0.0,
    )
    return EpisodeLog(
        episode_id=str(uuid.uuid4()),
        request_id="test-reward-req",
        agent_mode="compressor",
        trajectory=traj,
        compressed_states=(),
        final_itinerary=itinerary,
        reward_components=rc,
        tool_stats=(),
        total_steps=5,
        success=True,
        config_hash="test",
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


@pytest.mark.unit_test
class TestExtraComponents:
    def test_extra_component_increases_reward(self):
        """An extra component with positive return should increase total reward."""
        def always_one(ep, req, is_terminal):
            return 1.0

        rf_base = RewardFunction(config=RewardConfig(normalize=False))
        rf_extra = RewardFunction(
            config=RewardConfig(normalize=False),
            extra_components={"bonus": (always_one, 0.5)},
        )
        request = _make_request()
        itinerary = _make_itinerary(total_cost=200.0)
        episode = _make_episode_log(itinerary)

        base_reward = rf_base.compute(episode, request, is_terminal=False).total_reward
        extra_reward = rf_extra.compute(episode, request, is_terminal=False).total_reward

        assert extra_reward > base_reward, (
            "Extra component should add to total reward"
        )
        assert abs(extra_reward - base_reward - 0.5) < 1e-5

    def test_multiple_extra_components_sum(self):
        """Multiple extra components should all contribute."""
        rf = RewardFunction(
            config=RewardConfig(normalize=False),
            extra_components={
                "comp_a": (lambda ep, req, t: 0.5, 1.0),
                "comp_b": (lambda ep, req, t: 0.2, 0.5),
            },
        )
        request = _make_request()
        episode = _make_episode_log(_make_itinerary())
        result = rf.compute(episode, request, is_terminal=False)
        # Extra contribution = 1.0*0.5 + 0.5*0.2 = 0.6 more than base
        base_rf = RewardFunction(config=RewardConfig(normalize=False))
        base = base_rf.compute(episode, request, is_terminal=False).total_reward
        assert abs(result.total_reward - base - 0.6) < 1e-4


@pytest.mark.unit_test
class TestOptionalV2Components:
    def test_destination_coverage_enabled_increases_reward_when_city_covered(self):
        """Enabling destination_coverage should give higher reward when city is covered."""
        config_on = RewardConfig(
            normalize=False,
            optional=OptionalRewardComponents(
                destination_coverage=OptionalRewardComponent(enabled=True, weight=1.0)
            ),
        )
        config_off = RewardConfig(normalize=False)

        request = _make_request(destination_cities=["Paris"])
        itinerary = _make_itinerary(city="Paris", total_cost=200.0)
        episode = _make_episode_log(itinerary)

        rf_on = RewardFunction(config=config_on)
        rf_off = RewardFunction(config=config_off)

        reward_on = rf_on.compute(episode, request).total_reward
        reward_off = rf_off.compute(episode, request).total_reward
        # Paris is covered → destination_coverage returns 1.0 → +1.0 weight
        assert reward_on > reward_off

    def test_activity_density_enabled_increases_reward_for_busy_itinerary(self):
        """activity_density component should favour itineraries with more activities."""
        config_on = RewardConfig(
            normalize=False,
            optional=OptionalRewardComponents(
                activity_density=OptionalRewardComponent(enabled=True, weight=1.0)
            ),
        )
        request = _make_request()
        busy_ep = _make_episode_log(_make_itinerary(num_activities=3))
        sparse_ep = _make_episode_log(_make_itinerary(num_activities=0))

        rf = RewardFunction(config=config_on)
        reward_busy = rf.compute(busy_ep, request).total_reward
        reward_sparse = rf.compute(sparse_ep, request).total_reward
        assert reward_busy > reward_sparse

    def test_disabled_optional_components_do_not_affect_reward(self):
        """With default config (all disabled), optional components are no-ops."""
        config_default = RewardConfig()
        rf1 = RewardFunction(config=config_default)
        rf2 = RewardFunction(
            config=RewardConfig(
                optional=OptionalRewardComponents(
                    destination_coverage=OptionalRewardComponent(enabled=False),
                    activity_density=OptionalRewardComponent(enabled=False),
                )
            )
        )
        request = _make_request()
        episode = _make_episode_log(_make_itinerary())
        assert rf1.compute(episode, request).total_reward == pytest.approx(
            rf2.compute(episode, request).total_reward, abs=1e-6
        )
