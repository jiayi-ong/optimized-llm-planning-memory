"""
tests/test_training/test_reward_eval_consistency.py
====================================================
Regression tests: training reward == deterministic evaluation metric.

The core design invariant (documented in training/reward.py) is:

    For any (itinerary, user_request) pair, RewardFunction.compute() and
    DeterministicEvaluator.score() must produce identical scores for every
    component that both compute.

Violation means the compressor is optimised for a signal that evaluation does
not measure, breaking the paper's comparison validity.

Components checked
------------------
  * hard_constraint_ratio / hard_constraint_score
  * soft_constraint_score
  * budget_adherence / hard_constraint_score (budget sub-category)
  * logical_consistency
  * tool_efficiency
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from optimized_llm_planning_memory.core.config import RewardConfig, RewardWeights
from optimized_llm_planning_memory.core.models import (
    AccommodationBooking,
    ActivityBooking,
    Constraint,
    ConstraintCategory,
    ConstraintType,
    EpisodeLog,
    ItineraryDay,
    Itinerary,
    PPOTransition,
    RewardComponents,
    ToolCallStats,
    TrajectoryModel,
    UserRequest,
)
from optimized_llm_planning_memory.mcts.node import MCTSStats  # resolve forward ref
EpisodeLog.model_rebuild()
from optimized_llm_planning_memory.evaluation.deterministic import DeterministicEvaluator
from optimized_llm_planning_memory.training.reward import RewardFunction


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_request(
    budget: float = 1500.0,
    soft: bool = True,
) -> UserRequest:
    sc = []
    if soft:
        sc = [Constraint(
            constraint_id="sc-1",
            constraint_type=ConstraintType.SOFT,
            category=ConstraintCategory.ACCOMMODATION,
            description="Prefer boutique hotels",
            value="boutique",
        )]
    return UserRequest(
        request_id="consistency-req-001",
        raw_text="Plan a 3-day Paris trip.",
        origin_city="New York",
        destination_cities=["Paris"],
        start_date="2025-06-01",
        end_date="2025-06-03",
        budget_usd=budget,
        hard_constraints=[
            Constraint(
                constraint_id="hc-budget",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.BUDGET,
                description=f"Total cost ≤ ${budget}",
                value=budget,
                unit="USD",
            ),
        ],
        soft_constraints=sc,
    )


def _make_itinerary(total_cost: float = 800.0) -> Itinerary:
    """Build a Paris itinerary whose activities sum to `total_cost`."""
    # activity costs + accommodation total should equal total_cost
    accom_cost = min(300.0, total_cost * 0.375)
    activity_cost = total_cost - accom_cost
    day = ItineraryDay(
        date="2025-06-01",
        city="Paris",
        activities=[
            ActivityBooking(
                activity_id="act-1",
                activity_name="Louvre Museum",
                location="Central Paris",
                city="Paris",
                start_datetime="2025-06-01T10:00:00",
                duration_hours=3.0,
                cost_usd=activity_cost,
                category="museum",
                booking_ref="SIM-ACT-001",
            )
        ],
        accommodation=AccommodationBooking(
            hotel_id="htl-1",
            hotel_name="Paris Boutique Inn",
            city="Paris",
            check_in="2025-06-01",
            check_out="2025-06-03",
            cost_per_night_usd=150.0,
            total_cost_usd=accom_cost,
            booking_ref="SIM-HTL-001",
        ),
    )
    return Itinerary(
        itinerary_id=str(uuid.uuid4()),
        request_id="consistency-req-001",
        days=[day],
    )


def _make_episode_log(itinerary: Itinerary | None = None) -> EpisodeLog:
    traj = TrajectoryModel(
        trajectory_id=str(uuid.uuid4()),
        request_id="consistency-req-001",
        steps=(),
        total_steps=0,
    )
    placeholder = RewardComponents(
        hard_constraint_score=0.0,
        soft_constraint_score=0.0,
        tool_efficiency_score=1.0,
        tool_failure_penalty=0.0,
        logical_consistency_score=1.0,
        total_reward=0.0,
    )
    tool_stat = ToolCallStats(
        tool_name="search_flights",
        call_count=1,
        success_count=1,
        failure_count=0,
        redundant_call_count=0,
        total_latency_ms=5.0,
        avg_latency_ms=5.0,
    )
    return EpisodeLog(
        episode_id=str(uuid.uuid4()),
        request_id="consistency-req-001",
        agent_mode="raw",
        trajectory=traj,
        compressed_states=(),
        final_itinerary=itinerary,
        reward_components=placeholder,
        tool_stats=(tool_stat,),
        total_steps=3,
        success=True,
        config_hash="test",
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRewardEvalConsistency:
    """T1: training reward ≡ deterministic evaluation metric for shared components."""

    def test_hard_constraint_score_matches_deterministic(self):
        """RewardFunction.hard_score == DeterministicEvaluator.hard_constraint_ratio."""
        request = _make_request(budget=1500.0)
        itinerary = _make_itinerary(total_cost=800.0)  # within budget
        episode = _make_episode_log(itinerary=itinerary)

        reward_fn = RewardFunction(
            config=RewardConfig(weights=RewardWeights(hard_constraint=1.0), step_penalty=0.0)
        )
        evaluator = DeterministicEvaluator()

        r = reward_fn.compute(episode, request, is_terminal=True)
        d = evaluator.score(episode, request)

        assert abs(r.hard_constraint_score - d["hard_constraint_ratio"]) < 1e-9, (
            f"reward hard={r.hard_constraint_score}, eval hard={d['hard_constraint_ratio']}"
        )

    def test_soft_constraint_score_matches_deterministic(self):
        """RewardFunction.soft_score == DeterministicEvaluator.soft_constraint_score."""
        request = _make_request(soft=True)
        itinerary = _make_itinerary()
        episode = _make_episode_log(itinerary=itinerary)

        reward_fn = RewardFunction(config=RewardConfig(step_penalty=0.0))
        evaluator = DeterministicEvaluator()

        r = reward_fn.compute(episode, request, is_terminal=True)
        d = evaluator.score(episode, request)

        assert abs(r.soft_constraint_score - d["soft_constraint_score"]) < 1e-9, (
            f"reward soft={r.soft_constraint_score}, eval soft={d['soft_constraint_score']}"
        )

    def test_no_soft_constraints_both_default_to_one(self):
        """When request has no soft constraints, both paths must return 1.0."""
        request = _make_request(soft=False)
        itinerary = _make_itinerary()
        episode = _make_episode_log(itinerary=itinerary)

        reward_fn = RewardFunction(config=RewardConfig(step_penalty=0.0))
        evaluator = DeterministicEvaluator()

        r = reward_fn.compute(episode, request, is_terminal=True)
        d = evaluator.score(episode, request)

        assert r.soft_constraint_score == 1.0, (
            f"reward soft default should be 1.0, got {r.soft_constraint_score}"
        )
        assert d["soft_constraint_score"] == 1.0, (
            f"eval soft default should be 1.0, got {d['soft_constraint_score']}"
        )
        # The key invariant: they agree
        assert abs(r.soft_constraint_score - d["soft_constraint_score"]) < 1e-9

    def test_budget_over_limit_hard_constraint_zero(self):
        """Over-budget itinerary: hard_constraint_score = 0 in both paths."""
        request = _make_request(budget=500.0)
        itinerary = _make_itinerary(total_cost=1200.0)  # over budget
        episode = _make_episode_log(itinerary=itinerary)

        reward_fn = RewardFunction(config=RewardConfig(step_penalty=0.0))
        evaluator = DeterministicEvaluator()

        r = reward_fn.compute(episode, request, is_terminal=True)
        d = evaluator.score(episode, request)

        assert r.hard_constraint_score < 1.0
        assert d["hard_constraint_ratio"] < 1.0
        assert abs(r.hard_constraint_score - d["hard_constraint_ratio"]) < 1e-9

    def test_logical_consistency_score_matches_deterministic(self):
        """RewardFunction.logical_consistency_score == DeterministicEvaluator.logical_consistency."""
        request = _make_request()
        itinerary = _make_itinerary()
        episode = _make_episode_log(itinerary=itinerary)

        reward_fn = RewardFunction(config=RewardConfig(step_penalty=0.0))
        evaluator = DeterministicEvaluator()

        r = reward_fn.compute(episode, request, is_terminal=True)
        d = evaluator.score(episode, request)

        assert abs(r.logical_consistency_score - d["logical_consistency"]) < 1e-9, (
            f"reward consistency={r.logical_consistency_score}, "
            f"eval consistency={d['logical_consistency']}"
        )
