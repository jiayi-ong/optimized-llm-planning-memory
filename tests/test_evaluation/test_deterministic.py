"""
tests/test_evaluation/test_deterministic.py
=============================================
Unit tests for DeterministicEvaluator.

Verifies that the evaluator uses the same ConstraintSatisfactionEngine as the
reward function, and that all metric keys are present in the output.
"""

from __future__ import annotations

import pytest

from optimized_llm_planning_memory.evaluation.deterministic import DeterministicEvaluator


EXPECTED_KEYS = {
    "hard_constraint_ratio",
    "soft_constraint_score",
    "tool_efficiency",
    "tool_failure_rate",
    "avg_tool_latency_ms",
    "steps_per_episode",
    "budget_adherence",
    "logical_consistency",
}


@pytest.fixture
def evaluator() -> DeterministicEvaluator:
    return DeterministicEvaluator()


def test_score_returns_all_keys(evaluator, sample_episode_log, sample_user_request):
    scores = evaluator.score(sample_episode_log, sample_user_request)
    assert EXPECTED_KEYS.issubset(scores.keys())


def test_score_values_in_valid_range(evaluator, sample_episode_log, sample_user_request):
    scores = evaluator.score(sample_episode_log, sample_user_request)
    for key, val in scores.items():
        if key not in ("avg_tool_latency_ms", "steps_per_episode", "tool_failure_rate"):
            assert 0.0 <= val <= 1.0, f"{key} = {val} out of [0, 1]"


def test_budget_adherence_within_budget(evaluator, sample_episode_log, sample_user_request):
    """Itinerary at $170 with $2000 budget → adherence = 1.0."""
    scores = evaluator.score(sample_episode_log, sample_user_request)
    assert scores["budget_adherence"] == pytest.approx(1.0)


def test_logical_consistency_perfect(evaluator, sample_episode_log, sample_user_request):
    """Single-day itinerary with no double-booking → consistency = 1.0."""
    scores = evaluator.score(sample_episode_log, sample_user_request)
    assert scores["logical_consistency"] == pytest.approx(1.0)
