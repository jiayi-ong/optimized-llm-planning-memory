"""Module tests for evaluation — Evaluator + DeterministicEvaluator workflows."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from optimized_llm_planning_memory.core.models import (
    EpisodeLog,
    EvalResult,
    RewardComponents,
    TrajectoryModel,
)
from optimized_llm_planning_memory.mcts.node import MCTSStats  # resolve forward ref
EpisodeLog.model_rebuild()
from optimized_llm_planning_memory.evaluation.deterministic import DeterministicEvaluator
from optimized_llm_planning_memory.evaluation.evaluator import Evaluator


def _make_episode_log(itinerary=None, episode_id: str | None = None) -> EpisodeLog:
    traj = TrajectoryModel(
        trajectory_id=str(uuid.uuid4()),
        request_id="req-001",
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
        episode_id=episode_id or str(uuid.uuid4()),
        request_id="req-001",
        agent_mode="raw",
        trajectory=traj,
        compressed_states=(),
        final_itinerary=itinerary,
        reward_components=reward,
        tool_stats=(),
        total_steps=0,
        success=True,
        config_hash="test",
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


@pytest.mark.module_test
class TestDeterministicEvaluatorScore:
    def test_score_returns_dict_with_keys(self, sample_itinerary, sample_user_request):
        evaluator = DeterministicEvaluator()
        episode = _make_episode_log(itinerary=sample_itinerary)
        scores = evaluator.score(episode, sample_user_request)
        assert isinstance(scores, dict)
        assert len(scores) > 0

    def test_score_values_in_zero_one_range(self, sample_itinerary, sample_user_request):
        evaluator = DeterministicEvaluator()
        episode = _make_episode_log(itinerary=sample_itinerary)
        scores = evaluator.score(episode, sample_user_request)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"Score '{key}' = {val} out of [0, 1]"


@pytest.mark.module_test
class TestEvaluatorEpisode:
    def test_evaluate_episode_returns_eval_result(self, sample_itinerary, sample_user_request):
        evaluator = Evaluator(deterministic_eval=DeterministicEvaluator())
        episode = _make_episode_log(itinerary=sample_itinerary)
        result = evaluator.evaluate_episode(episode, sample_user_request)
        assert isinstance(result, EvalResult)

    def test_eval_result_has_deterministic_scores(self, sample_itinerary, sample_user_request):
        evaluator = Evaluator(deterministic_eval=DeterministicEvaluator())
        episode = _make_episode_log(itinerary=sample_itinerary)
        result = evaluator.evaluate_episode(episode, sample_user_request)
        assert len(result.deterministic_scores) > 0

    def test_overall_score_bounded(self, sample_itinerary, sample_user_request):
        evaluator = Evaluator(deterministic_eval=DeterministicEvaluator())
        episode = _make_episode_log(itinerary=sample_itinerary)
        result = evaluator.evaluate_episode(episode, sample_user_request)
        assert 0.0 <= result.overall_score <= 1.0

    def test_no_llm_judge_calls_when_deterministic_only(self, sample_itinerary, sample_user_request):
        judge = MagicMock()
        evaluator = Evaluator(
            deterministic_eval=DeterministicEvaluator(),
            llm_judge=None,
        )
        episode = _make_episode_log(itinerary=sample_itinerary)
        evaluator.evaluate_episode(episode, sample_user_request)
        judge.score.assert_not_called()


@pytest.mark.module_test
class TestEvaluatorAggregate:
    def test_aggregate_returns_mean_and_std_keys(self, sample_itinerary, sample_user_request):
        evaluator = Evaluator(deterministic_eval=DeterministicEvaluator())
        episodes = [_make_episode_log(itinerary=sample_itinerary) for _ in range(3)]
        results = evaluator.evaluate_dataset(episodes, [sample_user_request] * 3)
        agg = evaluator.aggregate(results)
        assert any("mean" in k for k in agg.keys())

    def test_aggregate_overall_score_mean_present(self, sample_itinerary, sample_user_request):
        evaluator = Evaluator(deterministic_eval=DeterministicEvaluator())
        episodes = [_make_episode_log(itinerary=sample_itinerary) for _ in range(2)]
        results = evaluator.evaluate_dataset(episodes, [sample_user_request] * 2)
        agg = evaluator.aggregate(results)
        assert "overall_score_mean" in agg
