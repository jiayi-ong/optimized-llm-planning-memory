"""Module tests for evaluation/ablation — AblationRunner full sweep."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from optimized_llm_planning_memory.core.models import (
    EpisodeLog,
    RewardComponents,
    TrajectoryModel,
)
from optimized_llm_planning_memory.mcts.node import MCTSStats  # resolve forward ref
EpisodeLog.model_rebuild()
from optimized_llm_planning_memory.evaluation.ablation import AblationResult, AblationRunner
from optimized_llm_planning_memory.evaluation.deterministic import DeterministicEvaluator
from optimized_llm_planning_memory.evaluation.evaluator import Evaluator


def _make_episode(itinerary=None) -> EpisodeLog:
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
        episode_id=str(uuid.uuid4()),
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
class TestAblationRunnerModule:
    def test_two_value_axis_produces_two_results(self, sample_itinerary, sample_user_request):
        evaluator = Evaluator(deterministic_eval=DeterministicEvaluator())

        def generator(overrides):
            eps = [_make_episode(itinerary=sample_itinerary)]
            reqs = [sample_user_request]
            return eps, reqs

        runner = AblationRunner(evaluator=evaluator, episode_generator=generator)
        results = runner.run(axes={"compressor_type": ["identity", "dummy"]})
        assert len(results) == 2

    def test_all_results_have_readable_labels(self, sample_itinerary, sample_user_request):
        evaluator = Evaluator(deterministic_eval=DeterministicEvaluator())

        def generator(overrides):
            return [_make_episode(itinerary=sample_itinerary)], [sample_user_request]

        runner = AblationRunner(evaluator=evaluator, episode_generator=generator)
        results = runner.run(axes={"mode": ["raw", "compressor"]})
        for r in results:
            assert len(r.label) > 0
            assert r.n_episodes >= 1

    def test_four_combo_cartesian_product(self, sample_itinerary, sample_user_request):
        evaluator = Evaluator(deterministic_eval=DeterministicEvaluator())

        def generator(overrides):
            return [_make_episode(itinerary=sample_itinerary)], [sample_user_request]

        runner = AblationRunner(evaluator=evaluator, episode_generator=generator)
        results = runner.run(axes={
            "mode": ["raw", "compressor"],
            "type": ["identity", "dummy"],
        })
        assert len(results) == 4
