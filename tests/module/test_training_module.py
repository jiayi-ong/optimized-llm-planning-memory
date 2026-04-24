"""Module tests for training — EpisodeBuffer, RewardFunction, CompressionEnv."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from optimized_llm_planning_memory.core.models import (
    EpisodeLog,
    PPOTransition,
    RewardComponents,
    TrajectoryModel,
)
from optimized_llm_planning_memory.mcts.node import MCTSStats  # resolve forward ref
EpisodeLog.model_rebuild()
from optimized_llm_planning_memory.training.episode_buffer import EpisodeBuffer
from optimized_llm_planning_memory.training.env import CompressionEnv


def _make_transition(reward: float = 0.5) -> PPOTransition:
    return PPOTransition(
        trajectory_text="trajectory text",
        compressed_state_text="compressed text",
        reward=reward,
        value_estimate=0.4,
        log_prob=-0.2,
        advantage=None,
    )


def _make_episode_log(itinerary=None) -> EpisodeLog:
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
class TestEpisodeBufferMinibackBatch:
    def test_twenty_items_batch_four_gives_five_batches(self):
        buf = EpisodeBuffer()
        for i in range(20):
            buf.add(_make_transition(float(i)))
        buf.fill_advantages([float(i) for i in range(20)])
        batches = list(buf.minibatches(batch_size=4, shuffle=False))
        assert len(batches) == 5
        assert all(len(b) == 4 for b in batches)

    def test_clear_then_refill(self):
        buf = EpisodeBuffer()
        for _ in range(5):
            buf.add(_make_transition())
        buf.clear()
        assert buf.is_empty()
        buf.add(_make_transition())
        assert len(buf) == 1


@pytest.mark.module_test
class TestRewardFunctionCompute:
    def test_compute_returns_reward_components(self, reward_fn, paris_request, sample_itinerary):
        episode = _make_episode_log(itinerary=sample_itinerary)
        components = reward_fn.compute(
            episode_log=episode,
            user_request=paris_request,
            is_terminal=True,
        )
        assert isinstance(components.total_reward, float)

    def test_changing_weights_changes_total(self, paris_request, sample_itinerary):
        from optimized_llm_planning_memory.training.reward import RewardFunction
        from optimized_llm_planning_memory.core.config import RewardConfig, RewardWeights

        episode = _make_episode_log(itinerary=sample_itinerary)

        rf1 = RewardFunction(config=RewardConfig(weights=RewardWeights(hard_constraint=2.0), step_penalty=0.0))
        rf2 = RewardFunction(config=RewardConfig(weights=RewardWeights(hard_constraint=0.1), step_penalty=0.0))

        r1 = rf1.compute(episode, paris_request, is_terminal=True)
        r2 = rf2.compute(episode, paris_request, is_terminal=True)
        assert r1.total_reward != r2.total_reward


@pytest.mark.module_test
class TestCompressionEnvResetStep:
    def test_reset_returns_zero_obs(self, paris_request, reward_fn, env_config_small):
        agent = MagicMock()
        env = CompressionEnv(
            agent_factory=lambda: agent,
            simulator_factory=lambda seed: MagicMock(),
            reward_fn=reward_fn,
            user_requests=[paris_request],
            config=env_config_small,
        )
        obs, info = env.reset()
        assert np.all(obs == 0)
        assert "request_id" in info
