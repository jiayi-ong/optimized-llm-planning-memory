"""Unit tests for training/env.py — CompressionEnv observation/action spaces."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from optimized_llm_planning_memory.core.config import EnvConfig
from optimized_llm_planning_memory.core.models import UserRequest
from optimized_llm_planning_memory.training.env import CompressionEnv
from optimized_llm_planning_memory.training.reward import RewardFunction


def _make_request(rid: str = "req-1") -> UserRequest:
    return UserRequest(
        request_id=rid,
        raw_text="Plan a trip to Paris.",
        origin_city="New York",
        destination_cities=["Paris"],
        start_date="2025-06-01",
        end_date="2025-06-07",
        budget_usd=3000.0,
    )


def _make_env(max_obs: int = 64, max_act: int = 32) -> CompressionEnv:
    reward_fn = MagicMock(spec=RewardFunction)
    reward_fn.compute.return_value = 0.5

    agent = MagicMock()
    # run_steps() now drives the env loop; must return (itinerary, done, error_msg)
    agent.run_steps.return_value = (None, False, None)

    sim = MagicMock()

    config = EnvConfig(max_obs_tokens=max_obs, max_action_tokens=max_act)
    env = CompressionEnv(
        agent_factory=lambda: agent,
        simulator_factory=lambda seed: sim,
        reward_fn=reward_fn,
        user_requests=[_make_request()],
        config=config,
    )
    return env


@pytest.mark.unit
class TestCompressionEnvSpaces:
    def test_observation_space_shape(self):
        env = _make_env(max_obs=64)
        assert env.observation_space.shape == (64,)

    def test_action_space_shape(self):
        env = _make_env(max_act=32)
        assert env.action_space.shape == (32,)

    def test_observation_space_dtype_int32(self):
        env = _make_env()
        assert env.observation_space.dtype == np.int32

    def test_action_space_dtype_int32(self):
        env = _make_env()
        assert env.action_space.dtype == np.int32

    def test_custom_dimensions_respected(self):
        env = _make_env(max_obs=128, max_act=64)
        assert env.observation_space.shape == (128,)
        assert env.action_space.shape == (64,)


@pytest.mark.unit
class TestCompressionEnvReset:
    def test_reset_returns_tuple(self):
        env = _make_env()
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_obs_is_ndarray(self):
        env = _make_env(max_obs=64)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (64,)

    def test_reset_obs_is_zero_padded(self):
        env = _make_env(max_obs=64)
        obs, info = env.reset()
        assert np.all(obs == 0)

    def test_reset_info_has_request_id(self):
        env = _make_env()
        _, info = env.reset()
        assert "request_id" in info

    def test_reset_seed_accepted(self):
        env = _make_env()
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
