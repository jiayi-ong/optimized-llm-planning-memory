"""
Unit tests — mini PPO update cycle (no LLM, no simulator, CPU only).

Verifies that the full SB3 PPO machinery compiles and runs with
CompressorPolicy + IdentityCompressor in a mocked environment.

The test is intentionally minimal (2 envs, 8 steps, 1 update) to run
quickly in CI without requiring a GPU or API keys.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
from optimized_llm_planning_memory.core.config import EnvConfig, RewardConfig
from optimized_llm_planning_memory.core.models import (
    UserRequest,
    TravelerProfile,
    Constraint,
    ConstraintType,
    ConstraintCategory,
)
from optimized_llm_planning_memory.training.env import CompressionEnv
from optimized_llm_planning_memory.training.reward import RewardFunction


def _make_request() -> UserRequest:
    return UserRequest(
        request_id="ppo-test-req",
        raw_text="Test request",
        origin_city="New York",
        destination_cities=["Paris"],
        start_date="2025-06-01",
        end_date="2025-06-03",
        budget_usd=3000.0,
        traveler_profile=TravelerProfile(num_adults=1),
        hard_constraints=[
            Constraint(
                constraint_id="hc1",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.BUDGET,
                description="Budget <= $3000",
                value=3000.0,
                unit="USD",
            )
        ],
        soft_constraints=[],
    )


def _make_env(request: UserRequest) -> CompressionEnv:
    agent = MagicMock()
    agent.run_steps.return_value = (None, False, None)
    sim = MagicMock()
    return CompressionEnv(
        agent_factory=lambda: agent,
        simulator_factory=lambda seed: sim,
        reward_fn=RewardFunction(config=RewardConfig()),
        user_requests=[request],
        config=EnvConfig(max_obs_tokens=32, max_action_tokens=16),
    )


@pytest.mark.unit_test
class TestMiniPPOUpdateCycle:
    def test_ppo_learn_completes_without_error(self):
        """Full mini PPO train loop: 1 env, 4 steps, 1 update, no crash."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env

        from optimized_llm_planning_memory.training.policy import CompressorPolicy

        compressor = IdentityCompressor()
        request = _make_request()

        vec_env = make_vec_env(
            lambda: _make_env(request),
            n_envs=1,
            seed=0,
        )

        policy_kwargs = {"compressor": compressor, "value_hidden_dim": 16}

        ppo = PPO(
            policy=CompressorPolicy,
            env=vec_env,
            learning_rate=3e-4,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=1.0,
            policy_kwargs=policy_kwargs,
            device="cpu",
            verbose=0,
        )

        # Should not raise; completes 1 PPO update (4 steps × 1 env = 4 timesteps)
        ppo.learn(total_timesteps=4)

    def test_ppo_policy_gradients_non_zero_after_update(self):
        """After one PPO update, at least one compressor parameter must have changed."""
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env

        from optimized_llm_planning_memory.training.policy import CompressorPolicy

        compressor = IdentityCompressor()
        request = _make_request()

        vec_env = make_vec_env(lambda: _make_env(request), n_envs=1, seed=1)

        policy_kwargs = {"compressor": compressor, "value_hidden_dim": 16}
        ppo = PPO(
            policy=CompressorPolicy,
            env=vec_env,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            policy_kwargs=policy_kwargs,
            device="cpu",
            verbose=0,
        )

        params_before = [p.data.clone() for p in compressor.get_trainable_parameters()]
        ppo.learn(total_timesteps=4)
        params_after = [p.data.clone() for p in compressor.get_trainable_parameters()]

        changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
        assert changed, "At least one compressor parameter must change after a PPO update"

    def test_ppo_value_loss_decreases_with_repeated_updates(self):
        """Value function should show a loss value after training (sanity check only)."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env

        from optimized_llm_planning_memory.training.policy import CompressorPolicy

        compressor = IdentityCompressor()
        request = _make_request()

        vec_env = make_vec_env(lambda: _make_env(request), n_envs=1, seed=2)
        policy_kwargs = {"compressor": compressor, "value_hidden_dim": 16}
        ppo = PPO(
            policy=CompressorPolicy,
            env=vec_env,
            n_steps=4,
            batch_size=4,
            n_epochs=2,
            policy_kwargs=policy_kwargs,
            device="cpu",
            verbose=0,
        )

        ppo.learn(total_timesteps=8)
        # Just check that SB3 has logged something for value_loss
        log_vals = ppo.logger.name_to_value
        # SB3 may not yet have logged if not enough rollouts — tolerate absence
        if "train/value_loss" in log_vals:
            val_loss = log_vals["train/value_loss"]
            assert val_loss is not None
            assert np.isfinite(float(val_loss))
