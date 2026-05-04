"""System tests — RL components: CompressionEnv gymnasium API + EpisodeBuffer cycle."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from optimized_llm_planning_memory.core.config import EnvConfig, RewardConfig
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
from optimized_llm_planning_memory.training.reward import RewardFunction


def _make_episode_log() -> EpisodeLog:
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
        final_itinerary=None,
        reward_components=reward,
        tool_stats=(),
        total_steps=0,
        success=True,
        config_hash="test",
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


def _make_env(paris_request, reward_fn, env_config_small: EnvConfig) -> CompressionEnv:
    agent = MagicMock()
    # run_steps() now drives the env loop; must return (itinerary, done, error_msg)
    agent.run_steps.return_value = (None, False, None)

    sim = MagicMock()

    return CompressionEnv(
        agent_factory=lambda: agent,
        simulator_factory=lambda seed: sim,
        reward_fn=reward_fn,
        user_requests=[paris_request],
        config=env_config_small,
    )


@pytest.mark.system_test
class TestCompressionEnvGymnasiumAPI:
    def test_reset_returns_obs_and_info(self, paris_request, reward_fn, env_config_small):
        env = _make_env(paris_request, reward_fn, env_config_small)
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_observation_space_contains_reset_obs(self, paris_request, reward_fn, env_config_small):
        env = _make_env(paris_request, reward_fn, env_config_small)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_reset_obs_is_zero(self, paris_request, reward_fn, env_config_small):
        env = _make_env(paris_request, reward_fn, env_config_small)
        obs, _ = env.reset()
        assert np.all(obs == 0)

    def test_reset_info_has_request_id(self, paris_request, reward_fn, env_config_small):
        env = _make_env(paris_request, reward_fn, env_config_small)
        _, info = env.reset()
        assert "request_id" in info

    def test_step_returns_five_tuple(self, paris_request, reward_fn, env_config_small):
        env = _make_env(paris_request, reward_fn, env_config_small)
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert isinstance(result, tuple) and len(result) == 5

    def test_step_reward_is_float(self, paris_request, reward_fn, env_config_small):
        env = _make_env(paris_request, reward_fn, env_config_small)
        env.reset()
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float))

    def test_step_info_has_episode_log(self, paris_request, reward_fn, env_config_small):
        env = _make_env(paris_request, reward_fn, env_config_small)
        env.reset()
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "episode_log" in info


@pytest.mark.system_test
class TestEpisodeBufferCycle:
    def test_add_fill_advantages_minibatch_clear_cycle(self):
        buf = EpisodeBuffer()
        n = 8
        for i in range(n):
            buf.add(PPOTransition(
                trajectory_text=f"traj {i}",
                compressed_state_text=f"comp {i}",
                reward=float(i),
                value_estimate=0.5,
                log_prob=-0.3,
            ))
        buf.fill_advantages([float(i) for i in range(n)])

        all_items = []
        for batch in buf.minibatches(batch_size=4, shuffle=False):
            all_items.extend(batch)
        assert len(all_items) == n

        buf.clear()
        assert buf.is_empty()


@pytest.mark.system_test
class TestCompressorPolicyCPU:
    def test_identity_compressor_evaluate_actions(self, identity_compressor):
        import torch
        trajectories = ["trajectory text one", "trajectory text two"]
        compressed = ["compressed one", "compressed two"]
        for traj, comp in zip(trajectories, compressed):
            lp = identity_compressor.get_log_probs(traj, comp)
            assert isinstance(lp, torch.Tensor)
            assert lp.dim() == 1

    def test_dummy_compressor_is_trainable(self, dummy_compressor_cpu):
        assert dummy_compressor_cpu.is_trainable() is True
        params = dummy_compressor_cpu.get_trainable_parameters()
        assert len(params) > 0


@pytest.mark.system_test
class TestRLRunLoggerSmokeWithJSONL:
    """End-to-end smoke: 2 env steps → JSONL is written → round-trips correctly."""

    def test_episode_callback_writes_jsonl_on_step(
        self, tmp_path, paris_request, reward_fn, env_config_small
    ):
        """EpisodeLogCallback must write a JSONL line after an episode completes."""
        import json

        from stable_baselines3.common.callbacks import CallbackList

        from optimized_llm_planning_memory.training.run_logger import RLRunLogger, load_episode_metrics
        from optimized_llm_planning_memory.training.trainer import EpisodeLogCallback

        # Build a minimal env
        agent = MagicMock()
        agent.run_steps.return_value = (None, True, None)  # episode terminates immediately

        env = CompressionEnv(
            agent_factory=lambda: agent,
            simulator_factory=lambda seed: MagicMock(),
            reward_fn=reward_fn,
            user_requests=[paris_request],
            config=env_config_small,
        )

        run_id = "smoke_test_run"
        with RLRunLogger(run_id=run_id, training_dir=tmp_path) as logger:
            cb = EpisodeLogCallback(run_logger=logger, verbose=0)

            # Simulate what SB3 does: set up the callback's internal state
            cb.locals = {}
            cb.globals = {}
            cb.num_timesteps = 1
            cb.model = MagicMock()
            # SB3's BaseCallback.logger is a read-only property (returns model.logger);
            # setting cb.model already provides a mock logger via cb.model.logger.

            # Inject a fake info dict (as if returned by CompressionEnv.step())
            from optimized_llm_planning_memory.core.models import RewardComponents
            fake_rc = RewardComponents(
                hard_constraint_score=0.8,
                soft_constraint_score=0.7,
                tool_efficiency_score=0.9,
                tool_failure_penalty=0.0,
                logical_consistency_score=1.0,
                total_reward=0.75,
            )
            cb.locals["infos"] = [
                {
                    "reward_components": fake_rc,
                    "episode_log": None,
                    "request_id": "req-001",
                }
            ]
            cb._on_step()

        records = load_episode_metrics(run_id, tmp_path)
        assert len(records) >= 1
        assert records[0].total_reward == pytest.approx(0.75, abs=1e-4)

    def test_run_manifest_saved_alongside_jsonl(self, tmp_path):
        """save_manifest() must create a valid manifest.json in the run directory."""
        from optimized_llm_planning_memory.training.run_manifest import (
            TrainingRunManifest,
            load_manifest,
            save_manifest,
        )

        manifest = TrainingRunManifest.create(
            run_id="smoke_manifest",
            compressor_type="IdentityCompressor",
            n_train_requests=10,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        save_manifest(manifest, tmp_path / "training" / "smoke_manifest")
        loaded = load_manifest("smoke_manifest", tmp_path / "training")

        assert loaded is not None
        assert loaded.compressor_type == "IdentityCompressor"
        assert loaded.n_train_requests == 10
