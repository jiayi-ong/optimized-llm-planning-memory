"""Unit tests for core/config.py — configuration schema defaults and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from optimized_llm_planning_memory.core.config import (
    AgentConfig,
    EnvConfig,
    RewardConfig,
    RewardWeights,
    PPOHyperparams,
    TrainingConfig,
    EvalConfig,
)


@pytest.mark.unit
class TestAgentConfig:
    def test_default_values(self):
        cfg = AgentConfig()
        assert cfg.max_steps == 30
        assert cfg.compress_every_n_steps == 5
        assert cfg.mode == "compressor"

    def test_max_steps_ge_one(self):
        with pytest.raises(ValidationError):
            AgentConfig(max_steps=0)

    def test_mode_string_raw(self):
        cfg = AgentConfig(mode="raw")
        assert cfg.mode == "raw"

    def test_compress_every_n_steps_ge_one(self):
        with pytest.raises(ValidationError):
            AgentConfig(compress_every_n_steps=0)


@pytest.mark.unit
class TestRewardConfig:
    def test_step_penalty_le_zero(self):
        with pytest.raises(ValidationError):
            RewardConfig(step_penalty=0.1)  # must be <= 0

    def test_step_penalty_negative_ok(self):
        cfg = RewardConfig(step_penalty=-0.05)
        assert cfg.step_penalty == pytest.approx(-0.05)

    def test_weights_accessible(self):
        cfg = RewardConfig()
        assert isinstance(cfg.weights, RewardWeights)

    def test_tool_failure_weight_le_zero(self):
        with pytest.raises(ValidationError):
            RewardWeights(tool_failure_penalty=0.5)


@pytest.mark.unit
class TestEnvConfig:
    def test_defaults(self):
        cfg = EnvConfig()
        assert cfg.max_obs_tokens == 2048
        assert cfg.max_action_tokens == 512

    def test_custom_values(self):
        cfg = EnvConfig(max_obs_tokens=64, max_action_tokens=32)
        assert cfg.max_obs_tokens == 64
        assert cfg.max_action_tokens == 32


@pytest.mark.unit
class TestPPOHyperparams:
    def test_defaults(self):
        p = PPOHyperparams()
        assert p.learning_rate == pytest.approx(3e-5)
        assert p.clip_epsilon == pytest.approx(0.2)
        assert p.gamma == pytest.approx(0.99)

    def test_learning_rate_gt_zero(self):
        with pytest.raises(ValidationError):
            PPOHyperparams(learning_rate=0.0)


@pytest.mark.unit
class TestEvalConfig:
    def test_deterministic_only_default_false(self):
        cfg = EvalConfig()
        assert cfg.deterministic_only is False

    def test_rubric_dimensions_populated(self):
        cfg = EvalConfig()
        assert len(cfg.rubric_dimensions) > 0
