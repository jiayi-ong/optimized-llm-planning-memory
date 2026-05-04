"""
Unit tests — CompressorPolicy forward + backward pass.

Tests verify that:
1. A forward pass through CompressorPolicy returns tensors of the correct shape.
2. A backward pass (loss.backward()) runs without errors and populates gradients
   on all get_trainable_parameters().
3. evaluate_actions() returns (values, log_probs, entropy) with correct shapes.

All compressor calls use IdentityCompressor (no HF model needed) so these tests
run on CPU in < 1 s with no API calls.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from gymnasium import spaces

from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
from optimized_llm_planning_memory.training.policy import CompressorPolicy


# ── Fixtures ──────────────────────────────────────────────────────────────────

MAX_OBS = 32
MAX_ACT = 16
BATCH = 2


@pytest.fixture
def obs_space() -> spaces.Box:
    return spaces.Box(low=0, high=127, shape=(MAX_OBS,), dtype=np.int32)


@pytest.fixture
def act_space() -> spaces.Box:
    return spaces.Box(low=0, high=127, shape=(MAX_ACT,), dtype=np.int32)


@pytest.fixture
def compressor() -> IdentityCompressor:
    return IdentityCompressor()


@pytest.fixture
def policy(obs_space, act_space, compressor) -> CompressorPolicy:
    return CompressorPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 3e-4,
        compressor=compressor,
        value_hidden_dim=32,
    )


@pytest.fixture
def obs_tensor() -> torch.Tensor:
    """Random int32 observation batch."""
    return torch.randint(0, 128, (BATCH, MAX_OBS), dtype=torch.int32)


@pytest.fixture
def act_tensor() -> torch.Tensor:
    """Random int32 action batch."""
    return torch.randint(0, 128, (BATCH, MAX_ACT), dtype=torch.int32)


# ── forward() tests ───────────────────────────────────────────────────────────

@pytest.mark.unit_test
class TestCompressorPolicyForward:
    def test_forward_returns_three_tensors(self, policy, obs_tensor):
        actions, values, log_probs = policy.forward(obs_tensor, deterministic=True)
        assert isinstance(actions, torch.Tensor)
        assert isinstance(values, torch.Tensor)
        assert isinstance(log_probs, torch.Tensor)

    def test_forward_action_shape(self, policy, obs_tensor):
        actions, _, _ = policy.forward(obs_tensor, deterministic=True)
        assert actions.shape == (BATCH, MAX_ACT)

    def test_forward_values_shape(self, policy, obs_tensor):
        _, values, _ = policy.forward(obs_tensor, deterministic=True)
        assert values.shape == (BATCH, 1)

    def test_forward_log_probs_shape(self, policy, obs_tensor):
        _, _, log_probs = policy.forward(obs_tensor, deterministic=True)
        assert log_probs.shape == (BATCH,)

    def test_forward_values_finite(self, policy, obs_tensor):
        _, values, _ = policy.forward(obs_tensor, deterministic=True)
        assert torch.isfinite(values).all()


# ── evaluate_actions() tests ──────────────────────────────────────────────────

@pytest.mark.unit_test
class TestCompressorPolicyEvaluateActions:
    def test_evaluate_actions_returns_three_tensors(self, policy, obs_tensor, act_tensor):
        values, log_probs, entropy = policy.evaluate_actions(obs_tensor, act_tensor)
        assert isinstance(values, torch.Tensor)
        assert isinstance(log_probs, torch.Tensor)
        assert isinstance(entropy, torch.Tensor)

    def test_evaluate_actions_values_shape(self, policy, obs_tensor, act_tensor):
        values, _, _ = policy.evaluate_actions(obs_tensor, act_tensor)
        assert values.shape == (BATCH, 1)

    def test_evaluate_actions_log_probs_shape(self, policy, obs_tensor, act_tensor):
        _, log_probs, _ = policy.evaluate_actions(obs_tensor, act_tensor)
        assert log_probs.shape == (BATCH,)

    def test_evaluate_actions_entropy_scalar(self, policy, obs_tensor, act_tensor):
        _, _, entropy = policy.evaluate_actions(obs_tensor, act_tensor)
        assert entropy.dim() == 0  # scalar

    def test_evaluate_actions_log_probs_negative(self, policy, obs_tensor, act_tensor):
        _, log_probs, _ = policy.evaluate_actions(obs_tensor, act_tensor)
        # Log-probs should be ≤ 0 for a valid probability distribution
        assert (log_probs <= 0).all(), "Log-probabilities must be non-positive"


# ── backward pass tests ───────────────────────────────────────────────────────

@pytest.mark.unit_test
class TestCompressorPolicyBackward:
    def test_value_loss_backward_populates_gradients(self, policy, obs_tensor):
        """Value network backward pass: gradients must flow to value net params."""
        policy.optimizer.zero_grad()
        _, values, _ = policy.forward(obs_tensor, deterministic=True)
        loss = values.mean()
        loss.backward()

        # Check that value net parameters received gradients
        value_params = list(policy._value_net.parameters())
        grads_populated = [p.grad is not None and p.grad.abs().sum() > 0 for p in value_params]
        assert any(grads_populated), "Value net should have non-zero gradients after backward()"

    def test_evaluate_actions_backward_populates_compressor_params(
        self, policy, obs_tensor, act_tensor
    ):
        """Policy loss backward: gradients must flow into compressor trainable params."""
        policy.optimizer.zero_grad()
        values, log_probs, entropy = policy.evaluate_actions(obs_tensor, act_tensor)

        # Minimal PPO-like surrogate loss
        advantages = torch.ones(BATCH)
        loss = -(log_probs * advantages).mean() + 0.5 * values.mean() - 0.01 * entropy
        loss.backward()

        trainable_params = policy.compressor.get_trainable_parameters()
        assert len(trainable_params) > 0, "Compressor must expose at least one trainable parameter"
        grads_exist = [p.grad is not None for p in trainable_params]
        assert all(grads_exist), (
            "All compressor trainable parameters must have gradients after backward(). "
            f"Missing: {[i for i, ok in enumerate(grads_exist) if not ok]}"
        )

    def test_optimizer_step_changes_params(self, policy, obs_tensor, act_tensor):
        """Optimizer step must change at least one trainable parameter value."""
        compressor_params = policy.compressor.get_trainable_parameters()
        before = [p.data.clone() for p in compressor_params]

        policy.optimizer.zero_grad()
        values, log_probs, entropy = policy.evaluate_actions(obs_tensor, act_tensor)
        loss = -(log_probs.mean()) + values.mean()
        loss.backward()
        policy.optimizer.step()

        after = [p.data.clone() for p in compressor_params]
        changed = [not torch.equal(b, a) for b, a in zip(before, after)]
        assert any(changed), "At least one parameter must change after an optimizer step"

    def test_predict_values_shape(self, policy, obs_tensor):
        values = policy.predict_values(obs_tensor)
        assert values.shape == (BATCH, 1)
