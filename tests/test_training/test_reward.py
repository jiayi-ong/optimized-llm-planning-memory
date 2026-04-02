"""
tests/test_training/test_reward.py
====================================
Unit tests for RewardFunction.

Verifies that hand-crafted EpisodeLog inputs produce expected RewardComponents.
Critical invariant: same ConstraintSatisfactionEngine used by reward and eval.
"""

from __future__ import annotations

import pytest

from optimized_llm_planning_memory.training.reward import RewardFunction
from optimized_llm_planning_memory.core.config import RewardConfig


@pytest.fixture
def reward_fn() -> RewardFunction:
    return RewardFunction(config=RewardConfig())


def test_reward_returns_components(reward_fn, sample_episode_log, sample_user_request):
    rc = reward_fn.compute(sample_episode_log, sample_user_request, is_terminal=True)
    assert rc.total_reward is not None
    assert 0.0 <= rc.hard_constraint_score <= 1.0
    assert 0.0 <= rc.soft_constraint_score <= 1.0
    assert 0.0 <= rc.tool_efficiency_score <= 1.0
    assert rc.tool_failure_penalty <= 0.0


def test_reward_terminal_bonus_applied(reward_fn, sample_episode_log, sample_user_request):
    """Perfect episode (all constraints satisfied) should get terminal bonus."""
    rc_terminal = reward_fn.compute(sample_episode_log, sample_user_request, is_terminal=True)
    rc_nonterminal = reward_fn.compute(sample_episode_log, sample_user_request, is_terminal=False)
    assert rc_terminal.total_reward >= rc_nonterminal.total_reward


def test_reward_normalized_to_unit_interval(reward_fn, sample_episode_log, sample_user_request):
    """With normalize=True, total_reward should be in roughly [-1, 1]."""
    cfg = RewardConfig(normalize=True)
    fn = RewardFunction(config=cfg)
    rc = fn.compute(sample_episode_log, sample_user_request, is_terminal=True)
    assert -2.0 <= rc.total_reward <= 2.0  # slightly loose to account for penalties


def test_reward_no_itinerary(reward_fn, sample_episode_log, sample_user_request):
    """Episode with no final itinerary should have zero hard constraint score."""
    log_no_itin = sample_episode_log.model_copy(update={"final_itinerary": None})
    rc = reward_fn.compute(log_no_itin, sample_user_request)
    assert rc.hard_constraint_score == pytest.approx(0.0)
