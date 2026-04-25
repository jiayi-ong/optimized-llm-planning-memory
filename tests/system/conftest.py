"""
tests/system/conftest.py
========================
System-test-level fixtures for cross-component end-to-end tests.

All LLM calls are mocked via litellm.completion patches.
The MockSimulator is used as-is (tool failures are handled gracefully by BaseTool).
"""

from __future__ import annotations

from unittest.mock import MagicMock
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from optimized_llm_planning_memory.compressor.dummy_compressor import DummyCompressor
from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
from optimized_llm_planning_memory.core.config import AgentConfig, EnvConfig, RewardConfig
from optimized_llm_planning_memory.training.reward import RewardFunction

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from test_integration.mock_simulator import MockSimulator, make_test_requests


def make_litellm_response(content: str) -> MagicMock:
    """Build a minimal mock that mimics litellm.completion return value."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


DONE_RESPONSE = make_litellm_response("Thought: I have gathered enough information.\nAction: DONE\n{}")


@pytest.fixture
def paris_request():
    return make_test_requests()[0]


@pytest.fixture
def rome_request():
    return make_test_requests()[1]


@pytest.fixture
def barcelona_request():
    return make_test_requests()[2]


@pytest.fixture
def all_requests():
    return make_test_requests()


@pytest.fixture
def mock_sim():
    return MockSimulator(seed=42)


@pytest.fixture
def dummy_compressor_cpu() -> DummyCompressor:
    return DummyCompressor(
        d_model=16, nhead=2, num_encoder_layers=1,
        num_decoder_layers=1, dim_feedforward=32,
        max_input_len=64, max_output_len=32, device="cpu",
    )


@pytest.fixture
def identity_compressor() -> IdentityCompressor:
    return IdentityCompressor()


@pytest.fixture
def reward_fn() -> RewardFunction:
    return RewardFunction(config=RewardConfig())


@pytest.fixture
def env_config_small() -> EnvConfig:
    return EnvConfig(max_obs_tokens=64, max_action_tokens=32)
