"""
tests/module/conftest.py
========================
Module-test-level fixtures.

Module tests wire together multiple real classes within one component.
LLM calls are still mocked; simulator uses MagicMock for tool-level tests
and the real MockSimulator for agent-level tests (where tool failures are ok).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
import uuid

import pytest

from optimized_llm_planning_memory.compressor.dummy_compressor import DummyCompressor
from optimized_llm_planning_memory.core.config import AgentConfig, EnvConfig, RewardConfig
from optimized_llm_planning_memory.core.models import (
    EpisodeLog, HardConstraintLedger, Itinerary, RewardComponents, TrajectoryModel,
)
from optimized_llm_planning_memory.mcts.node import MCTSStats  # resolve forward ref
EpisodeLog.model_rebuild()
from optimized_llm_planning_memory.tools.events import EventBus
from optimized_llm_planning_memory.tools.registry import ToolRegistry
from optimized_llm_planning_memory.tools.tracker import ToolCallTracker
from optimized_llm_planning_memory.training.reward import RewardFunction

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from test_integration.mock_simulator import MockSimulator, make_test_requests


def make_litellm_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


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
def mock_sim():
    return MockSimulator(seed=42)


@pytest.fixture
def mock_sim_protocol() -> MagicMock:
    """Protocol-compatible MagicMock simulator for tool module tests."""
    sim = MagicMock()
    sim.search_flights.return_value = [
        {"edge_id": "E001", "airline": "TestAir", "total_price": 300.0, "stops": 0}
    ]
    sim.get_available_routes.return_value = [
        {"origin_city_id": "NYC", "destination_city_id": "PAR",
         "origin_city_name": "New York", "destination_city_name": "Paris"}
    ]
    sim.search_hotels.return_value = [
        {"hotel_id": "HTL001", "name": "Test Hotel", "stars": 3, "price_per_night": 120.0}
    ]
    sim.book_hotel.return_value = {"booking_ref": "HTL-REF-001", "status": "confirmed"}
    sim.get_hotel_detail.return_value = {"hotel_id": "HTL001"}
    sim.search_attractions.return_value = [{"attraction_id": "ATT001"}]
    sim.get_attraction_detail.return_value = {"attraction_id": "ATT001"}
    sim.search_restaurants.return_value = [{"restaurant_id": "REST001"}]
    sim.search_events.return_value = [{"event_id": "EVT001"}]
    sim.book_event.return_value = {"booking_ref": "EVT-REF-001", "status": "confirmed"}
    sim.plan_route.return_value = [{"mode": "walk", "duration_minutes": 15}]
    sim.get_world_seed.return_value = 42
    return sim


@pytest.fixture
def dummy_compressor() -> DummyCompressor:
    return DummyCompressor(
        d_model=16, nhead=2, num_encoder_layers=1,
        num_decoder_layers=1, dim_feedforward=32,
        max_input_len=64, max_output_len=32, device="cpu",
    )


@pytest.fixture
def reward_fn() -> RewardFunction:
    return RewardFunction(config=RewardConfig())


@pytest.fixture
def fresh_tracker() -> ToolCallTracker:
    return ToolCallTracker()


@pytest.fixture
def fresh_event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def fresh_registry(mock_sim_protocol, fresh_tracker, fresh_event_bus) -> ToolRegistry:
    """ToolRegistry populated from config using the mock simulator."""
    return ToolRegistry.from_config(
        simulator=mock_sim_protocol,
        tracker=fresh_tracker,
        event_bus=fresh_event_bus,
    )


@pytest.fixture
def env_config_small():
    from optimized_llm_planning_memory.core.config import EnvConfig
    return EnvConfig(max_obs_tokens=64, max_action_tokens=32)


@pytest.fixture
def minimal_episode_log(sample_itinerary) -> EpisodeLog:
    """EpisodeLog fixture (relies on root conftest sample_itinerary)."""
    traj = TrajectoryModel(
        trajectory_id=str(uuid.uuid4()),
        request_id="test-req-001",
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
        request_id="test-req-001",
        agent_mode="raw",
        trajectory=traj,
        compressed_states=(),
        final_itinerary=sample_itinerary,
        reward_components=reward,
        tool_stats=(),
        total_steps=0,
        success=True,
        config_hash="test",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
