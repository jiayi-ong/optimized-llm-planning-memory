"""Module tests for agent — ReActAgent multi-step trajectory accumulation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from optimized_llm_planning_memory.agent.context_builder import ContextBuilder
from optimized_llm_planning_memory.agent.modes import AgentMode
from optimized_llm_planning_memory.agent.prompts import get_system_prompt
from optimized_llm_planning_memory.agent.react_agent import ReActAgent
from optimized_llm_planning_memory.core.config import AgentConfig
from optimized_llm_planning_memory.tools.events import EventBus
from optimized_llm_planning_memory.tools.registry import ToolRegistry
from optimized_llm_planning_memory.tools.tracker import ToolCallTracker


def _make_llm_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


DONE_RESPONSE = _make_llm_response(
    "Thought: I have enough information. Action: DONE\n{}"
)

SEARCH_RESPONSE = _make_llm_response(
    'Thought: I should search for flights.\n'
    'Action: search_flights({"origin_city_id": "NYC", "destination_city_id": "PAR", "departure_date": "2025-06-01"})'
)


def _build_agent(mock_sim, mode: AgentMode = AgentMode.RAW, max_steps: int = 5,
                 compressor=None) -> ReActAgent:
    tracker = ToolCallTracker()
    event_bus = EventBus()
    registry = ToolRegistry.from_config(simulator=mock_sim, tracker=tracker, event_bus=event_bus)
    config = AgentConfig(
        mode=mode.value,
        llm_model_id="mock/model",
        max_steps=max_steps,
        compress_every_n_steps=2,
        temperature=0.0,
    )
    context_builder = ContextBuilder(
        system_prompt=get_system_prompt("v1"),
        tool_registry=registry,
    )
    return ReActAgent(
        llm_model_id="mock/model",
        tool_registry=registry,
        compressor=compressor,
        context_builder=context_builder,
        config=config,
        mode=mode,
    )


@pytest.mark.module_test
class TestReActAgentEpisode:
    def test_one_step_done_episode_succeeds(self, paris_request, mock_sim_protocol):
        agent = _build_agent(mock_sim_protocol, mode=AgentMode.RAW, max_steps=5)

        with patch("litellm.completion", return_value=DONE_RESPONSE):
            log = agent.run_episode(paris_request, mock_sim_protocol)

        assert log.success is True
        assert log.total_steps >= 1

    def test_two_step_search_then_done(self, paris_request, mock_sim_protocol):
        agent = _build_agent(mock_sim_protocol, mode=AgentMode.RAW, max_steps=5)

        with patch("litellm.completion", side_effect=[SEARCH_RESPONSE, DONE_RESPONSE]):
            log = agent.run_episode(paris_request, mock_sim_protocol)

        assert log.total_steps >= 1

    def test_max_steps_exceeded_stops_episode(self, paris_request, mock_sim_protocol):
        agent = _build_agent(mock_sim_protocol, mode=AgentMode.RAW, max_steps=2)

        with patch("litellm.completion", return_value=SEARCH_RESPONSE):
            log = agent.run_episode(paris_request, mock_sim_protocol)

        assert log.total_steps <= 2

    def test_trajectory_accumulates_steps(self, paris_request, mock_sim_protocol):
        agent = _build_agent(mock_sim_protocol, mode=AgentMode.RAW, max_steps=5)

        responses = [SEARCH_RESPONSE, SEARCH_RESPONSE, DONE_RESPONSE]
        with patch("litellm.completion", side_effect=responses):
            log = agent.run_episode(paris_request, mock_sim_protocol)

        assert log.total_steps >= 2


@pytest.mark.module_test
class TestReActAgentCompressorMode:
    def test_compressor_mode_fires_at_threshold(self, paris_request, mock_sim_protocol, dummy_compressor):
        agent = _build_agent(
            mock_sim_protocol, mode=AgentMode.COMPRESSOR,
            max_steps=10, compressor=dummy_compressor
        )

        responses = [SEARCH_RESPONSE] * 4 + [DONE_RESPONSE]
        with patch("litellm.completion", side_effect=responses):
            log = agent.run_episode(paris_request, mock_sim_protocol)

        assert len(log.compressed_states) >= 1
