"""System tests — full episode RAW and COMPRESSOR mode end-to-end (LLM mocked)."""

from __future__ import annotations

import json
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
    "Thought: I have gathered enough information. Action: DONE\n{}"
)

SEARCH_FLIGHT_RESPONSE = _make_llm_response(
    'Thought: I should search for flights. '
    'Action: search_flights({"origin_city_id": "NYC", "destination_city_id": "PAR", "departure_date": "2025-06-01"})'
)

INVALID_TOOL_RESPONSE = _make_llm_response(
    'Thought: Let me try. '
    'Action: search_flights({})'
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


@pytest.mark.system_test
class TestRAWModeEpisode:
    def test_one_step_done_success(self, paris_request, mock_sim):
        agent = _build_agent(mock_sim, mode=AgentMode.RAW, max_steps=5)

        with patch("litellm.completion", return_value=DONE_RESPONSE):
            log = agent.run_episode(paris_request, mock_sim)

        assert log.success is True
        assert log.total_steps >= 1

    def test_two_step_search_then_done(self, paris_request, mock_sim):
        agent = _build_agent(mock_sim, mode=AgentMode.RAW, max_steps=5)

        with patch("litellm.completion", side_effect=[SEARCH_FLIGHT_RESPONSE, DONE_RESPONSE]):
            log = agent.run_episode(paris_request, mock_sim)

        tool_names = [
            step.action.tool_name
            for step in log.trajectory.steps
            if step.action is not None
        ]
        assert "search_flights" in tool_names

    def test_max_steps_exceeded_stops_episode(self, paris_request, mock_sim):
        agent = _build_agent(mock_sim, mode=AgentMode.RAW, max_steps=2)

        with patch("litellm.completion", return_value=SEARCH_FLIGHT_RESPONSE):
            log = agent.run_episode(paris_request, mock_sim)

        assert log.total_steps <= 2

    def test_tool_validation_failure_episode_continues(self, paris_request, mock_sim):
        agent = _build_agent(mock_sim, mode=AgentMode.RAW, max_steps=3)

        with patch("litellm.completion", side_effect=[INVALID_TOOL_RESPONSE, DONE_RESPONSE]):
            log = agent.run_episode(paris_request, mock_sim)

        assert log.total_steps >= 1

    def test_episode_log_serialisable_to_json(self, paris_request, mock_sim):
        agent = _build_agent(mock_sim, mode=AgentMode.RAW, max_steps=3)

        with patch("litellm.completion", return_value=DONE_RESPONSE):
            log = agent.run_episode(paris_request, mock_sim)

        json_str = log.model_dump_json()
        parsed = json.loads(json_str)
        assert "episode_id" in parsed


@pytest.mark.system_test
class TestCompressorModeEpisode:
    def test_compression_fires_at_threshold(self, paris_request, mock_sim, dummy_compressor_cpu):
        agent = _build_agent(
            mock_sim, mode=AgentMode.COMPRESSOR, max_steps=10,
            compressor=dummy_compressor_cpu,
        )

        responses = [SEARCH_FLIGHT_RESPONSE] * 4 + [DONE_RESPONSE]
        with patch("litellm.completion", side_effect=responses):
            log = agent.run_episode(paris_request, mock_sim)

        assert len(log.compressed_states) >= 1

    def test_context_after_compression_contains_headers(self, paris_request, mock_sim, identity_compressor):
        agent = _build_agent(
            mock_sim, mode=AgentMode.COMPRESSOR, max_steps=5,
            compressor=identity_compressor,
        )

        contexts_seen: list[str] = []

        def capture_context(*args, **kwargs):
            msgs = kwargs.get("messages", args[1] if len(args) > 1 else [])
            for m in msgs:
                if isinstance(m, dict):
                    contexts_seen.append(m.get("content", ""))
            return SEARCH_FLIGHT_RESPONSE if len(contexts_seen) < 4 else DONE_RESPONSE

        with patch("litellm.completion", side_effect=capture_context):
            log = agent.run_episode(paris_request, mock_sim)

        compressed_contexts = [c for c in contexts_seen if "[COMPRESSED MEMORY STATE]" in c]
        if log.compressed_states:
            assert len(compressed_contexts) > 0
