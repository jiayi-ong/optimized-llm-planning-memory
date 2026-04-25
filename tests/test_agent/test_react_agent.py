"""
tests/test_agent/test_react_agent.py
=====================================
Unit tests for ReActAgent in RAW mode (no compression, no real LLM calls).

Approach
--------
litellm.completion is patched via unittest.mock.patch so no API key is needed.
MockSimulator from tests/test_integration/mock_simulator.py provides realistic
tool responses without any external process.

Test categories
---------------
Parsing      — _parse_response with various LLM output formats
Tool exec    — _execute_tool with valid/invalid tool names
Compression  — _should_compress in RAW vs COMPRESSOR mode
run_episode  — full episode lifecycle (DONE, tool calls, max steps, tracking)
Tracking     — tool stats are populated correctly in EpisodeLog
ContextBuilder — builds correct sections for RAW mode
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from optimized_llm_planning_memory.agent.context_builder import ContextBuilder
from optimized_llm_planning_memory.agent.modes import AgentMode
from optimized_llm_planning_memory.agent.react_agent import ReActAgent
from optimized_llm_planning_memory.agent.trajectory import Trajectory
from optimized_llm_planning_memory.agent.prompts import get_system_prompt, SYSTEM_PROMPT_V1
from optimized_llm_planning_memory.core.config import AgentConfig
from optimized_llm_planning_memory.core.models import EpisodeLog, ToolCall, ToolResult
from optimized_llm_planning_memory.tools.events import EventBus
from optimized_llm_planning_memory.tools.registry import ToolRegistry
from optimized_llm_planning_memory.tools.tracker import ToolCallTracker

# Import the shared mock simulator
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from test_integration.mock_simulator import MockSimulator, make_test_requests


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_litellm_response(content: str) -> MagicMock:
    """Build a minimal mock that mimics litellm.completion return value."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_agent(
    mode: AgentMode = AgentMode.RAW,
    max_steps: int = 10,
    compress_every_n_steps: int = 5,
    compressor=None,
) -> tuple[ReActAgent, MockSimulator, ToolRegistry]:
    """Construct a ReActAgent with MockSimulator and a real ToolRegistry."""
    simulator = MockSimulator(seed=42)
    tracker = ToolCallTracker()
    event_bus = EventBus()
    registry = ToolRegistry.from_config(
        simulator=simulator, tracker=tracker, event_bus=event_bus
    )
    system_prompt = get_system_prompt("v1")
    config = AgentConfig(
        mode=mode.value,
        llm_model_id="groq/llama3-8b-8192",
        max_steps=max_steps,
        compress_every_n_steps=compress_every_n_steps,
        temperature=0.0,
    )
    context_builder = ContextBuilder(
        system_prompt=system_prompt,
        tool_registry=registry,
        llm_model_id="groq/llama3-8b-8192",
    )
    agent = ReActAgent(
        llm_model_id="groq/llama3-8b-8192",
        tool_registry=registry,
        compressor=compressor,
        context_builder=context_builder,
        config=config,
        mode=mode,
    )
    return agent, simulator, registry


# ══════════════════════════════════════════════════════════════════════════════
# 1. _parse_response tests
# ══════════════════════════════════════════════════════════════════════════════

class TestParseResponse:
    """Tests for ReActAgent._parse_response() with various LLM output formats."""

    @pytest.fixture
    def agent(self):
        a, _, _ = _make_agent()
        return a

    def test_thought_and_action_parsed(self, agent):
        text = (
            'Thought: I need to search for flights to Paris.\n'
            'Action: search_flights({"origin": "NYC", "destination": "Paris", '
            '"date": "2025-06-01", "num_passengers": 1})'
        )
        thought, tool_call = agent._parse_response(text)
        assert "Paris" in thought
        assert tool_call is not None
        assert tool_call.tool_name == "search_flights"
        assert tool_call.arguments["destination"] == "Paris"

    def test_done_signal_returns_none_action(self, agent):
        text = "Thought: Planning is complete.\nAction: DONE"
        thought, tool_call = agent._parse_response(text)
        assert "complete" in thought.lower()
        assert tool_call is None

    def test_done_case_insensitive(self, agent):
        text = "Thought: Done.\nAction: done"
        _, tool_call = agent._parse_response(text)
        assert tool_call is None

    def test_no_action_returns_none(self, agent):
        text = "Thought: Just thinking, no action needed."
        thought, tool_call = agent._parse_response(text)
        assert thought != ""
        assert tool_call is None

    def test_malformed_json_stored_in_raw(self, agent):
        """Bad JSON in arguments should not raise — stored as _raw key."""
        text = "Thought: Testing.\nAction: search_hotels(city: Paris, date: today)"
        _, tool_call = agent._parse_response(text)
        assert tool_call is not None
        assert "_raw" in tool_call.arguments

    def test_multiline_thought_parsed(self, agent):
        text = (
            "Thought: First I need to understand the budget.\n"
            "Then I should search for flights.\n"
            "Action: DONE"
        )
        thought, tool_call = agent._parse_response(text)
        assert "budget" in thought
        assert tool_call is None

    def test_empty_response_returns_empty_thought(self, agent):
        thought, tool_call = agent._parse_response("")
        assert thought == ""
        assert tool_call is None

    def test_action_args_parsed_correctly(self, agent):
        text = (
            'Thought: Booking flight.\n'
            'Action: book_flight({"flight_id": "FL001", "passenger_details": '
            '{"num_passengers": 2}})'
        )
        _, tool_call = agent._parse_response(text)
        assert tool_call.tool_name == "book_flight"
        assert tool_call.arguments["flight_id"] == "FL001"
        assert tool_call.arguments["passenger_details"]["num_passengers"] == 2


# ══════════════════════════════════════════════════════════════════════════════
# 2. _execute_tool tests
# ══════════════════════════════════════════════════════════════════════════════

class TestExecuteTool:

    @pytest.fixture
    def agent_and_registry(self):
        agent, sim, registry = _make_agent()
        return agent, registry

    def test_valid_tool_call_succeeds(self, agent_and_registry):
        agent, registry = agent_and_registry
        tool_call = ToolCall(
            tool_name="search_flights",
            arguments={
                "origin_city_id": "nyc-001",
                "destination_city_id": "par-001",
                "departure_date": "2025-06-01",
                "passengers": 1,
            },
            raw_text="search_flights(...)",
        )
        result = agent._execute_tool(registry, tool_call)
        assert result.success is True
        assert result.tool_name == "search_flights"
        assert result.result is not None

    def test_unknown_tool_returns_error_result(self, agent_and_registry):
        agent, registry = agent_and_registry
        tool_call = ToolCall(
            tool_name="fly_rocket",
            arguments={"destination": "moon"},
            raw_text="fly_rocket(...)",
        )
        result = agent._execute_tool(registry, tool_call)
        assert result.success is False
        assert "fly_rocket" in result.error_message
        assert "Available tools" in result.error_message

    def test_invalid_args_returns_error_result(self, agent_and_registry):
        """Missing required field → validation failure → ToolResult(success=False)."""
        agent, registry = agent_and_registry
        tool_call = ToolCall(
            tool_name="search_flights",
            arguments={"destination": "Paris"},  # missing origin, date, num_passengers
            raw_text="search_flights(...)",
        )
        result = agent._execute_tool(registry, tool_call)
        assert result.success is False
        assert result.error_message is not None


# ══════════════════════════════════════════════════════════════════════════════
# 3. _should_compress tests
# ══════════════════════════════════════════════════════════════════════════════

class TestShouldCompress:

    def test_raw_mode_never_compresses(self):
        agent, _, _ = _make_agent(mode=AgentMode.RAW)
        traj = Trajectory(request_id="r1")
        # Even after many steps, RAW mode should not compress
        for i in range(20):
            assert agent._should_compress(traj, i) is False

    def test_compressor_mode_triggers_at_interval(self):
        from optimized_llm_planning_memory.compressor.dummy_compressor import DummyCompressor
        compressor = DummyCompressor(d_model=16, nhead=2, num_encoder_layers=1,
                                      num_decoder_layers=1, dim_feedforward=32)
        agent, _, _ = _make_agent(
            mode=AgentMode.COMPRESSOR,
            compress_every_n_steps=3,
            compressor=compressor,
        )
        traj = Trajectory(request_id="r1")

        # Steps 0–2: not yet (3 steps since last compression = 0)
        for i in range(3):
            assert agent._should_compress(traj, i) is False
        # Step 3: should compress (3 steps since step 0)
        assert agent._should_compress(traj, 3) is True

    def test_compressor_mode_no_compressor_no_compress(self):
        """COMPRESSOR mode with compressor=None should never compress."""
        agent, _, _ = _make_agent(mode=AgentMode.COMPRESSOR, compressor=None)
        traj = Trajectory(request_id="r1")
        for i in range(20):
            assert agent._should_compress(traj, i) is False

    def test_after_mark_compression_resets_counter(self):
        from optimized_llm_planning_memory.compressor.dummy_compressor import DummyCompressor
        compressor = DummyCompressor(d_model=16, nhead=2, num_encoder_layers=1,
                                      num_decoder_layers=1, dim_feedforward=32)
        agent, _, _ = _make_agent(
            mode=AgentMode.COMPRESSOR,
            compress_every_n_steps=3,
            compressor=compressor,
        )
        traj = Trajectory(request_id="r1")
        # After mark_compression at step 3, last_compressed_step = 3
        # step 4: 4 - 3 = 1 < 3 → no compress
        traj.mark_compression(at_step=3)
        assert agent._should_compress(traj, 4) is False
        # step 6: 6 - 3 = 3 ≥ 3 → compress
        assert agent._should_compress(traj, 6) is True


# ══════════════════════════════════════════════════════════════════════════════
# 4. run_episode tests (mocked LLM)
# ══════════════════════════════════════════════════════════════════════════════

DONE_RESPONSE = "Thought: Planning complete.\nAction: DONE"

SEARCH_THEN_DONE = [
    # Step 0: search flights
    _make_litellm_response(
        'Thought: I need to find flights.\n'
        'Action: search_flights({"origin_city_id": "nyc-001", "destination_city_id": "par-001", '
        '"departure_date": "2025-06-01", "passengers": 1})'
    ),
    # Step 1: done
    _make_litellm_response("Thought: Found flights. Planning done.\nAction: DONE"),
]

BOOK_HOTEL_THEN_DONE = [
    _make_litellm_response(
        'Thought: Search hotels in Paris.\n'
        'Action: search_hotels({"city_id": "par-001", "check_in": "2025-06-01", '
        '"check_out": "2025-06-04", "guests": 1})'
    ),
    _make_litellm_response(
        'Thought: Book the boutique hotel.\n'
        'Action: book_hotel({"hotel_id": "HTL_PAR_BOUTIQUE", '
        '"check_in": "2025-06-01", "check_out": "2025-06-04"})'
    ),
    _make_litellm_response("Thought: Hotel booked. Done.\nAction: DONE"),
]


class TestRunEpisode:

    @pytest.fixture
    def user_request(self):
        return make_test_requests()[0]  # simple Paris request

    def test_done_immediately_returns_episode_log(self, user_request):
        agent, simulator, _ = _make_agent(mode=AgentMode.RAW)
        with patch("litellm.completion", return_value=_make_litellm_response(DONE_RESPONSE)):
            log = agent.run_episode(user_request, simulator)
        assert isinstance(log, EpisodeLog)
        assert log.request_id == user_request.request_id
        assert log.success is True

    def test_episode_log_has_agent_mode(self, user_request):
        agent, simulator, _ = _make_agent(mode=AgentMode.RAW)
        with patch("litellm.completion", return_value=_make_litellm_response(DONE_RESPONSE)):
            log = agent.run_episode(user_request, simulator)
        assert log.agent_mode == AgentMode.RAW.value

    def test_trajectory_step_count(self, user_request):
        agent, simulator, _ = _make_agent(mode=AgentMode.RAW)
        responses = iter(SEARCH_THEN_DONE)
        with patch("litellm.completion", side_effect=lambda **kw: next(responses)):
            log = agent.run_episode(user_request, simulator)
        # 2 LLM calls: search_flights step + DONE step
        assert log.total_steps == 2

    def test_tool_stats_populated_after_tool_call(self, user_request):
        agent, simulator, _ = _make_agent(mode=AgentMode.RAW)
        responses = iter(SEARCH_THEN_DONE)
        with patch("litellm.completion", side_effect=lambda **kw: next(responses)):
            log = agent.run_episode(user_request, simulator)
        tool_names = [s.tool_name for s in log.tool_stats]
        assert "search_flights" in tool_names

    def test_search_flights_recorded_as_success(self, user_request):
        agent, simulator, _ = _make_agent(mode=AgentMode.RAW)
        responses = iter(SEARCH_THEN_DONE)
        with patch("litellm.completion", side_effect=lambda **kw: next(responses)):
            log = agent.run_episode(user_request, simulator)
        stats = {s.tool_name: s for s in log.tool_stats}
        assert stats["search_flights"].success_count == 1
        assert stats["search_flights"].failure_count == 0

    def test_redundant_call_tracking(self, user_request):
        """Calling search_flights twice with the same args should increment redundancy."""
        agent, simulator, _ = _make_agent(mode=AgentMode.RAW)
        same_search = (
            'Thought: Let me search again.\n'
            'Action: search_flights({"origin_city_id": "nyc-001", "destination_city_id": "par-001", '
            '"departure_date": "2025-06-01", "passengers": 1})'
        )
        responses = [
            _make_litellm_response(same_search),
            _make_litellm_response(same_search),
            _make_litellm_response(DONE_RESPONSE),
        ]
        responses_iter = iter(responses)
        with patch("litellm.completion", side_effect=lambda **kw: next(responses_iter)):
            log = agent.run_episode(user_request, simulator)
        stats = {s.tool_name: s for s in log.tool_stats}
        assert stats["search_flights"].redundant_call_count == 1

    def test_multiple_tool_calls_tracked(self, user_request):
        agent, simulator, _ = _make_agent(mode=AgentMode.RAW)
        responses = iter(BOOK_HOTEL_THEN_DONE)
        with patch("litellm.completion", side_effect=lambda **kw: next(responses)):
            log = agent.run_episode(user_request, simulator)
        tool_names = {s.tool_name for s in log.tool_stats}
        assert "search_hotels" in tool_names
        assert "book_hotel" in tool_names

    def test_unknown_tool_sets_failure_in_log(self, user_request):
        """Agent calling a non-existent tool should record it as a failure."""
        agent, simulator, _ = _make_agent(mode=AgentMode.RAW)
        responses = [
            _make_litellm_response(
                'Thought: Try teleportation.\n'
                'Action: teleport({"destination": "Moon"})'
            ),
            _make_litellm_response(DONE_RESPONSE),
        ]
        responses_iter = iter(responses)
        with patch("litellm.completion", side_effect=lambda **kw: next(responses_iter)):
            log = agent.run_episode(user_request, simulator)
        # Episode should succeed overall (agent recovered)
        assert log.success is True

    def test_max_steps_hit_sets_success_false(self, user_request):
        """When the agent never sends DONE and max_steps is hit, success=False."""
        agent, simulator, _ = _make_agent(mode=AgentMode.RAW, max_steps=3)
        infinite_search = _make_litellm_response(
            'Thought: Keep searching.\n'
            'Action: get_city_info({"city": "Paris"})'
        )
        with patch("litellm.completion", return_value=infinite_search):
            log = agent.run_episode(user_request, simulator)
        assert log.success is False

    def test_episode_id_is_unique(self, user_request):
        """Two separate episodes should have distinct episode IDs."""
        agent, simulator, _ = _make_agent(mode=AgentMode.RAW)
        with patch("litellm.completion", return_value=_make_litellm_response(DONE_RESPONSE)):
            log1 = agent.run_episode(user_request, simulator)
            log2 = agent.run_episode(user_request, simulator)
        assert log1.episode_id != log2.episode_id

    def test_raw_mode_no_compressed_states(self, user_request):
        """RAW mode must never produce compressed states."""
        agent, simulator, _ = _make_agent(mode=AgentMode.RAW)
        responses = iter(SEARCH_THEN_DONE)
        with patch("litellm.completion", side_effect=lambda **kw: next(responses)):
            log = agent.run_episode(user_request, simulator)
        assert len(log.compressed_states) == 0

    def test_trajectory_stored_in_episode_log(self, user_request):
        agent, simulator, _ = _make_agent(mode=AgentMode.RAW)
        responses = iter(SEARCH_THEN_DONE)
        with patch("litellm.completion", side_effect=lambda **kw: next(responses)):
            log = agent.run_episode(user_request, simulator)
        assert log.trajectory is not None
        assert log.trajectory.total_steps > 0

    def test_tool_latency_is_nonnegative(self, user_request):
        agent, simulator, _ = _make_agent(mode=AgentMode.RAW)
        responses = iter(SEARCH_THEN_DONE)
        with patch("litellm.completion", side_effect=lambda **kw: next(responses)):
            log = agent.run_episode(user_request, simulator)
        for stat in log.tool_stats:
            assert stat.avg_latency_ms >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. ContextBuilder tests
# ══════════════════════════════════════════════════════════════════════════════

class TestContextBuilder:

    @pytest.fixture
    def builder_and_request(self):
        simulator = MockSimulator(seed=42)
        tracker = ToolCallTracker()
        event_bus = EventBus()
        registry = ToolRegistry.from_config(simulator=simulator, tracker=tracker, event_bus=event_bus)
        builder = ContextBuilder(
            system_prompt=SYSTEM_PROMPT_V1,
            tool_registry=registry,
            llm_model_id="groq/llama3-8b-8192",
        )
        request = make_test_requests()[0]
        return builder, request

    def test_raw_context_contains_system_prompt(self, builder_and_request):
        builder, request = builder_and_request
        traj = Trajectory(request_id=request.request_id)
        context = builder.build(traj, compressed_state=None, mode=AgentMode.RAW, request=request)
        assert "[SYSTEM]" in context
        assert "travel planning" in context.lower()

    def test_raw_context_contains_user_request(self, builder_and_request):
        builder, request = builder_and_request
        traj = Trajectory(request_id=request.request_id)
        context = builder.build(traj, compressed_state=None, mode=AgentMode.RAW, request=request)
        assert "[USER REQUEST]" in context
        assert request.raw_text in context

    def test_raw_context_lists_tools(self, builder_and_request):
        builder, request = builder_and_request
        traj = Trajectory(request_id=request.request_id)
        context = builder.build(traj, compressed_state=None, mode=AgentMode.RAW, request=request)
        assert "[AVAILABLE TOOLS]" in context
        assert "search_flights" in context

    def test_raw_context_empty_trajectory_says_no_steps(self, builder_and_request):
        builder, request = builder_and_request
        traj = Trajectory(request_id=request.request_id)
        context = builder.build(traj, compressed_state=None, mode=AgentMode.RAW, request=request)
        assert "No steps yet" in context or "[CONTEXT]" in context

    def test_compressor_mode_falls_back_to_raw_if_no_state(self, builder_and_request):
        """COMPRESSOR mode with no prior state falls back to raw history."""
        builder, request = builder_and_request
        traj = Trajectory(request_id=request.request_id)
        context = builder.build(traj, compressed_state=None, mode=AgentMode.COMPRESSOR, request=request)
        # Should still work (fallback to raw)
        assert "[SYSTEM]" in context

    def test_prompts_get_system_prompt_v1(self):
        prompt = get_system_prompt("v1")
        assert "travel planning" in prompt.lower()

    def test_prompts_get_system_prompt_unknown_version_raises(self):
        with pytest.raises(ValueError, match="Unknown system prompt version"):
            get_system_prompt("v99")
