"""Unit tests for agent/context_builder.py — ContextBuilder."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from optimized_llm_planning_memory.agent.context_builder import ContextBuilder
from optimized_llm_planning_memory.agent.modes import AgentMode
from optimized_llm_planning_memory.agent.trajectory import Trajectory
from optimized_llm_planning_memory.core.models import (
    CompressedState,
    HardConstraintLedger,
    ReActStep,
    UserRequest,
)


def _make_request() -> UserRequest:
    return UserRequest(
        request_id="req-test",
        raw_text="Plan a trip to Paris.",
        origin_city="New York",
        destination_cities=["Paris"],
        start_date="2025-06-01",
        end_date="2025-06-07",
        budget_usd=3000.0,
    )


def _make_step(idx: int) -> ReActStep:
    return ReActStep(
        step_index=idx,
        thought=f"Step {idx} thought",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _make_compressed_state() -> CompressedState:
    ledger = HardConstraintLedger(
        constraints=(), satisfied_ids=(), violated_ids=(), unknown_ids=()
    )
    return CompressedState(
        state_id="cs-001",
        trajectory_id="traj-001",
        step_index=2,
        hard_constraint_ledger=ledger,
        soft_constraints_summary="Budget is fine.",
        decisions_made=["book_flight"],
        open_questions=[],
        key_discoveries=["Cheap flights found"],
        current_itinerary_sketch="Day 1: Fly to Paris.",
        compression_method="identity",
        token_count=10,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_builder(registry: MagicMock | None = None) -> ContextBuilder:
    if registry is None:
        registry = MagicMock()
        registry.list_tools.return_value = [
            {"name": "search_flights", "description": "Search for flights"}
        ]
    return ContextBuilder(
        system_prompt="You are a travel planning assistant.",
        tool_registry=registry,
    )


@pytest.mark.unit
class TestContextBuilderSections:
    def test_build_contains_system_section(self):
        builder = _make_builder()
        traj = Trajectory(request_id="r1")
        context = builder.build(traj, None, AgentMode.RAW, _make_request())
        assert "[SYSTEM]" in context

    def test_build_contains_user_request_section(self):
        builder = _make_builder()
        traj = Trajectory(request_id="r1")
        context = builder.build(traj, None, AgentMode.RAW, _make_request())
        assert "[USER REQUEST]" in context
        assert "Plan a trip to Paris." in context

    def test_build_contains_available_tools_section(self):
        builder = _make_builder()
        traj = Trajectory(request_id="r1")
        context = builder.build(traj, None, AgentMode.RAW, _make_request())
        assert "[AVAILABLE TOOLS]" in context
        assert "search_flights" in context

    def test_build_contains_context_section(self):
        builder = _make_builder()
        traj = Trajectory(request_id="r1")
        context = builder.build(traj, None, AgentMode.RAW, _make_request())
        assert "[CONTEXT]" in context


@pytest.mark.unit
class TestContextBuilderRawMode:
    def test_empty_trajectory_does_not_raise(self):
        builder = _make_builder()
        traj = Trajectory(request_id="r1")
        context = builder.build(traj, None, AgentMode.RAW, _make_request())
        assert isinstance(context, str)
        assert len(context) > 0

    def test_empty_trajectory_has_begin_planning_hint(self):
        builder = _make_builder()
        traj = Trajectory(request_id="r1")
        context = builder.build(traj, None, AgentMode.RAW, _make_request())
        assert "begin planning" in context.lower() or "no steps" in context.lower()

    def test_raw_mode_includes_trajectory_text(self):
        builder = _make_builder()
        traj = Trajectory(request_id="r1")
        traj.add_step(_make_step(0))
        context = builder.build(traj, None, AgentMode.RAW, _make_request())
        assert "Step 0 thought" in context


@pytest.mark.unit
class TestContextBuilderCompressorMode:
    def test_compressor_mode_no_state_falls_back_to_raw(self):
        builder = _make_builder()
        traj = Trajectory(request_id="r1")
        traj.add_step(_make_step(0))
        context = builder.build(traj, None, AgentMode.COMPRESSOR, _make_request())
        assert "Step 0 thought" in context

    def test_compressor_mode_with_state_injects_headers(self):
        builder = _make_builder()
        traj = Trajectory(request_id="r1")
        compressed = _make_compressed_state()
        context = builder.build(traj, compressed, AgentMode.COMPRESSOR, _make_request())
        assert "[COMPRESSED MEMORY STATE]" in context

    def test_compressor_mode_with_recent_steps_shows_recent_section(self):
        builder = _make_builder()
        traj = Trajectory(request_id="r1")
        traj.add_step(_make_step(0))
        traj.add_step(_make_step(1))
        compressed = _make_compressed_state()
        context = builder.build(traj, compressed, AgentMode.COMPRESSOR, _make_request())
        assert "RECENT STEPS" in context


@pytest.mark.unit
class TestContextBuilderLLMSummaryMode:
    def test_llm_summary_no_state_falls_back_to_raw(self):
        builder = _make_builder()
        traj = Trajectory(request_id="r1")
        traj.add_step(_make_step(0))
        context = builder.build(traj, None, AgentMode.LLM_SUMMARY, _make_request())
        assert "Step 0 thought" in context

    def test_llm_summary_with_state_calls_llm(self):
        builder = _make_builder()
        traj = Trajectory(request_id="r1")
        traj.add_step(_make_step(0))
        compressed = _make_compressed_state()

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Summary text"
        with patch("litellm.completion", return_value=mock_resp) as mock_llm:
            context = builder.build(traj, compressed, AgentMode.LLM_SUMMARY, _make_request())
            mock_llm.assert_called_once()
        assert "SUMMARY" in context or "Summary text" in context
