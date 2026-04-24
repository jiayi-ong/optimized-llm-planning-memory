"""Unit tests for tools/base.py — BaseTool template method lifecycle."""

from __future__ import annotations

from unittest.mock import MagicMock, call
from typing import Any

import pytest
from pydantic import BaseModel

from optimized_llm_planning_memory.core.exceptions import ToolExecutionError
from optimized_llm_planning_memory.core.models import ToolResult
from optimized_llm_planning_memory.tools.base import BaseTool
from optimized_llm_planning_memory.tools.events import EventBus, ToolEvent
from optimized_llm_planning_memory.tools.tracker import ToolCallTracker


# ── Minimal concrete stub ─────────────────────────────────────────────────────

class _StubInput(BaseModel):
    city: str


class _StubTool(BaseTool):
    tool_name = "stub_tool"
    tool_description = "A stub tool for testing."
    input_schema = _StubInput

    def _execute(self, validated_input: _StubInput) -> Any:
        return self._simulator.get_city_info(validated_input.city)


class _RaisingTool(BaseTool):
    """Stub tool that raises ToolExecutionError on execute."""
    tool_name = "raising_tool"
    tool_description = "Always raises."
    input_schema = _StubInput

    def _execute(self, validated_input: _StubInput) -> Any:
        raise ToolExecutionError("intentional failure")


class _UnexpectedRaisingTool(BaseTool):
    """Stub tool that raises an unexpected RuntimeError."""
    tool_name = "unexpected_tool"
    tool_description = "Raises RuntimeError."
    input_schema = _StubInput

    def _execute(self, validated_input: _StubInput) -> Any:
        raise RuntimeError("unexpected failure")


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestBaseToolCallSuccess:
    def setup_method(self):
        self.sim = MagicMock()
        self.sim.get_city_info.return_value = {"city": "Paris", "country": "France"}
        self.tracker = ToolCallTracker()
        self.bus = EventBus()
        self.tool = _StubTool(simulator=self.sim, tracker=self.tracker, event_bus=self.bus)

    def test_valid_args_returns_success_result(self):
        result = self.tool.call({"city": "Paris"})
        assert isinstance(result, ToolResult)
        assert result.success is True

    def test_result_has_correct_tool_name(self):
        result = self.tool.call({"city": "Paris"})
        assert result.tool_name == "stub_tool"

    def test_latency_is_nonnegative(self):
        result = self.tool.call({"city": "Paris"})
        assert result.latency_ms >= 0.0

    def test_tracker_records_success(self):
        self.tool.call({"city": "Paris"})
        stats = self.tracker.get_stats()
        assert len(stats) == 1
        assert stats[0].tool_name == "stub_tool"
        assert stats[0].success_count == 1
        assert stats[0].failure_count == 0

    def test_event_bus_receives_event(self):
        received: list[ToolEvent] = []
        self.bus.subscribe("*", received.append)
        self.tool.call({"city": "Paris"})
        assert len(received) == 1
        assert received[0].tool_name == "stub_tool"
        assert received[0].success is True


@pytest.mark.unit
class TestBaseToolCallValidationFailure:
    def setup_method(self):
        self.sim = MagicMock()
        self.tracker = ToolCallTracker()
        self.bus = EventBus()
        self.tool = _StubTool(simulator=self.sim, tracker=self.tracker, event_bus=self.bus)

    def test_missing_required_field_returns_error(self):
        result = self.tool.call({})  # missing 'city'
        assert result.success is False

    def test_error_message_mentions_field(self):
        result = self.tool.call({})
        assert result.error_message is not None
        assert "city" in result.error_message

    def test_tracker_records_failure(self):
        self.tool.call({})
        stats = self.tracker.get_stats()
        assert stats[0].failure_count == 1
        assert stats[0].success_count == 0


@pytest.mark.unit
class TestBaseToolCallExecutionFailure:
    def setup_method(self):
        self.sim = MagicMock()
        self.tracker = ToolCallTracker()
        self.bus = EventBus()

    def test_execute_raises_tool_execution_error(self):
        tool = _RaisingTool(simulator=self.sim, tracker=self.tracker, event_bus=self.bus)
        result = tool.call({"city": "Paris"})
        assert result.success is False
        assert result.error_message is not None

    def test_execute_raises_unexpected_exception(self):
        tool = _UnexpectedRaisingTool(simulator=self.sim, tracker=self.tracker, event_bus=self.bus)
        result = tool.call({"city": "Paris"})
        assert result.success is False
        assert result.error_message is not None
        assert "RuntimeError" in result.error_message or "unexpected" in result.error_message.lower()


@pytest.mark.unit
class TestBaseToolGetSchema:
    def setup_method(self):
        self.sim = MagicMock()
        self.tracker = ToolCallTracker()
        self.bus = EventBus()
        self.tool = _StubTool(simulator=self.sim, tracker=self.tracker, event_bus=self.bus)

    def test_get_schema_has_name(self):
        schema = self.tool.get_schema_for_agent()
        assert schema["name"] == "stub_tool"

    def test_get_schema_has_description(self):
        schema = self.tool.get_schema_for_agent()
        assert "description" in schema
        assert len(schema["description"]) > 0

    def test_get_schema_has_parameters(self):
        schema = self.tool.get_schema_for_agent()
        assert "parameters" in schema
        assert isinstance(schema["parameters"], dict)
