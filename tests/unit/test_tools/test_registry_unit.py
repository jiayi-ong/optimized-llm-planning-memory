"""Unit tests for tools/registry.py — ToolRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from optimized_llm_planning_memory.core.exceptions import ToolNotFoundError
from optimized_llm_planning_memory.tools.events import EventBus
from optimized_llm_planning_memory.tools.registry import ToolRegistry
from optimized_llm_planning_memory.tools.tracker import ToolCallTracker


@pytest.mark.unit
class TestToolRegistryRegister:
    def setup_method(self):
        self.registry = ToolRegistry()
        self.sim = MagicMock()
        self.tracker = ToolCallTracker()
        self.bus = EventBus()

    def _make_tool(self, name: str):
        from pydantic import BaseModel
        from typing import Any
        from optimized_llm_planning_memory.tools.base import BaseTool

        class _Input(BaseModel):
            x: str = "x"

        class _Tool(BaseTool):
            tool_name = name
            tool_description = f"Tool {name}"
            input_schema = _Input

            def _execute(self, validated_input: _Input) -> Any:
                return {}

        return _Tool(simulator=self.sim, tracker=self.tracker, event_bus=self.bus)

    def test_register_and_get(self):
        tool = self._make_tool("tool_a")
        self.registry.register(tool)
        retrieved = self.registry.get("tool_a")
        assert retrieved is tool

    def test_duplicate_name_raises_value_error(self):
        tool = self._make_tool("tool_dup")
        self.registry.register(tool)
        with pytest.raises(ValueError):
            self.registry.register(self._make_tool("tool_dup"))

    def test_deregister_removes_tool(self):
        tool = self._make_tool("tool_del")
        self.registry.register(tool)
        self.registry.deregister("tool_del")
        with pytest.raises(ToolNotFoundError):
            self.registry.get("tool_del")

    def test_len_increases_on_register(self):
        assert len(self.registry) == 0
        self.registry.register(self._make_tool("t1"))
        assert len(self.registry) == 1
        self.registry.register(self._make_tool("t2"))
        assert len(self.registry) == 2

    def test_contains_true_for_registered(self):
        self.registry.register(self._make_tool("present"))
        assert "present" in self.registry

    def test_contains_false_for_unregistered(self):
        assert "absent" not in self.registry


@pytest.mark.unit
class TestToolRegistryGet:
    def test_get_unknown_raises_tool_not_found_error(self, fresh_registry):
        with pytest.raises(ToolNotFoundError) as exc_info:
            fresh_registry.get("does_not_exist")
        assert "does_not_exist" in str(exc_info.value)

    def test_list_tools_returns_all_schemas(self, fresh_registry):
        schemas = fresh_registry.list_tools()
        assert isinstance(schemas, list)
        assert len(schemas) > 0
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema

    def test_tool_names_sorted(self, fresh_registry):
        names = fresh_registry.tool_names()
        assert names == sorted(names)


@pytest.mark.unit
class TestToolRegistryFromConfig:
    def test_from_config_registers_all_tools(self, mock_sim_protocol):
        tracker = ToolCallTracker()
        bus = EventBus()
        registry = ToolRegistry.from_config(
            simulator=mock_sim_protocol, tracker=tracker, event_bus=bus
        )
        assert len(registry) == 13

    def test_from_config_subset(self, mock_sim_protocol):
        tracker = ToolCallTracker()
        bus = EventBus()
        registry = ToolRegistry.from_config(
            simulator=mock_sim_protocol, tracker=tracker, event_bus=bus,
            enabled_tools=["search_flights"],
        )
        assert len(registry) == 1
        assert "search_flights" in registry

    def test_from_config_tool_names_correct(self, mock_sim_protocol):
        tracker = ToolCallTracker()
        bus = EventBus()
        registry = ToolRegistry.from_config(
            simulator=mock_sim_protocol, tracker=tracker, event_bus=bus
        )
        names = registry.tool_names()
        assert "search_flights" in names
        assert "search_hotels" in names
        assert "book_hotel" in names
        assert "get_available_routes" in names
