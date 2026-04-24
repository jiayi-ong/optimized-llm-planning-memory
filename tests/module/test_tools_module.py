"""Module tests for tools — ToolRegistry + concrete tools working together."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from optimized_llm_planning_memory.tools.events import EventBus, ToolEvent
from optimized_llm_planning_memory.tools.registry import ToolRegistry
from optimized_llm_planning_memory.tools.tracker import ToolCallTracker


@pytest.mark.module_test
class TestToolRegistryWithConcreteTools:
    def test_search_flights_call_succeeds(self, fresh_registry):
        result = fresh_registry.get("search_flights").call({
            "origin_city_id": "NYC",
            "destination_city_id": "PAR",
            "departure_date": "2025-06-01",
        })
        assert result.success is True

    def test_invalid_call_returns_structured_error(self, fresh_registry):
        result = fresh_registry.get("search_flights").call({})
        assert result.success is False
        assert result.error_message is not None

    def test_all_expected_tool_names_present(self, fresh_registry):
        names = fresh_registry.tool_names()
        expected = [
            "search_flights", "select_flight",
            "search_hotels", "book_hotel", "get_hotel_detail",
            "search_attractions", "get_attraction_detail",
            "search_restaurants", "search_events", "book_event",
            "plan_route", "get_available_routes",
        ]
        for name in expected:
            assert name in names, f"Expected tool '{name}' not found in registry"

    def test_redundant_calls_tracked(self, mock_sim_protocol):
        tracker = ToolCallTracker()
        bus = EventBus()
        registry = ToolRegistry.from_config(
            simulator=mock_sim_protocol, tracker=tracker, event_bus=bus
        )
        tool = registry.get("search_flights")
        args = {"origin_city_id": "NYC", "destination_city_id": "PAR", "departure_date": "2025-06-01"}
        tool.call(args)
        tool.call(args)  # second identical call = redundant
        stats = tracker.get_stats()
        sf_stats = next(s for s in stats if s.tool_name == "search_flights")
        assert sf_stats.success_count == 2

    def test_event_bus_subscriber_receives_events(self, mock_sim_protocol):
        tracker = ToolCallTracker()
        bus = EventBus()
        registry = ToolRegistry.from_config(
            simulator=mock_sim_protocol, tracker=tracker, event_bus=bus
        )
        received: list[ToolEvent] = []
        bus.subscribe("*", received.append)
        registry.get("search_flights").call({
            "origin_city_id": "NYC",
            "destination_city_id": "PAR",
            "departure_date": "2025-06-01",
        })
        assert len(received) == 1
        assert received[0].tool_name == "search_flights"


@pytest.mark.module_test
class TestToolRegistryFromConfig:
    def test_from_config_subset_enabled_tools(self, mock_sim_protocol):
        tracker = ToolCallTracker()
        bus = EventBus()
        registry = ToolRegistry.from_config(
            simulator=mock_sim_protocol, tracker=tracker, event_bus=bus,
            enabled_tools=["search_flights", "search_hotels"],
        )
        assert len(registry) == 2
        assert "search_flights" in registry
        assert "search_hotels" in registry
        assert "book_hotel" not in registry

    def test_tool_schemas_are_complete(self, fresh_registry):
        schemas = fresh_registry.list_tools()
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
