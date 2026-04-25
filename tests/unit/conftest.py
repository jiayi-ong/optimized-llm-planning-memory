"""
tests/unit/conftest.py
======================
Unit-test-level fixtures that extend the root tests/conftest.py.

All fixtures here are fast (< 100ms) and use no real I/O, LLM calls,
or external services. External deps (litellm, simulator) are always mocked.

Root fixtures (sample_user_request, sample_itinerary, sample_episode_log,
mock_simulator) are automatically inherited via pytest fixture discovery.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from optimized_llm_planning_memory.tools.events import EventBus
from optimized_llm_planning_memory.tools.registry import ToolRegistry
from optimized_llm_planning_memory.tools.tracker import ToolCallTracker


# ── Shared LLM mock helper ────────────────────────────────────────────────────

def make_litellm_response(content: str) -> MagicMock:
    """Build a minimal mock that mimics litellm.completion return value."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── Tool infrastructure fixtures ──────────────────────────────────────────────

@pytest.fixture
def fresh_tracker() -> ToolCallTracker:
    return ToolCallTracker()


@pytest.fixture
def fresh_event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def mock_sim_protocol() -> MagicMock:
    """MagicMock satisfying SimulatorProtocol with sensible return values.

    Uses MagicMock (not MockSimulator) so keyword arg names match the protocol.
    """
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
    sim.get_hotel_detail.return_value = {"hotel_id": "HTL001", "name": "Test Hotel"}
    sim.search_attractions.return_value = [
        {"attraction_id": "ATT001", "name": "City Museum", "category": "culture"}
    ]
    sim.get_attraction_detail.return_value = {"attraction_id": "ATT001"}
    sim.search_restaurants.return_value = [
        {"restaurant_id": "REST001", "name": "Cafe Bistro", "cuisine": "french"}
    ]
    sim.search_events.return_value = [
        {"event_id": "EVT001", "name": "Summer Festival", "cost": 0.0}
    ]
    sim.book_event.return_value = {"booking_ref": "EVT-REF-001", "status": "confirmed"}
    sim.plan_route.return_value = [
        {"mode": "walk", "duration_minutes": 15, "distance_km": 1.2}
    ]
    sim.get_world_seed.return_value = 42
    return sim


@pytest.fixture
def fresh_registry(mock_sim_protocol, fresh_tracker, fresh_event_bus) -> ToolRegistry:
    """ToolRegistry populated from config using the mock simulator."""
    return ToolRegistry.from_config(
        simulator=mock_sim_protocol,
        tracker=fresh_tracker,
        event_bus=fresh_event_bus,
    )
