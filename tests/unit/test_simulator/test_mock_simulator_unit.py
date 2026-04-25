"""Unit tests for tests/test_integration/mock_simulator.py — MockSimulator interface."""

from __future__ import annotations

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from test_integration.mock_simulator import MockSimulator


@pytest.mark.unit
class TestMockSimulatorSearchMethods:
    def test_search_flights_returns_list(self):
        sim = MockSimulator(seed=42)
        results = sim.search_flights("nyc-001", "par-001", "2025-06-01", 1)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_flights_result_has_required_keys(self):
        sim = MockSimulator(seed=42)
        results = sim.search_flights("nyc-001", "par-001", "2025-06-01", 2)
        assert all("edge_id" in r for r in results)
        assert all("airline" in r for r in results)
        assert all("price_per_person" in r for r in results)
        assert all("stops" in r for r in results)

    def test_search_hotels_returns_list(self):
        sim = MockSimulator(seed=42)
        results = sim.search_hotels("par-001", "2025-06-01", "2025-06-04", 2)
        assert isinstance(results, list)
        assert len(results) >= 2

    def test_search_hotels_has_hotel_id_and_city_id(self):
        sim = MockSimulator(seed=42)
        results = sim.search_hotels("par-001", "2025-06-01", "2025-06-04", 1)
        for r in results:
            assert "hotel_id" in r
            assert "city_id" in r

    def test_search_attractions_returns_list(self):
        sim = MockSimulator(seed=42)
        results = sim.search_attractions("par-001")
        assert isinstance(results, list)
        assert len(results) >= 2

    def test_get_available_routes_returns_routes(self):
        sim = MockSimulator(seed=42)
        routes = sim.get_available_routes()
        assert isinstance(routes, list)
        assert len(routes) > 0
        assert all("origin_city_id" in r for r in routes)
        assert all("destination_city_id" in r for r in routes)

    def test_search_events_returns_list(self):
        sim = MockSimulator(seed=42)
        events = sim.search_events("par-001", start_date="2025-06-01", end_date="2025-06-30")
        assert isinstance(events, list)
        assert all("event_id" in e for e in events)
        assert all("name" in e for e in events)


@pytest.mark.unit
class TestMockSimulatorBookMethods:
    def test_book_hotel_returns_confirmed(self):
        sim = MockSimulator(seed=42)
        result = sim.book_hotel("HTL_PAR_BOUTIQUE", "2025-06-01", "2025-06-04")
        assert result["status"] == "confirmed"
        assert "booking_ref" in result

    def test_book_event_returns_confirmed(self):
        sim = MockSimulator(seed=42)
        result = sim.book_event("EVT_PAR_FESTIVAL", quantity=1)
        assert result["status"] == "confirmed"
        assert "booking_ref" in result


@pytest.mark.unit
class TestMockSimulatorWorldManagement:
    def test_get_world_seed_returns_seed(self):
        sim = MockSimulator(seed=99)
        assert sim.get_world_seed() == 99

    def test_reset_changes_seed(self):
        sim = MockSimulator(seed=42)
        sim.reset(seed=7)
        assert sim.get_world_seed() == 7

    def test_reset_without_seed_preserves_current(self):
        sim = MockSimulator(seed=42)
        sim.reset()
        assert sim.get_world_seed() == 42
