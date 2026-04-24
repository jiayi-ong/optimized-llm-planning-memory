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
        results = sim.search_flights("NYC", "PAR", "2025-06-01", 1)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_flights_result_has_required_keys(self):
        sim = MockSimulator(seed=42)
        results = sim.search_flights("NYC", "PAR", "2025-06-01", 2)
        assert all("flight_id" in r for r in results)
        assert all("airline" in r for r in results)
        assert all("price_per_person" in r for r in results)
        assert all("stops" in r for r in results)

    def test_search_hotels_returns_list(self):
        sim = MockSimulator(seed=42)
        results = sim.search_hotels("Paris", "2025-06-01", "2025-06-04", 2)
        assert isinstance(results, list)
        assert len(results) >= 2

    def test_search_hotels_has_hotel_id_and_city(self):
        sim = MockSimulator(seed=42)
        results = sim.search_hotels("Paris", "2025-06-01", "2025-06-04", 1)
        for r in results:
            assert "hotel_id" in r
            assert "city" in r

    def test_search_activities_returns_three(self):
        sim = MockSimulator(seed=42)
        results = sim.search_activities("Paris", "2025-06-01")
        assert len(results) == 3

    def test_get_city_info_returns_dict(self):
        sim = MockSimulator(seed=42)
        info = sim.get_city_info("Paris")
        assert isinstance(info, dict)
        assert "city" in info
        assert "country" in info

    def test_get_events_returns_list(self):
        sim = MockSimulator(seed=42)
        events = sim.get_events("Paris", "2025-06-01", "2025-06-30")
        assert isinstance(events, list)
        assert all("event_id" in e for e in events)
        assert all("name" in e for e in events)


@pytest.mark.unit
class TestMockSimulatorBookMethods:
    def test_book_flight_returns_confirmed(self):
        sim = MockSimulator(seed=42)
        result = sim.book_flight("FL_NYC_PAR_001", {"passenger": "Alice"})
        assert result["status"] == "confirmed"
        assert "booking_ref" in result

    def test_book_hotel_returns_confirmed(self):
        sim = MockSimulator(seed=42)
        result = sim.book_hotel("HTL_PAR_BOUTIQUE", {"check_in": "2025-06-01"})
        assert result["status"] == "confirmed"
        assert "booking_ref" in result

    def test_book_activity_returns_confirmed(self):
        sim = MockSimulator(seed=42)
        result = sim.book_activity("ACT_PAR_MUSEUM", {"participants": 1})
        assert result["status"] == "confirmed"


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
