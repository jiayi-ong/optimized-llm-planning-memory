"""Module tests for simulator — MockSimulator multi-step workflows."""

from __future__ import annotations

import pytest


@pytest.mark.module_test
class TestSimulatorFlightWorkflow:
    def test_search_flights_returns_results(self, mock_sim):
        results = mock_sim.search_flights(
            origin_city_id="nyc-001",
            destination_city_id="par-001",
            departure_date="2025-06-01",
            passengers=1,
        )
        assert len(results) >= 1
        for r in results:
            assert "edge_id" in r or "airline" in r

    def test_search_then_select_flight(self, mock_sim):
        results = mock_sim.search_flights(
            origin_city_id="nyc-001",
            destination_city_id="par-001",
            departure_date="2025-06-01",
            passengers=1,
        )
        assert len(results) > 0
        assert "edge_id" in results[0]

    def test_search_two_passengers(self, mock_sim):
        results = mock_sim.search_flights(
            origin_city_id="nyc-001",
            destination_city_id="par-001",
            departure_date="2025-06-01",
            passengers=2,
        )
        assert len(results) >= 1


@pytest.mark.module_test
class TestSimulatorHotelWorkflow:
    def test_search_hotels_returns_results(self, mock_sim):
        results = mock_sim.search_hotels(
            city_id="par-001",
            check_in="2025-06-01",
            check_out="2025-06-03",
            guests=1,
        )
        assert len(results) > 0
        assert "hotel_id" in results[0]

    def test_search_then_book_hotel(self, mock_sim):
        results = mock_sim.search_hotels(
            city_id="par-001",
            check_in="2025-06-01",
            check_out="2025-06-03",
            guests=1,
        )
        assert len(results) > 0
        hotel_id = results[0]["hotel_id"]

        booking = mock_sim.book_hotel(
            hotel_id=hotel_id,
            check_in="2025-06-01",
            check_out="2025-06-03",
        )
        assert "booking_ref" in booking or "status" in booking


@pytest.mark.module_test
class TestSimulatorReset:
    def test_reset_changes_world_seed(self, mock_sim):
        mock_sim.reset(seed=7)
        new_seed = mock_sim.get_world_seed()
        assert new_seed == 7

    def test_reset_does_not_raise(self, mock_sim):
        mock_sim.reset(seed=123)

    def test_get_world_seed_returns_integer(self, mock_sim):
        seed = mock_sim.get_world_seed()
        assert isinstance(seed, int)


@pytest.mark.module_test
class TestSimulatorActivitiesSearch:
    def test_search_attractions_returns_list(self, mock_sim):
        results = mock_sim.search_attractions(city_id="par-001")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_get_available_routes_has_city_ids(self, mock_sim):
        routes = mock_sim.get_available_routes()
        assert isinstance(routes, list)
        assert all("origin_city_id" in r for r in routes)
