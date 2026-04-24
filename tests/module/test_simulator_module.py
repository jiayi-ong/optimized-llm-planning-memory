"""Module tests for simulator — MockSimulator multi-step workflows."""

from __future__ import annotations

import pytest


@pytest.mark.module_test
class TestSimulatorFlightWorkflow:
    def test_search_flights_returns_results(self, mock_sim):
        results = mock_sim.search_flights(
            origin="New York",
            destination="Paris",
            date="2025-06-01",
            num_passengers=1,
        )
        assert len(results) >= 1
        for r in results:
            assert "flight_id" in r or "airline" in r

    def test_search_then_book_flight(self, mock_sim):
        results = mock_sim.search_flights(
            origin="New York",
            destination="Paris",
            date="2025-06-01",
            num_passengers=1,
        )
        assert len(results) > 0
        flight_id = results[0]["flight_id"]

        booking = mock_sim.book_flight(flight_id=flight_id, passenger_details={"adults": 1})
        assert "booking_ref" in booking or "status" in booking

    def test_search_two_passengers(self, mock_sim):
        results = mock_sim.search_flights(
            origin="New York",
            destination="Paris",
            date="2025-06-01",
            num_passengers=2,
        )
        assert len(results) >= 1


@pytest.mark.module_test
class TestSimulatorHotelWorkflow:
    def test_search_hotels_returns_results(self, mock_sim):
        results = mock_sim.search_hotels(
            city="Paris",
            check_in="2025-06-01",
            check_out="2025-06-03",
            num_guests=1,
        )
        assert len(results) > 0
        assert "hotel_id" in results[0]

    def test_search_then_book_hotel(self, mock_sim):
        results = mock_sim.search_hotels(
            city="Paris",
            check_in="2025-06-01",
            check_out="2025-06-03",
            num_guests=1,
        )
        assert len(results) > 0
        hotel_id = results[0]["hotel_id"]

        booking = mock_sim.book_hotel(
            hotel_id=hotel_id,
            guest_details={"adults": 1},
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
    def test_search_activities_returns_list(self, mock_sim):
        results = mock_sim.search_activities(city="Paris", date="2025-06-01")
        assert isinstance(results, list)

    def test_city_info_has_expected_keys(self, mock_sim):
        info = mock_sim.get_city_info(city="Paris")
        assert isinstance(info, dict)
