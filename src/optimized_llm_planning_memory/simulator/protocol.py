"""
simulator/protocol.py
=====================
SimulatorProtocol — structural typing contract for the travel simulator.

Design pattern: Protocol (structural subtyping)
------------------------------------------------
typing.Protocol with @runtime_checkable is used instead of an ABC because:

1. SimulatorAdapter calls the external travel_world library which has its own
   class hierarchy. We cannot force that class to inherit from our ABC.

2. Structural typing lets SimulatorAdapter satisfy the contract just by
   implementing the required methods — no explicit class declaration needed.

3. MockSimulator in tests only needs to implement the methods the test calls.

4. isinstance(obj, SimulatorProtocol) runtime checks work via @runtime_checkable.

All other modules type-hint simulator arguments as SimulatorProtocol.
They never import SimulatorAdapter directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SimulatorProtocol(Protocol):
    """
    Interface contract for the travel simulator backend.

    Any object that implements all methods below satisfies this protocol,
    regardless of its inheritance hierarchy.

    City IDs vs. city names
    -----------------------
    All search/booking methods use city_id (an opaque string identifier from
    the simulator world), not human-readable city names. Use get_available_routes()
    to discover which city IDs exist in a given world and their names.
    """

    # ── Flight methods ────────────────────────────────────────────────────────

    def search_flights(
        self,
        origin_city_id: str,
        destination_city_id: str,
        departure_date: str,
        passengers: int = 1,
    ) -> list[dict]:
        """
        Return available flight options between two cities on a given date.

        Parameters
        ----------
        origin_city_id      : City ID for departure.
        destination_city_id : City ID for arrival.
        departure_date      : ISO 8601 date ('YYYY-MM-DD').
        passengers          : Number of passengers.

        Returns
        -------
        list[dict] — each dict matches schemas.FlightOption.
        """
        ...

    def get_available_routes(self) -> list[dict]:
        """
        Return all city-pair routes with at least one flight edge.

        Primary city-discovery method. Each dict contains:
            origin_city_id, origin_city_name,
            destination_city_id, destination_city_name,
            origin_hub_id, destination_hub_id
        """
        ...

    # ── Hotel methods ─────────────────────────────────────────────────────────

    def search_hotels(
        self,
        city_id: str,
        check_in: str,
        check_out: str,
        guests: int = 1,
        max_price: float | None = None,
        min_stars: float | None = None,
    ) -> list[dict]:
        """
        Return available hotel options in a city for the given stay period.

        Returns
        -------
        list[dict] — each dict matches schemas.HotelOption.
        """
        ...

    def book_hotel(
        self,
        hotel_id: str,
        check_in: str,
        check_out: str,
    ) -> dict:
        """
        Confirm a hotel booking (decrements availability).

        Returns
        -------
        dict matching schemas.BookingConfirmation.
        """
        ...

    def get_hotel_detail(self, hotel_id: str) -> dict:
        """Full hotel record including 30-day availability calendar."""
        ...

    # ── Attraction methods ────────────────────────────────────────────────────

    def search_attractions(
        self,
        city_id: str,
        category: str | None = None,
        free_only: bool = False,
    ) -> list[dict]:
        """
        Return attractions in a city, optionally filtered by category.

        Returns
        -------
        list[dict] — each dict matches schemas.AttractionOption.
        """
        ...

    def get_attraction_detail(self, attraction_id: str) -> dict:
        """Full attraction record including opening hours and capacity."""
        ...

    # ── Restaurant methods ────────────────────────────────────────────────────

    def search_restaurants(
        self,
        city_id: str,
        cuisine: str | None = None,
        max_avg_spend: float | None = None,
    ) -> list[dict]:
        """
        Return restaurants in a city, optionally filtered by cuisine.

        Returns
        -------
        list[dict] — each dict matches schemas.RestaurantOption.
        """
        ...

    # ── Event methods ─────────────────────────────────────────────────────────

    def search_events(
        self,
        city_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        category: str | None = None,
        max_price: float | None = None,
    ) -> list[dict]:
        """
        Return events in a city within a date range.

        Returns
        -------
        list[dict] — each dict matches schemas.EventOption.
        """
        ...

    def book_event(self, event_id: str, quantity: int = 1) -> dict:
        """
        Book tickets for an event (decrements ticket count).

        Returns
        -------
        dict matching schemas.BookingConfirmation.
        """
        ...

    # ── Routing methods ───────────────────────────────────────────────────────

    def plan_route(
        self,
        origin_location_id: str,
        destination_location_id: str,
        departure_datetime: str,
        modes: list[str] | None = None,
        optimize_for: str = "time",
    ) -> list[dict]:
        """
        Plan routes between two locations for each available transport mode.

        Returns
        -------
        list[dict] — one per mode, each matching schemas.RouteOption.
        """
        ...

    # ── World management ──────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> None:
        """
        Reset world state, optionally with a new seed.

        After reset all bookings are cleared and the world is re-initialised.
        Used at episode boundaries in RL training.
        """
        ...

    def get_world_seed(self) -> int:
        """Return the seed used to generate the current world instance."""
        ...
