"""
simulator/protocol.py
=====================
SimulatorProtocol — structural typing contract for the travel simulator.

Design pattern: Protocol (structural subtyping)
------------------------------------------------
``typing.Protocol`` with ``@runtime_checkable`` is used instead of an ABC
because:

1. The external simulator library has its own class hierarchy. We cannot make
   that class inherit from our ABC without monkey-patching.

2. Structural typing lets ``SimulatorAdapter`` satisfy the contract just by
   implementing the required methods — no explicit ``class Foo(SimulatorProtocol)``
   declaration needed.

3. ``MockSimulator`` in tests only needs to implement the methods called by
   the specific test, without inheriting the full ABC.

4. ``isinstance(obj, SimulatorProtocol)`` runtime check works because of
   ``@runtime_checkable``.

All other modules type-hint simulator arguments as ``SimulatorProtocol``.
They never import ``SimulatorAdapter`` directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SimulatorProtocol(Protocol):
    """
    Interface contract for the travel simulator backend.

    Any object that implements all methods below satisfies this protocol,
    regardless of its inheritance hierarchy.

    Method naming follows REST-style verb-noun patterns so that pydantic-ai
    tool names align naturally with the protocol methods.
    """

    # ── Search (read-only) ────────────────────────────────────────────────────

    def search_flights(
        self,
        origin: str,
        destination: str,
        date: str,
        num_passengers: int = 1,
    ) -> list[dict]:
        """
        Return available flight options between two cities on a given date.

        Parameters
        ----------
        origin, destination : City names recognised by the simulator world.
        date                : ISO 8601 date string (e.g., '2025-06-01').
        num_passengers      : Total number of passengers (adults + children).

        Returns
        -------
        list[dict]
            Each dict matches ``schemas.FlightOption``.
        """
        ...

    def search_hotels(
        self,
        city: str,
        check_in: str,
        check_out: str,
        num_guests: int = 1,
    ) -> list[dict]:
        """
        Return available hotel options in a city for the given stay period.

        Returns
        -------
        list[dict]
            Each dict matches ``schemas.HotelOption``.
        """
        ...

    def search_activities(
        self,
        city: str,
        date: str,
        category: str | None = None,
    ) -> list[dict]:
        """
        Return available activities in a city on a given date.

        Parameters
        ----------
        category : Optional filter (e.g., 'outdoor', 'museum', 'food').

        Returns
        -------
        list[dict]
            Each dict matches ``schemas.ActivityOption``.
        """
        ...

    def get_city_info(self, city: str) -> dict:
        """
        Return city metadata: districts, landmark nodes, connectivity graph.

        Returns
        -------
        dict
            Matches ``schemas.CityInfo``.
        """
        ...

    def get_location_details(self, location_id: str) -> dict:
        """
        Return detailed attributes for a specific location node (hotel, venue, etc.).

        Returns
        -------
        dict
            Matches ``schemas.LocationDetails``.
        """
        ...

    def get_events(
        self,
        city: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """
        Return special events occurring in a city within a date range.

        Returns
        -------
        list[dict]
            Each dict matches ``schemas.EventOption``.
        """
        ...

    # ── Booking (write) ───────────────────────────────────────────────────────

    def book_flight(self, flight_id: str, passenger_details: dict) -> dict:
        """
        Confirm a flight booking.

        Returns
        -------
        dict
            Matches ``schemas.BookingConfirmation``.
        """
        ...

    def book_hotel(self, hotel_id: str, guest_details: dict) -> dict:
        """
        Confirm a hotel booking.

        Returns
        -------
        dict
            Matches ``schemas.BookingConfirmation``.
        """
        ...

    def book_activity(self, activity_id: str, participant_details: dict) -> dict:
        """
        Confirm an activity booking.

        Returns
        -------
        dict
            Matches ``schemas.BookingConfirmation``.
        """
        ...

    # ── World management ──────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> None:
        """
        Reset world state, optionally with a new seed.

        After reset, all previously booked resources are cleared and the world
        is re-initialised from scratch. Used at episode boundaries.
        """
        ...

    def get_world_seed(self) -> int:
        """Return the seed used to generate the current world instance."""
        ...
