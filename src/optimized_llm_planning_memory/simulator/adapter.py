"""
simulator/adapter.py
====================
SimulatorAdapter — thin wrapper over the travel_world Python library.

Design principle: Single responsibility
-----------------------------------------
The adapter has exactly one job: instantiate travel_world services and
translate their return types into our Pydantic schemas (simulator/schemas.py).

It intentionally contains:
  - No tracking, usage metrics, or logging (→ tools/tracker.py)
  - No retry logic or error feedback generation (→ tools/base.py)
  - No business logic (→ agent or compressor)

Session management
------------------
travel_world booking methods (HotelService.book, EventService.book_ticket)
require a session_id. The adapter creates one internal session per world
instance and reuses it for all bookings. This keeps the tool interface
clean — tools never need to pass a session_id explicitly.

Parallelism note
----------------
Multiple SimulatorAdapter(seed=N) instances — one per RL rollout worker —
are completely safe. Each creates its own WorldState and session.

Usage
-----
    from optimized_llm_planning_memory.simulator.adapter import SimulatorAdapter

    sim = SimulatorAdapter(seed=42)
    routes = sim.get_available_routes()           # discover city IDs
    flights = sim.search_flights(
        origin_city_id=routes[0]["origin_city_id"],
        destination_city_id=routes[0]["destination_city_id"],
        departure_date="2026-06-01",
    )

To swap the adapter for a mock in tests::

    class MockSimulator:
        def get_available_routes(self): return [...]
        def search_flights(self, ...): return [...]
        # ... implement only the methods your test needs

    # isinstance(MockSimulator(), SimulatorProtocol) → True  (structural subtyping)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from optimized_llm_planning_memory.core.exceptions import SimulatorError
from optimized_llm_planning_memory.simulator.schemas import (
    AttractionOption,
    BookingConfirmation,
    EventOption,
    FlightOption,
    HotelOption,
    RestaurantOption,
    RouteOption,
)


class SimulatorAdapter:
    """
    Wraps the travel_world Python library services.

    All public methods match SimulatorProtocol exactly. Return values are
    validated through our Pydantic schemas for a stable contract downstream.

    Parameters
    ----------
    seed       : World generation seed. Identical seeds → identical worlds.
    worlds_dir : Directory where WorldManager stores world files.
                 Created on first use if it does not exist.
    """

    def __init__(self, seed: int, worlds_dir: str | Path = "./worlds") -> None:
        self._seed = seed
        self._worlds_dir = Path(worlds_dir)
        self._worlds_dir.mkdir(parents=True, exist_ok=True)
        self._init_services(seed)

    def _init_services(self, seed: int) -> None:
        """Instantiate the world and all service objects."""
        try:
            from travel_world.manager.world_manager import WorldManager
            from travel_world.services.attraction_service import AttractionService
            from travel_world.services.event_service import EventService
            from travel_world.services.flight_service import FlightService
            from travel_world.services.hotel_service import HotelService
            from travel_world.services.restaurant_service import RestaurantService
            from travel_world.services.routing_service import RoutingService
            from travel_world.services.session_service import SessionService
        except ImportError as exc:
            raise SimulatorError(
                "travel_world package is not installed. "
                "Install it with: pip install travel_world"
            ) from exc

        try:
            manager = WorldManager(self._worlds_dir)
            self._ws = manager.create_world(seed=seed)
        except Exception as exc:
            raise SimulatorError(
                f"Failed to create travel_world world with seed={seed}: {exc}"
            ) from exc

        self._flights = FlightService(self._ws)
        self._hotels = HotelService(self._ws)
        self._events = EventService(self._ws)
        self._routing = RoutingService(self._ws)
        self._attractions = AttractionService(self._ws)
        self._restaurants = RestaurantService(self._ws)
        self._sessions = SessionService()

        # Create one internal session per world instance
        world_id = getattr(self._ws, "world_id", f"world_{seed}")
        try:
            session = self._sessions.create_session(world_id=world_id)
            self._session_id: str = session.session_id
        except Exception as exc:
            raise SimulatorError(
                f"Failed to create session for world_id={world_id}: {exc}"
            ) from exc

    # ── Flight methods ────────────────────────────────────────────────────────

    def search_flights(
        self,
        origin_city_id: str,
        destination_city_id: str,
        departure_date: str,
        passengers: int = 1,
    ) -> list[dict]:
        """
        Search for available flights between two cities.

        Parameters
        ----------
        origin_city_id      : City ID for departure (from get_available_routes).
        destination_city_id : City ID for arrival.
        departure_date      : ISO 8601 date string ('YYYY-MM-DD').
        passengers          : Number of passengers (default 1).

        Returns
        -------
        list[dict] — each dict matches FlightOption schema.
        """
        try:
            raw = self._flights.search(
                origin_city_id=origin_city_id,
                destination_city_id=destination_city_id,
                departure_date=departure_date,
                passengers=passengers,
            )
            return [FlightOption.model_validate(r).model_dump() for r in raw]
        except Exception as exc:
            raise SimulatorError(f"search_flights failed: {exc}") from exc

    def get_available_routes(self) -> list[dict]:
        """
        Return all city-pair routes that have at least one flight edge, or —
        in single-city worlds — a city descriptor for each city in the world.

        This is the primary city-discovery method. The agent calls this first
        to learn which city_id values exist. In the current travel_world build
        each world contains exactly one city, so the flight-route list is
        empty and we fall back to reading the geo layer directly.

        Returns
        -------
        list[dict], each with keys:
            city_id, city_name, description, vibe_summary,
            dominant_cuisines, dominant_attraction_categories,
            dominant_event_categories
        (For multi-city worlds that have flight edges the dicts additionally
        contain origin_city_id / destination_city_id / hub_id fields.)
        """
        try:
            routes = self._flights.get_available_routes()
            if routes:
                return routes
            # Single-city world: expose city info from the geo layer so the
            # agent can discover the city_id it should use for tool calls.
            geo = self._ws.get_layer("geo")
            result = []
            for cid, city in geo.cities.items():
                result.append({
                    "city_id": cid,
                    "city_name": getattr(city, "name", cid),
                    "description": getattr(city, "description", ""),
                    "vibe_summary": getattr(city, "vibe_summary", ""),
                    "dominant_cuisines": list(getattr(city, "dominant_cuisines", [])),
                    "dominant_attraction_categories": list(
                        getattr(city, "dominant_attraction_categories", [])
                    ),
                    "dominant_event_categories": list(
                        getattr(city, "dominant_event_categories", [])
                    ),
                })
            return result
        except Exception as exc:
            raise SimulatorError(f"get_available_routes failed: {exc}") from exc

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
        Search for available hotels in a city.

        Parameters
        ----------
        city_id   : City ID (from get_available_routes).
        check_in  : ISO 8601 date ('YYYY-MM-DD').
        check_out : ISO 8601 date ('YYYY-MM-DD').
        guests    : Number of guests (default 1).
        max_price : Optional max price per night (USD).
        min_stars : Optional minimum star rating (0.0–5.0).

        Returns
        -------
        list[dict] — each dict matches HotelOption schema.
        """
        try:
            raw = self._hotels.search(
                city_id=city_id,
                check_in=check_in,
                check_out=check_out,
                guests=guests,
                max_price_per_night=max_price,
                min_stars=min_stars,
            )
            return [HotelOption.model_validate(r).model_dump() for r in raw]
        except Exception as exc:
            raise SimulatorError(f"search_hotels failed: {exc}") from exc

    def book_hotel(
        self,
        hotel_id: str,
        check_in: str,
        check_out: str,
    ) -> dict:
        """
        Confirm a hotel booking (decrements availability).

        Parameters
        ----------
        hotel_id  : Hotel ID from a prior search_hotels result.
        check_in  : ISO 8601 date ('YYYY-MM-DD').
        check_out : ISO 8601 date ('YYYY-MM-DD').

        Returns
        -------
        dict matching BookingConfirmation schema.

        Raises
        ------
        SimulatorError if the hotel is unavailable for the dates.
        """
        try:
            raw = self._hotels.book(
                hotel_id=hotel_id,
                check_in=check_in,
                check_out=check_out,
                session_id=self._session_id,
            )
            return BookingConfirmation.model_validate(raw).model_dump()
        except Exception as exc:
            raise SimulatorError(f"book_hotel failed: {exc}") from exc

    def get_hotel_detail(self, hotel_id: str) -> dict:
        """Full hotel record + 30-day availability summary."""
        try:
            return self._hotels.get_hotel_detail(hotel_id=hotel_id)
        except Exception as exc:
            raise SimulatorError(f"get_hotel_detail failed: {exc}") from exc

    # ── Attraction methods ────────────────────────────────────────────────────

    def search_attractions(
        self,
        city_id: str,
        category: str | None = None,
        free_only: bool = False,
    ) -> list[dict]:
        """
        Search for attractions in a city.

        Parameters
        ----------
        city_id   : City ID.
        category  : Optional AttractionCategory string filter.
        free_only : If True, only return free-entry attractions.

        Returns
        -------
        list[dict] — each dict matches AttractionOption schema.
        """
        try:
            raw = self._attractions.search(
                city_id=city_id,
                category=category,
                free_only=free_only,
            )
            return [AttractionOption.model_validate(r).model_dump() for r in raw]
        except Exception as exc:
            raise SimulatorError(f"search_attractions failed: {exc}") from exc

    def get_attraction_detail(self, attraction_id: str) -> dict:
        """Full attraction record including opening hours and capacity."""
        try:
            return self._attractions.get_detail(attraction_id=attraction_id)
        except Exception as exc:
            raise SimulatorError(f"get_attraction_detail failed: {exc}") from exc

    # ── Restaurant methods ────────────────────────────────────────────────────

    def search_restaurants(
        self,
        city_id: str,
        cuisine: str | None = None,
        max_avg_spend: float | None = None,
    ) -> list[dict]:
        """
        Search for restaurants in a city.

        Parameters
        ----------
        city_id       : City ID.
        cuisine       : Optional cuisine type string (e.g. 'italian').
        max_avg_spend : Optional max average spend per person (USD).

        Returns
        -------
        list[dict] — each dict matches RestaurantOption schema.
        """
        try:
            raw = self._restaurants.search(
                city_id=city_id,
                cuisine=cuisine,
                max_avg_spend=max_avg_spend,
            )
            return [RestaurantOption.model_validate(r).model_dump() for r in raw]
        except Exception as exc:
            raise SimulatorError(f"search_restaurants failed: {exc}") from exc

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
        Search for events in a city within a date range.

        Returns
        -------
        list[dict] — each dict matches EventOption schema.
        """
        try:
            raw = self._events.search(
                city_id=city_id,
                start_date=start_date,
                end_date=end_date,
                category=category,
                max_price=max_price,
            )
            return [EventOption.model_validate(r).model_dump() for r in raw]
        except Exception as exc:
            raise SimulatorError(f"search_events failed: {exc}") from exc

    def book_event(self, event_id: str, quantity: int = 1) -> dict:
        """
        Book tickets for an event (decrements ticket count).

        Returns
        -------
        dict matching BookingConfirmation schema.

        Raises
        ------
        SimulatorError if insufficient tickets remain.
        """
        try:
            raw = self._events.book_ticket(
                event_id=event_id,
                quantity=quantity,
                session_id=self._session_id,
            )
            return BookingConfirmation.model_validate(raw).model_dump()
        except Exception as exc:
            raise SimulatorError(f"book_event failed: {exc}") from exc

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
        Plan routes between two locations using one or more transport modes.

        Parameters
        ----------
        origin_location_id      : Location node ID (hub, venue, etc.).
        destination_location_id : Destination location node ID.
        departure_datetime      : ISO 8601 datetime string.
        modes                   : List of TransportMode strings; None = all modes.
        optimize_for            : 'time' | 'cost' | 'balanced'.

        Returns
        -------
        list[dict] — one entry per available transport mode.
        """
        try:
            raw = self._routing.plan(
                origin_location_id=origin_location_id,
                destination_location_id=destination_location_id,
                departure_datetime=departure_datetime,
                modes=modes,
                optimize_for=optimize_for,
            )
            return [RouteOption.model_validate(r).model_dump() for r in raw]
        except Exception as exc:
            raise SimulatorError(f"plan_route failed: {exc}") from exc

    # ── World management ──────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> None:
        """Re-initialise the world and session, optionally with a new seed."""
        self._seed = seed if seed is not None else self._seed
        self._init_services(self._seed)

    def get_world_seed(self) -> int:
        return self._seed
