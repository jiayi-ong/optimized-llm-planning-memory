"""
simulator/adapter.py
====================
SimulatorAdapter — thin Python wrapper over the external travel simulator library.

Design principle: Single responsibility
-----------------------------------------
The adapter has exactly one job: call the external library and translate its
return types into our Pydantic schemas (``simulator/schemas.py``).

It intentionally contains:
  - No tracking, usage metrics, or logging (→ tools/tracker.py)
  - No retry logic or error feedback generation (→ tools/base.py)
  - No business logic (→ agent or compressor)

This makes the adapter trivially testable and easy to update when the
external library changes.

Parallelism note
----------------
The simulator instantiates a world in memory with no global state, so creating
multiple ``SimulatorAdapter(seed=N)`` instances — one per RL rollout worker —
is completely safe. Each instance owns its own world.

Usage
-----
    from optimized_llm_planning_memory.simulator.adapter import SimulatorAdapter

    sim = SimulatorAdapter(seed=42)
    flights = sim.search_flights("Paris", "Rome", "2025-06-10", num_passengers=2)

To swap the adapter for a mock in tests::

    class MockSimulator:
        def search_flights(self, ...): return [{"flight_id": "FL001", ...}]
        # ... implement only the methods your test needs

    # isinstance(MockSimulator(), SimulatorProtocol) → True  (structural subtyping)
"""

from __future__ import annotations

from optimized_llm_planning_memory.core.exceptions import SimulatorError
from optimized_llm_planning_memory.simulator.schemas import (
    ActivityOption,
    BookingConfirmation,
    CityInfo,
    EventOption,
    FlightOption,
    HotelOption,
    LocationDetails,
)


class SimulatorAdapter:
    """
    Wraps the external travel simulator Python library.

    All public methods match ``SimulatorProtocol`` exactly. Return types are
    translated from the library's native types into our ``schemas.py`` Pydantic
    models and then serialised to ``dict`` for JSON-compatibility downstream.

    Parameters
    ----------
    seed : int
        World generation seed. Two adapters with the same seed produce identical
        worlds, enabling reproducible episodes and fair ablation comparisons.
    world_params : dict
        Extra kwargs forwarded to the simulator constructor (e.g., world size,
        number of cities). Sourced from ``SimulatorConfig.world_params``.

    Notes
    -----
    The ``# type: ignore`` comments below mark the import of the external library.
    Replace with the actual import once the library is installed.
    """

    def __init__(self, seed: int, world_params: dict | None = None) -> None:
        self._seed = seed
        self._world_params = world_params or {}
        self._world = self._init_world(seed)

    def _init_world(self, seed: int):  # type: ignore[return]
        """
        Instantiate the external simulator world.

        TODO: Replace the placeholder below with the actual import and
              constructor call once the external library is installed.

              Example::

                  from travel_simulator import TravelWorld
                  return TravelWorld(seed=seed, **self._world_params)
        """
        try:
            # Placeholder — replace with real import
            # from travel_simulator import TravelWorld
            # return TravelWorld(seed=seed, **self._world_params)
            raise ImportError("travel_simulator library not yet installed.")
        except ImportError as exc:
            raise SimulatorError(
                f"Could not initialise the travel simulator world (seed={seed}). "
                f"Is the 'travel_simulator' package installed? Original error: {exc}"
            ) from exc

    # ── Read methods ──────────────────────────────────────────────────────────

    def search_flights(
        self,
        origin: str,
        destination: str,
        date: str,
        num_passengers: int = 1,
    ) -> list[dict]:
        """Delegates to world.search_flights() and validates via FlightOption."""
        raw: list[dict] = self._world.search_flights(  # type: ignore[attr-defined]
            origin=origin,
            destination=destination,
            date=date,
            num_passengers=num_passengers,
        )
        return [FlightOption.model_validate(r).model_dump() for r in raw]

    def search_hotels(
        self,
        city: str,
        check_in: str,
        check_out: str,
        num_guests: int = 1,
    ) -> list[dict]:
        """Delegates to world.search_hotels() and validates via HotelOption."""
        raw: list[dict] = self._world.search_hotels(  # type: ignore[attr-defined]
            city=city,
            check_in=check_in,
            check_out=check_out,
            num_guests=num_guests,
        )
        return [HotelOption.model_validate(r).model_dump() for r in raw]

    def search_activities(
        self,
        city: str,
        date: str,
        category: str | None = None,
    ) -> list[dict]:
        """Delegates to world.search_activities() and validates via ActivityOption."""
        raw: list[dict] = self._world.search_activities(  # type: ignore[attr-defined]
            city=city,
            date=date,
            category=category,
        )
        return [ActivityOption.model_validate(r).model_dump() for r in raw]

    def get_city_info(self, city: str) -> dict:
        raw: dict = self._world.get_city_info(city=city)  # type: ignore[attr-defined]
        return CityInfo.model_validate(raw).model_dump()

    def get_location_details(self, location_id: str) -> dict:
        raw: dict = self._world.get_location_details(  # type: ignore[attr-defined]
            location_id=location_id
        )
        return LocationDetails.model_validate(raw).model_dump()

    def get_events(
        self,
        city: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        raw: list[dict] = self._world.get_events(  # type: ignore[attr-defined]
            city=city,
            start_date=start_date,
            end_date=end_date,
        )
        return [EventOption.model_validate(r).model_dump() for r in raw]

    # ── Write methods ─────────────────────────────────────────────────────────

    def book_flight(self, flight_id: str, passenger_details: dict) -> dict:
        raw: dict = self._world.book_flight(  # type: ignore[attr-defined]
            flight_id=flight_id,
            passenger_details=passenger_details,
        )
        return BookingConfirmation.model_validate(raw).model_dump()

    def book_hotel(self, hotel_id: str, guest_details: dict) -> dict:
        raw: dict = self._world.book_hotel(  # type: ignore[attr-defined]
            hotel_id=hotel_id,
            guest_details=guest_details,
        )
        return BookingConfirmation.model_validate(raw).model_dump()

    def book_activity(self, activity_id: str, participant_details: dict) -> dict:
        raw: dict = self._world.book_activity(  # type: ignore[attr-defined]
            activity_id=activity_id,
            participant_details=participant_details,
        )
        return BookingConfirmation.model_validate(raw).model_dump()

    # ── World management ──────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> None:
        """Re-initialise the world, optionally with a new seed."""
        self._seed = seed if seed is not None else self._seed
        self._world = self._init_world(self._seed)

    def get_world_seed(self) -> int:
        return self._seed
