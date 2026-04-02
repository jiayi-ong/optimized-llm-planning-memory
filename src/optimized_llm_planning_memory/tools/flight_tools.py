"""
tools/flight_tools.py
=====================
SearchFlights and BookFlight tool implementations.

Each tool subclasses BaseTool. The only responsibility here is:
  1. Define the Pydantic input schema.
  2. Implement ``_execute()`` by calling the appropriate simulator method.

All validation, tracking, event emission, and error handling are handled
by ``BaseTool.call()`` — the Template Method base.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from optimized_llm_planning_memory.tools.base import BaseTool


# ── Input schemas ─────────────────────────────────────────────────────────────

class SearchFlightsInput(BaseModel):
    origin: str = Field(description="Departure city name (must exist in the simulated world).")
    destination: str = Field(description="Arrival city name.")
    date: str = Field(description="Travel date in ISO 8601 format (YYYY-MM-DD).")
    num_passengers: int = Field(default=1, ge=1, le=20, description="Total number of passengers.")


class BookFlightInput(BaseModel):
    flight_id: str = Field(description="The flight_id from a prior SearchFlights result.")
    passenger_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Dict of passenger info (name, passport_no, etc.) for the booking.",
    )


# ── Tool classes ──────────────────────────────────────────────────────────────

class SearchFlights(BaseTool):
    """Search for available flights between two cities on a specific date."""

    tool_name = "search_flights"
    tool_description = (
        "Search for available flights between two cities on a specific date. "
        "Returns a list of flight options with prices, schedules, and availability. "
        "Use this before BookFlight to find valid flight_ids."
    )
    input_schema = SearchFlightsInput

    def _execute(self, validated_input: SearchFlightsInput) -> Any:
        return self._simulator.search_flights(
            origin=validated_input.origin,
            destination=validated_input.destination,
            date=validated_input.date,
            num_passengers=validated_input.num_passengers,
        )

    def _generate_error_feedback(self, error: Exception, arguments: dict[str, Any]) -> str:
        return (
            f"Tool 'search_flights' failed: {error}. "
            f"Try: verify that origin='{arguments.get('origin')}' and "
            f"destination='{arguments.get('destination')}' are valid city names "
            f"in this world (use get_city_info to check), and that date is in YYYY-MM-DD format."
        )


class BookFlight(BaseTool):
    """Confirm a flight booking using a flight_id from a prior search."""

    tool_name = "book_flight"
    tool_description = (
        "Book a specific flight using its flight_id from a prior search_flights call. "
        "Returns a booking confirmation with a booking_ref. "
        "WARNING: bookings are final — do not book without confirming the flight is correct."
    )
    input_schema = BookFlightInput

    def _execute(self, validated_input: BookFlightInput) -> Any:
        return self._simulator.book_flight(
            flight_id=validated_input.flight_id,
            passenger_details=validated_input.passenger_details,
        )

    def _generate_error_feedback(self, error: Exception, arguments: dict[str, Any]) -> str:
        return (
            f"Tool 'book_flight' failed: {error}. "
            f"Try: confirm flight_id='{arguments.get('flight_id')}' exists in a prior "
            f"search_flights result. The flight may no longer be available."
        )
