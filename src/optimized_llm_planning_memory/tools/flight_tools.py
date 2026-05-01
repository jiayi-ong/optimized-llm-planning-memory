"""
tools/flight_tools.py
=====================
SearchFlights and SelectFlight tool implementations.

Note on SelectFlight
---------------------
travel_world has no FlightService.book() method — flights are scheduled
routes, not bookable inventory. SelectFlight is therefore a *pseudo-booking*:
it validates the edge_id format and returns a synthetic BookingConfirmation
that the agent can use to record the flight in its itinerary. No simulator
state is mutated.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from optimized_llm_planning_memory.tools.base import BaseTool


# ── Input schemas ─────────────────────────────────────────────────────────────

class SearchFlightsInput(BaseModel):
    origin_city_id: str = Field(
        description="City ID for departure. Obtain from get_available_routes."
    )
    destination_city_id: str = Field(
        description="City ID for arrival. Obtain from get_available_routes."
    )
    departure_date: str = Field(
        description="Travel date in ISO 8601 format (YYYY-MM-DD)."
    )
    passengers: int = Field(
        default=1, ge=1, le=20,
        description="Total number of passengers. Always pass this to match your group size."
    )
    max_results: int = Field(
        default=10, ge=1, le=50,
        description="Maximum number of results to return, sorted by total price (cheapest first)."
    )


class SelectFlightInput(BaseModel):
    edge_id: str = Field(
        description=(
            "The edge_id from a prior search_flights result. "
            "This records the agent's flight selection in the itinerary."
        )
    )
    origin_city_name: str = Field(
        default="",
        description="Human-readable origin city name (for itinerary display)."
    )
    destination_city_name: str = Field(
        default="",
        description="Human-readable destination city name (for itinerary display)."
    )
    departure_datetime: str = Field(
        default="",
        description="Departure datetime string from the search result."
    )
    arrival_datetime: str = Field(
        default="",
        description="Arrival datetime string from the search result."
    )
    total_price: float = Field(
        default=0.0, ge=0.0,
        description="Total flight price in USD from the search result."
    )


# ── Tool classes ──────────────────────────────────────────────────────────────

class SearchFlights(BaseTool):
    """Search for available flights between two cities on a specific date."""

    tool_name = "search_flights"
    tool_description = (
        "Search for available flights between two cities on a specific date. "
        "Requires city IDs — call get_available_routes first to discover them. "
        "Returns up to 10 flights sorted by total price (cheapest first). "
        "ALWAYS pass 'passengers' matching your group size to filter correctly. "
        "Use the edge_id with select_flight to confirm your choice."
    )
    input_schema = SearchFlightsInput

    def _execute(self, validated_input: SearchFlightsInput) -> Any:
        results = self._simulator.search_flights(
            origin_city_id=validated_input.origin_city_id,
            destination_city_id=validated_input.destination_city_id,
            departure_date=validated_input.departure_date,
            passengers=validated_input.passengers,
        )
        results.sort(key=lambda r: r.get("total_price", float("inf")))
        return results[: validated_input.max_results]

    def _generate_error_feedback(self, error: Exception, arguments: dict[str, Any]) -> str:
        return (
            f"Tool 'search_flights' failed: {error}. "
            f"Check that origin_city_id='{arguments.get('origin_city_id')}' and "
            f"destination_city_id='{arguments.get('destination_city_id')}' are valid "
            f"(use get_available_routes to list valid city IDs), and that "
            f"departure_date is in YYYY-MM-DD format."
        )


class SelectFlight(BaseTool):
    """
    Record a flight selection in the trip plan.

    travel_world flights are scheduled routes, not bookable inventory.
    SelectFlight validates the edge_id and creates a booking record without
    mutating simulator state. Always call this after search_flights to
    confirm which flight you are choosing for the itinerary.
    """

    tool_name = "select_flight"
    tool_description = (
        "Confirm a flight selection using an edge_id from a prior search_flights call. "
        "Returns a booking confirmation for itinerary tracking. "
        "NOTE: Flights in this world are scheduled routes — call select_flight to "
        "record your choice; no seats are decremented."
    )
    input_schema = SelectFlightInput

    def _execute(self, validated_input: SelectFlightInput) -> Any:
        # Pseudo-booking: validate format and return synthetic confirmation
        if not validated_input.edge_id:
            raise ValueError("edge_id must not be empty.")
        return {
            "booking_id": f"FLT-{str(uuid.uuid4())[:8].upper()}",
            "edge_id": validated_input.edge_id,
            "origin_city_name": validated_input.origin_city_name,
            "destination_city_name": validated_input.destination_city_name,
            "departure_datetime": validated_input.departure_datetime,
            "arrival_datetime": validated_input.arrival_datetime,
            "total_price": validated_input.total_price,
            "status": "selected",
            "confirmation_datetime": datetime.now(tz=timezone.utc).isoformat(),
        }

    def _generate_error_feedback(self, error: Exception, arguments: dict[str, Any]) -> str:
        return (
            f"Tool 'select_flight' failed: {error}. "
            f"Ensure edge_id='{arguments.get('edge_id')}' was returned by a recent "
            f"search_flights call."
        )
