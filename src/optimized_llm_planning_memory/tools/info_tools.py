"""tools/info_tools.py — Read-only world information tools (no bookings)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from optimized_llm_planning_memory.tools.base import BaseTool


class GetCityInfoInput(BaseModel):
    city: str = Field(description="Name of the city to retrieve information for.")


class GetLocationDetailsInput(BaseModel):
    location_id: str = Field(description="The location_id of a specific node in the city graph.")


class GetEventsInput(BaseModel):
    city: str = Field(description="City in which to look for events.")
    start_date: str = Field(description="Start of the date range (YYYY-MM-DD, inclusive).")
    end_date: str = Field(description="End of the date range (YYYY-MM-DD, inclusive).")


class GetCityInfo(BaseTool):
    """Retrieve city metadata: districts, landmarks, transport hubs, connectivity."""

    tool_name = "get_city_info"
    tool_description = (
        "Get general information about a city: its districts, landmark locations, "
        "airport hubs, and the location graph. Use this to understand city layout "
        "before searching for hotels or activities."
    )
    input_schema = GetCityInfoInput

    def _execute(self, validated_input: GetCityInfoInput) -> Any:
        return self._simulator.get_city_info(city=validated_input.city)


class GetLocationDetails(BaseTool):
    """Retrieve detailed attributes for a specific location node."""

    tool_name = "get_location_details"
    tool_description = (
        "Get detailed attributes for a specific location node (hotel, venue, transport hub, etc.). "
        "Includes accessibility features, nearby nodes, and tags. "
        "Requires a valid location_id from a prior get_city_info or search result."
    )
    input_schema = GetLocationDetailsInput

    def _execute(self, validated_input: GetLocationDetailsInput) -> Any:
        return self._simulator.get_location_details(location_id=validated_input.location_id)


class GetEvents(BaseTool):
    """Retrieve special events occurring in a city within a date range."""

    tool_name = "get_events"
    tool_description = (
        "Get special events (festivals, concerts, exhibitions) in a city "
        "within a start–end date range. Use this to find event-based activities "
        "and plan around them."
    )
    input_schema = GetEventsInput

    def _execute(self, validated_input: GetEventsInput) -> Any:
        return self._simulator.get_events(
            city=validated_input.city,
            start_date=validated_input.start_date,
            end_date=validated_input.end_date,
        )
