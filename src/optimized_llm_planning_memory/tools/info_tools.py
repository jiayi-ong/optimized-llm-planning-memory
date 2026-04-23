"""tools/info_tools.py — GetAvailableRoutes: primary world-discovery tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from optimized_llm_planning_memory.tools.base import BaseTool


class GetAvailableRoutesInput(BaseModel):
    """No parameters needed — returns all routes in the current world."""
    pass


class GetAvailableRoutes(BaseTool):
    """
    Discover which cities exist in this world and which routes connect them.

    This is the FIRST tool to call at the start of any planning episode.
    It returns all flight-connected city pairs with their city IDs and names.
    Use the city IDs with search_flights, search_hotels, search_attractions, etc.
    """

    tool_name = "get_available_routes"
    tool_description = (
        "Return all available flight routes in this world. "
        "ALWAYS call this first to discover valid city IDs and names. "
        "Each result includes: origin_city_id, origin_city_name, "
        "destination_city_id, destination_city_name. "
        "Use city_id values with other search tools."
    )
    input_schema = GetAvailableRoutesInput

    def _execute(self, validated_input: GetAvailableRoutesInput) -> Any:
        return self._simulator.get_available_routes()
