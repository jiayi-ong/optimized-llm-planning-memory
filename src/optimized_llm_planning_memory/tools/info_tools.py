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
        "Discover all cities available in this simulation world. "
        "Returns a list of city descriptors — each entry contains: "
        "city_id (use with all other search tools), city_name, description, "
        "vibe_summary, dominant_cuisines, dominant_attraction_categories, "
        "dominant_event_categories. "
        "Call this FIRST before any other tool. "
        "IMPORTANT: if none of the returned city_names match the user's requested "
        "destinations, those cities do not exist in this world — terminate with Action: DONE."
    )
    input_schema = GetAvailableRoutesInput

    def _execute(self, validated_input: GetAvailableRoutesInput) -> Any:
        return self._simulator.get_available_routes()
