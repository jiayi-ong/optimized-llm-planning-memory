"""tools/routing_tools.py — PlanRoute tool implementation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from optimized_llm_planning_memory.tools.base import BaseTool


class PlanRouteInput(BaseModel):
    origin_location_id: str = Field(
        description=(
            "Location node ID for the departure point (e.g. a hotel, airport, or venue). "
            "Location IDs appear in hotel, attraction, and event search results."
        )
    )
    destination_location_id: str = Field(
        description="Location node ID for the arrival point."
    )
    departure_datetime: str = Field(
        description="Departure datetime in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)."
    )
    optimize_for: str = Field(
        default="time",
        description="Optimisation objective: 'time' | 'cost' | 'balanced'."
    )


class PlanRoute(BaseTool):
    """
    Plan a route between two location nodes using available transport modes.

    Useful for estimating travel time and cost between an airport and hotel,
    between two attractions, or between districts. Returns one option per
    available transport mode (walking, taxi, transit, etc.).
    """

    tool_name = "plan_route"
    tool_description = (
        "Plan a route between two specific locations within a city or between cities. "
        "Returns one option per available transport mode with total duration, cost, "
        "and distance. Optimise for 'time', 'cost', or 'balanced'. "
        "Use location IDs from hotel, attraction, or event search results."
    )
    input_schema = PlanRouteInput

    def _execute(self, validated_input: PlanRouteInput) -> Any:
        return self._simulator.plan_route(
            origin_location_id=validated_input.origin_location_id,
            destination_location_id=validated_input.destination_location_id,
            departure_datetime=validated_input.departure_datetime,
            optimize_for=validated_input.optimize_for,
        )

    def _generate_error_feedback(self, error: Exception, arguments: dict[str, Any]) -> str:
        return (
            f"Tool 'plan_route' failed: {error}. "
            f"Check that origin_location_id='{arguments.get('origin_location_id')}' and "
            f"destination_location_id='{arguments.get('destination_location_id')}' are valid "
            f"location IDs from a prior search result, and that departure_datetime is "
            f"in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)."
        )
