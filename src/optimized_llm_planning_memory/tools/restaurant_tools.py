"""tools/restaurant_tools.py — SearchRestaurants tool implementation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from optimized_llm_planning_memory.tools.base import BaseTool


class SearchRestaurantsInput(BaseModel):
    city_id: str = Field(
        description="City ID to search restaurants in. Use get_available_routes to find city IDs."
    )
    cuisine: str | None = Field(
        default=None,
        description=(
            "Optional cuisine filter (case-insensitive), e.g. 'italian', 'japanese', "
            "'french', 'mexican', 'thai', 'indian', 'american'. "
            "Pass this when the user has a food preference."
        ),
    )
    max_avg_spend: float | None = Field(
        default=None, ge=0.0,
        description=(
            "Optional maximum average spend per person in USD. "
            "Set based on per-meal budget to exclude expensive options."
        ),
    )
    max_results: int = Field(
        default=10, ge=1, le=50,
        description="Maximum number of results to return, sorted by rating (highest first)."
    )


class SearchRestaurants(BaseTool):
    """Search for restaurants in a city."""

    tool_name = "search_restaurants"
    tool_description = (
        "Search for restaurants in a city. "
        "Returns up to 10 restaurants sorted by average rating (highest first). "
        "ALWAYS pass 'cuisine' when the user has a food preference, and "
        "'max_avg_spend' based on per-meal budget to exclude expensive options. "
        "Returns name, cuisine types, average spend per person, Michelin stars, and rating."
    )
    input_schema = SearchRestaurantsInput

    def _execute(self, validated_input: SearchRestaurantsInput) -> Any:
        results = self._simulator.search_restaurants(
            city_id=validated_input.city_id,
            cuisine=validated_input.cuisine,
            max_avg_spend=validated_input.max_avg_spend,
        )
        results.sort(key=lambda r: r.get("average_rating", 0.0), reverse=True)
        return results[: validated_input.max_results]

    def _generate_error_feedback(self, error: Exception, arguments: dict[str, Any]) -> str:
        return (
            f"Tool 'search_restaurants' failed: {error}. "
            f"Check that city_id='{arguments.get('city_id')}' is valid "
            f"(use get_available_routes to verify)."
        )
