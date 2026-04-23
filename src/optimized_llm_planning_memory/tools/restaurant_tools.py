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
            "'french', 'mexican', 'thai', 'indian', 'american'."
        ),
    )
    max_avg_spend: float | None = Field(
        default=None, ge=0.0,
        description="Optional maximum average spend per person in USD."
    )


class SearchRestaurants(BaseTool):
    """Search for restaurants in a city."""

    tool_name = "search_restaurants"
    tool_description = (
        "Search for restaurants in a city. "
        "Returns restaurant name, cuisine types, average spend per person, "
        "price tier, Michelin stars, reservation requirement, and ratings. "
        "Optionally filter by cuisine type or max spend."
    )
    input_schema = SearchRestaurantsInput

    def _execute(self, validated_input: SearchRestaurantsInput) -> Any:
        return self._simulator.search_restaurants(
            city_id=validated_input.city_id,
            cuisine=validated_input.cuisine,
            max_avg_spend=validated_input.max_avg_spend,
        )

    def _generate_error_feedback(self, error: Exception, arguments: dict[str, Any]) -> str:
        return (
            f"Tool 'search_restaurants' failed: {error}. "
            f"Check that city_id='{arguments.get('city_id')}' is valid "
            f"(use get_available_routes to verify)."
        )
