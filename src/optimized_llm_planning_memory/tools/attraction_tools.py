"""tools/attraction_tools.py — SearchAttractions and GetAttractionDetail tool implementations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from optimized_llm_planning_memory.tools.base import BaseTool


class SearchAttractionsInput(BaseModel):
    city_id: str = Field(
        description="City ID to search attractions in. Use get_available_routes to find city IDs."
    )
    category: str | None = Field(
        default=None,
        description=(
            "Optional category filter, e.g. 'museum', 'park', 'landmark', "
            "'entertainment', 'shopping', 'nature'. "
            "Pass this when the user has stated a preference to focus results."
        ),
    )
    free_only: bool = Field(
        default=False,
        description="If true, only return free-entry attractions. Use when budget is tight."
    )
    max_results: int = Field(
        default=10, ge=1, le=50,
        description="Maximum number of results to return, sorted by popularity (highest first)."
    )


class GetAttractionDetailInput(BaseModel):
    attraction_id: str = Field(
        description="The attraction_id from a prior search_attractions result."
    )


class SearchAttractions(BaseTool):
    """Search for tourist attractions in a city."""

    tool_name = "search_attractions"
    tool_description = (
        "Search for tourist attractions and points of interest in a city. "
        "Returns up to 10 attractions sorted by popularity (highest first). "
        "Pass 'category' to match user preferences (e.g. 'museum', 'park', 'landmark'). "
        "Use 'free_only=true' when the user has a tight budget. "
        "Returns name, category, ticket price, duration, and popularity."
    )
    input_schema = SearchAttractionsInput

    def _execute(self, validated_input: SearchAttractionsInput) -> Any:
        results = self._simulator.search_attractions(
            city_id=validated_input.city_id,
            category=validated_input.category,
            free_only=validated_input.free_only,
        )
        results.sort(key=lambda r: r.get("popularity_score", 0.0), reverse=True)
        return results[: validated_input.max_results]

    def _generate_error_feedback(self, error: Exception, arguments: dict[str, Any]) -> str:
        return (
            f"Tool 'search_attractions' failed: {error}. "
            f"Check that city_id='{arguments.get('city_id')}' is valid "
            f"(use get_available_routes to verify)."
        )


class GetAttractionDetail(BaseTool):
    """Get full details for a specific attraction including opening hours."""

    tool_name = "get_attraction_detail"
    tool_description = (
        "Get full details for a specific attraction: opening hours, capacity, "
        "ticket price, ratings, and a crowding forecast. "
        "Requires a valid attraction_id from a prior search_attractions call."
    )
    input_schema = GetAttractionDetailInput

    def _execute(self, validated_input: GetAttractionDetailInput) -> Any:
        return self._simulator.get_attraction_detail(
            attraction_id=validated_input.attraction_id
        )
