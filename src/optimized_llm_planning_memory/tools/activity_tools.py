"""tools/activity_tools.py — SearchActivities and BookActivity tool implementations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from optimized_llm_planning_memory.tools.base import BaseTool


class SearchActivitiesInput(BaseModel):
    city: str = Field(description="City in which to search for activities.")
    date: str = Field(description="Date of the activity (YYYY-MM-DD).")
    category: str | None = Field(
        default=None,
        description="Optional category filter (e.g., 'outdoor', 'museum', 'food', 'nightlife').",
    )


class BookActivityInput(BaseModel):
    activity_id: str = Field(description="The activity_id from a prior SearchActivities result.")
    participant_details: dict[str, Any] = Field(default_factory=dict)


class SearchActivities(BaseTool):
    """Search for available activities in a city on a specific date."""

    tool_name = "search_activities"
    tool_description = (
        "Search for activities available in a city on a specific date. "
        "Optionally filter by category (outdoor, museum, food, nightlife, etc.). "
        "Returns activity name, duration, price, and available capacity."
    )
    input_schema = SearchActivitiesInput

    def _execute(self, validated_input: SearchActivitiesInput) -> Any:
        return self._simulator.search_activities(
            city=validated_input.city,
            date=validated_input.date,
            category=validated_input.category,
        )


class BookActivity(BaseTool):
    """Confirm an activity booking using an activity_id from a prior search."""

    tool_name = "book_activity"
    tool_description = (
        "Book a specific activity using its activity_id from a prior search_activities call. "
        "Returns a booking confirmation with a booking_ref."
    )
    input_schema = BookActivityInput

    def _execute(self, validated_input: BookActivityInput) -> Any:
        return self._simulator.book_activity(
            activity_id=validated_input.activity_id,
            participant_details=validated_input.participant_details,
        )
