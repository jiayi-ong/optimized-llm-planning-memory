"""tools/event_tools.py — SearchEvents and BookEvent tool implementations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from optimized_llm_planning_memory.tools.base import BaseTool


class SearchEventsInput(BaseModel):
    city_id: str = Field(
        description="City ID to search events in. Use get_available_routes to find city IDs."
    )
    start_date: str | None = Field(
        default=None,
        description="Start of date range (YYYY-MM-DD, inclusive). Defaults to world's sim_date."
    )
    end_date: str | None = Field(
        default=None,
        description="End of date range (YYYY-MM-DD, inclusive)."
    )
    category: str | None = Field(
        default=None,
        description=(
            "Optional category filter, e.g. 'concert', 'festival', 'sport', "
            "'exhibition', 'theater', 'cultural'."
        ),
    )
    max_price: float | None = Field(
        default=None, ge=0.0,
        description="Optional maximum ticket price in USD."
    )


class BookEventInput(BaseModel):
    event_id: str = Field(
        description="The event_id from a prior search_events result."
    )
    quantity: int = Field(
        default=1, ge=1, le=50,
        description="Number of tickets to book."
    )


class SearchEvents(BaseTool):
    """Search for events in a city within a date range."""

    tool_name = "search_events"
    tool_description = (
        "Search for special events (concerts, festivals, exhibitions, sports) in a city. "
        "Returns event name, category, venue, dates, ticket price, and availability. "
        "Optionally filter by date range, category, or max ticket price."
    )
    input_schema = SearchEventsInput

    def _execute(self, validated_input: SearchEventsInput) -> Any:
        return self._simulator.search_events(
            city_id=validated_input.city_id,
            start_date=validated_input.start_date,
            end_date=validated_input.end_date,
            category=validated_input.category,
            max_price=validated_input.max_price,
        )

    def _generate_error_feedback(self, error: Exception, arguments: dict[str, Any]) -> str:
        return (
            f"Tool 'search_events' failed: {error}. "
            f"Check that city_id='{arguments.get('city_id')}' is valid and that "
            f"date strings are in YYYY-MM-DD format."
        )


class BookEvent(BaseTool):
    """Book tickets for an event using an event_id from a prior search."""

    tool_name = "book_event"
    tool_description = (
        "Book tickets for a specific event using its event_id from search_events. "
        "Returns a booking confirmation with total cost and remaining ticket count. "
        "WARNING: bookings are final and reduce available ticket count."
    )
    input_schema = BookEventInput

    def _execute(self, validated_input: BookEventInput) -> Any:
        return self._simulator.book_event(
            event_id=validated_input.event_id,
            quantity=validated_input.quantity,
        )

    def _generate_error_feedback(self, error: Exception, arguments: dict[str, Any]) -> str:
        return (
            f"Tool 'book_event' failed: {error}. "
            f"Check that event_id='{arguments.get('event_id')}' is from a recent "
            f"search_events result and that enough tickets remain."
        )
