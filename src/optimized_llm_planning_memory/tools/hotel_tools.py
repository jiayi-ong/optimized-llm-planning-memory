"""tools/hotel_tools.py — SearchHotels, BookHotel, and GetHotelDetail tool implementations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from optimized_llm_planning_memory.tools.base import BaseTool


class SearchHotelsInput(BaseModel):
    city_id: str = Field(
        description="City ID in which to search for hotels. Use get_available_routes to find city IDs."
    )
    check_in: str = Field(description="Check-in date (YYYY-MM-DD).")
    check_out: str = Field(description="Check-out date (YYYY-MM-DD).")
    guests: int = Field(default=1, ge=1, le=20, description="Number of guests.")
    max_price_per_night: float | None = Field(
        default=None, ge=0.0,
        description=(
            "Optional maximum price per night in USD. "
            "ALWAYS set this to remaining_budget / num_nights to filter out unaffordable hotels."
        ),
    )
    min_stars: float | None = Field(
        default=None, ge=0.0, le=5.0,
        description="Optional minimum star rating (0.0–5.0). Set if the user has a quality preference."
    )
    max_results: int = Field(
        default=10, ge=1, le=50,
        description="Maximum number of results to return, sorted by price per night (cheapest first)."
    )


class BookHotelInput(BaseModel):
    hotel_id: str = Field(description="The hotel_id from a prior search_hotels result.")
    check_in: str = Field(description="Check-in date (YYYY-MM-DD).")
    check_out: str = Field(description="Check-out date (YYYY-MM-DD).")


class GetHotelDetailInput(BaseModel):
    hotel_id: str = Field(description="The hotel_id to retrieve full details for.")


class SearchHotels(BaseTool):
    """Search for available hotels in a city for a given stay period."""

    tool_name = "search_hotels"
    tool_description = (
        "Search for available hotels in a city for a given check-in/check-out period. "
        "Returns up to 10 hotels sorted by price per night (cheapest first). "
        "ALWAYS pass 'max_price_per_night' = remaining_budget / num_nights to exclude "
        "unaffordable options; pass 'min_stars' if the user has a quality preference. "
        "Requires a city_id from get_available_routes."
    )
    input_schema = SearchHotelsInput

    def _execute(self, validated_input: SearchHotelsInput) -> Any:
        results = self._simulator.search_hotels(
            city_id=validated_input.city_id,
            check_in=validated_input.check_in,
            check_out=validated_input.check_out,
            guests=validated_input.guests,
            max_price=validated_input.max_price_per_night,
            min_stars=validated_input.min_stars,
        )
        results.sort(key=lambda r: r.get("price_per_night", float("inf")))
        return results[: validated_input.max_results]

    def _generate_error_feedback(self, error: Exception, arguments: dict[str, Any]) -> str:
        return (
            f"Tool 'search_hotels' failed: {error}. "
            f"Check that city_id='{arguments.get('city_id')}' is valid "
            f"and that check_in < check_out (both YYYY-MM-DD)."
        )


class BookHotel(BaseTool):
    """Confirm a hotel booking using a hotel_id from a prior search."""

    tool_name = "book_hotel"
    tool_description = (
        "Book a specific hotel using its hotel_id from a prior search_hotels call. "
        "Returns a booking confirmation with total cost. "
        "WARNING: bookings are final and reduce room availability — "
        "only book after confirming the hotel fits the itinerary."
    )
    input_schema = BookHotelInput

    def _execute(self, validated_input: BookHotelInput) -> Any:
        return self._simulator.book_hotel(
            hotel_id=validated_input.hotel_id,
            check_in=validated_input.check_in,
            check_out=validated_input.check_out,
        )

    def _generate_error_feedback(self, error: Exception, arguments: dict[str, Any]) -> str:
        return (
            f"Tool 'book_hotel' failed: {error}. "
            f"Check that hotel_id='{arguments.get('hotel_id')}' is from a recent "
            f"search_hotels result and that the hotel has availability for "
            f"{arguments.get('check_in')} to {arguments.get('check_out')}."
        )


class GetHotelDetail(BaseTool):
    """Retrieve full details for a specific hotel including availability."""

    tool_name = "get_hotel_detail"
    tool_description = (
        "Get full details for a specific hotel: room types, availability calendar, "
        "reviews, and nearby attractions. Requires a valid hotel_id from search_hotels."
    )
    input_schema = GetHotelDetailInput

    def _execute(self, validated_input: GetHotelDetailInput) -> Any:
        return self._simulator.get_hotel_detail(hotel_id=validated_input.hotel_id)
