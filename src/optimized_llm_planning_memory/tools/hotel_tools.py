"""tools/hotel_tools.py — SearchHotels and BookHotel tool implementations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from optimized_llm_planning_memory.tools.base import BaseTool


class SearchHotelsInput(BaseModel):
    city: str = Field(description="City in which to search for hotels.")
    check_in: str = Field(description="Check-in date (YYYY-MM-DD).")
    check_out: str = Field(description="Check-out date (YYYY-MM-DD).")
    num_guests: int = Field(default=1, ge=1, le=20)


class BookHotelInput(BaseModel):
    hotel_id: str = Field(description="The hotel_id from a prior SearchHotels result.")
    guest_details: dict[str, Any] = Field(default_factory=dict)


class SearchHotels(BaseTool):
    """Search for available hotels in a city for a given stay period."""

    tool_name = "search_hotels"
    tool_description = (
        "Search for available hotels in a city for a given check-in/check-out period. "
        "Returns price per night, star rating, district, and amenities."
    )
    input_schema = SearchHotelsInput

    def _execute(self, validated_input: SearchHotelsInput) -> Any:
        return self._simulator.search_hotels(
            city=validated_input.city,
            check_in=validated_input.check_in,
            check_out=validated_input.check_out,
            num_guests=validated_input.num_guests,
        )

    def _generate_error_feedback(self, error: Exception, arguments: dict[str, Any]) -> str:
        return (
            f"Tool 'search_hotels' failed: {error}. "
            f"Try: verify city='{arguments.get('city')}' is valid and that "
            f"check_in < check_out (both YYYY-MM-DD)."
        )


class BookHotel(BaseTool):
    """Confirm a hotel booking using a hotel_id from a prior search."""

    tool_name = "book_hotel"
    tool_description = (
        "Book a specific hotel using its hotel_id from a prior search_hotels call. "
        "Returns a booking confirmation with a booking_ref."
    )
    input_schema = BookHotelInput

    def _execute(self, validated_input: BookHotelInput) -> Any:
        return self._simulator.book_hotel(
            hotel_id=validated_input.hotel_id,
            guest_details=validated_input.guest_details,
        )
