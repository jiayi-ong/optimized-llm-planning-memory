"""
tools/itinerary_tools.py
========================
Itinerary manipulation meta-tools.

These tools allow the agent to modify its in-flight itinerary without
interacting with the simulator. The actual state change (removal) is handled
by ReActAgent._try_extract_itinerary() — the tool itself only validates the
request and returns a confirmation token.

Why a separate tool rather than an in-agent command?
----------------------------------------------------
Keeping removal as a named tool means it follows the same Thought → Action →
Observation format as every other tool call. This keeps the ReAct loop uniform
and ensures the tracker/event_bus record all itinerary mutations for diagnostics.

Simulator cancellation note
---------------------------
Hotels booked via book_hotel() decrement room inventory in the simulator.
cancel_booking() does NOT call a simulator cancel method (none is defined in
SimulatorProtocol). Room inventory is therefore not restored. This is an
acceptable limitation for the current simulation scope.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from optimized_llm_planning_memory.tools.base import BaseTool


class CancelBookingInput(BaseModel):
    booking_ref: str = Field(
        description=(
            "The booking reference string to cancel. "
            "Use the booking_ref field from a prior select_flight, book_hotel, "
            "or book_event confirmation."
        )
    )


class CancelBooking(BaseTool):
    """
    Remove a confirmed booking from the current itinerary.

    Use this to correct a booking mistake (wrong city, wrong dates, over budget).
    Always cancel before re-booking — the system will reject a duplicate.

    After cancellation, the item is removed from [CURRENT ITINERARY STATE]
    and the cost is recalculated.
    """

    tool_name = "cancel_booking"
    tool_description = (
        "Remove a confirmed item from the current itinerary by its booking_ref. "
        "Use this to fix mistakes — cancel the wrong booking first, then re-book. "
        "Valid booking_refs appear in [CURRENT ITINERARY STATE] and in prior "
        "select_flight / book_hotel / book_event observations."
    )
    input_schema = CancelBookingInput

    def _execute(self, validated_input: CancelBookingInput) -> Any:
        # The tool simply confirms the cancellation request.
        # ReActAgent._try_extract_itinerary() performs the actual removal from
        # the Itinerary object when it detects tool_name == "cancel_booking".
        return {
            "cancelled_booking_ref": validated_input.booking_ref,
            "status": "cancelled",
        }

    def _generate_error_feedback(self, error: Exception, arguments: dict[str, Any]) -> str:
        return (
            f"Tool 'cancel_booking' failed: {error}. "
            f"Ensure booking_ref='{arguments.get('booking_ref')}' matches a "
            f"confirmed booking visible in [CURRENT ITINERARY STATE]."
        )
