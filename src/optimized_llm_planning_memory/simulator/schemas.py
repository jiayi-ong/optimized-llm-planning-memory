"""
simulator/schemas.py
====================
Pydantic models for the travel_world service response shapes.

Purpose
-------
These schemas serve as a versioned contract between SimulatorAdapter (which
calls the travel_world library) and the rest of the system (which consumes
the results).

When the library changes a field name or type, only this file and adapter.py
need updating — all downstream code that creates Itinerary items from these
schemas remains stable.

Design notes
------------
- All models use ConfigDict(extra="ignore") so they stay forward-compatible
  as travel_world adds new fields in future releases.
- These are RESPONSE schemas (what the simulator returns), not itinerary models
  (which live in core/models.py). The tool layer maps response → core model.
- Field names mirror the travel_world service return keys exactly (e.g.
  FlightService.search() returns dicts with key "edge_id", not "flight_id").
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ── Flight schemas ─────────────────────────────────────────────────────────────

class FlightOption(BaseModel):
    """A single flight option returned by FlightService.search()."""
    model_config = ConfigDict(frozen=True, extra="ignore")

    edge_id: str = Field(description="Unique flight edge identifier — use this as flight_id.")
    airline: str
    flight_number: str = Field(default="")
    origin_city_id: str
    destination_city_id: str
    origin_hub_id: str = Field(default="")
    destination_hub_id: str = Field(default="")
    departure_datetime: str
    arrival_datetime: str
    duration_min: int = Field(ge=0, alias="duration_min", default=0)
    price_per_person: float = Field(ge=0.0, default=0.0)
    total_price: float = Field(ge=0.0, default=0.0)
    passengers: int = Field(ge=1, default=1)
    seats_available: int = Field(ge=0, default=0)
    cabin_class: str = Field(default="economy")
    is_direct: bool = Field(default=True)
    distance_km: float = Field(ge=0.0, default=0.0)
    baggage_included: bool = Field(default=False)
    expected_delay_min: int = Field(ge=0, default=0)

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)


# ── Hotel schemas ──────────────────────────────────────────────────────────────

class HotelOption(BaseModel):
    """A single hotel option returned by HotelService.search()."""
    model_config = ConfigDict(frozen=True, extra="ignore")

    hotel_id: str
    name: str
    city_id: str
    district_name: str = Field(default="")
    star_rating: float = Field(ge=0.0, le=5.0, default=3.0)
    price_per_night: float = Field(ge=0.0, default=0.0)
    total_cost: float = Field(ge=0.0, default=0.0)
    num_nights: int = Field(ge=0, default=0)
    check_in: str = Field(default="")
    check_out: str = Field(default="")
    amenities: list[str] = Field(default_factory=list)
    average_rating: float = Field(ge=0.0, le=5.0, default=3.0)
    review_count: int = Field(ge=0, default=0)
    description: str = Field(default="")
    neighborhood_score: float = Field(ge=0.0, le=1.0, default=0.5)


# ── Attraction schemas ─────────────────────────────────────────────────────────

class AttractionOption(BaseModel):
    """A single attraction returned by AttractionService.search()."""
    model_config = ConfigDict(frozen=True, extra="ignore")

    attraction_id: str
    name: str
    city_id: str
    district_name: str = Field(default="")
    category: str = Field(default="")
    ticket_price: float = Field(ge=0.0, default=0.0)
    free_entry: bool = Field(default=False)
    duration_hours: float = Field(ge=0.0, default=1.0)
    popularity_score: float = Field(ge=0.0, le=1.0, default=0.5)
    crowding_level: float = Field(ge=0.0, le=1.0, default=0.5)
    wait_time_min: int = Field(ge=0, default=0)
    average_rating: float = Field(ge=0.0, le=5.0, default=3.0)
    review_count: int = Field(ge=0, default=0)
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)


# ── Restaurant schemas ─────────────────────────────────────────────────────────

class RestaurantOption(BaseModel):
    """A single restaurant returned by RestaurantService.search()."""
    model_config = ConfigDict(frozen=True, extra="ignore")

    restaurant_id: str
    name: str
    city_id: str
    district_name: str = Field(default="")
    cuisine_types: list[str] = Field(default_factory=list)
    average_spend: float = Field(ge=0.0, default=0.0)
    price_tier: int = Field(ge=1, le=4, default=2)
    price_tier_label: str = Field(default="$$")
    michelin_stars: int = Field(ge=0, le=3, default=0)
    reservation_required: bool = Field(default=False)
    popularity_score: float = Field(ge=0.0, le=1.0, default=0.5)
    average_rating: float = Field(ge=0.0, le=5.0, default=3.0)
    review_count: int = Field(ge=0, default=0)
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)


# ── Event schemas ──────────────────────────────────────────────────────────────

class EventOption(BaseModel):
    """A special event returned by EventService.search()."""
    model_config = ConfigDict(frozen=True, extra="ignore")

    event_id: str
    name: str
    category: str = Field(default="")
    venue_name: str = Field(default="")
    city_id: str
    start_datetime: str
    end_datetime: str = Field(default="")
    base_ticket_price: float = Field(ge=0.0, default=0.0)
    requires_booking: bool = Field(default=False)
    tickets_remaining: int = Field(ge=0, default=100)
    capacity: int = Field(ge=0, default=100)
    popularity: float = Field(ge=0.0, le=1.0, default=0.5)
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)
    average_rating: float = Field(ge=0.0, le=5.0, default=3.0)


# ── Route schemas ──────────────────────────────────────────────────────────────

class RouteOption(BaseModel):
    """A single route option returned by RoutingService.plan()."""
    model_config = ConfigDict(frozen=True, extra="ignore")

    mode: str
    origin_id: str = Field(default="")
    destination_id: str = Field(default="")
    total_duration_min: float = Field(ge=0.0, default=0.0)
    total_cost: float = Field(ge=0.0, default=0.0)
    total_distance_km: float = Field(ge=0.0, default=0.0)
    num_transfers: int = Field(ge=0, default=0)
    optimize_for: str = Field(default="time")


# ── Booking confirmation ───────────────────────────────────────────────────────

class BookingConfirmation(BaseModel):
    """Returned by book_hotel() and book_event() on success."""
    model_config = ConfigDict(frozen=True, extra="ignore")

    booking_id: str = Field(default="")
    # hotel bookings
    hotel_id: str = Field(default="")
    hotel_name: str = Field(default="")
    check_in: str = Field(default="")
    check_out: str = Field(default="")
    num_nights: int = Field(ge=0, default=0)
    # event bookings
    event_id: str = Field(default="")
    event_name: str = Field(default="")
    quantity: int = Field(ge=0, default=0)
    # common
    total_cost: float = Field(ge=0.0, default=0.0)
    price_per_night: float = Field(ge=0.0, default=0.0)
    session_id: str = Field(default="")
    status: str = Field(default="confirmed")
