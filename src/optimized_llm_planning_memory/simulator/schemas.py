"""
simulator/schemas.py
====================
Pydantic models for the simulator's response shapes.

Purpose
-------
These schemas serve as a versioned contract between ``SimulatorAdapter``
(which calls the external library) and the rest of the system (which
consumes the results).

When the simulator library changes a field name or type, only this file
and ``adapter.py`` need updating — all downstream code that creates
``TransportSegment``, ``AccommodationBooking``, etc., from these schemas
remains stable.

Design note
-----------
These are RESPONSE schemas (what the simulator returns) not the same as the
itinerary models in ``core/models.py`` (what the agent builds). The adapter
maps from simulator response → core model. This separation allows the
simulator's data vocabulary to evolve independently from the itinerary format.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FlightOption(BaseModel):
    """A single flight option returned by search_flights()."""
    model_config = ConfigDict(frozen=True)

    flight_id: str
    airline: str
    origin: str
    destination: str
    departure_datetime: str
    arrival_datetime: str
    duration_minutes: int = Field(ge=0)
    price_usd: float = Field(ge=0.0)
    seats_available: int = Field(ge=0)
    cabin_class: str = Field(default="economy")


class HotelOption(BaseModel):
    """A single hotel option returned by search_hotels()."""
    model_config = ConfigDict(frozen=True)

    hotel_id: str
    hotel_name: str
    city: str
    district: str
    stars: float = Field(ge=0.0, le=5.0)
    price_per_night_usd: float = Field(ge=0.0)
    rooms_available: int = Field(ge=0)
    amenities: list[str] = Field(default_factory=list)
    location_id: str | None = None


class ActivityOption(BaseModel):
    """A single activity option returned by search_activities()."""
    model_config = ConfigDict(frozen=True)

    activity_id: str
    activity_name: str
    category: str
    city: str
    location_id: str
    date: str
    start_time: str
    duration_hours: float = Field(gt=0.0)
    price_usd: float = Field(ge=0.0)
    capacity_remaining: int = Field(ge=0)
    description: str = ""


class EventOption(BaseModel):
    """A special event returned by get_events()."""
    model_config = ConfigDict(frozen=True)

    event_id: str
    event_name: str
    city: str
    location_id: str | None
    start_datetime: str
    end_datetime: str
    category: str
    price_usd: float = Field(ge=0.0)
    description: str = ""


class LocationDetails(BaseModel):
    """Detailed attributes for a location node."""
    model_config = ConfigDict(frozen=True)

    location_id: str
    name: str
    city: str
    district: str
    location_type: str = Field(description="'hotel', 'attraction', 'transport_hub', 'venue', etc.")
    coordinates: tuple[float, float] | None = None
    accessibility_features: list[str] = Field(default_factory=list)
    nearby_location_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class CityInfo(BaseModel):
    """Metadata for a city, including its districts and connectivity graph."""
    model_config = ConfigDict(frozen=True)

    city_name: str
    country: str
    districts: list[str] = Field(default_factory=list)
    population: int | None = None
    timezone: str | None = None
    airport_location_ids: list[str] = Field(default_factory=list)
    landmark_location_ids: list[str] = Field(default_factory=list)


class BookingConfirmation(BaseModel):
    """Returned by any book_*() call on success."""
    model_config = ConfigDict(frozen=True)

    booking_ref: str
    resource_id: str = Field(description="flight_id, hotel_id, or activity_id that was booked.")
    resource_type: str = Field(description="'flight' | 'hotel' | 'activity'")
    total_cost_usd: float = Field(ge=0.0)
    confirmation_datetime: str
    details: dict = Field(default_factory=dict)
