"""
tests/test_integration/mock_simulator.py
==========================================
MockSimulator and canonical test UserRequest objects.

MockSimulator implements ``SimulatorProtocol`` as a concrete Python class
(not a MagicMock) so it can be passed to ``ToolRegistry.from_config()`` and
have its methods called by real ``BaseTool`` subclasses.

The test requests cover three scenario types:
  1. Simple single-city trip within budget.
  2. Multi-city trip with hard budget constraint.
  3. Family trip with accessibility and dietary constraints.
"""

from __future__ import annotations

from optimized_llm_planning_memory.core.models import (
    Constraint,
    ConstraintCategory,
    ConstraintType,
    TravelerProfile,
    UserRequest,
)


# ── MockSimulator ─────────────────────────────────────────────────────────────

class MockSimulator:
    """
    In-memory travel simulator that returns deterministic canned responses.

    Satisfies ``SimulatorProtocol`` as a structural subtype.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    # ── Search methods ─────────────────────────────────────────────────────────

    def search_flights(
        self,
        origin: str,
        destination: str,
        date: str,
        num_passengers: int,
    ) -> list[dict]:
        return [
            {
                "flight_id": f"FL_{origin[:3].upper()}_{destination[:3].upper()}_001",
                "airline": "MockAir",
                "origin": origin,
                "destination": destination,
                "departure_datetime": f"{date}T08:00:00",
                "arrival_datetime": f"{date}T14:00:00",
                "price_per_person": 350.0,
                "total_price": 350.0 * num_passengers,
                "stops": 0,
                "duration_hours": 6.0,
            },
            {
                "flight_id": f"FL_{origin[:3].upper()}_{destination[:3].upper()}_002",
                "airline": "BudgetFly",
                "origin": origin,
                "destination": destination,
                "departure_datetime": f"{date}T14:00:00",
                "arrival_datetime": f"{date}T20:30:00",
                "price_per_person": 220.0,
                "total_price": 220.0 * num_passengers,
                "stops": 1,
                "duration_hours": 6.5,
            },
        ]

    def search_hotels(
        self,
        city: str,
        check_in: str,
        check_out: str,
        num_guests: int,
    ) -> list[dict]:
        return [
            {
                "hotel_id": f"HTL_{city[:3].upper()}_BOUTIQUE",
                "name": f"{city} Boutique Inn",
                "city": city,
                "stars": 3,
                "category": "boutique",
                "check_in": check_in,
                "check_out": check_out,
                "price_per_night": 120.0,
                "total_price": 120.0 * 3,  # assume 3 nights
                "amenities": ["wifi", "breakfast"],
            },
            {
                "hotel_id": f"HTL_{city[:3].upper()}_GRAND",
                "name": f"Grand {city} Hotel",
                "city": city,
                "stars": 5,
                "category": "luxury",
                "check_in": check_in,
                "check_out": check_out,
                "price_per_night": 280.0,
                "total_price": 280.0 * 3,
                "amenities": ["wifi", "spa", "pool", "breakfast"],
            },
        ]

    def search_activities(
        self,
        city: str,
        date: str,
        category: str | None = None,
    ) -> list[dict]:
        return [
            {
                "activity_id": f"ACT_{city[:3].upper()}_MUSEUM",
                "name": f"{city} National Museum",
                "city": city,
                "location": f"Central {city}",
                "date": date,
                "category": "culture",
                "duration_hours": 3.0,
                "cost_per_person": 18.0,
                "max_participants": 50,
            },
            {
                "activity_id": f"ACT_{city[:3].upper()}_TOUR",
                "name": f"{city} Walking Tour",
                "city": city,
                "location": f"Old Town {city}",
                "date": date,
                "category": "tour",
                "duration_hours": 2.0,
                "cost_per_person": 12.0,
                "max_participants": 20,
            },
            {
                "activity_id": f"ACT_{city[:3].upper()}_FOOD",
                "name": f"{city} Food Market Tour",
                "city": city,
                "location": f"Market District {city}",
                "date": date,
                "category": "food",
                "duration_hours": 2.5,
                "cost_per_person": 25.0,
                "max_participants": 15,
            },
        ]

    def get_city_info(self, city: str) -> dict:
        return {
            "city": city,
            "country": "TestLand",
            "timezone": "UTC+1",
            "currency": "USD",
            "language": "English",
            "population": 1_500_000,
            "highlights": [f"{city} Old Town", f"{city} Museum", f"{city} Park"],
            "avg_daily_cost_usd": 80.0,
        }

    def get_location_details(self, location_id: str) -> dict:
        return {
            "location_id": location_id,
            "name": f"Location {location_id}",
            "type": "point_of_interest",
            "city": "MockCity",
            "coordinates": {"lat": 48.85, "lon": 2.35},
            "description": "A mock location for testing purposes.",
            "rating": 4.2,
        }

    def get_events(
        self, city: str, start_date: str, end_date: str
    ) -> list[dict]:
        return [
            {
                "event_id": f"EVT_{city[:3].upper()}_FESTIVAL",
                "name": f"{city} Summer Festival",
                "city": city,
                "start_date": start_date,
                "end_date": start_date,
                "venue": f"{city} Main Square",
                "category": "festival",
                "cost": 0.0,
                "description": "Annual city festival with music and food.",
            }
        ]

    # ── Booking methods ────────────────────────────────────────────────────────

    def book_flight(self, flight_id: str, passenger_details: dict) -> dict:
        return {
            "booking_ref": f"BK_FL_{flight_id[-3:]}_001",
            "flight_id": flight_id,
            "status": "confirmed",
            "total_cost": 700.0,
            "passenger_details": passenger_details,
        }

    def book_hotel(self, hotel_id: str, guest_details: dict) -> dict:
        return {
            "booking_ref": f"BK_HTL_{hotel_id[-6:]}_001",
            "hotel_id": hotel_id,
            "status": "confirmed",
            "total_cost": 360.0,
            "check_in": guest_details.get("check_in", "2025-06-01"),
            "check_out": guest_details.get("check_out", "2025-06-04"),
        }

    def book_activity(self, activity_id: str, participant_details: dict) -> dict:
        return {
            "booking_ref": f"BK_ACT_{activity_id[-6:]}_001",
            "activity_id": activity_id,
            "status": "confirmed",
            "total_cost": 36.0,
            "participant_details": participant_details,
        }

    # ── Utility methods ────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed

    def get_world_seed(self) -> int:
        return self._seed


# ── Test UserRequest fixtures ─────────────────────────────────────────────────

def make_test_requests() -> list[UserRequest]:
    """Return 3 canonical UserRequest objects for integration tests."""
    return [_request_simple_paris(), _request_multiday_rome(), _request_family_barcelona()]


def _request_simple_paris() -> UserRequest:
    """Scenario 1: 3-day Paris trip for 1 adult, $1500 budget."""
    return UserRequest(
        request_id="test-req-paris-001",
        raw_text=(
            "Plan a 3-day trip to Paris for 1 adult from June 1–3 2025. "
            "Budget: $1500. I love art museums and good coffee."
        ),
        origin_city="New York",
        destination_cities=["Paris"],
        start_date="2025-06-01",
        end_date="2025-06-03",
        budget_usd=1500.0,
        traveler_profile=TravelerProfile(num_adults=1),
        hard_constraints=[
            Constraint(
                constraint_id="hc_budget_paris",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.BUDGET,
                description="Total cost must not exceed $1500",
                value=1500.0,
                unit="USD",
            ),
            Constraint(
                constraint_id="hc_city_paris",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.CITY,
                description="Destination must be Paris",
                value="Paris",
            ),
        ],
        soft_constraints=[
            Constraint(
                constraint_id="sc_museum_paris",
                constraint_type=ConstraintType.SOFT,
                category=ConstraintCategory.ACTIVITY,
                description="Include at least one art museum visit",
                value="art_museum",
            ),
        ],
        preferences=["art museums", "café culture"],
    )


def _request_multiday_rome() -> UserRequest:
    """Scenario 2: 5-day Rome + Florence for 2 adults, $3000 budget."""
    return UserRequest(
        request_id="test-req-rome-002",
        raw_text=(
            "Plan a 5-day Italy trip visiting Rome (3 nights) and Florence (2 nights) "
            "for 2 adults, June 10–14 2025. Budget is $3000 total. "
            "We prefer boutique hotels and are interested in history."
        ),
        origin_city="London",
        destination_cities=["Rome", "Florence"],
        start_date="2025-06-10",
        end_date="2025-06-14",
        budget_usd=3000.0,
        traveler_profile=TravelerProfile(num_adults=2),
        hard_constraints=[
            Constraint(
                constraint_id="hc_budget_italy",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.BUDGET,
                description="Total cost must not exceed $3000",
                value=3000.0,
                unit="USD",
            ),
            Constraint(
                constraint_id="hc_duration_italy",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.DURATION,
                description="Trip must be exactly 5 days",
                value=5,
                unit="days",
            ),
        ],
        soft_constraints=[
            Constraint(
                constraint_id="sc_hotel_italy",
                constraint_type=ConstraintType.SOFT,
                category=ConstraintCategory.ACCOMMODATION,
                description="Prefer boutique hotels over large chains",
                value="boutique",
            ),
        ],
        preferences=["history", "Renaissance art", "local cuisine"],
    )


def _request_family_barcelona() -> UserRequest:
    """Scenario 3: 4-day Barcelona for 2 adults + 1 child, $2500 budget, vegetarian."""
    return UserRequest(
        request_id="test-req-barcelona-003",
        raw_text=(
            "Family trip to Barcelona June 20–23 2025. "
            "2 adults and 1 child (age 8). Budget $2500. "
            "We need vegetarian dining options and child-friendly activities."
        ),
        origin_city="Amsterdam",
        destination_cities=["Barcelona"],
        start_date="2025-06-20",
        end_date="2025-06-23",
        budget_usd=2500.0,
        traveler_profile=TravelerProfile(
            num_adults=2,
            num_children=1,
            dietary_restrictions=["vegetarian"],
        ),
        hard_constraints=[
            Constraint(
                constraint_id="hc_budget_bcn",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.BUDGET,
                description="Total cost must not exceed $2500",
                value=2500.0,
                unit="USD",
            ),
        ],
        soft_constraints=[
            Constraint(
                constraint_id="sc_diet_bcn",
                constraint_type=ConstraintType.SOFT,
                category=ConstraintCategory.PREFERENCE,
                description="Vegetarian dining options at all meals",
                value="vegetarian",
            ),
            Constraint(
                constraint_id="sc_child_bcn",
                constraint_type=ConstraintType.SOFT,
                category=ConstraintCategory.ACTIVITY,
                description="Child-friendly activities included",
                value="child_friendly",
            ),
        ],
        preferences=["beach", "architecture", "vegetarian food"],
    )
