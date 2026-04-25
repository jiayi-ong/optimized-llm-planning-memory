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
    Method signatures match the keyword arguments that BaseTool subclasses pass.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    # ── Discovery ──────────────────────────────────────────────────────────────

    def get_available_routes(self) -> list[dict]:
        return [
            {
                "edge_id": "EDGE_NYC_PAR_001",
                "origin_city_id": "nyc-001",
                "origin_city_name": "New York",
                "destination_city_id": "par-001",
                "destination_city_name": "Paris",
            },
            {
                "edge_id": "EDGE_LON_ROM_001",
                "origin_city_id": "lon-001",
                "origin_city_name": "London",
                "destination_city_id": "rom-001",
                "destination_city_name": "Rome",
            },
            {
                "edge_id": "EDGE_AMS_BCN_001",
                "origin_city_id": "ams-001",
                "origin_city_name": "Amsterdam",
                "destination_city_id": "bcn-001",
                "destination_city_name": "Barcelona",
            },
        ]

    # ── Flight methods ─────────────────────────────────────────────────────────

    def search_flights(
        self,
        origin_city_id: str,
        destination_city_id: str,
        departure_date: str,
        passengers: int = 1,
    ) -> list[dict]:
        prefix = f"{origin_city_id[:3].upper()}_{destination_city_id[:3].upper()}"
        return [
            {
                "edge_id": f"EDGE_{prefix}_001",
                "flight_id": f"FL_{prefix}_001",
                "airline": "MockAir",
                "origin_city_id": origin_city_id,
                "destination_city_id": destination_city_id,
                "departure_datetime": f"{departure_date}T08:00:00",
                "arrival_datetime": f"{departure_date}T14:00:00",
                "price_per_person": 350.0,
                "total_price": 350.0 * passengers,
                "stops": 0,
                "duration_hours": 6.0,
            },
            {
                "edge_id": f"EDGE_{prefix}_002",
                "flight_id": f"FL_{prefix}_002",
                "airline": "BudgetFly",
                "origin_city_id": origin_city_id,
                "destination_city_id": destination_city_id,
                "departure_datetime": f"{departure_date}T14:00:00",
                "arrival_datetime": f"{departure_date}T20:30:00",
                "price_per_person": 220.0,
                "total_price": 220.0 * passengers,
                "stops": 1,
                "duration_hours": 6.5,
            },
        ]

    # ── Hotel methods ──────────────────────────────────────────────────────────

    def search_hotels(
        self,
        city_id: str,
        check_in: str,
        check_out: str,
        guests: int = 1,
        max_price: float | None = None,
        min_stars: float | None = None,
    ) -> list[dict]:
        city_tag = city_id[:3].upper()
        return [
            {
                "hotel_id": f"HTL_{city_tag}_BOUTIQUE",
                "name": f"{city_id} Boutique Inn",
                "city_id": city_id,
                "stars": 3,
                "category": "boutique",
                "check_in": check_in,
                "check_out": check_out,
                "price_per_night": 120.0,
                "total_price": 120.0 * 3,
                "amenities": ["wifi", "breakfast"],
            },
            {
                "hotel_id": f"HTL_{city_tag}_GRAND",
                "name": f"Grand {city_id} Hotel",
                "city_id": city_id,
                "stars": 5,
                "category": "luxury",
                "check_in": check_in,
                "check_out": check_out,
                "price_per_night": 280.0,
                "total_price": 280.0 * 3,
                "amenities": ["wifi", "spa", "pool", "breakfast"],
            },
        ]

    def book_hotel(self, hotel_id: str, check_in: str, check_out: str) -> dict:
        return {
            "booking_ref": f"SIM-HTL-{hotel_id[-6:]}-001",
            "hotel_id": hotel_id,
            "status": "confirmed",
            "total_cost": 360.0,
            "check_in": check_in,
            "check_out": check_out,
        }

    def get_hotel_detail(self, hotel_id: str) -> dict:
        return {
            "hotel_id": hotel_id,
            "name": f"Hotel {hotel_id}",
            "stars": 3,
            "description": "A mock hotel for testing.",
            "amenities": ["wifi", "breakfast"],
            "price_per_night": 120.0,
        }

    # ── Attraction methods ─────────────────────────────────────────────────────

    def search_attractions(
        self,
        city_id: str,
        category: str | None = None,
        free_only: bool = False,
    ) -> list[dict]:
        city_tag = city_id[:3].upper()
        return [
            {
                "attraction_id": f"ATT_{city_tag}_MUSEUM",
                "name": f"{city_id} National Museum",
                "city_id": city_id,
                "category": "museum",
                "duration_hours": 3.0,
                "admission_fee": 18.0,
                "is_free": False,
                "description": "A world-class museum.",
            },
            {
                "attraction_id": f"ATT_{city_tag}_PARK",
                "name": f"{city_id} Central Park",
                "city_id": city_id,
                "category": "park",
                "duration_hours": 2.0,
                "admission_fee": 0.0,
                "is_free": True,
                "description": "A beautiful park.",
            },
        ]

    def get_attraction_detail(self, attraction_id: str) -> dict:
        return {
            "attraction_id": attraction_id,
            "name": f"Attraction {attraction_id}",
            "description": "A mock attraction for testing.",
            "admission_fee": 18.0,
            "duration_hours": 3.0,
            "opening_hours": "09:00-18:00",
        }

    # ── Event methods ──────────────────────────────────────────────────────────

    def search_events(
        self,
        city_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        category: str | None = None,
    ) -> list[dict]:
        city_tag = city_id[:3].upper()
        return [
            {
                "event_id": f"EVT_{city_tag}_FESTIVAL",
                "name": f"{city_id} Summer Festival",
                "city_id": city_id,
                "start_date": start_date or "2025-06-01",
                "end_date": end_date or "2025-06-01",
                "venue": f"{city_id} Main Square",
                "category": "festival",
                "price_per_ticket": 0.0,
                "description": "Annual city festival.",
            }
        ]

    def book_event(self, event_id: str, quantity: int = 1) -> dict:
        return {
            "booking_ref": f"SIM-EVT-{event_id[-6:]}-001",
            "event_id": event_id,
            "status": "confirmed",
            "quantity": quantity,
            "total_cost": 0.0,
        }

    # ── Restaurant methods ─────────────────────────────────────────────────────

    def search_restaurants(
        self,
        city_id: str,
        cuisine: str | None = None,
        max_avg_spend: float | None = None,
    ) -> list[dict]:
        city_tag = city_id[:3].upper()
        return [
            {
                "restaurant_id": f"RST_{city_tag}_BISTRO",
                "name": f"{city_id} Bistro",
                "city_id": city_id,
                "cuisine": cuisine or "international",
                "avg_spend_per_person": 35.0,
                "rating": 4.2,
            },
        ]

    # ── Routing methods ────────────────────────────────────────────────────────

    def plan_route(
        self,
        origin_location_id: str,
        destination_location_id: str,
        departure_datetime: str | None = None,
        optimize_for: str | None = None,
    ) -> dict:
        return {
            "route_id": f"ROUTE_{origin_location_id[:4]}_{destination_location_id[:4]}",
            "origin_location_id": origin_location_id,
            "destination_location_id": destination_location_id,
            "duration_minutes": 45,
            "distance_km": 12.5,
            "mode": "transit",
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
