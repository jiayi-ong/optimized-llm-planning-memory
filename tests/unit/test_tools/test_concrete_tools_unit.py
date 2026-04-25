"""Unit tests for concrete tool classes — schema validation and instantiation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from optimized_llm_planning_memory.tools.events import EventBus
from optimized_llm_planning_memory.tools.tracker import ToolCallTracker
from optimized_llm_planning_memory.tools.flight_tools import SearchFlights, SelectFlight
from optimized_llm_planning_memory.tools.hotel_tools import BookHotel, GetHotelDetail, SearchHotels
from optimized_llm_planning_memory.tools.attraction_tools import GetAttractionDetail, SearchAttractions
from optimized_llm_planning_memory.tools.restaurant_tools import SearchRestaurants
from optimized_llm_planning_memory.tools.event_tools import BookEvent, SearchEvents
from optimized_llm_planning_memory.tools.routing_tools import PlanRoute
from optimized_llm_planning_memory.tools.info_tools import GetAvailableRoutes


ALL_TOOL_CLASSES = [
    SearchFlights,
    SelectFlight,
    SearchHotels,
    BookHotel,
    GetHotelDetail,
    SearchAttractions,
    GetAttractionDetail,
    SearchRestaurants,
    SearchEvents,
    BookEvent,
    PlanRoute,
    GetAvailableRoutes,
]


@pytest.fixture
def sim():
    return MagicMock()


@pytest.fixture
def tracker():
    return ToolCallTracker()


@pytest.fixture
def bus():
    return EventBus()


@pytest.mark.unit
class TestConcreteToolInstantiation:
    def test_all_tools_have_tool_name(self, sim, tracker, bus):
        for cls in ALL_TOOL_CLASSES:
            tool = cls(simulator=sim, tracker=tracker, event_bus=bus)
            assert isinstance(tool.tool_name, str)
            assert len(tool.tool_name) > 0, f"{cls.__name__} has empty tool_name"

    def test_all_tools_have_description(self, sim, tracker, bus):
        for cls in ALL_TOOL_CLASSES:
            tool = cls(simulator=sim, tracker=tracker, event_bus=bus)
            assert isinstance(tool.tool_description, str)
            assert len(tool.tool_description) > 0

    def test_all_tools_have_input_schema(self, sim, tracker, bus):
        from pydantic import BaseModel
        for cls in ALL_TOOL_CLASSES:
            tool = cls(simulator=sim, tracker=tracker, event_bus=bus)
            assert issubclass(tool.input_schema, BaseModel)


@pytest.mark.unit
class TestSearchFlightsSchema:
    def test_valid_input_passes(self):
        from optimized_llm_planning_memory.tools.flight_tools import SearchFlightsInput
        inp = SearchFlightsInput(
            origin_city_id="NYC",
            destination_city_id="PAR",
            departure_date="2025-06-01",
            passengers=2,
        )
        assert inp.origin_city_id == "NYC"

    def test_missing_required_field_raises(self):
        from optimized_llm_planning_memory.tools.flight_tools import SearchFlightsInput
        with pytest.raises(ValidationError):
            SearchFlightsInput(destination_city_id="PAR", departure_date="2025-06-01")

    def test_passengers_default_is_one(self):
        from optimized_llm_planning_memory.tools.flight_tools import SearchFlightsInput
        inp = SearchFlightsInput(
            origin_city_id="NYC",
            destination_city_id="PAR",
            departure_date="2025-06-01",
        )
        assert inp.passengers == 1


@pytest.mark.unit
class TestSelectFlightSchema:
    def test_valid_edge_id(self):
        from optimized_llm_planning_memory.tools.flight_tools import SelectFlightInput
        inp = SelectFlightInput(edge_id="edge_001")
        assert inp.edge_id == "edge_001"

    def test_missing_edge_id_raises(self):
        from optimized_llm_planning_memory.tools.flight_tools import SelectFlightInput
        with pytest.raises(ValidationError):
            SelectFlightInput()


@pytest.mark.unit
class TestSearchHotelsSchema:
    def test_valid_input(self):
        from optimized_llm_planning_memory.tools.hotel_tools import SearchHotelsInput
        inp = SearchHotelsInput(city_id="PAR", check_in="2025-06-01", check_out="2025-06-03")
        assert inp.city_id == "PAR"

    def test_missing_city_id_raises(self):
        from optimized_llm_planning_memory.tools.hotel_tools import SearchHotelsInput
        with pytest.raises(ValidationError):
            SearchHotelsInput(check_in="2025-06-01", check_out="2025-06-03")


@pytest.mark.unit
class TestBookEventSchema:
    def test_quantity_ge_one(self):
        from optimized_llm_planning_memory.tools.event_tools import BookEventInput
        with pytest.raises(ValidationError):
            BookEventInput(event_id="EVT001", quantity=0)

    def test_valid_input(self):
        from optimized_llm_planning_memory.tools.event_tools import BookEventInput
        inp = BookEventInput(event_id="EVT001", quantity=2)
        assert inp.quantity == 2


@pytest.mark.unit
class TestPlanRouteSchema:
    def test_optimize_for_default(self):
        from optimized_llm_planning_memory.tools.routing_tools import PlanRouteInput
        inp = PlanRouteInput(
            origin_location_id="loc_001",
            destination_location_id="loc_002",
            departure_datetime="2025-06-01T10:00:00",
        )
        assert inp.optimize_for == "time"
