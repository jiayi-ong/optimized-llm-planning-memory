# Tools — Middleware Layer

The tools module wraps the `travel_world` simulator with validation, call tracking, event emission, and structured error feedback. The agent calls tools by name with a JSON argument dict; the middleware handles everything else.

---

## Files

| File | Role |
|---|---|
| `tools/base.py` | `BaseTool` ABC — Template Method pattern |
| `tools/registry.py` | `ToolRegistry` — name → instance lookup, schema generation |
| `tools/tracker.py` | `ToolCallTracker`, `EpisodeTimer` — per-episode usage stats |
| `tools/events.py` | `EventBus`, `ToolEvent` — episode-scoped event pub/sub |
| `tools/flight_tools.py` | `SearchFlights`, `SelectFlight` |
| `tools/hotel_tools.py` | `SearchHotels`, `BookHotel`, `GetHotelDetail` |
| `tools/attraction_tools.py` | `SearchAttractions`, `GetAttractionDetail` |
| `tools/restaurant_tools.py` | `SearchRestaurants` |
| `tools/event_tools.py` | `SearchEvents`, `BookEvent` |
| `tools/routing_tools.py` | `PlanRoute` |
| `tools/info_tools.py` | `GetAvailableRoutes` |

---

## BaseTool — Template Method Pattern

Every tool call goes through this fixed lifecycle:

```python
class BaseTool(ABC):
    tool_name: str          # must match the string the LLM produces
    tool_description: str   # shown in the agent's system prompt
    input_schema: type[BaseModel]  # Pydantic model; call() validates against this

    def call(self, raw_arguments: dict) -> ToolResult:
        # 1. Validate raw_arguments against input_schema
        # 2. Call self._execute(validated_input)
        # 3. Record result to ToolCallTracker
        # 4. Emit ToolEvent to EventBus
        # 5. Return ToolResult (success or failure)

    @abstractmethod
    def _execute(self, validated_input: BaseModel) -> Any:
        # Subclasses implement ONLY this method
        pass
```

Subclasses never override `call()`. Validation, tracking, and events are guaranteed for every tool automatically.

---

## Complete Tool Inventory

### `get_available_routes`

**Always call this first.** Returns all flight-connected city pairs with their `city_id` values. Use these IDs with every other search tool.

```
Input: {} (no parameters)
Returns: list of {origin_city_id, origin_city_name, destination_city_id, destination_city_name, edge_id}
```

---

### `search_flights`

```
Input:
  origin_city_id       str   — from get_available_routes
  destination_city_id  str   — from get_available_routes
  departure_date       str   — YYYY-MM-DD
  passengers           int   — default 1, range [1, 20]

Returns: list of {edge_id, airline, departure_time, arrival_time, duration_min, price_usd, seats_available}
```

---

### `select_flight`

Pseudo-booking: validates the `edge_id` and returns a synthetic `BookingConfirmation`. No simulator state is mutated. Use this after `search_flights` to record a flight selection in the itinerary.

```
Input:
  edge_id                str   — from search_flights
  origin_city_name       str   — optional, for display
  destination_city_name  str   — optional, for display
  departure_datetime     str   — optional, ISO 8601

Returns: {booking_ref, edge_id, confirmation_status}
```

---

### `search_hotels`

```
Input:
  city_id              str    — from get_available_routes
  check_in             str    — YYYY-MM-DD
  check_out            str    — YYYY-MM-DD
  guests               int    — default 1, range [1, 20]
  max_price_per_night  float  — optional, USD
  min_stars            float  — optional, [0.0, 5.0]

Returns: list of {hotel_id, name, stars, price_per_night_usd, district, amenities, availability}
```

---

### `book_hotel`

```
Input:
  hotel_id   str — from search_hotels
  check_in   str — YYYY-MM-DD
  check_out  str — YYYY-MM-DD

Returns: {booking_ref, hotel_id, hotel_name, total_cost_usd, confirmation_status}
```

---

### `get_hotel_detail`

```
Input:
  hotel_id  str — from search_hotels

Returns: full hotel profile including rooms, policies, nearby attractions
```

---

### `search_attractions`

```
Input:
  city_id   str  — from get_available_routes
  category  str  — optional: 'museum' | 'park' | 'landmark' | 'entertainment' | 'shopping' | 'nature'
  free_only bool — default False

Returns: list of {attraction_id, name, category, ticket_price_usd, duration_hours, popularity, wait_time_min}
```

---

### `get_attraction_detail`

```
Input:
  attraction_id  str — from search_attractions

Returns: full attraction profile including opening hours, location_id, reviews
```

---

### `search_restaurants`

```
Input:
  city_id       str   — from get_available_routes
  cuisine       str   — optional: 'italian' | 'japanese' | 'french' | 'mexican' | ... (case-insensitive)
  max_avg_spend float — optional, USD per person

Returns: list of {restaurant_id, name, cuisine, avg_spend_usd, price_tier, michelin_stars, reservation_required}
```

---

### `search_events`

```
Input:
  city_id    str   — from get_available_routes
  start_date str   — optional, YYYY-MM-DD
  end_date   str   — optional, YYYY-MM-DD
  category   str   — optional: 'concert' | 'festival' | 'sport' | 'exhibition' | 'theater' | 'cultural'
  max_price  float — optional, USD per ticket

Returns: list of {event_id, name, category, date, venue, ticket_price_usd, availability}
```

---

### `book_event`

```
Input:
  event_id  str — from search_events
  quantity  int — default 1, range [1, 50]

Returns: {booking_ref, event_id, event_name, total_cost_usd, seats_booked}
```

---

### `plan_route`

```
Input:
  origin_location_id       str  — location node ID (from hotel, attraction, or event results)
  destination_location_id  str  — location node ID
  departure_datetime       str  — ISO 8601 (YYYY-MM-DDTHH:MM:SS)
  optimize_for             str  — 'time' | 'cost' | 'balanced' (default 'time')

Returns: list of transport options, one per available mode (walking/taxi/transit), each with
         {mode, duration_min, cost_usd, distance_km}
```

---

## ToolRegistry

`ToolRegistry` is the single source of truth for which tools exist.

```python
# Construction (from scripts and tests)
from optimized_llm_planning_memory.tools.registry import ToolRegistry

registry = ToolRegistry.from_config(
    simulator=adapter,
    tracker=tracker,
    event_bus=event_bus,
    enabled_tools=None,    # None = all tools; list of names to restrict
)

# At runtime (inside ReActAgent)
result = registry.call("search_hotels", {"city_id": "par-001", ...})
schemas = registry.get_tool_schemas()   # injected into the agent prompt
```

---

## Adding a New Tool

Follow these five steps. Nothing else needs changing.

### Step 1 — Write the input schema

Create a Pydantic model in a new or existing file under `tools/`:

```python
# tools/my_tool.py
from pydantic import BaseModel, Field
from optimized_llm_planning_memory.tools.base import BaseTool

class MyToolInput(BaseModel):
    city_id: str = Field(description="City ID. Use get_available_routes to find IDs.")
    some_param: str = Field(description="What this param means to the LLM.")
    optional_param: float | None = Field(default=None, description="Optional filter.")
```

Keep descriptions LLM-readable — they appear verbatim in the agent's tool schema prompt.

### Step 2 — Implement the tool class

```python
class MyTool(BaseTool):
    tool_name = "my_tool"             # must be snake_case; this is what the LLM produces
    tool_description = (
        "One-sentence description of what this tool does. "
        "Include what it returns and any key caveats."
    )
    input_schema = MyToolInput

    def _execute(self, validated_input: MyToolInput) -> Any:
        return self._simulator.my_simulator_method(
            city_id=validated_input.city_id,
            some_param=validated_input.some_param,
            optional_param=validated_input.optional_param,
        )

    # Optional: override for custom error hints shown back to the agent
    def _generate_error_feedback(self, error: Exception, arguments: dict) -> str:
        return (
            f"Tool 'my_tool' failed: {error}. "
            "Hint: ensure city_id comes from get_available_routes."
        )
```

### Step 3 — Register in ToolRegistry

Open `tools/registry.py` and add your tool to `from_config()`:

```python
from optimized_llm_planning_memory.tools.my_tool import MyTool

class ToolRegistry:
    @classmethod
    def from_config(cls, simulator, tracker, event_bus, enabled_tools=None):
        registry = cls()
        all_tools = [
            ...existing tools...,
            MyTool(simulator=simulator, tracker=tracker, event_bus=event_bus),
        ]
        for tool in all_tools:
            if enabled_tools is None or tool.tool_name in enabled_tools:
                registry.register(tool)
        return registry
```

### Step 4 — Update `SimulatorProtocol` (if needed)

If your tool calls a new simulator method, add it to `simulator/protocol.py`:

```python
class SimulatorProtocol(Protocol):
    ...
    def my_simulator_method(
        self,
        city_id: str,
        some_param: str,
        optional_param: float | None = None,
    ) -> list[dict]: ...
```

Also add a matching stub to `tests/test_integration/mock_simulator.py` so tests keep working.

### Step 5 — Add few-shot examples

Add at least one `Thought / Action / Observation` example for the new tool to `data/few_shot_examples/react_tool_use.json`. Use real-looking IDs matching the mock simulator's convention (e.g., `"nyc-001"`, `"SIM-HTL-001"`).

### Step 6 — Write a unit test

```python
# tests/test_tools/test_my_tool.py
from optimized_llm_planning_memory.tools.my_tool import MyTool

def test_my_tool_succeeds(mock_simulator, tracker, event_bus):
    tool = MyTool(simulator=mock_simulator, tracker=tracker, event_bus=event_bus)
    result = tool.call({"city_id": "par-001", "some_param": "value"})
    assert result.success is True
```

---

## ToolCallTracker

`ToolCallTracker` accumulates per-tool statistics that feed into the reward function and evaluation metrics.

```python
stats = tracker.get_stats()
# Returns list[ToolCallStats]:
# {tool_name, call_count, success_count, failure_count,
#  redundant_call_count, total_latency_ms, avg_latency_ms}
```

A call is counted as **redundant** when the same tool is called with the same arguments a second time in the same episode without any new information arriving in between.

---

## EventBus

`EventBus` is a simple in-memory pub/sub for episode-scoped events. Tools emit `ToolEvent` objects; other components (e.g., live logging, MCTS node evaluator) can subscribe.

```python
bus.subscribe(lambda event: print(event.tool_name, event.success))
bus.emit(ToolEvent(tool_name="search_flights", success=True, latency_ms=42.0, ...))
```

The bus is created once per episode and passed into every tool at construction time. Do not reuse a bus across episodes.
