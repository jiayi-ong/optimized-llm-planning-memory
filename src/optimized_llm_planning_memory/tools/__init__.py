"""
tools/
======
Tool middleware layer. The planning agent calls ONLY tools; it never calls
the simulator adapter directly.

Design principle: Separation of concerns
-----------------------------------------
``simulator/`` answers the question "what can the world provide?"
``tools/`` answers the question "what is the agent allowed to ask?"

Every tool call passes through ``BaseTool.call()``, which provides:
  - Input validation (Pydantic)
  - Execution (delegated to ``_execute()``)
  - Usage tracking (ToolCallTracker)
  - Event emission (EventBus)
  - Structured failure feedback for the agent's next thought

Design pattern: Template Method (BaseTool.call → _execute)
-----------------------------------------------------------
The public ``call()`` method is the template: it orchestrates validation,
execution, tracking, and event emission. Subclasses override only the
``_execute()`` step, receiving already-validated input.

This means every tool gets tracking and error handling for free — the
concrete tool classes are only responsible for the actual simulator call.

Contents
--------
base.py          — BaseTool ABC (Template Method pattern)
tracker.py       — ToolCallTracker (thread-safe usage recorder)
events.py        — ToolEvent + EventBus (pub/sub for tool outcomes)
registry.py      — ToolRegistry (name→tool map; pydantic-ai schema adapter)
flight_tools.py  — SearchFlights, BookFlight
hotel_tools.py   — SearchHotels, BookHotel
activity_tools.py— SearchActivities, BookActivity
info_tools.py    — GetCityInfo, GetLocationDetails, GetEvents
"""
