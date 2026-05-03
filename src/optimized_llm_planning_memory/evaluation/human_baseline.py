"""
evaluation/human_baseline.py
=============================
HumanPlanningSession — notebook-driven session for human-generated travel plans.

Design: Adapter + Session pattern
-----------------------------------
The session wraps the same tool middleware the ReAct agent uses (ToolRegistry,
ToolCallTracker, EventBus) so that human tool calls produce the same statistics
and feedback as agent calls. Every call is logged as a ``ReActStep``, building
an ``EpisodeLog`` in the exact same format consumed by the evaluator.

This makes the human plan a directly comparable reference: feed the resulting
``EpisodeLog`` into ``DeterministicEvaluator`` or ``LLMJudge`` exactly as you
would an agent episode.

Usage flow (notebook):
----------------------
1. Create session with a UserRequest and a world seed.
2. Call tool methods (search_flights, book_hotel, …). Each call:
     a. Invokes the tool through the middleware.
     b. Logs a ReActStep (thought + action + observation).
     c. Returns the raw result for inspection.
3. Call add_*_to_itinerary() helpers to build the plan. Each call:
     a. Updates the mutable Itinerary.
     b. Logs a separate ReActStep capturing the itinerary snapshot.
4. Call finalize() + save() to write the EpisodeLog to disk.
5. Evaluate with DeterministicEvaluator.score(episode_log, user_request).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from optimized_llm_planning_memory.core.constraints import ConstraintSatisfactionEngine
from optimized_llm_planning_memory.core.models import (
    AccommodationBooking,
    ActivityBooking,
    EpisodeLog,
    Itinerary,
    ItineraryDay,
    ReActStep,
    RewardComponents,
    ToolCall,
    ToolResult,
    TrajectoryModel,
    TransportSegment,
    UserRequest,
)
from optimized_llm_planning_memory.simulator.adapter import SimulatorAdapter
from optimized_llm_planning_memory.tools.events import EventBus
from optimized_llm_planning_memory.tools.registry import ToolRegistry
from optimized_llm_planning_memory.tools.tracker import ToolCallTracker
from optimized_llm_planning_memory.utils.episode_io import save_episode


class HumanPlanningSession:
    """
    Interactive travel planning session that produces an agent-compatible EpisodeLog.

    Every tool call and itinerary decision is logged as a ReActStep so the
    resulting EpisodeLog can be evaluated with the same metrics used for agent
    episodes — enabling apples-to-apples comparison.

    Parameters
    ----------
    user_request : The travel request the human is planning against.
    seed         : World seed passed to SimulatorAdapter. Use the same seed as
                   the agent episodes you want to compare against.
    output_dir   : Default directory for save(). Created if absent.
    """

    AGENT_MODE = "human_baseline"

    def __init__(
        self,
        user_request: UserRequest,
        seed: int = 42,
        output_dir: str | Path = "outputs/human_episodes",
        world_config: dict | None = None,
    ) -> None:
        self._request = user_request
        self._output_dir = Path(output_dir)
        self._episode_id = uuid.uuid4().hex[:12]

        # Same infrastructure the ReAct agent uses — ensures comparable stats.
        self._simulator = SimulatorAdapter(seed=seed, world_config=world_config)
        self._tracker = ToolCallTracker()
        self._event_bus = EventBus()
        self._registry = ToolRegistry.from_config(
            simulator=self._simulator,
            tracker=self._tracker,
            event_bus=self._event_bus,
        )

        # Mutable episode state
        self._steps: list[ReActStep] = []
        self._itinerary = Itinerary(
            itinerary_id=uuid.uuid4().hex[:12],
            request_id=user_request.request_id,
            days=[],
        )
        self._finalized = False
        self._episode_log: EpisodeLog | None = None

        print(f"╔{'═' * 58}╗")
        print(f"║  Human Planning Session                                  ║")
        print(f"╠{'═' * 58}╣")
        print(f"║  episode_id  : {self._episode_id:<42}║")
        print(f"║  request_id  : {user_request.request_id:<42}║")
        print(f"║  world seed  : {self._simulator.get_world_seed()!s:<42}║")
        print(f"║  budget      : ${user_request.budget_usd:<41,.2f}║")
        print(f"║  dates       : {user_request.start_date} → {user_request.end_date:<28}║")
        print(f"╚{'═' * 58}╝")
        print(f"\nRequest: {user_request.raw_text}\n")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def itinerary(self) -> Itinerary:
        """Current (mutable) itinerary being built."""
        return self._itinerary

    @property
    def budget_remaining(self) -> float:
        """Budget left after all itinerary costs so far."""
        return self._request.budget_usd - self._itinerary.total_cost_usd

    @property
    def steps_taken(self) -> int:
        return len(self._steps)

    @property
    def episode_id(self) -> str:
        return self._episode_id

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call_tool(self, tool_name: str, thought: str, arguments: dict[str, Any]) -> Any:
        """
        Invoke a registered tool, log the step, and return the raw result.

        Filters None values from arguments so optional tool fields use defaults.
        On tool error, prints the message and returns None — the notebook cell
        does not raise so the human can adjust arguments and retry.
        """
        if self._finalized:
            raise RuntimeError(
                "Session is already finalized. Create a new HumanPlanningSession to continue."
            )

        clean_args = {k: v for k, v in arguments.items() if v is not None}
        tool = self._registry.get(tool_name)
        result: ToolResult = tool.call(clean_args)

        self._steps.append(ReActStep(
            step_index=len(self._steps),
            thought=thought,
            action=ToolCall(
                tool_name=tool_name,
                arguments=clean_args,
                raw_text=f"Action: {tool_name}({clean_args})",
            ),
            observation=result,
            itinerary_snapshot=None,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        ))

        if result.success:
            return result.result
        print(f"[TOOL ERROR — {tool_name}] {result.error_message}")
        return None

    def _log_itinerary_update(self, thought: str) -> None:
        """
        Append a step that records an itinerary mutation (no tool call).

        Separating tool calls from itinerary decisions mirrors the agent's
        trajectory structure and keeps the trail readable.
        """
        import copy
        self._steps.append(ReActStep(
            step_index=len(self._steps),
            thought=thought,
            action=None,
            observation=None,
            itinerary_snapshot=copy.deepcopy(self._itinerary),
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        ))

    def _ensure_day(self, date_str: str, city: str) -> ItineraryDay:
        """Return the ItineraryDay for date_str, creating and inserting it if absent."""
        for day in self._itinerary.days:
            if day.date == date_str:
                return day
        new_day = ItineraryDay(date=date_str, city=city)
        self._itinerary.days.append(new_day)
        self._itinerary.days.sort(key=lambda d: d.date)
        return new_day

    # ── Tool call methods ─────────────────────────────────────────────────────
    # Thin wrappers over the registry. Named parameters make arg names
    # discoverable by tab-completion; `thought` is prepended to every call
    # so the human's reasoning appears in the trail alongside the action.

    def get_available_routes(
        self,
        thought: str = "Discovering available cities and connections.",
    ) -> list[dict]:
        """List all cities in the world. Call this first — city_ids are required everywhere."""
        return self._call_tool("get_available_routes", thought, {}) or []

    def search_flights(
        self,
        thought: str,
        origin_city_id: str,
        destination_city_id: str,
        departure_date: str,
        passengers: int = 1,
        max_results: int = 10,
    ) -> list[dict]:
        """Search flights between two cities. Returns options sorted cheapest first."""
        return self._call_tool("search_flights", thought, {
            "origin_city_id": origin_city_id,
            "destination_city_id": destination_city_id,
            "departure_date": departure_date,
            "passengers": passengers,
            "max_results": max_results,
        }) or []

    def select_flight(
        self,
        thought: str,
        edge_id: str,
        origin_city_name: str = "",
        destination_city_name: str = "",
        departure_datetime: str = "",
        arrival_datetime: str = "",
        total_price: float = 0.0,
    ) -> dict | None:
        """Confirm a flight choice by edge_id. Returns booking confirmation."""
        return self._call_tool("select_flight", thought, {
            "edge_id": edge_id,
            "origin_city_name": origin_city_name,
            "destination_city_name": destination_city_name,
            "departure_datetime": departure_datetime,
            "arrival_datetime": arrival_datetime,
            "total_price": total_price,
        })

    def search_hotels(
        self,
        thought: str,
        city_id: str,
        check_in: str,
        check_out: str,
        guests: int = 1,
        max_price_per_night: float | None = None,
        min_stars: float | None = None,
        max_results: int = 10,
    ) -> list[dict]:
        """Search hotels in a city. Set max_price_per_night = budget_remaining / nights."""
        return self._call_tool("search_hotels", thought, {
            "city_id": city_id,
            "check_in": check_in,
            "check_out": check_out,
            "guests": guests,
            "max_price_per_night": max_price_per_night,
            "min_stars": min_stars,
            "max_results": max_results,
        }) or []

    def book_hotel(
        self,
        thought: str,
        hotel_id: str,
        check_in: str,
        check_out: str,
    ) -> dict | None:
        """Book a hotel. Returns confirmation with total cost."""
        return self._call_tool("book_hotel", thought, {
            "hotel_id": hotel_id,
            "check_in": check_in,
            "check_out": check_out,
        })

    def get_hotel_detail(self, thought: str, hotel_id: str) -> dict | None:
        """Get full hotel details including availability calendar."""
        return self._call_tool("get_hotel_detail", thought, {"hotel_id": hotel_id})

    def search_attractions(
        self,
        thought: str,
        city_id: str,
        category: str | None = None,
        free_only: bool = False,
    ) -> list[dict]:
        """Search attractions. Common categories: hiking, museum, park, landmark."""
        return self._call_tool("search_attractions", thought, {
            "city_id": city_id,
            "category": category,
            "free_only": free_only,
        }) or []

    def get_attraction_detail(self, thought: str, attraction_id: str) -> dict | None:
        """Get full attraction details: hours, capacity, pricing."""
        return self._call_tool("get_attraction_detail", thought, {"attraction_id": attraction_id})

    def search_restaurants(
        self,
        thought: str,
        city_id: str,
        cuisine: str | None = None,
        max_avg_spend: float | None = None,
    ) -> list[dict]:
        """Search restaurants, optionally filtered by cuisine or per-person spend cap."""
        return self._call_tool("search_restaurants", thought, {
            "city_id": city_id,
            "cuisine": cuisine,
            "max_avg_spend": max_avg_spend,
        }) or []

    def search_events(
        self,
        thought: str,
        city_id: str,
        start_date: str,
        end_date: str,
        category: str | None = None,
        max_price: float | None = None,
    ) -> list[dict]:
        """Search events in a city within a date range."""
        return self._call_tool("search_events", thought, {
            "city_id": city_id,
            "start_date": start_date,
            "end_date": end_date,
            "category": category,
            "max_price": max_price,
        }) or []

    def book_event(self, thought: str, event_id: str, quantity: int = 1) -> dict | None:
        """Book event tickets. Returns confirmation."""
        return self._call_tool("book_event", thought, {"event_id": event_id, "quantity": quantity})

    def plan_route(
        self,
        thought: str,
        origin_location_id: str,
        destination_location_id: str,
        departure_datetime: str,
        modes: list[str] | None = None,
        optimize_for: str = "time",
    ) -> list[dict]:
        """Get routing options between two locations. optimize_for: 'time' | 'cost' | 'distance'."""
        return self._call_tool("plan_route", thought, {
            "origin_location_id": origin_location_id,
            "destination_location_id": destination_location_id,
            "departure_datetime": departure_datetime,
            "modes": modes or ["taxi", "walk"],
            "optimize_for": optimize_for,
        }) or []

    def cancel_booking(self, thought: str, booking_ref: str) -> dict | None:
        """Cancel a hotel or event booking by its booking reference."""
        return self._call_tool("cancel_booking", thought, {"booking_ref": booking_ref})

    # ── Itinerary building helpers ────────────────────────────────────────────
    # These translate raw booking dicts (whose field names are dictated by the
    # simulator) into the project's Pydantic models, then log an itinerary step.

    def add_flight_to_itinerary(
        self,
        date_str: str,
        booking: dict,
        *,
        city: str = "",
        thought: str = "",
    ) -> None:
        """
        Record a select_flight() result as a TransportSegment.

        Parameters
        ----------
        date_str : ISO date the flight departs (YYYY-MM-DD).
        booking  : Dict returned by select_flight().
        city     : Day city label (defaults to flight's origin).
        thought  : Shown in the trail; auto-generated if omitted.
        """
        origin = booking.get("origin_city_name", "") or city
        dest = booking.get("destination_city_name", "")
        cost = float(booking.get("total_price", 0.0))

        day = self._ensure_day(date_str, city or origin)
        day.transport_segments.append(TransportSegment(
            mode="flight",
            from_location=origin,
            to_location=dest,
            departure_datetime=booking.get("departure_datetime", f"{date_str}T00:00:00"),
            arrival_datetime=booking.get("arrival_datetime", f"{date_str}T02:00:00"),
            cost_usd=cost,
            booking_ref=booking.get("booking_id"),
        ))
        self._itinerary.recompute_total_cost()

        desc = thought or f"Added flight: {origin} → {dest} on {date_str} (${cost:.2f})"
        self._log_itinerary_update(desc)
        print(f"  ✅ {desc}")

    def add_hotel_to_itinerary(
        self,
        booking: dict,
        *,
        city: str = "",
        hotel_name: str = "",
        star_rating: float | None = None,
        cost_per_night: float | None = None,
        thought: str = "",
    ) -> None:
        """
        Record a book_hotel() result as an AccommodationBooking.

        The simulator's booking dict uses 'total_cost' and 'price_per_night';
        override fields are provided for cases where those keys are absent.

        Parameters
        ----------
        booking        : Dict returned by book_hotel().
        city           : City name if not in booking dict.
        hotel_name     : Override if not in booking dict.
        star_rating    : Override if not in booking dict.
        cost_per_night : Override if not in booking dict.
        thought        : Shown in the trail; auto-generated if omitted.
        """
        check_in = booking.get("check_in", "")
        check_out = booking.get("check_out", "")
        total_cost = float(
            booking.get("total_cost", 0.0)
            or booking.get("total_price", 0.0)
            or booking.get("total_cost_usd", 0.0)
        )
        nights = _nights_between(check_in, check_out)
        name = hotel_name or booking.get("hotel_name", booking.get("name", booking.get("hotel_id", "Hotel")))
        derived_city = city or booking.get("city", "")
        ppn = cost_per_night or booking.get("price_per_night", 0.0) or (total_cost / nights)
        stars = star_rating if star_rating is not None else booking.get("star_rating")

        day = self._ensure_day(check_in, derived_city)
        day.accommodation = AccommodationBooking(
            hotel_id=booking.get("hotel_id", ""),
            hotel_name=name,
            city=derived_city,
            check_in=check_in,
            check_out=check_out,
            cost_per_night_usd=ppn,
            total_cost_usd=total_cost,
            star_rating=stars,
            booking_ref=booking.get("booking_ref") or booking.get("booking_id"),
        )
        self._itinerary.recompute_total_cost()

        desc = thought or f"Added hotel: {name} in {derived_city} ({check_in} → {check_out}, ${total_cost:.2f})"
        self._log_itinerary_update(desc)
        print(f"  ✅ {desc}")

    def add_activity_to_itinerary(
        self,
        date_str: str,
        city: str,
        *,
        activity_id: str,
        activity_name: str,
        location: str = "",
        start_time: str = "10:00",
        duration_hours: float = 2.0,
        cost_usd: float = 0.0,
        category: str = "sightseeing",
        booking_ref: str | None = None,
        thought: str = "",
    ) -> None:
        """
        Add any activity, attraction, or restaurant visit to a day.

        Parameters
        ----------
        date_str      : ISO date (YYYY-MM-DD).
        city          : City where the activity is located.
        activity_id   : ID from the search result (attraction_id, restaurant_id, etc.).
        activity_name : Human-readable name.
        start_time    : HH:MM string, e.g. "09:30".
        duration_hours: Estimated duration.
        cost_usd      : Per-person cost (or total if already multiplied by group size).
        category      : e.g. "hiking", "museum", "restaurant", "event".
        booking_ref   : From book_event() if tickets were reserved.
        thought       : Shown in the trail; auto-generated if omitted.
        """
        day = self._ensure_day(date_str, city)
        day.activities.append(ActivityBooking(
            activity_id=activity_id,
            activity_name=activity_name,
            location=location or city,
            city=city,
            start_datetime=f"{date_str}T{start_time}:00",
            duration_hours=duration_hours,
            cost_usd=cost_usd,
            category=category,
            booking_ref=booking_ref,
        ))
        self._itinerary.recompute_total_cost()

        desc = thought or f"Added {category}: {activity_name} on {date_str} at {start_time} (${cost_usd:.2f})"
        self._log_itinerary_update(desc)
        print(f"  ✅ {desc}")

    def add_event_to_itinerary(
        self,
        booking: dict,
        city: str,
        *,
        event_name: str = "",
        category: str = "event",
        duration_hours: float = 2.0,
        thought: str = "",
    ) -> None:
        """
        Record a book_event() result as an ActivityBooking.

        Extracts date and start time from the booking's start_datetime field
        and delegates to add_activity_to_itinerary().
        """
        start_dt: str = booking.get("start_datetime", booking.get("start_time", ""))
        date_str = start_dt[:10] if len(start_dt) >= 10 else ""
        start_time = start_dt[11:16] if len(start_dt) > 10 else "19:00"
        cost = float(booking.get("total_price", 0.0) or booking.get("cost", 0.0))
        name = event_name or booking.get("event_name", booking.get("name", "Event"))

        self.add_activity_to_itinerary(
            date_str=date_str,
            city=city,
            activity_id=booking.get("event_id", ""),
            activity_name=name,
            location=booking.get("venue", city),
            start_time=start_time,
            duration_hours=duration_hours,
            cost_usd=cost,
            category=category,
            booking_ref=booking.get("booking_ref") or booking.get("booking_id"),
            thought=thought or f"Added event: {name} in {city} on {date_str} (${cost:.2f})",
        )

    def add_transport_to_itinerary(
        self,
        date_str: str,
        city: str,
        *,
        mode: str,
        from_location: str,
        to_location: str,
        departure_datetime: str,
        arrival_datetime: str,
        cost_usd: float = 0.0,
        booking_ref: str | None = None,
        thought: str = "",
    ) -> None:
        """
        Manually record any transport segment (train, taxi, ferry, etc.).

        Use this when plan_route() identifies the leg but there is no
        dedicated booking tool — you record the decision directly.
        """
        day = self._ensure_day(date_str, city)
        day.transport_segments.append(TransportSegment(
            mode=mode,
            from_location=from_location,
            to_location=to_location,
            departure_datetime=departure_datetime,
            arrival_datetime=arrival_datetime,
            cost_usd=cost_usd,
            booking_ref=booking_ref,
        ))
        self._itinerary.recompute_total_cost()

        desc = thought or f"Added {mode}: {from_location} → {to_location} on {date_str} (${cost_usd:.2f})"
        self._log_itinerary_update(desc)
        print(f"  ✅ {desc}")

    # ── Display helpers ───────────────────────────────────────────────────────

    def show_itinerary(self) -> None:
        """Pretty-print the current itinerary with per-day cost breakdown."""
        itin = self._itinerary
        print("═" * 64)
        print(f"  ITINERARY  —  {self._request.request_id}")
        print("═" * 64)
        if not itin.days:
            print("  (no days planned yet)")
        for day in itin.days:
            print(f"\n  📅 {day.date}  |  {day.city}  (${day.total_cost_usd:.2f})")
            for seg in day.transport_segments:
                ref = f"  [{seg.booking_ref}]" if seg.booking_ref else ""
                print(f"     ✈  {seg.mode.upper()}: {seg.from_location} → {seg.to_location}"
                      f"  {seg.departure_datetime[11:16]} → {seg.arrival_datetime[11:16]}"
                      f"  ${seg.cost_usd:.2f}{ref}")
            if day.accommodation:
                a = day.accommodation
                stars = f"  ({'★' * int(a.star_rating or 0)})" if a.star_rating else ""
                print(f"     🏨  {a.hotel_name}{stars}  {a.check_in} → {a.check_out}  ${a.total_cost_usd:.2f}")
            for act in day.activities:
                ref = f"  [{act.booking_ref}]" if act.booking_ref else ""
                print(f"     🎯  {act.start_datetime[11:16]}  [{act.category}]  {act.activity_name}  ${act.cost_usd:.2f}{ref}")
        print(f"\n  Spent: ${itin.total_cost_usd:,.2f}  /  Budget: ${self._request.budget_usd:,.2f}"
              f"  /  Remaining: ${self.budget_remaining:,.2f}"
              + ("  ⚠️  OVER BUDGET" if self.budget_remaining < 0 else ""))
        print("═" * 64)

    def show_trail(self) -> None:
        """Print the full episode trail (all logged ReActStep entries)."""
        print("═" * 64)
        print(f"  EPISODE TRAIL  —  {self._episode_id}  ({len(self._steps)} steps)")
        print("═" * 64)
        if not self._steps:
            print("  (no steps yet)")
        for step in self._steps:
            label = "TOOL" if step.action else "PLAN"
            print(f"\n  [{step.step_index:02d}] [{label}]  {step.thought}")
            if step.action:
                print(f"        Action: {step.action.tool_name}({str(step.action.arguments)[:80]})")
            if step.observation:
                status = "✓" if step.observation.success else "✗"
                print(f"        Obs [{status}]: {str(step.observation.result)[:100]}")
            if step.itinerary_snapshot:
                snap = step.itinerary_snapshot
                print(f"        Itinerary: {len(snap.days)} day(s), ${snap.total_cost_usd:.2f} total")

    def show_budget(self) -> None:
        """Print a budget breakdown by category."""
        itin = self._itinerary
        transport = sum(seg.cost_usd for day in itin.days for seg in day.transport_segments)
        accommodation = sum(day.accommodation.total_cost_usd for day in itin.days if day.accommodation)
        activities = sum(act.cost_usd for day in itin.days for act in day.activities)
        print(f"  Budget            : ${self._request.budget_usd:>10,.2f}")
        print(f"  Transport         : ${transport:>10,.2f}")
        print(f"  Accommodation     : ${accommodation:>10,.2f}")
        print(f"  Activities/Events : ${activities:>10,.2f}")
        print(f"  {'─' * 29}")
        print(f"  Spent             : ${itin.total_cost_usd:>10,.2f}")
        print(f"  Remaining         : ${self.budget_remaining:>10,.2f}")
        if self.budget_remaining < 0:
            print(f"  ⚠️  Over budget by ${abs(self.budget_remaining):.2f}")

    # ── Session finalization ──────────────────────────────────────────────────

    def finalize(self, termination_reason: str = "DONE_ITINERARY") -> EpisodeLog:
        """
        Assemble and return the complete EpisodeLog.

        termination_reason:
            'DONE_ITINERARY'  — plan is complete (default).
            'HUMAN_INCOMPLETE' — plan is partial (stopped early).
            'BUDGET_EXCEEDED'  — plan exceeds the budget.

        The returned log is identical in structure to agent-generated episodes
        and can be passed directly to DeterministicEvaluator or LLMJudge.
        """
        if self._finalized and self._episode_log is not None:
            return self._episode_log

        self._itinerary.is_complete = (termination_reason == "DONE_ITINERARY")
        self._itinerary.version += 1

        trajectory = TrajectoryModel(
            trajectory_id=uuid.uuid4().hex[:12],
            request_id=self._request.request_id,
            steps=tuple(self._steps),
            total_steps=len(self._steps),
        )

        reward_components = self._compute_reward_components()

        self._episode_log = EpisodeLog(
            episode_id=self._episode_id,
            request_id=self._request.request_id,
            agent_mode=self.AGENT_MODE,
            trajectory=trajectory,
            compressed_states=(),
            final_itinerary=self._itinerary,
            reward_components=reward_components,
            tool_stats=tuple(self._tracker.get_stats()),
            total_steps=len(self._steps),
            total_tokens_used=None,
            mcts_stats=None,
            world_seed=self._simulator.get_world_seed(),
            success=(termination_reason == "DONE_ITINERARY"),
            error=None,
            termination_reason=termination_reason,
            user_request=self._request,
            config_hash="human_baseline_session",
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        self._finalized = True

        print(f"\n  Episode finalized — {len(self._steps)} steps, "
              f"${self._itinerary.total_cost_usd:.2f} spent, "
              f"hard_score={reward_components.hard_constraint_score:.2f}")
        return self._episode_log

    def save(self, directory: str | Path | None = None) -> Path:
        """Serialize the EpisodeLog to ep_{episode_id}.json. Calls finalize() if needed."""
        if not self._finalized or self._episode_log is None:
            self.finalize()
        assert self._episode_log is not None
        path = save_episode(self._episode_log, Path(directory) if directory else self._output_dir)
        print(f"  Saved → {path}")
        return path

    # ── Reward computation ────────────────────────────────────────────────────

    def _compute_reward_components(self) -> RewardComponents:
        """
        Compute RewardComponents without an EpisodeLog.

        DeterministicEvaluator.score() requires an EpisodeLog, but EpisodeLog
        requires RewardComponents — chicken-and-egg. We break the cycle by
        calling ConstraintSatisfactionEngine directly, the same class used by
        both RewardFunction and DeterministicEvaluator.
        """
        engine = ConstraintSatisfactionEngine()
        itin = self._itinerary

        hard_ratio = 0.0
        if self._request.hard_constraints and itin.days:
            results = engine.evaluate(itin, list(self._request.hard_constraints))
            hard_ratio = engine.hard_satisfaction_ratio(results, list(self._request.hard_constraints))

        # 1.0 when no soft constraints — matches reward.py behaviour.
        soft_score = 1.0
        if self._request.soft_constraints and itin.days:
            results = engine.evaluate(itin, list(self._request.soft_constraints))
            soft_score = engine.soft_satisfaction_score(results, list(self._request.soft_constraints))

        stats = self._tracker.get_stats()
        total_calls = sum(s.call_count for s in stats)
        redundant = sum(s.redundant_call_count for s in stats)
        failures = sum(s.failure_count for s in stats)

        efficiency = max(0.0, 1.0 - redundant / total_calls) if total_calls > 0 else 1.0
        failure_rate = failures / total_calls if total_calls > 0 else 0.0

        # Days must be sorted with no duplicates for a logically consistent plan.
        dates = [d.date for d in itin.days]
        consistency = 1.0 if dates == sorted(set(dates)) else 0.5

        total = 0.40 * hard_ratio + 0.20 * soft_score + 0.15 * efficiency + 0.15 * consistency - 0.10 * failure_rate

        return RewardComponents(
            hard_constraint_score=hard_ratio,
            soft_constraint_score=soft_score,
            tool_efficiency_score=efficiency,
            tool_failure_penalty=-failure_rate,
            logical_consistency_score=consistency,
            terminal_itinerary_score=hard_ratio,
            total_reward=total,
        )


# ── Module-level utility ──────────────────────────────────────────────────────

def _nights_between(check_in: str, check_out: str) -> int:
    """Return number of nights between two ISO date strings. Minimum 1."""
    try:
        return max(1, (date.fromisoformat(check_out) - date.fromisoformat(check_in)).days)
    except (ValueError, TypeError):
        return 1


# ── Display helpers ───────────────────────────────────────────────────────────

def _df_display(rows: list[dict], cols: list[str]) -> None:
    """Render rows as a DataFrame, or plain text if pandas/IPython unavailable."""
    try:
        import pandas as pd
        df = pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows])
        df.index += 1
        try:
            from IPython.display import display
            display(df)
        except ImportError:
            print(df.to_string())
    except ImportError:
        for i, r in enumerate(rows, 1):
            print(f"  [{i}]  " + "  |  ".join(f"{c}: {r.get(c, '')}" for c in cols))


def display_routes(routes: list[dict]) -> None:
    """Display get_available_routes() results."""
    if not routes:
        print("No routes returned.")
        return
    rows = [
        {
            "city_name": r.get("city_name", r.get("name", "")),
            "city_id": r.get("city_id", r.get("id", "")),
            "connects_to": ", ".join(
                c.get("city_name", c.get("name", ""))
                for c in r.get("connections", [])
                if isinstance(r.get("connections"), list)
            ),
        }
        for r in routes
    ]
    _df_display(rows, ["city_name", "city_id", "connects_to"])


def display_flights(flights: list[dict]) -> None:
    """Display search_flights() results."""
    if not flights:
        print("No flights found.")
        return
    _df_display(flights, [
        "edge_id", "origin_city_id", "destination_city_id",
        "departure_datetime", "arrival_datetime", "total_price",
    ])


def display_hotels(hotels: list[dict]) -> None:
    """Display search_hotels() results."""
    if not hotels:
        print("No hotels found.")
        return
    _df_display(hotels, ["hotel_id", "name", "star_rating", "price_per_night", "total_cost"])


def display_attractions(attractions: list[dict]) -> None:
    """Display search_attractions() results."""
    if not attractions:
        print("No attractions found.")
        return
    _df_display(attractions, ["attraction_id", "name", "category", "admission_fee", "location"])


def display_restaurants(restaurants: list[dict]) -> None:
    """Display search_restaurants() results."""
    if not restaurants:
        print("No restaurants found.")
        return
    _df_display(restaurants, ["restaurant_id", "name", "cuisine", "avg_spend_per_person"])


def display_events(events: list[dict]) -> None:
    """Display search_events() results."""
    if not events:
        print("No events found.")
        return
    _df_display(events, ["event_id", "name", "category", "start_datetime", "price", "available_tickets"])


def display_scores(scores: dict[str, float]) -> None:
    """Display DeterministicEvaluator scores grouped by category with bar charts."""
    if not scores:
        print("No scores to display.")
        return
    groups = {
        "Constraint satisfaction": [
            "hard_constraint_ratio", "soft_constraint_score", "budget_adherence",
        ],
        "Tool usage": [
            "tool_efficiency", "tool_failure_rate", "avg_tool_latency_ms", "steps_per_episode",
        ],
        "Trip quality": [
            "destination_coverage_ratio", "accommodation_coverage_ratio",
            "activity_density_score", "rest_day_ratio", "schedule_overlap_score",
            "intra_city_feasibility", "logical_consistency",
        ],
    }
    print("─" * 52)
    print("  EVALUATION SCORES")
    print("─" * 52)
    for group, keys in groups.items():
        print(f"\n  {group}:")
        for k in keys:
            if k in scores:
                v = scores[k]
                bar = _bar(v) if 0.0 <= v <= 1.0 else ""
                print(f"    {k:<35} {v:6.3f}  {bar}")
    print("─" * 52)


def _bar(value: float, width: int = 10) -> str:
    filled = round(value * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"
