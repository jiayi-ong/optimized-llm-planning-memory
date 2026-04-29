"""
scripts/run_baseline_eval.py
==============================
Run a single-episode baseline evaluation against the first hand-crafted
user request (data/user_requests/test/aeloria_4day_v001.json).

Two modes
---------
LLM mode (default when ANTHROPIC_API_KEY or OPENAI_API_KEY is set):
    Runs the full ReActAgent driven by the configured LLM.

Scripted mode (fallback, no API key needed):
    Runs a deterministic agent that makes the "reasonable" sequence of tool
    calls an expert planner would make.  This produces a real EpisodeLog with
    real simulator results, so the deterministic evaluator scores are genuine.
    Use this to: (a) verify the pipeline end-to-end, (b) establish a
    rule-based planning baseline before the LLM is wired in.

Usage
-----
    python scripts/run_baseline_eval.py
    python scripts/run_baseline_eval.py agent.mode=raw   # once API key set
    python scripts/run_baseline_eval.py scripted=true    # force scripted mode
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Scripted agent ────────────────────────────────────────────────────────────


def run_scripted_episode(request, simulator, registry=None, compressor=None):
    """
    Execute a deterministic planning sequence using real tool calls.

    Sequence
    --------
    1. get_available_routes       → discover city_id
    2. search_hotels              → find 3+ star options
    3. book_hotel                 → book the cheapest qualifying hotel
    4. search_attractions         → find attractions (up to 3)
    5. search_restaurants         → find local cuisine restaurants (up to 2)
    6. search_events              → find affordable events (ticket < $60)
    7. book_event                 → book the cheapest qualifying event (if any)
    8. DONE
    """
    if registry is None:
        from optimized_llm_planning_memory.tools.tracker import ToolCallTracker
        from optimized_llm_planning_memory.tools.events import EventBus
        from optimized_llm_planning_memory.tools.registry import ToolRegistry
        registry = ToolRegistry.from_config(
            simulator=simulator,
            tracker=ToolCallTracker(),
            event_bus=EventBus(),
        )
    if compressor is None:
        from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
        compressor = IdentityCompressor()

    from optimized_llm_planning_memory.agent.trajectory import Trajectory
    from optimized_llm_planning_memory.core.models import (
        AccommodationBooking,
        ActivityBooking,
        CompressedState,
        EpisodeLog,
        Itinerary,
        ItineraryDay,
        ReActStep,
        RewardComponents,
        ToolCall,
        ToolResult,
        TransportSegment,
    )
    from optimized_llm_planning_memory.tools.tracker import ToolCallTracker
    from optimized_llm_planning_memory.tools.events import EventBus

    episode_id = str(uuid.uuid4())
    trajectory = Trajectory(request_id=request.request_id)
    final_itinerary = Itinerary(
        itinerary_id=request.request_id,
        request_id=request.request_id,
    )

    steps = []
    step_i = 0

    def _call(tool_name: str, args: dict, thought: str) -> ToolResult:
        nonlocal step_i
        tc = ToolCall(tool_name=tool_name, arguments=args, raw_text=f"{tool_name}({args})")
        try:
            tool = registry.get(tool_name)
            result = tool.call(args)
        except Exception as exc:
            result = ToolResult(
                tool_name=tool_name, success=False,
                result=None, error_message=str(exc), latency_ms=0.0,
            )
        step = ReActStep(
            step_index=step_i,
            thought=thought,
            action=tc,
            observation=result,
            itinerary_snapshot=None,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )
        steps.append(step)
        trajectory.add_step(step)
        step_i += 1
        return result

    # ── Step 1: city discovery ─────────────────────────────────────────────────
    r1 = _call("get_available_routes", {},
               "I need to discover the city_id for Aeloria before making any searches.")
    city_id = None
    city_name = "Aeloria"
    if r1.success and isinstance(r1.result, list) and r1.result:
        city_id = r1.result[0].get("city_id")
        city_name = r1.result[0].get("city_name", "Aeloria")
    if not city_id:
        print("WARN: could not discover city_id from get_available_routes")
        city_id = "unknown"

    check_in, check_out = request.start_date, request.end_date

    # Pre-populate one ItineraryDay per night (check-in through check-out) so that:
    #   (a) hc-dates constraint can verify days[0] and days[-1] cover the full range
    #   (b) activities can be spread across nights instead of stacked on check-in
    from datetime import date as _date, timedelta as _td
    _cur = _date.fromisoformat(check_in)
    _end = _date.fromisoformat(check_out)
    while _cur <= _end:
        _get_or_create_day(final_itinerary, _cur.isoformat(), city_name)
        _cur += _td(days=1)
    # Night dates only (excludes check-out morning) — used for activity distribution.
    _trip_nights = [d.date for d in final_itinerary.days if d.date < check_out]

    # ── Step 2: hotel search ───────────────────────────────────────────────────
    r2 = _call("search_hotels",
               {"city_id": city_id, "check_in": check_in, "check_out": check_out,
                "guests": 2, "min_stars": 3.0},
               "Searching for 3-star+ hotels within my budget.")
    booked_hotel = None
    if r2.success and isinstance(r2.result, list):
        # Pick cheapest hotel that fits in budget (hotel cost ≤ 50% of total budget)
        # Pick cheapest qualifying hotel regardless of price;
        # budget_adherence metric in the evaluator handles overspend.
        candidates = [h for h in r2.result
                      if isinstance(h, dict) and h.get("star_rating", 0) >= 3]
        candidates.sort(key=lambda h: h.get("total_cost", 9999))
        if candidates:
            booked_hotel = candidates[0]

    # ── Step 3: hotel booking ──────────────────────────────────────────────────
    hotel_booking = None
    if booked_hotel:
        r3 = _call("book_hotel",
                   {"hotel_id": booked_hotel["hotel_id"],
                    "check_in": check_in, "check_out": check_out},
                   f"Booking {booked_hotel['name']} ({booked_hotel['star_rating']}*, "
                   f"${booked_hotel['total_cost']:.0f} total).")
        if r3.success and isinstance(r3.result, dict):
            res = r3.result
            hotel_booking = AccommodationBooking(
                hotel_id=res.get("hotel_id", booked_hotel["hotel_id"]),
                hotel_name=res.get("hotel_name", booked_hotel.get("name", "")),
                city=city_name,
                check_in=check_in,
                check_out=check_out,
                cost_per_night_usd=booked_hotel.get("price_per_night", 0.0),
                total_cost_usd=res.get("total_cost", booked_hotel.get("total_cost", 0.0)),
                star_rating=booked_hotel.get("star_rating"),
                booking_ref=res.get("booking_id"),
            )
            # Attach to check-in day only; AccommodationBooking.check_out covers full stay.
            # Attaching to every night would make the duplicate-hotel consistency check flag it.
            day = _get_or_create_day(final_itinerary, check_in, city_name)
            day.accommodation = hotel_booking

    # ── Step 4: attraction search ──────────────────────────────────────────────
    r4 = _call("search_attractions", {"city_id": city_id},
               "Searching for attractions to visit during the stay.")
    attraction_count = 0
    if r4.success and isinstance(r4.result, list):
        # Pick up to 3 attractions affordable for 2 people; sort by ticket price
        affordable = sorted(
            [a for a in r4.result if isinstance(a, dict)],
            key=lambda a: a.get("ticket_price", 0.0) or 0.0,
        )[:5]
        for i, attr in enumerate(affordable[:3]):
            # One attraction per trip night so no same-day time overlaps occur.
            target_date = _trip_nights[i % len(_trip_nights)] if _trip_nights else check_in
            ticket = attr.get("ticket_price", 0.0) or 0.0
            activity = ActivityBooking(
                activity_id=attr.get("attraction_id", str(uuid.uuid4())),
                activity_name=attr.get("name", "Attraction"),
                location=attr.get("district_name", city_name),
                city=city_name,
                start_datetime=f"{target_date}T09:00:00",
                duration_hours=attr.get("duration_hours", 2.0) or 2.0,
                cost_usd=ticket * 2,  # 2 adults
                category=attr.get("category", "attraction"),
                booking_ref=None,
            )
            day = _get_or_create_day(final_itinerary, target_date, city_name)
            day.activities.append(activity)
            attraction_count += 1

    # ── Step 5: restaurant search + add dining activities ────────────────────
    r5 = _call("search_restaurants", {"city_id": city_id},
               "Looking for local restaurants for a couple of dinners.")
    if r5.success and isinstance(r5.result, list):
        restaurants = [r for r in r5.result if isinstance(r, dict)][:2]
        for i, rest in enumerate(restaurants):
            target_date = _trip_nights[i % len(_trip_nights)] if _trip_nights else check_in
            meal_cost = float(
                rest.get("avg_meal_cost_per_person") or rest.get("avg_cost_per_person") or 20.0
            )
            restaurant_activity = ActivityBooking(
                activity_id=rest.get("restaurant_id", str(uuid.uuid4())),
                activity_name=rest.get("name", "Restaurant"),
                location=rest.get("district_name", city_name),
                city=city_name,
                start_datetime=f"{target_date}T12:00:00",
                duration_hours=1.5,
                cost_usd=meal_cost * 2,  # 2 adults
                category="restaurant",
                booking_ref=None,
            )
            day = _get_or_create_day(final_itinerary, target_date, city_name)
            day.activities.append(restaurant_activity)

    # ── Step 6: event search ───────────────────────────────────────────────────
    r6 = _call("search_events",
               {"city_id": city_id, "start_date": check_in,
                "end_date": check_out, "max_price": 60.0},
               "Checking for affordable evening events (ticket price < $60).")
    affordable_events = []
    if r6.success and isinstance(r6.result, list):
        affordable_events = [
            e for e in r6.result
            if isinstance(e, dict) and (e.get("base_ticket_price") or 0) <= 60.0
        ]

    # ── Step 7: event booking ──────────────────────────────────────────────────
    event_booked = False
    if affordable_events:
        evt = affordable_events[0]
        r7 = _call("book_event",
                   {"event_id": evt["event_id"], "quantity": 2},
                   f"Booking event '{evt.get('name', 'event')}' "
                   f"(${evt.get('base_ticket_price', 0):.0f}/ticket × 2).")
        if r7.success and isinstance(r7.result, dict):
            res = r7.result
            event_date = _trip_nights[-1] if _trip_nights else check_in
            event_activity = ActivityBooking(
                activity_id=res.get("event_id", evt["event_id"]),
                activity_name=res.get("event_name", evt.get("name", "Event")),
                location=evt.get("venue_name", city_name),
                city=city_name,
                start_datetime=f"{event_date}T19:00:00",
                duration_hours=2.0,
                cost_usd=res.get("total_cost", 0.0),
                category="event",
                booking_ref=res.get("booking_id"),
            )
            day = _get_or_create_day(final_itinerary, event_date, city_name)
            day.activities.append(event_activity)
            event_booked = True

    # ── Step 8: DONE ──────────────────────────────────────────────────────────
    done_step = ReActStep(
        step_index=step_i,
        thought=(
            f"Planning complete. Hotel booked: {bool(hotel_booking)}. "
            f"Attractions added: {attraction_count}. "
            f"Event booked: {event_booked}."
        ),
        action=None,
        observation=None,
        itinerary_snapshot=final_itinerary,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )
    steps.append(done_step)
    trajectory.add_step(done_step)

    final_itinerary.recompute_total_cost()
    final_itinerary.is_complete = bool(hotel_booking)

    return EpisodeLog(
        episode_id=episode_id,
        request_id=request.request_id,
        agent_mode="scripted_baseline",
        trajectory=trajectory.to_model(),
        compressed_states=(),
        final_itinerary=final_itinerary,
        reward_components=RewardComponents(
            hard_constraint_score=0.0,
            soft_constraint_score=0.0,
            tool_efficiency_score=0.0,
            tool_failure_penalty=0.0,
            logical_consistency_score=0.0,
            total_reward=0.0,
        ),
        tool_stats=(),
        total_steps=len(steps),
        success=bool(hotel_booking),
        error=None,
        config_hash="scripted",
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


def _get_or_create_day(itinerary, date: str, city: str):
    from optimized_llm_planning_memory.core.models import ItineraryDay
    for day in itinerary.days:
        if day.date == date:
            return day
    new_day = ItineraryDay(date=date, city=city)
    itinerary.days.append(new_day)
    return new_day


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    import json
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from optimized_llm_planning_memory.core.models import UserRequest
    from optimized_llm_planning_memory.evaluation.deterministic import DeterministicEvaluator
    from optimized_llm_planning_memory.simulator.adapter import SimulatorAdapter
    from optimized_llm_planning_memory.tools.events import EventBus
    from optimized_llm_planning_memory.tools.registry import ToolRegistry
    from optimized_llm_planning_memory.tools.tracker import ToolCallTracker
    from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor

    REQUEST_FILE = Path("data/user_requests/test/aeloria_4day_v001.json")
    SEED = 42

    # ── Load request ──────────────────────────────────────────────────────────
    request = UserRequest.model_validate(json.loads(REQUEST_FILE.read_text()))
    print(f"Request: {request.request_id}")
    print(f"  Destination: {request.destination_cities[0]}")
    print(f"  Dates: {request.start_date} to {request.end_date}")
    print(f"  Budget: ${request.budget_usd:,.0f} USD")
    print(f"  Hard constraints: {len(request.hard_constraints)}")
    print(f"  Soft constraints: {len(request.soft_constraints)}")
    print()

    # ── Build simulator ───────────────────────────────────────────────────────
    print(f"Initialising simulator (seed={SEED})...")
    sim = SimulatorAdapter(seed=SEED)
    routes = sim.get_available_routes()
    if routes:
        city = routes[0]
        print(f"  World city: {city['city_name']} (id: {city['city_id'][:30]}...)")
        print(f"  Vibe: {city.get('vibe_summary', '')[:80]}...")
    print()

    tracker = ToolCallTracker()
    event_bus = EventBus()
    registry = ToolRegistry.from_config(simulator=sim, tracker=tracker, event_bus=event_bus)
    compressor = IdentityCompressor()

    # ── Choose mode ───────────────────────────────────────────────────────────
    force_scripted = "--scripted" in sys.argv or "scripted=true" in sys.argv
    has_api_key = bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )

    if has_api_key and not force_scripted:
        print("API key found — running full LLM agent...")
        from optimized_llm_planning_memory.agent.react_agent import ReActAgent
        from optimized_llm_planning_memory.agent.context_builder import ContextBuilder
        from optimized_llm_planning_memory.agent.prompts import get_system_prompt
        from optimized_llm_planning_memory.agent.modes import AgentMode
        from optimized_llm_planning_memory.core.config import AgentConfig

        llm_model = os.environ.get("AGENT_LLM_MODEL_ID", "anthropic/claude-haiku-4-5-20251001")
        agent_config = AgentConfig(
            mode="compressor",
            llm_model_id=llm_model,
            max_steps=20,
            compress_every_n_steps=999,
        )
        context_builder = ContextBuilder(
            system_prompt=get_system_prompt("v1"),
            tool_registry=registry,
            llm_model_id=llm_model,
        )
        agent = ReActAgent(
            llm_model_id=llm_model,
            tool_registry=registry,
            compressor=compressor,
            context_builder=context_builder,
            config=agent_config,
            mode=AgentMode.COMPRESSOR,
        )
        episode_log = agent.run_episode(request=request, simulator=sim)
        mode_label = f"LLM ({llm_model})"
    else:
        if not has_api_key:
            print("No API key found - running scripted baseline agent.")
            print("  Set ANTHROPIC_API_KEY or OPENAI_API_KEY to run the LLM agent.")
        else:
            print("Scripted mode forced via CLI.")
        print()
        episode_log = run_scripted_episode(request, sim, registry, compressor)
        mode_label = "Scripted Baseline"

    # ── Print episode summary ─────────────────────────────────────────────────
    print("=" * 60)
    print(f"EPISODE COMPLETE — mode: {mode_label}")
    print("=" * 60)
    print(f"  success      : {episode_log.success}")
    print(f"  total_steps  : {episode_log.total_steps}")
    if episode_log.error:
        print(f"  error        : {episode_log.error}")

    it = episode_log.final_itinerary
    if it:
        print(f"  total_cost   : ${it.total_cost_usd:.2f}")
        print(f"  days planned : {len(it.days)}")
        hotels = [d.accommodation for d in it.days if d.accommodation]
        print(f"  hotel nights : {len(hotels)}")
        acts = [a for d in it.days for a in d.activities]
        print(f"  activities   : {len(acts)}")
        if hotels:
            h = hotels[0]
            print(f"  hotel        : {h.hotel_name} ({h.cost_per_night_usd:.0f}/night, ${h.total_cost_usd:.0f} total)")

    # ── Deterministic evaluation ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print("DETERMINISTIC EVALUATION METRICS")
    print("=" * 60)
    evaluator = DeterministicEvaluator()
    metrics = evaluator.score(episode_log, request)
    max_key = max(len(k) for k in metrics)
    for k, v in metrics.items():
        bar = "#" * int(v * 20) if 0.0 <= v <= 1.0 else ""
        print(f"  {k:<{max_key}}  {v:6.3f}  {bar}")

    # ── Tool call breakdown ───────────────────────────────────────────────────
    print()
    print("TOOL CALL BREAKDOWN")
    print("-" * 60)
    for step in episode_log.trajectory.steps:
        if step.action:
            status = "OK" if step.observation and step.observation.success else "FAIL"
            result_summary = ""
            if step.observation and step.observation.success and isinstance(step.observation.result, list):
                result_summary = f"({len(step.observation.result)} results)"
            elif step.observation and step.observation.success and isinstance(step.observation.result, dict):
                keys_shown = {k: step.observation.result[k] for k in list(step.observation.result.keys())[:3]}
                result_summary = f"({keys_shown})"
            print(f"  [{status}] step {step.step_index:2d}  {step.action.tool_name:<28s}  {result_summary}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = Path("outputs/baseline_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    episode_path = out_dir / f"episode_{ts}.json"
    metrics_path = out_dir / f"metrics_{ts}.json"

    episode_path.write_text(episode_log.model_dump_json(indent=2), encoding="utf-8")
    metrics_output = {
        "request_id": request.request_id,
        "mode": mode_label,
        "timestamp": ts,
        "episode_id": episode_log.episode_id,
        "success": episode_log.success,
        "total_steps": episode_log.total_steps,
        "total_cost_usd": it.total_cost_usd if it else None,
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(metrics_output, indent=2), encoding="utf-8")

    print()
    print(f"Episode log  : {episode_path}")
    print(f"Metrics JSON : {metrics_path}")

    return metrics


if __name__ == "__main__":
    main()
