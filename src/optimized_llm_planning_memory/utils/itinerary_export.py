"""
utils/itinerary_export.py
=========================
Convert the agent-internal Itinerary model to the ItineraryManifest format
defined by the adjacent my-travel-world library.

Why two representations?
-------------------------
- ``Itinerary`` (this project) is a *mutable, incremental* working copy built
  up as the agent calls booking tools during a single episode.
- ``ItineraryManifest`` (my-travel-world) is a *self-contained, evaluable*
  artifact that carries full location context (lat/lon, world_id) so an
  external evaluator can run all three evaluation tiers without re-querying
  the world API.

Conversion fidelity notes
--------------------------
- ``ItineraryItem.location_id`` and ``coordinates`` cannot be populated from
  the agent-internal model alone; they default to empty strings and
  ``{"lat": 0.0, "lon": 0.0}`` unless a ``simulator`` is supplied for lookup.
- ``ItineraryTransitSegment`` objects are not produced; the agent does not
  currently track explicit transit legs between activities.
- ``trip_date_range`` is derived from the earliest and latest dates across all
  days in the itinerary.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from optimized_llm_planning_memory.core.models import Itinerary, UserRequest

if TYPE_CHECKING:
    from optimized_llm_planning_memory.simulator.protocol import SimulatorProtocol


def to_itinerary_manifest(
    itinerary: Itinerary,
    request: UserRequest,
    world_id: str,
    episode_id: str,
    simulator: "SimulatorProtocol | None" = None,
) -> object:
    """
    Convert an agent-internal ``Itinerary`` to a ``ItineraryManifest``.

    Parameters
    ----------
    itinerary  : The completed (or partial) agent Itinerary.
    request    : The UserRequest that prompted the episode — used to populate
                 user_preferences, constraints_declared, and budget_declared.
    world_id   : Identifier of the simulator world instance.
    episode_id : The episode ID; stored as ``tool_calls_log_ref`` for replay.
    simulator  : Optional SimulatorProtocol instance.  When provided, location
                 lookups are attempted to populate ``location_id`` and
                 ``coordinates`` on each ItineraryItem.  When None, these fields
                 default to empty/zero values.

    Returns
    -------
    ItineraryManifest
        A my-travel-world ItineraryManifest populated from the agent itinerary.

    Raises
    ------
    ImportError
        If the ``travel_world`` package is not installed or importable.
    """
    try:
        from travel_world.core.itinerary import ItineraryItem, ItineraryManifest
    except ImportError as exc:
        raise ImportError(
            "Cannot import travel_world.core.itinerary. "
            "Ensure the my-travel-world package is installed or on sys.path."
        ) from exc

    items: list[ItineraryItem] = []
    total_by_category: dict[str, float] = {"flights": 0.0, "hotels": 0.0, "activities": 0.0}
    all_dates: list[date] = []

    for day in itinerary.days:
        day_date = _parse_date(day.date)
        if day_date:
            all_dates.append(day_date)

        for seg in day.transport_segments:
            dep = _parse_datetime(seg.departure_datetime)
            arr = _parse_datetime(seg.arrival_datetime)
            if dep is None:
                dep = datetime.now(tz=timezone.utc)
            if arr is None:
                arr = dep

            location_id, coords = _resolve_location(seg.from_location, simulator)
            item = ItineraryItem(
                item_id=str(uuid.uuid4()),
                item_type="flight",
                entity_id=seg.booking_ref or "",
                title=f"{seg.from_location} → {seg.to_location}",
                start_datetime=dep,
                end_datetime=arr,
                location_id=location_id,
                coordinates=coords,
                city_id=seg.from_location,
                confirmed_price=seg.cost_usd,
                currency="USD",
                booking_reference=seg.booking_ref,
                metadata={"mode": seg.mode, "destination": seg.to_location},
            )
            items.append(item)
            total_by_category["flights"] += seg.cost_usd

        if day.accommodation:
            acc = day.accommodation
            check_in_dt = _parse_datetime(acc.check_in + "T14:00:00Z")
            check_out_dt = _parse_datetime(acc.check_out + "T11:00:00Z")
            if check_in_dt is None:
                check_in_dt = datetime.now(tz=timezone.utc)
            if check_out_dt is None:
                check_out_dt = check_in_dt

            location_id, coords = _resolve_location(acc.hotel_id, simulator)
            item = ItineraryItem(
                item_id=str(uuid.uuid4()),
                item_type="hotel",
                entity_id=acc.hotel_id,
                title=acc.hotel_name,
                start_datetime=check_in_dt,
                end_datetime=check_out_dt,
                location_id=location_id,
                coordinates=coords,
                city_id=acc.city,
                confirmed_price=acc.total_cost_usd,
                currency="USD",
                booking_reference=acc.booking_ref,
                metadata={
                    "cost_per_night": acc.cost_per_night_usd,
                    "check_in": acc.check_in,
                    "check_out": acc.check_out,
                },
            )
            items.append(item)
            total_by_category["hotels"] += acc.total_cost_usd
            if day_date:
                all_dates.append(_parse_date(acc.check_in) or day_date)
                checkout = _parse_date(acc.check_out)
                if checkout:
                    all_dates.append(checkout)

        for act in day.activities:
            start = _parse_datetime(act.start_datetime)
            if start is None:
                start = datetime.now(tz=timezone.utc)
            end = datetime(start.year, start.month, start.day,
                           min(start.hour + int(act.duration_hours), 23),
                           tzinfo=start.tzinfo)

            location_id, coords = _resolve_location(act.activity_id, simulator)
            item = ItineraryItem(
                item_id=str(uuid.uuid4()),
                item_type="event" if act.category == "event" else "attraction",
                entity_id=act.activity_id,
                title=act.activity_name,
                start_datetime=start,
                end_datetime=end,
                location_id=location_id,
                coordinates=coords,
                city_id=act.city,
                confirmed_price=act.cost_usd,
                currency="USD",
                booking_reference=act.booking_ref,
                metadata={"category": act.category, "venue": act.location},
            )
            items.append(item)
            total_by_category["activities"] += act.cost_usd

    # Derive trip date range from collected dates
    if all_dates:
        trip_start = min(all_dates)
        trip_end = max(all_dates)
    else:
        trip_start = trip_end = date.today()

    # Collect destination cities from days and activities
    destination_city_ids = list(
        dict.fromkeys(
            [day.city for day in itinerary.days if day.city]
            + [act.city for day in itinerary.days for act in day.activities if act.city]
        )
    )

    # Budget from constraints
    budget: float | None = None
    for c in request.hard_constraints + request.soft_constraints:
        if c.category.value == "budget" and c.value is not None:
            try:
                budget = float(c.value)
            except (TypeError, ValueError):
                pass
            break

    manifest = ItineraryManifest(
        manifest_id=str(uuid.uuid4()),
        world_id=world_id,
        session_id=episode_id,
        trip_date_range=(trip_start, trip_end),
        origin_city_id="",  # origin not tracked in Itinerary; fill from UserRequest if available
        destination_city_ids=destination_city_ids,
        user_preferences={c.description: c.value for c in request.hard_constraints + request.soft_constraints},
        constraints_declared=[c.description for c in request.hard_constraints + request.soft_constraints],
        items=items,
        transit_segments=[],  # agent does not produce explicit transit legs
        total_cost=itinerary.total_cost_usd,
        total_cost_by_category=total_by_category,
        budget_declared=budget,
        tool_calls_log_ref=episode_id,
    )
    return manifest


# ── Private helpers ────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> date | None:
    """Parse an ISO date string (YYYY-MM-DD or full ISO 8601); return None on failure."""
    if not date_str:
        return None
    try:
        return date.fromisoformat(date_str[:10])
    except ValueError:
        return None


def _parse_datetime(dt_str: str) -> datetime | None:
    """Parse an ISO 8601 datetime string; return None on failure."""
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def _resolve_location(
    entity_id: str,
    simulator: "SimulatorProtocol | None",
) -> tuple[str, dict]:
    """
    Attempt to resolve a location_id and coordinates from the simulator.

    Falls back to empty string and zero coordinates when the simulator is
    unavailable or the lookup fails.
    """
    if simulator is None or not entity_id:
        return "", {"lat": 0.0, "lon": 0.0}
    try:
        # SimulatorProtocol may expose get_location(entity_id); attempt it
        loc = simulator.get_location(entity_id)  # type: ignore[attr-defined]
        if isinstance(loc, dict):
            return (
                str(loc.get("location_id", entity_id)),
                {
                    "lat": float(loc.get("lat", 0.0)),
                    "lon": float(loc.get("lon", 0.0)),
                },
            )
    except Exception:
        pass
    return entity_id, {"lat": 0.0, "lon": 0.0}
