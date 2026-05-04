"""
scripts/generate_user_requests.py
===================================
Generate synthetic UserRequest JSON files for train/val/test splits using
real city IDs and names discovered from the travel_world simulator.

Diversity axes baked into this generator:
  - Archetypes:  5 distinct traveler scenarios cycled round-robin so every
                 split has balanced, non-redundant coverage.
  - Tone:        3 raw_text variants per archetype (casual / neutral / detailed),
                 rotated across archetype rounds to maximize surface diversity.
  - Dates:       2025-2027 (3-year window) so temporal distribution varies.
  - Constraints: 5 hard + 3-4 soft per request (archetype-specific categories).
  - Cities:      Human-readable names used in raw_text; stable IDs stored in
                 the model for simulator lookups.

Archetypes
----------
  1. solo_adventurer   — 1 adult, 5-10 days, $1 200-$2 500
  2. family_trip       — 2 adults + 2 children, 7-14 days, $4 000-$8 000
  3. vegan_couple      — 2 adults (vegan), 5-10 days, $2 500-$5 500
  4. short_budget_stay — 2 adults, 2-3 days, $400-$900
  5. luxury_extended   — 2 adults, 10-14 days, $12 000-$20 000

Constraint engine compatibility notes (see core/constraints.py)
---------------------------------------------------------------
  - TRANSPORT value must be a real simulator mode: flight/train/bus/walk/taxi/ferry.
    "economy" and "public_only" are seat-class / policy strings — the engine does
    an exact mode match, so those would always score 0.0.
  - ACTIVITY text-search branch matches constraint.value against activity.category
    and activity.activity_name; use real simulator category strings.
  - PREFERENCE dining path fires only when value contains one of:
    "cuisine", "dining", "restaurant", "food" — use this to get proper scoring
    for diet-related hard constraints.
  - GROUP always returns satisfied=True (stub TODO in engine); safe to include
    as it won't artificially lower scores.
  - ACCOMMODATION unit="min_stars" branch is fully implemented.

Usage
-----
    python scripts/generate_user_requests.py
    python scripts/generate_user_requests.py data.n_train=40 data.n_val=10 data.n_test=10
    python scripts/generate_user_requests.py project.seed=123
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import hydra
from omegaconf import DictConfig, OmegaConf

from optimized_llm_planning_memory.utils.logging import configure_logging, get_logger
from optimized_llm_planning_memory.utils.seed import set_seed


# ── City metadata carrier ──────────────────────────────────────────────────────

class CityPair(NamedTuple):
    """Bundles simulator IDs (for API calls) with human names (for raw_text)."""
    origin_id: str
    origin_name: str
    dest_ids: list[str]
    dest_names: list[str]


# ── Traveler archetypes ────────────────────────────────────────────────────────
# Each archetype is a self-contained scenario bundle:
#   profile              — TravelerProfile fields
#   preferences          — free-form strings injected into raw_text and the model
#   duration_days        — candidate trip lengths (one sampled per request)
#   budget_usd           — candidate budgets (one sampled per request)
#   raw_text_templates   — 3 tone variants: casual / neutral / detailed
#                          placeholders: {duration} {origin_name} {dest_names}
#                                        {budget} {pref0} {pref1} {pref2}
#   extra_hard           — one archetype-specific hard constraint on top of the
#                          four shared ones (budget, date, duration, group)
#   soft_constraints     — 3-4 archetype-specific soft constraint dicts
#
# Engine-compatibility decisions for extra_hard:
#   solo_adventurer  → ACTIVITY / "hiking"      text-searched in activity names ✓
#   family_trip      → ACTIVITY / "park"        real simulator category string  ✓
#   vegan_couple     → PREFERENCE / "plant-based dining"  triggers dining path  ✓
#   short_budget_stay→ TRANSPORT / "bus"        real simulator transport mode   ✓
#   luxury_extended  → ACCOMMODATION / 5 / min_stars  fully implemented        ✓

_ARCHETYPES: list[dict] = [
    # ── 1. Solo adventurer ──────────────────────────────────────────────────────
    {
        "label": "solo_adventurer",
        "profile": {
            "num_adults": 1,
            "num_children": 0,
            "accessibility_needs": [],
            "dietary_restrictions": [],
        },
        "preferences": ["hiking", "local cuisine", "cultural sites", "photography", "street art"],
        "duration_days": [5, 7, 10],
        "budget_usd": [1200.0, 1800.0, 2500.0],
        "raw_text_templates": [
            # casual
            "Solo trip from {origin_name} to {dest_names} — {duration} days, ${budget:.0f}. "
            "I'm big on {pref0} and {pref1}. Keep it local and adventurous.",
            # neutral
            "Traveling solo from {origin_name} to {dest_names} for {duration} days. "
            "Budget: ${budget:.0f}. Main interests: {pref0}, {pref1}, and {pref2}.",
            # detailed
            "Planning a {duration}-day solo adventure departing {origin_name}, arriving in "
            "{dest_names}. Total budget: ${budget:.0f} USD. I love {pref0} and {pref1} — "
            "the more off-the-beaten-path the better. Please include some {pref2} too.",
        ],
        # ACTIVITY text-search: engine checks value against activity.category and
        # activity.activity_name — "hiking" is a real simulator activity category.
        "extra_hard": {
            "category": "activity",
            "description": "Itinerary must include at least one hiking activity.",
            "value": "hiking",
            "unit": None,
        },
        "soft_constraints": [
            {
                "category": "accommodation",
                "description": "Hostels or budget hotels (1-2 stars) are acceptable.",
                "value": 1,
                "unit": "min_stars",
            },
            {
                "category": "activity",
                "description": "Include at least one outdoor or adventure activity per day.",
                "value": "outdoor",
                "unit": "preference",
            },
            {
                "category": "preference",
                "description": "Favour local community experiences over tourist-trap attractions.",
                "value": "local_authentic",
                "unit": "preference",
            },
        ],
    },

    # ── 2. Family trip ──────────────────────────────────────────────────────────
    {
        "label": "family_trip",
        "profile": {
            "num_adults": 2,
            "num_children": 2,
            "accessibility_needs": [],
            "dietary_restrictions": [],
        },
        "preferences": ["family-friendly activities", "theme parks", "kid-friendly dining", "beaches", "safe neighborhoods"],
        "duration_days": [7, 10, 14],
        "budget_usd": [4000.0, 6000.0, 8000.0],
        "raw_text_templates": [
            # casual
            "Family of 4 (2 adults, 2 kids under 12) heading from {origin_name} to {dest_names} "
            "for {duration} days. Budget is ${budget:.0f}. We need {pref0} and {pref1}.",
            # neutral
            "Family trip: {origin_name} → {dest_names}, {duration} days, ${budget:.0f} total "
            "for 2 adults and 2 children. We want {pref0} and everything child-safe.",
            # detailed
            "We're a family of four planning {duration} days from {origin_name} to {dest_names}. "
            "Kids are 7 and 10. Budget cap: ${budget:.0f} USD. We need {pref0}, {pref1}, and "
            "{pref2}. Child safety is non-negotiable for all bookings.",
        ],
        # ACTIVITY text-search: "park" is a real simulator attraction category and
        # will be found in family-friendly itineraries (parks, theme parks, etc.).
        "extra_hard": {
            "category": "activity",
            "description": "Itinerary must include at least one park or family-friendly outdoor activity.",
            "value": "park",
            "unit": None,
        },
        "soft_constraints": [
            {
                "category": "accommodation",
                "description": "Prefer family rooms or suites at 3-star hotels or above.",
                "value": 3,
                "unit": "min_stars",
            },
            {
                "category": "transport",
                "description": "Prefer direct flights to minimise travel time with young children.",
                "value": "direct",
                "unit": "preference",
            },
            {
                "category": "activity",
                "description": "Include at least one major family attraction (theme park, beach, or zoo) per 3 days.",
                "value": "family_attraction",
                "unit": "preference",
            },
            {
                "category": "preference",
                "description": "Hotel should have a pool or dedicated play area for children.",
                "value": "child_amenities",
                "unit": "preference",
            },
        ],
    },

    # ── 3. Vegan couple ─────────────────────────────────────────────────────────
    {
        "label": "vegan_couple",
        "profile": {
            "num_adults": 2,
            "num_children": 0,
            "accessibility_needs": [],
            "dietary_restrictions": ["vegan"],
        },
        "preferences": ["vegan restaurants", "farmers markets", "nature walks", "art galleries", "sustainable travel"],
        "duration_days": [5, 7, 10],
        "budget_usd": [2500.0, 4000.0, 5500.0],
        "raw_text_templates": [
            # casual
            "My partner and I are vegan, planning {duration} days from {origin_name} to "
            "{dest_names} on ${budget:.0f}. Food scene matters a lot — {pref0} and {pref1} please.",
            # neutral
            "Vegan couple, {duration} days, {origin_name} to {dest_names}, ${budget:.0f}. "
            "Need dedicated vegan dining and prefer {pref0} and {pref2}.",
            # detailed
            "Two adults traveling together from {origin_name} to {dest_names} for {duration} days. "
            "Budget: ${budget:.0f}. Both strictly vegan — only restaurants with proper vegan menus. "
            "Into {pref0}, {pref1}, and sustainable travel.",
        ],
        # PREFERENCE dining path: value contains "dining" → engine checks for any
        # dining/restaurant activity in the itinerary instead of exact text match.
        # This is the best available proxy for vegan dietary compliance.
        "extra_hard": {
            "category": "preference",
            "description": "Both travelers are strictly vegan; itinerary must include dedicated plant-based dining options.",
            "value": "plant-based dining",
            "unit": "dietary",
        },
        "soft_constraints": [
            {
                "category": "accommodation",
                "description": "Prefer eco-certified or sustainability-rated properties.",
                "value": "eco_certified",
                "unit": "preference",
            },
            {
                "category": "activity",
                "description": "Prefer eco-friendly and sustainability-certified tour operators.",
                "value": "sustainable",
                "unit": "preference",
            },
            {
                "category": "transport",
                "description": "Prefer train or low-emission transport over air travel where feasible.",
                "value": "low_emission",
                "unit": "preference",
            },
            {
                "category": "preference",
                "description": "Prioritise plant-based food experiences: vegan cooking classes, farmers markets, plant-based food tours.",
                "value": "plant_based_experiences",
                "unit": "preference",
            },
        ],
    },

    # ── 4. Short budget stay ────────────────────────────────────────────────────
    {
        "label": "short_budget_stay",
        "profile": {
            "num_adults": 2,
            "num_children": 0,
            "accessibility_needs": [],
            "dietary_restrictions": [],
        },
        "preferences": ["street food", "free attractions", "walking tours", "local markets", "public transit"],
        "duration_days": [2, 3],
        "budget_usd": [400.0, 600.0, 900.0],
        "raw_text_templates": [
            # casual
            "Quick {duration}-day trip from {origin_name} to {dest_names}, very tight ${budget:.0f} "
            "total. We want {pref0} and {pref1}. As cheap as possible.",
            # neutral
            "Short getaway: {origin_name} → {dest_names}, {duration} days, ${budget:.0f} hard cap "
            "for 2. Focus on {pref0} and free/cheap {pref1}.",
            # detailed
            "We're squeezing in {duration} days from {origin_name} to {dest_names} with just "
            "${budget:.0f} between us. Happy to skip luxury. We enjoy {pref0} and {pref1} — "
            "seeing more matters more than comfort.",
        ],
        # TRANSPORT exact-mode match: "bus" is a real TransportSegment mode in the
        # simulator. Satisfied when the itinerary includes at least one bus segment.
        "extra_hard": {
            "category": "transport",
            "description": "Trip must use bus or public transit for at least one journey leg.",
            "value": "bus",
            "unit": "mode",
        },
        "soft_constraints": [
            {
                "category": "accommodation",
                "description": "Budget accommodation only — hostels or 1-2 star hotels.",
                "value": 1,
                "unit": "min_stars",
            },
            {
                "category": "activity",
                "description": "Prioritise free or low-cost attractions (under $20 per person per activity).",
                "value": "free_or_cheap",
                "unit": "preference",
            },
            {
                "category": "preference",
                "description": "Maximise the number of distinct locations visited within the time and budget.",
                "value": "coverage",
                "unit": "preference",
            },
        ],
    },

    # ── 5. Luxury extended ──────────────────────────────────────────────────────
    {
        "label": "luxury_extended",
        "profile": {
            "num_adults": 2,
            "num_children": 0,
            "accessibility_needs": [],
            "dietary_restrictions": [],
        },
        "preferences": ["fine dining", "5-star hotels", "private tours", "spa treatments", "exclusive experiences", "wine tasting"],
        "duration_days": [10, 12, 14],
        "budget_usd": [12000.0, 16000.0, 20000.0],
        "raw_text_templates": [
            # casual
            "Luxury {duration}-day trip from {origin_name} to {dest_names}. Budget ${budget:.0f}, "
            "want only the best — {pref0}, {pref1}, and {pref2}.",
            # neutral
            "Premium trip: {origin_name} → {dest_names}, {duration} days, ${budget:.0f}. "
            "Require {pref0}, 5-star accommodation, and private transfers. Include {pref1} and {pref2}.",
            # detailed
            "We're planning an extended {duration}-day luxury trip from {origin_name} to "
            "{dest_names}, budget ${budget:.0f}. Expect 5-star properties throughout, {pref0}, "
            "{pref1}, and curated {pref2} experiences. No compromises on quality.",
        ],
        # ACCOMMODATION unit="min_stars" branch is fully implemented in the engine:
        # checks booking.star_rating >= value for the first hotel booking.
        "extra_hard": {
            "category": "accommodation",
            "description": "All properties must be 5-star rated or equivalent luxury boutique hotels.",
            "value": 5,
            "unit": "min_stars",
        },
        "soft_constraints": [
            {
                "category": "transport",
                "description": "Business or first class for all flights; private transfers for ground transport.",
                "value": "business_class",
                "unit": "preference",
            },
            {
                "category": "activity",
                "description": "Prioritise private, exclusive, or curated experiences over group tours.",
                "value": "private_exclusive",
                "unit": "preference",
            },
            {
                "category": "preference",
                "description": "Include Michelin-starred or top-rated dining (at least one per 3 days).",
                "value": "fine_dining",
                "unit": "preference",
            },
            {
                "category": "preference",
                "description": "Include at least one full-day spa or wellness experience.",
                "value": "spa_wellness",
                "unit": "preference",
            },
        ],
    },
]


# ── World-aware helpers ────────────────────────────────────────────────────────

def load_world_cities(world_path: Path) -> tuple[str, str, dict[str, str]]:
    """Read a world folder directly (no simulator needed).

    Returns (world_id, sim_date, {city_id: city_name}) sourced from
    meta.json and geo_layer.json. This is the canonical way to bind
    generated requests to a specific reproducible world.

    The geo_layer stores cities under geo["data"]["cities"] as either a
    dict {city_id: city_obj} or a list[city_obj] — both are handled.
    """
    meta = json.loads((world_path / "meta.json").read_text(encoding="utf-8"))
    geo = json.loads((world_path / "geo_layer.json").read_text(encoding="utf-8"))

    cities_raw = geo.get("data", geo).get("cities", {})
    if isinstance(cities_raw, dict):
        registry = {cid: city["name"] for cid, city in cities_raw.items() if "name" in city}
    else:
        registry = {c["city_id"]: c["name"] for c in cities_raw if "city_id" in c and "name" in c}

    return meta["world_id"], meta["sim_date"], registry


def _anchor_date_range(
    sim_date: str, duration_days: int, rng: random.Random
) -> tuple[str, str]:
    """Start date is sim_date + 7..60 days so trips are temporally coherent with the world."""
    base = datetime.fromisoformat(sim_date)
    offset = rng.randint(7, 60)
    start = base + timedelta(days=offset)
    end = start + timedelta(days=duration_days - 1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def generate_from_world_dir(
    world_path: Path,
    n: int,
    split: str,
    output_dir: Path,
    rng: random.Random,
    log,
) -> None:
    """Generate ``n`` UserRequest JSON files bound to the given world directory.

    Unlike the Hydra ``main()`` path, this function reads city data directly
    from ``geo_layer.json`` without instantiating the simulator. All generated
    requests store ``world_id`` so evaluations can be traced back to the exact
    world that produced them.
    """
    from optimized_llm_planning_memory.core.models import UserRequest

    world_id, sim_date, city_registry = load_world_cities(world_path)

    if len(city_registry) < 2:
        raise ValueError(
            f"World at {world_path} has fewer than 2 cities ({len(city_registry)}). "
            "Cannot generate city pairs."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    city_ids = list(city_registry.keys())
    rng.shuffle(city_ids)

    city_pairs: list[CityPair] = [
        CityPair(
            origin_id=city_ids[i],
            origin_name=city_registry[city_ids[i]],
            dest_ids=[city_ids[(i + 1) % len(city_ids)]],
            dest_names=[city_registry[city_ids[(i + 1) % len(city_ids)]]],
        )
        for i in range(len(city_ids))
        if city_ids[i] != city_ids[(i + 1) % len(city_ids)]
    ]

    pairs_cycle = itertools.cycle(city_pairs)
    n_archetypes = len(_ARCHETYPES)
    saved = 0

    for i in range(n):
        pair = next(pairs_cycle)
        archetype = _ARCHETYPES[i % n_archetypes]
        tone_idx = i // n_archetypes
        budget = rng.choice(archetype["budget_usd"])
        duration = rng.choice(archetype["duration_days"])
        start_date, end_date = _anchor_date_range(sim_date, duration, rng)

        req_dict = _make_request(pair, archetype, budget, start_date, end_date, tone_idx)
        req_dict["world_id"] = world_id
        req_dict["metadata"]["world_id"] = world_id  # belt-and-suspenders for raw JSON readers

        try:
            req = UserRequest.model_validate(req_dict)
            out_path = output_dir / f"request_{req.request_id}.json"
            out_path.write_text(req.model_dump_json(indent=2), encoding="utf-8")
            saved += 1
        except Exception as exc:
            log.warning(
                "request_validation_failed",
                split=split,
                archetype=archetype["label"],
                error=str(exc),
            )

    log.info("split_generated", split=split, n=saved, world_id=world_id)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _random_date_range(rng: random.Random, duration_days: int) -> tuple[str, str]:
    """Return a (start_date, end_date) pair spread across 2025–2027.

    Duration is the number of calendar days inclusive. E.g., 10 days from
    2025-01-01 to 2025-01-10 (not to 2025-01-11).
    """
    start = datetime(2025, 1, 1) + timedelta(days=rng.randint(0, 1094))
    end = start + timedelta(days=duration_days - 1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _make_request(
    pair: CityPair,
    archetype: dict,
    budget: float,
    start_date: str,
    end_date: str,
    tone_idx: int,
) -> dict:
    """Build a UserRequest dict driven by the given archetype.

    Hard constraints produced (5 total):
      1. budget    — total cost cap            (numeric; BUDGET validator requires this)
      2. date      — fixed travel window       (string;  DATE validator requires this)
      3. duration  — exact trip length in days (numeric; DURATION validator requires this)
      4. group     — full party size           (numeric; GROUP stub always satisfied)
      5. archetype-specific extra hard         (category/value chosen for engine compat)
    """
    prefs = archetype["preferences"]
    duration = (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days
    dest_name_str = " and ".join(pair.dest_names)

    template = archetype["raw_text_templates"][tone_idx % len(archetype["raw_text_templates"])]
    raw_text = template.format(
        duration=duration,
        origin_name=pair.origin_name,
        dest_names=dest_name_str,
        budget=budget,
        pref0=prefs[0],
        pref1=prefs[1],
        pref2=prefs[2] if len(prefs) > 2 else prefs[0],
    )

    profile = archetype["profile"]
    group_size = profile["num_adults"] + profile["num_children"]
    group_desc = (
        f"{profile['num_adults']} adult(s)"
        + (f", {profile['num_children']} child(ren)" if profile["num_children"] else "")
    )
    extra = archetype["extra_hard"]

    hard_constraints = [
        {
            "constraint_id": f"hc-budget-{uuid.uuid4().hex[:6]}",
            "constraint_type": "hard",
            "category": "budget",
            "description": f"Total trip cost must not exceed ${budget:.0f} USD.",
            "value": budget,
            "unit": "USD",
        },
        {
            "constraint_id": f"hc-dates-{uuid.uuid4().hex[:6]}",
            "constraint_type": "hard",
            "category": "date",
            "description": f"Trip must start on {start_date} and end by {end_date}.",
            "value": f"{start_date} to {end_date}",
            "unit": "date_range",
        },
        {
            "constraint_id": f"hc-duration-{uuid.uuid4().hex[:6]}",
            "constraint_type": "hard",
            "category": "duration",
            "description": f"Trip must be exactly {duration} days long.",
            "value": duration,
            "unit": "days",
        },
        {
            "constraint_id": f"hc-group-{uuid.uuid4().hex[:6]}",
            "constraint_type": "hard",
            "category": "group",
            "description": (
                f"Party of {group_size} ({group_desc}); "
                "all pricing and reservations must accommodate the full group."
            ),
            "value": group_size,
            "unit": "people",
        },
        {
            "constraint_id": f"hc-{extra['category']}-{uuid.uuid4().hex[:6]}",
            "constraint_type": "hard",
            "category": extra["category"],
            "description": extra["description"],
            "value": extra["value"],
            "unit": extra.get("unit"),
        },
    ]

    soft_constraints = [
        {
            "constraint_id": f"sc-{sc['category']}-{uuid.uuid4().hex[:6]}",
            "constraint_type": "soft",
            "category": sc["category"],
            "description": sc["description"],
            "value": sc["value"],
            "unit": sc.get("unit"),
        }
        for sc in archetype["soft_constraints"]
    ]

    return {
        "request_id": str(uuid.uuid4()),
        "raw_text": raw_text,
        "origin_city": pair.origin_id,
        "destination_cities": pair.dest_ids,
        "start_date": start_date,
        "end_date": end_date,
        "budget_usd": budget,
        "traveler_profile": {
            "num_adults": profile["num_adults"],
            "num_children": profile["num_children"],
            "accessibility_needs": profile["accessibility_needs"],
            "dietary_restrictions": profile["dietary_restrictions"],
        },
        "hard_constraints": hard_constraints,
        "soft_constraints": soft_constraints,
        "preferences": prefs,
        "metadata": {
            "generated_by": "generate_user_requests.py",
            "template_version": "3.0",
            "archetype": archetype["label"],
            "tone_variant": tone_idx % len(archetype["raw_text_templates"]),
            "origin_name": pair.origin_name,
            "dest_names": pair.dest_names,
        },
    }


def generate_and_save(
    n: int,
    split: str,
    output_dir: Path,
    city_pairs: list[CityPair],
    rng: random.Random,
    log,
) -> None:
    """Generate ``n`` UserRequest JSON files into ``output_dir``.

    Archetypes are assigned by ``i % len(_ARCHETYPES)`` (round-robin).
    Tone variants rotate by ``i // len(_ARCHETYPES)`` so each archetype
    exhausts all three tones before any tone repeats.
    """
    from optimized_llm_planning_memory.core.models import UserRequest

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_cycle = itertools.cycle(city_pairs)
    n_archetypes = len(_ARCHETYPES)

    saved = 0
    for i in range(n):
        pair = next(pairs_cycle)
        archetype = _ARCHETYPES[i % n_archetypes]
        tone_idx = i // n_archetypes
        budget = rng.choice(archetype["budget_usd"])
        duration = rng.choice(archetype["duration_days"])
        start_date, end_date = _random_date_range(rng, duration)

        req_dict = _make_request(pair, archetype, budget, start_date, end_date, tone_idx)
        try:
            req = UserRequest.model_validate(req_dict)
            out_path = output_dir / f"request_{req.request_id}.json"
            out_path.write_text(req.model_dump_json(indent=2), encoding="utf-8")
            saved += 1
        except Exception as exc:
            log.warning(
                "request_validation_failed",
                split=split,
                archetype=archetype["label"],
                error=str(exc),
            )

    log.info("split_generated", split=split, n=saved)


# ── City discovery ─────────────────────────────────────────────────────────────

def _discover_cities(sim, log) -> dict[str, str]:
    """Return a city_id → city_name registry from the simulator.

    Handles both single-city world format (dict has 'city_id' + 'city_name')
    and multi-city route format (dict has 'origin_city_id'/'destination_city_id').
    Falls back to city_id as the display name when no name field is present.
    """
    try:
        routes = sim.get_available_routes()
    except Exception as exc:
        log.warning("routes_unavailable", error=str(exc))
        return {}

    registry: dict[str, str] = {}
    for r in routes:
        rd = r if isinstance(r, dict) else vars(r)

        # Single-city entry (most common in single-world mode)
        if "city_id" in rd:
            cid = str(rd["city_id"])
            name = rd.get("city_name") or rd.get("name") or cid
            registry[cid] = str(name)

        # Route entry with explicit origin/destination sides
        for side in ("origin", "destination"):
            cid_key = f"{side}_city_id"
            if cid_key in rd:
                cid = str(rd[cid_key])
                name = rd.get(f"{side}_city_name") or rd.get("city_name") or cid
                registry.setdefault(cid, str(name))

    return registry


# ── Entry point ────────────────────────────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    configure_logging(level=cfg.logging.level)
    log = get_logger(__name__)
    seed = cfg.project.seed
    set_seed(seed)
    rng = random.Random(seed)

    n_train = int(OmegaConf.select(cfg, "data.n_train", default=40))
    n_val   = int(OmegaConf.select(cfg, "data.n_val",   default=10))
    n_test  = int(OmegaConf.select(cfg, "data.n_test",  default=10))

    from optimized_llm_planning_memory.simulator.adapter import SimulatorAdapter
    worlds_dir = OmegaConf.select(cfg, "simulator.worlds_dir", default="./worlds")
    sim = SimulatorAdapter(seed=seed, worlds_dir=worlds_dir)

    city_registry = _discover_cities(sim, log)

    if len(city_registry) < 2:
        log.warning(
            "insufficient_cities",
            n=len(city_registry),
            hint="Using placeholder city IDs — replace with real IDs after world init.",
        )
        city_registry = {f"city_{i:03d}": f"City {i}" for i in range(10)}

    city_ids = list(city_registry.keys())
    rng.shuffle(city_ids)

    city_pairs: list[CityPair] = [
        CityPair(
            origin_id=city_ids[i],
            origin_name=city_registry[city_ids[i]],
            dest_ids=[city_ids[(i + 1) % len(city_ids)]],
            dest_names=[city_registry[city_ids[(i + 1) % len(city_ids)]]],
        )
        for i in range(len(city_ids))
        if city_ids[i] != city_ids[(i + 1) % len(city_ids)]
    ]

    # Write a template request so run_episode.py has a working default
    if city_pairs:
        template_req = _make_request(
            city_pairs[0], _ARCHETYPES[0], 2500.0, "2025-07-01", "2025-07-08", tone_idx=1
        )
        template_path = _REPO_ROOT / "data/user_requests/templates/request_template.json"
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(json.dumps(template_req, indent=2), encoding="utf-8")
        log.info("template_updated", path=str(template_path))

    base = _REPO_ROOT / "data/user_requests"
    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        log.info("generating_split", split=split, n=n)
        generate_and_save(n, split, base / split, city_pairs, rng, log)

    log.info("done", total=n_train + n_val + n_test)


def cli_main() -> None:
    """Argparse entry point for world-aligned request generation.

    Use when you already have a world folder and want requests that reference
    real city names and dates from that specific world.

    Example
    -------
    python scripts/generate_user_requests.py \\
        --world_dir worlds/world_42_20260502_084804 \\
        --n_train 40 --n_val 10 --n_test 10 --seed 42
    """
    parser = argparse.ArgumentParser(
        description="Generate world-aligned UserRequest JSON files.",
        add_help=True,
    )
    parser.add_argument(
        "--world_dir",
        type=Path,
        required=True,
        help="Path to a world folder (e.g. worlds/world_42_20260502_084804).",
    )
    parser.add_argument("--n_train", type=int, default=40)
    parser.add_argument("--n_val", type=int, default=10)
    parser.add_argument("--n_test", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/user_requests"),
        help="Base output directory; split sub-dirs are created automatically.",
    )

    # When Hydra is invoked it injects its own sys.argv. Guard against that here.
    args, _ = parser.parse_known_args()

    configure_logging(level="INFO")
    log = get_logger(__name__)
    set_seed(args.seed)
    rng = random.Random(args.seed)

    log.info("world_dir_mode", world_dir=str(args.world_dir))
    for split, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
        log.info("generating_split", split=split, n=n)
        generate_from_world_dir(args.world_dir, n, split, args.output_dir / split, rng, log)
    log.info("done", total=args.n_train + args.n_val + args.n_test)


if __name__ == "__main__":
    # When --world_dir is present, use the world-aligned path; otherwise fall
    # back to Hydra for backward compatibility with the config-driven workflow.
    if "--world_dir" in sys.argv:
        cli_main()
    else:
        main()
