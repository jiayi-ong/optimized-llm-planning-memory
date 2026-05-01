"""
scripts/generate_user_requests.py
===================================
Generate synthetic UserRequest JSON files for train/val/test splits using
real city IDs discovered from the travel_world simulator.

Unlike the original version (which required an LLM), this script uses the
SimulatorAdapter to list available city pairs from travel_world, then templates
diverse UserRequest objects from those pairs.  No API key is needed.

Usage
-----
    python scripts/generate_user_requests.py
    python scripts/generate_user_requests.py n_train=40 n_val=10 n_test=10
    python scripts/generate_user_requests.py project.seed=123
"""

from __future__ import annotations

import itertools
import json
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from omegaconf import DictConfig, OmegaConf

from optimized_llm_planning_memory.utils.logging import configure_logging, get_logger
from optimized_llm_planning_memory.utils.seed import set_seed


# ── Scenario templates ─────────────────────────────────────────────────────────

_PROFILES = [
    {"num_adults": 2, "num_children": 0, "accessibility_needs": [], "dietary_restrictions": []},
    {"num_adults": 1, "num_children": 0, "accessibility_needs": [], "dietary_restrictions": ["vegetarian"]},
    {"num_adults": 2, "num_children": 2, "accessibility_needs": [], "dietary_restrictions": []},
    {"num_adults": 4, "num_children": 0, "accessibility_needs": [], "dietary_restrictions": []},
    {"num_adults": 1, "num_children": 0, "accessibility_needs": ["wheelchair"], "dietary_restrictions": []},
]

_PREFERENCES = [
    ["art museums", "walking tours", "local cuisine"],
    ["beach activities", "water sports", "seafood restaurants"],
    ["hiking", "nature parks", "budget dining"],
    ["luxury hotels", "fine dining", "spa treatments"],
    ["historical sites", "guided tours", "street food"],
    ["nightlife", "concerts", "rooftop bars"],
]

_DURATIONS = [5, 7, 10, 14]
_BUDGETS = [1500.0, 2500.0, 4000.0, 6000.0, 10000.0]


def _random_date_range(rng: random.Random, duration_days: int) -> tuple[str, str]:
    """Return a (start_date, end_date) pair in 2025."""
    start_offset = rng.randint(0, 300)
    start = datetime(2025, 1, 1) + timedelta(days=start_offset)
    end = start + timedelta(days=duration_days)
    if end.year > 2025:
        start = datetime(2025, 1, 1)
        end = start + timedelta(days=duration_days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _make_request(
    origin_city: str,
    dest_cities: list[str],
    profile: dict,
    prefs: list[str],
    budget: float,
    start_date: str,
    end_date: str,
) -> dict:
    """Build a UserRequest dict with hard and soft constraints."""
    dest_str = " and ".join(dest_cities)
    duration = (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days
    profile_desc = f"{profile['num_adults']} adult(s)"
    if profile["num_children"]:
        profile_desc += f", {profile['num_children']} child(ren)"

    return {
        "request_id": str(uuid.uuid4()),
        "raw_text": (
            f"Plan a {duration}-day trip from {origin_city} to {dest_str} "
            f"for {profile_desc}. Budget is ${budget:.0f} USD. "
            f"Preferences: {', '.join(prefs[:2])}."
        ),
        "origin_city": origin_city,
        "destination_cities": dest_cities,
        "start_date": start_date,
        "end_date": end_date,
        "budget_usd": budget,
        "traveler_profile": {
            "num_adults": profile["num_adults"],
            "num_children": profile["num_children"],
            "accessibility_needs": profile["accessibility_needs"],
            "dietary_restrictions": profile["dietary_restrictions"],
        },
        "hard_constraints": [
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
        ],
        "soft_constraints": [
            {
                "constraint_id": f"sc-hotel-{uuid.uuid4().hex[:6]}",
                "constraint_type": "soft",
                "category": "accommodation",
                "description": "Prefer mid-range or above hotels (3+ stars).",
                "value": 3,
                "unit": "min_stars",
            },
            {
                "constraint_id": f"sc-activity-{uuid.uuid4().hex[:6]}",
                "constraint_type": "soft",
                "category": "activity",
                "description": f"Include activities matching preference: {prefs[0]}.",
                "value": prefs[0],
                "unit": "preference",
            },
        ],
        "preferences": prefs,
        "metadata": {"generated_by": "generate_user_requests.py", "template_version": "2.0"},
    }


def generate_and_save(
    n: int,
    split: str,
    output_dir: Path,
    city_pairs: list[tuple[str, list[str]]],
    rng: random.Random,
    log,
) -> None:
    """Generate ``n`` UserRequest JSON files into ``output_dir``."""
    from optimized_llm_planning_memory.core.models import UserRequest

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_cycle = itertools.cycle(city_pairs)

    saved = 0
    for _ in range(n):
        origin, dests = next(pairs_cycle)
        profile = rng.choice(_PROFILES)
        prefs = rng.choice(_PREFERENCES)
        budget = rng.choice(_BUDGETS)
        duration = rng.choice(_DURATIONS)
        start_date, end_date = _random_date_range(rng, duration)

        req_dict = _make_request(origin, dests, profile, prefs, budget, start_date, end_date)
        try:
            req = UserRequest.model_validate(req_dict)
            out_path = output_dir / f"request_{req.request_id}.json"
            out_path.write_text(req.model_dump_json(indent=2), encoding="utf-8")
            saved += 1
        except Exception as exc:
            log.warning("request_validation_failed", split=split, error=str(exc))

    log.info("split_generated", split=split, n=saved)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    configure_logging(level=cfg.logging.level)
    log = get_logger(__name__)
    seed = cfg.project.seed
    set_seed(seed)
    rng = random.Random(seed)

    n_train = int(OmegaConf.select(cfg, "n_train", default=40))
    n_val = int(OmegaConf.select(cfg, "n_val", default=10))
    n_test = int(OmegaConf.select(cfg, "n_test", default=10))

    # Discover real city IDs from travel_world
    from optimized_llm_planning_memory.simulator.adapter import SimulatorAdapter
    worlds_dir = OmegaConf.select(cfg, "simulator.worlds_dir", default="./worlds")
    sim = SimulatorAdapter(seed=seed, worlds_dir=worlds_dir)

    try:
        routes = sim.get_available_routes()
    except Exception as exc:
        log.warning("routes_unavailable", error=str(exc),
                    hint="Falling back to placeholder city IDs.")
        routes = []

    # Build a list of (origin_city_id, [dest_city_id]) pairs
    city_ids: list[str] = []
    if routes:
        # Routes is a list of dicts with origin/destination city info
        seen = set()
        for r in routes:
            for key in ("origin_city_id", "destination_city_id", "city_id"):
                cid = r.get(key) if isinstance(r, dict) else getattr(r, key, None)
                if cid and cid not in seen:
                    city_ids.append(str(cid))
                    seen.add(str(cid))

    if len(city_ids) < 2:
        log.warning("insufficient_cities",
                    n=len(city_ids),
                    hint="Using placeholder city IDs — replace with real IDs after world init.")
        city_ids = [f"city_{i:03d}" for i in range(10)]

    # Generate diverse pairs (single-destination; extend for multi-city later)
    rng.shuffle(city_ids)
    city_pairs: list[tuple[str, list[str]]] = []
    for i in range(len(city_ids)):
        origin = city_ids[i]
        dest = city_ids[(i + 1) % len(city_ids)]
        if origin != dest:
            city_pairs.append((origin, [dest]))

    # Also write one pair back to the template file so run_episode.py has a
    # working default without any CLI args
    if city_pairs:
        o, d = city_pairs[0]
        template_req = _make_request(
            o, d, _PROFILES[0], _PREFERENCES[0], 3000.0, "2025-07-01", "2025-07-08"
        )
        template_path = Path("data/user_requests/templates/request_template.json")
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(json.dumps(template_req, indent=2), encoding="utf-8")
        log.info("template_updated", path=str(template_path))

    base = Path("data/user_requests")
    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        log.info("generating_split", split=split, n=n)
        generate_and_save(n, split, base / split, city_pairs, rng, log)

    log.info("done", total=n_train + n_val + n_test)


if __name__ == "__main__":
    main()
