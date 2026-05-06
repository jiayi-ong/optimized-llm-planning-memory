"""
scripts/generate_worlds.py
===========================
Generate a batch of synthetic travel worlds and save them inside a named
world-set sub-folder.

Output layout
-------------
    worlds/
    └── {set_name}/            ← world-set folder (one per experiment / batch)
        ├── world_42_.../      ← individual world (seed 42)
        ├── world_43_.../      ← individual world (seed 43)
        └── world_44_.../      ← individual world (seed 44)

The world-set folder name becomes the "worlds directory" identifier used by
``generate_eval_dataset.py --world_set worlds/{set_name}``.

Usage
-----
    python scripts/generate_worlds.py \\
        --set_name batch_1 --n_worlds 3 --base_seed 42

    # Custom root directory and world config overrides
    python scripts/generate_worlds.py \\
        --set_name small_worlds --n_worlds 5 --base_seed 0 \\
        --worlds_dir /data/worlds \\
        --num_cities_per_region 2 --num_districts_per_city 3

Progress
--------
Each world is printed as it completes. The UI tracks progress by counting
``world_*/`` sub-folders in the set directory, so each successful creation
immediately advances the progress bar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a batch of travel worlds into a named world-set folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--set_name",
        required=True,
        help="World-set sub-folder name (e.g. 'batch_1', 'ablation_seeds').",
    )
    parser.add_argument(
        "--n_worlds",
        type=int,
        default=3,
        help="Number of worlds to generate (default: 3).",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="World i gets seed base_seed + i (default: 42).",
    )
    parser.add_argument(
        "--worlds_dir",
        type=Path,
        default=_REPO_ROOT / "worlds",
        help="Root worlds directory (default: <repo>/worlds).",
    )
    # Optional world generator config overrides
    parser.add_argument("--num_regions", type=int, default=None)
    parser.add_argument("--num_cities_per_region", type=int, default=None)
    parser.add_argument("--num_districts_per_city", type=int, default=None)
    parser.add_argument("--num_hotels_per_district", type=int, default=None)
    parser.add_argument("--num_attractions_per_district", type=int, default=None)
    parser.add_argument("--num_restaurants_per_district", type=int, default=None)
    parser.add_argument("--num_transport_hubs_per_city", type=int, default=None)
    parser.add_argument("--date_range_days", type=int, default=None)
    parser.add_argument("--num_events_per_city", type=int, default=None)
    parser.add_argument("--num_flights_per_route", type=int, default=None)
    return parser.parse_args()


def _build_world_config(args: argparse.Namespace) -> dict | None:
    """Return a world config dict from CLI args, or None if all defaults."""
    config_keys = [
        "num_regions", "num_cities_per_region", "num_districts_per_city",
        "num_hotels_per_district", "num_attractions_per_district",
        "num_restaurants_per_district", "num_transport_hubs_per_city",
        "date_range_days", "num_events_per_city", "num_flights_per_route",
    ]
    overrides = {k: getattr(args, k) for k in config_keys if getattr(args, k) is not None}
    return overrides if overrides else None


def main() -> None:
    args = _parse_args()

    set_dir = args.worlds_dir / args.set_name
    set_dir.mkdir(parents=True, exist_ok=True)

    world_config = _build_world_config(args)
    if world_config:
        print(f"World config overrides: {world_config}")

    from optimized_llm_planning_memory.simulator.adapter import SimulatorAdapter

    created: list[str] = []
    for i in range(args.n_worlds):
        seed = args.base_seed + i
        print(f"[{i + 1}/{args.n_worlds}] Generating world (seed={seed})…", flush=True)
        try:
            sim = SimulatorAdapter(
                seed=seed,
                worlds_dir=str(set_dir),
                world_config=world_config,
            )
            world_id = sim.world_id
            created.append(world_id)
            print(f"  ✓ {world_id}", flush=True)
        except Exception as exc:
            print(f"  ✗ ERROR: {exc}", flush=True)

    print(f"\n{'─' * 52}")
    print(f"  Generated : {len(created)} / {args.n_worlds} worlds")
    print(f"  Set folder: {set_dir.resolve()}")
    print(f"{'─' * 52}\n")


if __name__ == "__main__":
    main()
