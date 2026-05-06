"""
scripts/generate_eval_dataset.py
==================================
Standalone CLI for multi-world user request generation.

Unlike the Hydra-based ``generate_user_requests.py``, this script accepts plain
argparse flags and is designed for programmatic use (UI, CI, shell scripts).

Usage
-----
    # Generate 30 requests spread across 3 world dirs into the val split
    python scripts/generate_eval_dataset.py \\
        --world_dirs worlds/w0 worlds/w1 worlds/w2 \\
        --n_total 30 --split val --seed 42

    # Bias toward hard requests (0% low, 40% medium, 60% high)
    python scripts/generate_eval_dataset.py \\
        --world_dirs worlds/w0 \\
        --n_total 50 --split train \\
        --complexity_weights 0 40 60

    # Custom output directory
    python scripts/generate_eval_dataset.py \\
        --world_dirs worlds/w0 --n_total 10 --split val \\
        --output_dir data/user_requests_exp2

Distribution logic
------------------
  per_world = n_total // n_worlds
  First (n_total % n_worlds) worlds get one extra request each.

Complexity weighting
--------------------
  Requests are generated with the standard archetype round-robin. After
  generation, requests whose complexity_tier does not match the desired
  distribution are dropped and regenerated (rejection sampling with up to
  5× overgeneration to fill the quota). Unrecognised tiers are accepted.

  --complexity_weights takes 3 integers: <low> <medium> <high>
  They need not sum to 100; they are normalised to proportions internally.
  Default is uniform (equal weight for each tier).
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Early diagnostic — proves the script started before any other imports can fail
print(f"[generate_eval_dataset] starting, args={sys.argv[1:]}", flush=True)

from optimized_llm_planning_memory.utils.logging import configure_logging, get_logger
from optimized_llm_planning_memory.utils.seed import set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate world-aligned UserRequest JSON files across multiple worlds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n-----")[1],
    )
    world_grp = parser.add_mutually_exclusive_group(required=True)
    world_grp.add_argument(
        "--world_set",
        type=Path,
        metavar="PATH",
        help=(
            "World-set folder (e.g. worlds/batch_1). All world_* sub-directories "
            "inside are used as individual world sources."
        ),
    )
    world_grp.add_argument(
        "--world_dirs",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="One or more individual world directories (legacy / explicit mode).",
    )
    parser.add_argument(
        "--n_total",
        type=int,
        default=20,
        help="Total number of requests to generate across all worlds (default: 20).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help=(
            "Name for this collection of requests — used as the sub-folder under "
            "--output_dir (e.g. 'ablation_hard', 'val_seed42'). "
            "Defaults to a UTC timestamp (YYYYMMDD_HHMMSS) if not specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_REPO_ROOT / "data" / "user_requests",
        help="Base output directory; <collection> name is appended automatically.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--complexity_weights",
        nargs=3,
        type=float,
        metavar=("LOW", "MED", "HIGH"),
        default=None,
        help=(
            "Relative weights for low/medium/high complexity tiers. "
            "E.g. '0 40 60' for 0%% low, 40%% medium, 60%% high. "
            "Default: uniform (equal weight for each tier)."
        ),
    )
    return parser.parse_args()


def _tier_quotas(n: int, weights: list[float] | None) -> dict[str, int]:
    """Return {tier: count} for n requests given optional weights.

    When weights is None (uniform), each tier gets n // 3 requests with
    remainders assigned round-robin to low → medium → high.
    """
    tiers = ["low", "medium", "high"]
    if weights is None:
        weights = [1.0, 1.0, 1.0]

    total_w = sum(weights)
    if total_w == 0:
        weights = [1.0, 1.0, 1.0]
        total_w = 3.0

    raw = [w / total_w * n for w in weights]
    counts = [int(r) for r in raw]
    remainder = n - sum(counts)
    for i in range(remainder):
        counts[i % 3] += 1

    return {t: c for t, c in zip(tiers, counts)}


def _generate_for_world(
    world_path: Path,
    n: int,
    output_dir: Path,
    rng: random.Random,
    log,
    tier_quotas: dict[str, int] | None,
) -> int:
    """Generate n requests for a single world, applying tier-based rejection sampling.

    Returns the number of requests successfully saved.
    """
    # Import generation primitives from the existing script.
    # We stub hydra/omegaconf in sys.modules before exec_module so that Hydra's
    # @hydra.main decorator (which runs at import time) never tries to load a
    # config file. The stub provides a no-op @hydra.main and minimal omegaconf
    # stubs; since we never call main(), these are sufficient.
    import importlib.util
    import types

    def _make_stub(name: str) -> types.ModuleType:
        stub = types.ModuleType(name)
        stub.main = lambda *a, **kw: (lambda f: f)  # no-op decorator
        stub.DictConfig = dict
        class _OmegaConf:
            @staticmethod
            def select(cfg, key, default=None):
                return default
        stub.OmegaConf = _OmegaConf()
        return stub

    _saved: dict[str, object] = {}
    for _name in ("hydra", "omegaconf"):
        if _name not in sys.modules:
            _saved[_name] = None
            sys.modules[_name] = _make_stub(_name)

    try:
        spec = importlib.util.spec_from_file_location(
            "gen_requests",
            _REPO_ROOT / "scripts" / "generate_user_requests.py",
        )
        gen_mod = types.ModuleType("gen_requests")
        # __file__ must be set before exec_module so top-level code in the imported
        # script can resolve _REPO_ROOT via Path(__file__).parent.parent.
        gen_mod.__file__ = str(_REPO_ROOT / "scripts" / "generate_user_requests.py")
        assert spec and spec.loader
        spec.loader.exec_module(gen_mod)  # type: ignore[union-attr]
    finally:
        for _name, _orig in _saved.items():
            if _orig is None:
                sys.modules.pop(_name, None)
            else:
                sys.modules[_name] = _orig  # type: ignore[assignment]

    generate_from_world_dir = gen_mod.generate_from_world_dir
    load_world_cities = gen_mod.load_world_cities
    _make_request = gen_mod._make_request
    _inject_complexity = gen_mod._inject_complexity
    _anchor_date_range = gen_mod._anchor_date_range
    _ARCHETYPES: list[dict] = gen_mod._ARCHETYPES

    from optimized_llm_planning_memory.core.models import UserRequest

    world_id, sim_date, city_registry = load_world_cities(world_path)

    if len(city_registry) < 2:
        log.warning(
            "insufficient_cities",
            world=str(world_path),
            n=len(city_registry),
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    city_ids = list(city_registry.keys())
    rng.shuffle(city_ids)

    # Single-destination pairs (A→B) give geo_score=0; multi-destination pairs
    # (A→{B,C}) give geo_score=0.5, enabling "high" complexity tier.  Both are
    # included so the cycle covers the full complexity range.
    city_pairs = [
        gen_mod.CityPair(
            origin_id=city_ids[i],
            origin_name=city_registry[city_ids[i]],
            dest_ids=[city_ids[(i + 1) % len(city_ids)]],
            dest_names=[city_registry[city_ids[(i + 1) % len(city_ids)]]],
        )
        for i in range(len(city_ids))
        if city_ids[i] != city_ids[(i + 1) % len(city_ids)]
    ]
    if len(city_ids) >= 3:
        for i in range(len(city_ids)):
            others = [cid for j, cid in enumerate(city_ids) if j != i]
            if len(others) >= 2:
                city_pairs.append(gen_mod.CityPair(
                    origin_id=city_ids[i],
                    origin_name=city_registry[city_ids[i]],
                    dest_ids=others[:2],
                    dest_names=[city_registry[cid] for cid in others[:2]],
                ))

    if not city_pairs:
        log.warning("no_city_pairs", world=str(world_path))
        return 0

    pairs_cycle = itertools.cycle(city_pairs)
    n_archetypes = len(_ARCHETYPES)

    if tier_quotas is None:
        # No complexity constraint — generate straight through
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
            req_dict["metadata"]["world_id"] = world_id
            _inject_complexity(req_dict)
            try:
                req = UserRequest.model_validate(req_dict)
                out_path = output_dir / f"request_{req.request_id}.json"
                out_path.write_text(req.model_dump_json(indent=2), encoding="utf-8")
                saved += 1
            except Exception as exc:
                log.warning("validation_failed", error=str(exc))
        return saved

    # Tier-based rejection sampling
    buckets: dict[str, list[dict]] = {t: [] for t in tier_quotas}
    max_attempts = n * 20  # overgenerate up to 20× to fill quotas
    attempt = 0
    i = 0

    while attempt < max_attempts and any(
        len(buckets[t]) < tier_quotas[t] for t in tier_quotas
    ):
        pair = next(pairs_cycle)
        archetype = _ARCHETYPES[i % n_archetypes]
        tone_idx = i // n_archetypes
        budget = rng.choice(archetype["budget_usd"])
        duration = rng.choice(archetype["duration_days"])
        start_date, end_date = _anchor_date_range(sim_date, duration, rng)
        req_dict = _make_request(pair, archetype, budget, start_date, end_date, tone_idx)
        req_dict["world_id"] = world_id
        req_dict["metadata"]["world_id"] = world_id
        _inject_complexity(req_dict)

        tier = req_dict.get("metadata", {}).get("complexity_breakdown", {}).get(
            "complexity_tier", "unknown"
        )
        if tier in buckets and len(buckets[tier]) < tier_quotas[tier]:
            buckets[tier].append(req_dict)
        i += 1
        attempt += 1

    # Fallback: if some tier quotas still unmet (world can't produce that tier),
    # fill the deficit with unconstrained requests so the total count is met.
    n_filled = sum(len(v) for v in buckets.values())
    deficit = n - n_filled
    if deficit > 0:
        log.warning(
            "tier_quota_fallback",
            world=str(world_path),
            deficit=deficit,
            detail="world cannot produce all requested complexity tiers; filling with unconstrained requests",
        )
        for _ in range(deficit * 5):  # allow up to 5× attempts to fill the deficit
            if deficit <= 0:
                break
            pair = next(pairs_cycle)
            archetype = _ARCHETYPES[i % n_archetypes]
            tone_idx = i // n_archetypes
            budget = rng.choice(archetype["budget_usd"])
            duration = rng.choice(archetype["duration_days"])
            start_date, end_date = _anchor_date_range(sim_date, duration, rng)
            req_dict = _make_request(pair, archetype, budget, start_date, end_date, tone_idx)
            req_dict["world_id"] = world_id
            req_dict["metadata"]["world_id"] = world_id
            _inject_complexity(req_dict)
            # Accept any tier to fill the gap
            t = req_dict.get("metadata", {}).get("complexity_breakdown", {}).get("complexity_tier", "unknown")
            if t in buckets:
                buckets[t].append(req_dict)
                deficit -= 1
            i += 1

    saved = 0
    for tier_reqs in buckets.values():
        for req_dict in tier_reqs:
            try:
                req = UserRequest.model_validate(req_dict)
                out_path = output_dir / f"request_{req.request_id}.json"
                out_path.write_text(req.model_dump_json(indent=2), encoding="utf-8")
                saved += 1
            except Exception as exc:
                log.warning("validation_failed", error=str(exc))

    return saved


def main() -> None:
    args = _parse_args()
    configure_logging(level="INFO")
    log = get_logger(__name__)
    set_seed(args.seed)
    rng = random.Random(args.seed)

    # Resolve world directories from either --world_set or --world_dirs
    if args.world_set:
        world_set_dir = Path(args.world_set)
        if not world_set_dir.exists():
            log.error("world_set_missing", path=str(world_set_dir))
            sys.exit(1)
        world_dirs = sorted(world_set_dir.glob("world_*"))
        if not world_dirs:
            log.error("no_worlds_in_set", path=str(world_set_dir))
            print(f"ERROR: No world_* sub-folders found in {world_set_dir}")
            sys.exit(1)
        log.info("world_set_expanded", set_dir=str(world_set_dir), n_worlds=len(world_dirs))
    else:
        world_dirs = [Path(p) for p in args.world_dirs]

    n_worlds = len(world_dirs)

    for wd in world_dirs:
        if not wd.exists():
            log.error("world_dir_missing", path=str(wd))
            sys.exit(1)

    # Compute per-world request counts
    per_world = args.n_total // n_worlds
    remainders = args.n_total % n_worlds  # first R worlds get one extra
    world_counts = [per_world + (1 if i < remainders else 0) for i in range(n_worlds)]

    # Complexity quotas — None means no constraint
    tier_quotas_total: dict[str, int] | None = None
    if args.complexity_weights is not None:
        tier_quotas_total = _tier_quotas(args.n_total, list(args.complexity_weights))
        log.info("complexity_quotas", **tier_quotas_total)

    collection = args.collection or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / collection
    total_saved = 0

    for world_dir, n_for_world in zip(world_dirs, world_counts):
        if n_for_world == 0:
            continue

        # Compute per-world tier quotas proportional to world count.
        # No floor of 1: a tier that rounds to 0 for a given world simply
        # won't be demanded from it, avoiding unsatisfiable quotas in small worlds.
        world_tier_quotas: dict[str, int] | None = None
        if tier_quotas_total is not None:
            fraction = n_for_world / args.n_total
            world_tier_quotas = {
                t: max(0, round(q * fraction))
                for t, q in tier_quotas_total.items()
            }
            # Adjust to exact n_for_world (add/remove from medium bucket)
            diff = n_for_world - sum(world_tier_quotas.values())
            if diff != 0:
                world_tier_quotas["medium"] = max(0, world_tier_quotas["medium"] + diff)

        log.info("generating_world", world_dir=str(world_dir), n=n_for_world)
        saved = _generate_for_world(
            world_path=world_dir,
            n=n_for_world,
            output_dir=output_dir,
            rng=rng,
            log=log,
            tier_quotas=world_tier_quotas,
        )
        log.info("world_done", world_dir=str(world_dir), saved=saved)
        total_saved += saved

    print(f"\n{'-' * 50}")
    print(f"  Total generated : {total_saved} / {args.n_total} requests")
    print(f"  Collection      : {collection}")
    print(f"  Output dir      : {output_dir.resolve()}")
    print(f"  Worlds used     : {n_worlds}")
    if tier_quotas_total:
        print(f"  Complexity mix  : {tier_quotas_total}")
    print(f"{'-' * 50}\n")


if __name__ == "__main__":
    main()
