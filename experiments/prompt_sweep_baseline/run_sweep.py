"""
experiments/prompt_sweep_baseline/run_sweep.py
===============================================
Prompt-design sweep: run all 7 variants (sweep_A–G) against 3 control user
requests and measure whether each produces a non-degenerate itinerary.

Non-degenerate threshold: at least 1 booked transport + 1 booked hotel.
Measured via completion_rate > 0 AND accommodation_coverage_ratio > 0.

Usage
-----
    # Full sweep (21 episodes: 7 variants × 3 requests)
    python experiments/prompt_sweep_baseline/run_sweep.py

    # Subset (faster iteration)
    python experiments/prompt_sweep_baseline/run_sweep.py --variants sweep_A sweep_D
    python experiments/prompt_sweep_baseline/run_sweep.py --requests req_solo_simple

    # Dry run (no API calls; verifies imports, world loading, request parsing)
    python experiments/prompt_sweep_baseline/run_sweep.py --dry-run

Outputs (in experiments/prompt_sweep_baseline/results/)
---------
    results/<variant>/ep_<request_id>.json     — full EpisodeLog
    results/<variant>/eval_<request_id>.json   — deterministic scores
    summary_results.json                        — aggregated table

Design notes
------------
- All episodes use world_42_20260504_075144 for reproducibility.
- Variants run sequentially to avoid API rate-limit collisions.
- Exceptions are caught per (variant, request); the sweep continues.
- IdentityCompressor is used (no compression) for all variants.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
_EXPERIMENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXPERIMENT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Load API keys before litellm reads os.environ
from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env", override=False)

# ── Variant catalogue ─────────────────────────────────────────────────────────

@dataclass
class VariantConfig:
    name: str
    system_prompt_version: str
    mode: str           # "raw" or "stateless"
    max_steps: int
    description: str


VARIANTS: list[VariantConfig] = [
    VariantConfig("sweep_A", "sweep_A", "raw",       30, "Grow-then-Prune"),
    VariantConfig("sweep_B", "sweep_B", "raw",       30, "Ordered Pipeline"),
    VariantConfig("sweep_C", "sweep_C", "raw",       30, "Commit-Every-Step"),
    VariantConfig("sweep_D", "sweep_D", "raw",       20, "Minimal Viable First"),
    VariantConfig("sweep_E", "sweep_E", "raw",       25, "Step-Budget Phased"),
    VariantConfig("sweep_F", "sweep_F", "stateless", 25, "Non-ReAct Stateless"),
    VariantConfig("sweep_G", "sweep_G", "stateless", 20, "Non-ReAct Commitment-Gated"),
]

VARIANT_LOOKUP = {v.name: v for v in VARIANTS}

# ── Request catalogue ─────────────────────────────────────────────────────────

REQUEST_FILES: dict[str, Path] = {
    "req_solo_simple":  _EXPERIMENT_DIR / "user_requests" / "req_solo_simple.json",
    "req_couple_medium": _EXPERIMENT_DIR / "user_requests" / "req_couple_medium.json",
    "req_family_constrained": _REPO_ROOT / "data" / "user_requests" / "test" /
                               "request_0e8ed622-7fb8-4ca1-bf89-022601b94eaa.json",
}

# ── World settings ────────────────────────────────────────────────────────────

WORLD_ID   = "world_42_20260504_075144"
WORLDS_DIR = str(_REPO_ROOT / "worlds")
SEED       = 42
LLM_MODEL  = "openai/gpt-4o-mini"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_request(path: Path):
    """Load and validate a UserRequest from disk."""
    from optimized_llm_planning_memory.core.models import UserRequest
    return UserRequest.model_validate(json.loads(path.read_text()))


def _is_non_degenerate(episode_log, scores: dict) -> bool:
    """
    Return True if the episode produced a non-degenerate itinerary.

    A non-degenerate itinerary has:
      - completion_rate > 0 (at least some days booked)
      - accommodation_coverage_ratio > 0 (at least 1 hotel booking)
      - At least 1 transport segment in final_itinerary
    """
    it = episode_log.final_itinerary
    has_transport = any(
        seg for day in (it.days if it else []) for seg in day.transport_segments
    )
    has_hotel = scores.get("accommodation_coverage_ratio", 0.0) > 0
    has_days  = scores.get("completion_rate", 0.0) > 0
    return has_transport and has_hotel and has_days


def _print_summary_table(results: list[dict]) -> None:
    """Print a compact ASCII table of key metrics per (variant, request)."""
    cols = ["variant", "request", "non_deg", "completion", "accommodation", "steps", "term_reason"]
    widths = [12, 24, 7, 10, 13, 5, 18]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print("\n" + "=" * len(header))
    print("PROMPT SWEEP SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            row = [
                r["variant"][:widths[0]],
                r["request"][:widths[1]],
                "ERROR",
                "-", "-", "-",
                r["error"][:widths[6]],
            ]
        else:
            nd = "YES" if r["non_degenerate"] else "NO"
            row = [
                r["variant"][:widths[0]],
                r["request"][:widths[1]],
                nd,
                f"{r['scores'].get('completion_rate', 0):.2f}",
                f"{r['scores'].get('accommodation_coverage_ratio', 0):.2f}",
                str(int(r['scores'].get('steps_per_episode', 0))),
                (r.get("termination_reason") or "?")[:widths[6]],
            ]
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    print("=" * len(header))


# ── Core: run one episode ─────────────────────────────────────────────────────

def _build_agent(variant: VariantConfig, simulator):
    """Construct a ReActAgent for the given variant (no compressor)."""
    from optimized_llm_planning_memory.tools.registry import ToolRegistry
    from optimized_llm_planning_memory.tools.tracker import ToolCallTracker
    from optimized_llm_planning_memory.tools.events import EventBus
    from optimized_llm_planning_memory.agent.react_agent import ReActAgent
    from optimized_llm_planning_memory.agent.modes import AgentMode
    from optimized_llm_planning_memory.agent.context_builder import ContextBuilder
    from optimized_llm_planning_memory.agent.prompts import get_system_prompt
    from optimized_llm_planning_memory.core.config import AgentConfig
    from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor

    tracker   = ToolCallTracker()
    event_bus = EventBus()
    registry  = ToolRegistry.from_config(simulator=simulator, tracker=tracker, event_bus=event_bus)

    compressor = IdentityCompressor()
    system_prompt = get_system_prompt(variant.system_prompt_version)
    context_builder = ContextBuilder(
        system_prompt=system_prompt,
        tool_registry=registry,
        llm_model_id=LLM_MODEL,
    )
    agent_config = AgentConfig(
        mode=variant.mode,
        llm_model_id=LLM_MODEL,
        max_steps=variant.max_steps,
        compress_every_n_steps=9999,
    )
    agent = ReActAgent(
        llm_model_id=LLM_MODEL,
        tool_registry=registry,
        compressor=compressor,
        context_builder=context_builder,
        config=agent_config,
        mode=AgentMode(variant.mode),
    )
    return agent


def _run_one(
    variant: VariantConfig,
    request_name: str,
    request_path: Path,
    results_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """
    Run a single (variant, request) episode and return a result summary dict.

    Returns a dict with keys: variant, request, non_degenerate, scores,
    termination_reason, episode_path, eval_path.
    On error: variant, request, error (str).
    """
    from optimized_llm_planning_memory.simulator.adapter import SimulatorAdapter
    from optimized_llm_planning_memory.evaluation.deterministic import DeterministicEvaluator

    print(f"\n[{variant.name}] {request_name} — {variant.description}", flush=True)

    user_request = _load_request(request_path)
    out_dir = results_dir / f"{variant.name}_{'react' if variant.mode == 'raw' else 'stateless'}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"  [DRY RUN] request_id={user_request.request_id}, mode={variant.mode}, "
              f"prompt={variant.system_prompt_version}, max_steps={variant.max_steps}")
        return {
            "variant": variant.name,
            "request": request_name,
            "non_degenerate": None,
            "scores": {},
            "termination_reason": "dry_run",
            "episode_path": None,
            "eval_path": None,
        }

    simulator = SimulatorAdapter(
        seed=SEED,
        worlds_dir=WORLDS_DIR,
        world_id=WORLD_ID,
    )
    agent = _build_agent(variant, simulator)
    episode_id = str(uuid.uuid4())

    episode_log = agent.run_episode(
        request=user_request,
        simulator=simulator,
        episode_id=episode_id,
    )

    # Save episode log
    ep_path = out_dir / f"ep_{user_request.request_id}.json"
    ep_path.write_text(episode_log.model_dump_json(indent=2))
    print(f"  Episode saved ->{ep_path.relative_to(_REPO_ROOT)}")
    print(f"  Steps: {episode_log.total_steps}  Termination: {episode_log.termination_reason}")

    # Deterministic evaluation
    evaluator = DeterministicEvaluator()
    scores = evaluator.score(episode_log, user_request)

    eval_path = out_dir / f"eval_{user_request.request_id}.json"
    eval_path.write_text(json.dumps(scores, indent=2))
    print(f"  Eval saved   ->{eval_path.relative_to(_REPO_ROOT)}")

    nd = _is_non_degenerate(episode_log, scores)
    status = "NON-DEGENERATE" if nd else "DEGENERATE"
    print(f"  Itinerary: {status} | "
          f"completion={scores.get('completion_rate', 0):.2f} | "
          f"accommodation={scores.get('accommodation_coverage_ratio', 0):.2f}")

    return {
        "variant": variant.name,
        "request": request_name,
        "non_degenerate": nd,
        "scores": scores,
        "termination_reason": episode_log.termination_reason,
        "episode_path": str(ep_path),
        "eval_path": str(eval_path),
    }


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(
    variants: list[str] | None,
    requests: list[str] | None,
    dry_run: bool,
) -> None:
    from optimized_llm_planning_memory.utils.seed import set_seed
    set_seed(SEED)

    # Filter variants and requests
    active_variants = [VARIANT_LOOKUP[v] for v in (variants or list(VARIANT_LOOKUP))]
    active_requests = {k: v for k, v in REQUEST_FILES.items()
                       if requests is None or k in requests}

    if not active_variants:
        print("ERROR: No valid variants selected.", file=sys.stderr)
        sys.exit(1)
    if not active_requests:
        print("ERROR: No valid requests selected.", file=sys.stderr)
        sys.exit(1)

    results_dir = _EXPERIMENT_DIR / "results"
    all_results: list[dict] = []

    total = len(active_variants) * len(active_requests)
    done  = 0

    for variant in active_variants:
        variant_non_deg = 0
        for req_name, req_path in active_requests.items():
            done += 1
            print(f"\n{'-' * 60}", flush=True)
            print(f"  [{done}/{total}] variant={variant.name}, request={req_name}")
            try:
                result = _run_one(variant, req_name, req_path, results_dir, dry_run)
                all_results.append(result)
                if result.get("non_degenerate"):
                    variant_non_deg += 1
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"  ERROR: {exc}", flush=True)
                print(tb, flush=True)
                all_results.append({
                    "variant": variant.name,
                    "request": req_name,
                    "error": str(exc),
                })

        # Early report: did this variant produce non-degenerate results for all requests?
        n_requests = len(active_requests)
        if not dry_run and variant_non_deg == n_requests:
            print(f"\n*** EARLY REPORT: {variant.name} ({variant.description}) ***")
            print(f"    Non-degenerate on ALL {n_requests} requests. "
                  f"This is a candidate winning prompt variant.")

    # Save aggregated summary
    summary_path = _EXPERIMENT_DIR / "summary_results.json"
    summary = {
        "sweep_config": {
            "variants":      [v.name for v in active_variants],
            "requests":      list(active_requests.keys()),
            "world_id":      WORLD_ID,
            "llm_model":     LLM_MODEL,
            "seed":          SEED,
            "dry_run":       dry_run,
        },
        "results": all_results,
        "non_degenerate_counts": {
            v.name: sum(1 for r in all_results
                        if r.get("variant") == v.name and r.get("non_degenerate"))
            for v in active_variants
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n\nSummary saved ->{summary_path.relative_to(_REPO_ROOT)}")

    _print_summary_table(all_results)

    # Final assessment
    nd_variants = [v.name for v in active_variants
                   if summary["non_degenerate_counts"].get(v.name, 0) == len(active_requests)]
    if not dry_run:
        if nd_variants:
            print(f"\nWINNING VARIANTS (non-degenerate on all {len(active_requests)} requests):")
            for name in nd_variants:
                v = VARIANT_LOOKUP[name]
                print(f"  - {name}: {v.description}")
        else:
            print("\nNO variant produced non-degenerate results on all requests.")
            print("Consider reviewing the episode logs for failure modes.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt-design sweep: test 7 agent variants for non-degenerate itinerary production.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANT_LOOKUP),
        default=None,
        metavar="VARIANT",
        help="Subset of variants to run (default: all 7). E.g. --variants sweep_A sweep_D",
    )
    parser.add_argument(
        "--requests",
        nargs="+",
        choices=list(REQUEST_FILES),
        default=None,
        metavar="REQUEST",
        help="Subset of requests to run (default: all 3). E.g. --requests req_solo_simple",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Verify imports, world loading, and request parsing without making LLM API calls.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_sweep(
        variants=args.variants,
        requests=args.requests,
        dry_run=args.dry_run,
    )
