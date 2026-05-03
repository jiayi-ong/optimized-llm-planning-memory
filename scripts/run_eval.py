"""
scripts/run_eval.py
====================
Standalone CLI for evaluating saved EpisodeLogs — without re-running the agent.

Design: pure evaluation pass
------------------------------
This script reads existing ``ep_*.json`` files, scores them with the current
DeterministicEvaluator (and optionally the LLMJudge), then writes a timestamped
eval run directory.  It never touches the agent or creates new episodes.

This isolation means eval can be re-applied any time (e.g., after bumping
METRIC_VERSION from v1 to v2) without re-running the full planning pipeline.

Output directory naming
-----------------------
    outputs/eval_results/{YYYYMMDD_HHMMSS}_{run_id_short}/
                          ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^
                          sortable prefix   8-char UUID fragment

This ensures ``ls`` returns runs in chronological order.

Usage
-----
    # Score ALL episodes in the default episodes directory (explicit form)
    python scripts/run_eval.py --all

    # Score ALL episodes, skip LLM judge (fast mode)
    python scripts/run_eval.py --all --deterministic_only

    # Score specific episodes by UUID
    python scripts/run_eval.py --episode_ids 08dff70b-548a-... a1b2c3d4-...

    # Score all episodes that belong to specific requests
    python scripts/run_eval.py --request_ids req-abc req-def

    # Use a different LLM judge model
    python scripts/run_eval.py --all --judge_model openai/gpt-4o

    # Filter by agent_mode and add a note
    python scripts/run_eval.py --all --agent_mode raw --note "baseline re-eval after v2 metrics"

Selection flags (mutually exclusive)
-------------------------------------
    --all           Score every ep_*.json in episodes_dir.
    --episode_ids   Score a specific list of episode UUIDs.
    --request_ids   Score all episodes whose request_id matches.

    Omitting all three is equivalent to --all (backward compatible).
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

from optimized_llm_planning_memory.core.models import EpisodeLog, UserRequest
from optimized_llm_planning_memory.evaluation.deterministic import DeterministicEvaluator, METRIC_VERSION
from optimized_llm_planning_memory.evaluation.evaluator import Evaluator
from optimized_llm_planning_memory.evaluation.manifest import EvalRunManifest
from optimized_llm_planning_memory.utils.episode_io import (
    list_episodes,
    list_episodes_by_request,
    save_eval_run,
)
from optimized_llm_planning_memory.utils.logging import configure_logging, get_logger


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_episodes_for_eval(
    episodes_dir: Path,
    episode_ids: list[str] | None,
    request_ids: list[str] | None,
    agent_mode: str | None,
) -> list[EpisodeLog]:
    """Return the filtered list of EpisodeLogs to score.

    Priority:
    1. If ``episode_ids`` provided → load only those files.
    2. Elif ``request_ids`` provided → use list_episodes_by_request (efficient).
    3. Else → load all episodes.

    Then optionally filter by ``agent_mode``.
    """
    if episode_ids:
        episodes = []
        for eid in episode_ids:
            path = episodes_dir / f"ep_{eid}.json"
            if not path.exists():
                # Also try the raw ID as filename
                path = episodes_dir / eid if (episodes_dir / eid).exists() else path
            if path.exists():
                try:
                    from optimized_llm_planning_memory.utils.episode_io import load_episode
                    episodes.append(load_episode(path))
                except Exception as e:
                    print(f"  WARN: could not load {path}: {e}", file=sys.stderr)
    elif request_ids:
        episodes = list_episodes_by_request(episodes_dir, set(request_ids))
    else:
        episodes = list_episodes(episodes_dir)

    if agent_mode:
        episodes = [ep for ep in episodes if ep.agent_mode == agent_mode]

    return episodes


def resolve_user_request(
    episode: EpisodeLog,
    fallback_dirs: list[Path],
) -> UserRequest | None:
    """Return the UserRequest for an episode.

    Checks ``episode.user_request`` first (embedded at save time). Falls back
    to searching ``fallback_dirs`` by ``request_id`` if not embedded.
    """
    if episode.user_request is not None:
        return episode.user_request

    for search_dir in fallback_dirs:
        if not search_dir.exists():
            continue
        for json_file in search_dir.glob("request_*.json"):
            try:
                req = UserRequest.model_validate_json(json_file.read_text(encoding="utf-8"))
                if req.request_id == episode.request_id:
                    return req
            except Exception:
                continue

    return None


def _print_summary(results) -> None:
    """Print a compact summary table to stdout."""
    if not results:
        print("  (no results)")
        return
    header = f"  {'request_id':<36}  {'agent_mode':<20}  {'overall':>7}  {'hard_constr':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        hcr = r.deterministic_scores.get("hard_constraint_ratio", float("nan"))
        print(
            f"  {r.request_id:<36}  {r.agent_mode:<20}  "
            f"{r.overall_score:>7.3f}  {hcr:>12.3f}"
        )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved EpisodeLogs and write timestamped eval results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n-----")[1] if "Usage\n-----" in __doc__ else "",
    )
    # ── Episode selection (mutually exclusive) ────────────────────────────────
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all",
        action="store_true",
        dest="rerun_all",
        help="Score every ep_*.json file in episodes_dir. Equivalent to omitting all selection flags.",
    )
    selection.add_argument(
        "--episode_ids",
        nargs="+",
        metavar="ID",
        help="Score specific episodes by UUID (e.g. --episode_ids 08dff70b-548a... a1b2c3d4-...).",
    )
    selection.add_argument(
        "--request_ids",
        nargs="+",
        metavar="ID",
        help="Score all episodes whose request_id matches one of the given IDs.",
    )
    parser.add_argument(
        "--episodes_dir",
        type=Path,
        default=Path("outputs/episodes"),
        help="Directory containing ep_*.json files (default: outputs/episodes).",
    )
    parser.add_argument(
        "--eval_dir",
        type=Path,
        default=Path("outputs/eval_results"),
        help="Base directory for eval output (default: outputs/eval_results).",
    )
    parser.add_argument(
        "--requests_dir",
        type=Path,
        default=Path("data/user_requests"),
        help="Base directory to search for UserRequest JSON files when not embedded.",
    )
    parser.add_argument(
        "--deterministic_only",
        action="store_true",
        help="Skip LLM judge (fast mode).",
    )
    parser.add_argument(
        "--judge_model",
        default="openai/gpt-4o-mini",
        help="litellm model string for the LLM judge (default: openai/gpt-4o-mini).",
    )
    parser.add_argument(
        "--agent_mode",
        default=None,
        help="Filter episodes to a specific agent_mode (raw, llm_summary, compressor, ...).",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Free-text note recorded in the eval run manifest.",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        metavar="PATH",
        help=(
            "Append structured logs to this file in addition to stdout. "
            "Parent directories are created automatically. "
            "Example: --log_file outputs/logs/run_eval.log"
        ),
    )
    args = parser.parse_args()

    configure_logging(level="INFO", log_file=args.log_file)
    log = get_logger(__name__)

    # ── Resolve selection mode and log it ─────────────────────────────────────
    if args.episode_ids:
        mode_desc = f"specific episodes: {args.episode_ids}"
    elif args.request_ids:
        mode_desc = f"episodes for request_ids: {args.request_ids}"
    else:
        mode_desc = "all episodes"
    print(f"Selection  : {mode_desc}")
    if args.agent_mode:
        print(f"Agent mode : {args.agent_mode} (additional filter)")
    print()

    # ── Load episodes ──────────────────────────────────────────────────────────
    log.info("loading_episodes", episodes_dir=str(args.episodes_dir), mode=mode_desc)
    episodes = load_episodes_for_eval(
        args.episodes_dir,
        args.episode_ids,
        args.request_ids,
        args.agent_mode,
    )
    if not episodes:
        print("No episodes found matching the given criteria. Exiting.")
        sys.exit(0)
    log.info("episodes_loaded", n=len(episodes))

    # ── Resolve UserRequests ──────────────────────────────────────────────────
    fallback_dirs = [
        args.requests_dir / split
        for split in ("train", "val", "test")
        if (args.requests_dir / split).exists()
    ] + [args.requests_dir]

    pairs: list[tuple[EpisodeLog, UserRequest]] = []
    skipped = 0
    for ep in episodes:
        req = resolve_user_request(ep, fallback_dirs)
        if req is None:
            log.warning("request_not_found", episode_id=ep.episode_id, request_id=ep.request_id)
            skipped += 1
        else:
            pairs.append((ep, req))

    if not pairs:
        print("Could not resolve UserRequest for any episode. Exiting.")
        sys.exit(1)
    if skipped:
        log.warning("skipped_episodes", n=skipped)

    # ── Build evaluator ───────────────────────────────────────────────────────
    judge = None
    if not args.deterministic_only:
        try:
            from optimized_llm_planning_memory.evaluation.llm_judge import LLMJudge
            judge = LLMJudge(judge_model_id=args.judge_model)
            log.info("llm_judge_ready", model=args.judge_model)
        except Exception as e:
            log.warning("llm_judge_unavailable", error=str(e))

    evaluator = Evaluator(
        deterministic_eval=DeterministicEvaluator(),
        llm_judge=judge,
    )

    # ── Score episodes ────────────────────────────────────────────────────────
    log.info("scoring_episodes", n=len(pairs))
    results = []
    for ep, req in pairs:
        try:
            result = evaluator.evaluate_episode(ep, req)
            results.append(result)
        except Exception as e:
            log.warning("score_failed", episode_id=ep.episode_id, error=str(e))

    if not results:
        print("All scoring attempts failed. Check logs.")
        sys.exit(1)

    # ── Build manifest ────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id_short = uuid.uuid4().hex[:8]
    run_id = f"{ts}_{run_id_short}"

    unique_seeds = sorted({ep.world_seed for ep, _ in pairs if ep.world_seed is not None})
    unique_modes = sorted({ep.agent_mode for ep, _ in pairs})
    # Use the agent_mode from results (single mode if filtered, else multi)
    agent_mode_str = args.agent_mode or (unique_modes[0] if len(unique_modes) == 1 else "mixed")

    manifest = EvalRunManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        compressor_type="unknown",  # run_eval.py evaluates existing episodes; compressor type is in episode
        agent_mode=agent_mode_str,
        judge_model_id=args.judge_model if not args.deterministic_only else "none",
        config_hash="manual",
        metric_version=METRIC_VERSION,
        request_ids=[req.request_id for _, req in pairs],
        n_episodes=len(results),
        deterministic_only=args.deterministic_only,
        world_seeds=unique_seeds,
        episode_source="saved_episodes",
        notes=args.note,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    run_dir = save_eval_run(manifest, results, args.eval_dir)
    log.info("eval_run_saved", run_dir=str(run_dir), n_results=len(results))

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\nEval run: {run_id}")
    print(f"Metric version: {METRIC_VERSION}  |  Episodes scored: {len(results)}")
    print(f"Output: {run_dir}\n")
    _print_summary(results)
    print()


if __name__ == "__main__":
    main()
