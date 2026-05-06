"""
scripts/run_agent_batch.py
===========================
Batch agent runner — runs the ReAct agent on a set of user requests and saves
each resulting EpisodeLog to outputs/episodes/.

Each request carries a ``world_id`` field linking it to the travel world it was
generated from. The runner searches WORLDS_ROOT (and its immediate sub-folders,
e.g. worlds/batch_1/) for that world directory and instantiates the correct
SimulatorAdapter so that tool calls hit the exact same city/hotel/flight graph
that was used when the request was created.

This is essential for ablation studies: re-running the same request set with a
different agent configuration (agent_mode, augmentation_id) requires routing
each request back to its original world so the constraint satisfaction problem
is identical across conditions.

Parallelism
-----------
By default (--concurrency 5) episodes run in parallel using a thread pool.
Each episode gets its own SimulatorAdapter instance — world state (bookings)
must not be shared across concurrent episodes. The compressor, system prompt,
and MCTS controller are constructed once and shared read-only across threads.

If the agent uses a trained transformer compressor (not IdentityCompressor or
LLMCompressor) set --concurrency 1 to avoid concurrent model-inference races.

Usage
-----
    # All requests in a named collection — raw baseline (no compression)
    python scripts/run_agent_batch.py --collection val --agent_mode raw

    # Same requests, trained compressor from registry
    python scripts/run_agent_batch.py --collection val \\
        --agent_mode compressor --augmentation_id tgad-trained-001

    # Re-run the exact request set from a prior eval run with a new agent config
    python scripts/run_agent_batch.py \\
        --from_eval_run outputs/eval_results/20260506_abc12345 \\
        --agent_mode compressor --augmentation_id tgad-trained-002

    # Specific request IDs — quick smoke test
    python scripts/run_agent_batch.py \\
        --request_ids req_abc123 req_def456 --agent_mode llm_summary

Progress
--------
Each completed episode prints ``EPISODE_DONE: {n}/{total} {episode_id}`` to
stdout. The UI tracks progress by counting episode files newer than the job
start timestamp.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env", override=False)

from optimized_llm_planning_memory.utils.logging import configure_logging, get_logger
from optimized_llm_planning_memory.utils.seed import set_seed


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ReAct agent on a batch of user requests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    req_grp = parser.add_mutually_exclusive_group(required=True)
    req_grp.add_argument(
        "--collection", metavar="NAME",
        help="Run on all requests in this named collection sub-folder (e.g. 'ablation_hard', '20260506_143022').",
    )
    req_grp.add_argument(
        "--request_ids", nargs="+", metavar="ID",
        help="Run on specific request IDs.",
    )
    req_grp.add_argument(
        "--from_eval_run", type=Path, metavar="PATH",
        help="Re-run the request_ids listed in a prior eval run's manifest.json.",
    )

    parser.add_argument(
        "--agent_mode", required=True,
        choices=["raw", "llm_summary", "compressor", "mcts_compressor"],
    )
    parser.add_argument("--augmentation_id", default=None,
                        help="AugmentationRegistry ID (takes priority over --agent_mode for compressor config).")
    parser.add_argument("--prompt_id", default=None,
                        help="PromptRegistry ID for system prompt selection.")
    parser.add_argument("--llm_model_id", default="openai/gpt-4o-mini")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--compress_every_n_steps", type=int, default=5)

    parser.add_argument("--worlds_root", type=Path, default=_REPO_ROOT / "worlds",
                        help="Root worlds directory. Searched one level deep for world sub-folders.")
    parser.add_argument("--requests_dir", type=Path, default=_REPO_ROOT / "data" / "user_requests")
    parser.add_argument("--output_dir", type=Path, default=_REPO_ROOT / "outputs" / "episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help=(
            "Number of episodes to run in parallel (default 5). "
            "Set to 1 for sequential execution. "
            "Reduce if hitting OpenAI rate limits or using a trained transformer compressor."
        ),
    )

    return parser.parse_args()


# ── World resolution ──────────────────────────────────────────────────────────

def _find_worlds_dir(world_id: str | None, worlds_root: Path) -> Path:
    """Return the parent directory of the named world folder.

    Searches worlds_root and its immediate subdirectories (world-set folders
    like worlds/batch_1/) so that requests generated against any world set
    are found automatically.
    """
    if world_id is None:
        return worlds_root
    if (worlds_root / world_id).exists():
        return worlds_root
    for subdir in sorted(worlds_root.iterdir()):
        if subdir.is_dir() and (subdir / world_id).exists():
            return subdir
    return worlds_root


# ── Request loading ───────────────────────────────────────────────────────────

def _load_requests(args: argparse.Namespace, log) -> list:
    from optimized_llm_planning_memory.core.models import UserRequest

    requests_dir = Path(args.requests_dir)
    files: list[Path] = []

    if args.collection:
        coll_dir = requests_dir / args.collection
        if not coll_dir.exists():
            log.error("collection_dir_missing", path=str(coll_dir))
            sys.exit(1)
        files = sorted(coll_dir.glob("request_*.json"))
        if not files:
            log.error("no_requests_in_collection", collection=args.collection, path=str(coll_dir))
            sys.exit(1)

    elif args.request_ids:
        for req_id in args.request_ids:
            found = list(requests_dir.rglob(f"*{req_id}*.json"))
            if not found:
                log.warning("request_not_found", request_id=req_id)
            else:
                files.append(found[0])

    elif args.from_eval_run:
        run_path = Path(args.from_eval_run)
        manifest_path = run_path / "manifest.json" if run_path.is_dir() else run_path
        if not manifest_path.exists():
            log.error("manifest_missing", path=str(manifest_path))
            sys.exit(1)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for req_id in data.get("request_ids", []):
            found = list(requests_dir.rglob(f"*{req_id}*.json"))
            if found:
                files.append(found[0])
            else:
                log.warning("request_not_found", request_id=req_id)

    requests = []
    for f in files:
        try:
            requests.append(UserRequest.model_validate(json.loads(f.read_text(encoding="utf-8"))))
        except Exception as exc:
            log.warning("request_parse_failed", path=str(f), error=str(exc))

    return requests


# ── Compressor + system prompt (built once, reused across episodes) ────────────

def _build_shared_components(args: argparse.Namespace, log):
    """Return (compressor, system_prompt, mcts_controller).

    These are expensive to construct (weight loading for transformer compressor)
    so they are built once and shared across all episodes in the batch.
    """
    from optimized_llm_planning_memory.agent.prompts import get_system_prompt

    system_prompt = get_system_prompt("v2")
    mcts_controller = None

    if args.augmentation_id or args.prompt_id:
        try:
            from optimized_llm_planning_memory.core.registry import AugmentationRegistry, PromptRegistry
            from optimized_llm_planning_memory.core.run_spec import (
                RunSpec, resolve_run_spec, build_compressor_from_entry,
            )
            from omegaconf import OmegaConf

            spec = RunSpec(
                prompt_id=args.prompt_id or "v2",
                augmentation_id=args.augmentation_id or "raw-default",
            )
            resolved = resolve_run_spec(
                spec,
                prompt_registry=PromptRegistry.load(),
                aug_registry=AugmentationRegistry.load(),
            )
            cfg = OmegaConf.create({
                "compressor": {"device": "auto", "model_name_or_path": ""},
                "agent": {"llm_model_id": args.llm_model_id},
            })
            compressor = build_compressor_from_entry(resolved.augmentation_entry, cfg, _REPO_ROOT, log)
            system_prompt = resolved.system_prompt
            log.info("registry.resolved", aug_id=args.augmentation_id, prompt_id=args.prompt_id)
        except Exception as exc:
            log.warning("registry.resolution_failed", error=str(exc), fallback="agent_mode_default")
            compressor = _compressor_from_mode(args, log)
    else:
        compressor = _compressor_from_mode(args, log)

    if args.agent_mode == "mcts_compressor":
        try:
            from optimized_llm_planning_memory.mcts.config import MCTSConfig
            from optimized_llm_planning_memory.mcts.controller import MCTSController
            from optimized_llm_planning_memory.mcts.node_evaluator import NodeEvaluator

            mcts_cfg = MCTSConfig()
            evaluator = NodeEvaluator(model_id=mcts_cfg.evaluator_model_id, config=mcts_cfg)
            mcts_controller = MCTSController(
                evaluator=evaluator,
                llm_model_id=args.llm_model_id,
                config=mcts_cfg,
            )
        except Exception as exc:
            log.warning("mcts.build_failed", error=str(exc))

    return compressor, system_prompt, mcts_controller


def _compressor_from_mode(args: argparse.Namespace, log):
    mode = args.agent_mode
    if mode == "llm_summary":
        from optimized_llm_planning_memory.compressor.llm_compressor import LLMCompressor
        return LLMCompressor(model_id=args.llm_model_id)
    if mode in ("compressor", "mcts_compressor"):
        log.warning(
            "no_augmentation_id",
            detail="Using IdentityCompressor; pass --augmentation_id for a trained checkpoint.",
        )
    from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
    return IdentityCompressor()


# ── Per-episode agent construction ────────────────────────────────────────────

def _build_agent_for_episode(args, simulator, compressor, system_prompt, mcts_controller):
    """Build a ReActAgent with fresh tracker/event_bus per episode.

    ToolCallTracker accumulates state per episode; reusing it across episodes
    would corrupt tool-efficiency metrics. Only the compressor and simulator
    are reused.
    """
    from optimized_llm_planning_memory.tools.registry import ToolRegistry
    from optimized_llm_planning_memory.tools.tracker import ToolCallTracker
    from optimized_llm_planning_memory.tools.events import EventBus
    from optimized_llm_planning_memory.agent.react_agent import ReActAgent
    from optimized_llm_planning_memory.agent.modes import AgentMode
    from optimized_llm_planning_memory.agent.context_builder import ContextBuilder
    from optimized_llm_planning_memory.core.config import AgentConfig

    tracker = ToolCallTracker()
    event_bus = EventBus()
    tool_registry = ToolRegistry.from_config(
        simulator=simulator,
        tracker=tracker,
        event_bus=event_bus,
    )
    config = AgentConfig(
        mode=args.agent_mode,
        llm_model_id=args.llm_model_id,
        max_steps=args.max_steps,
        compress_every_n_steps=args.compress_every_n_steps,
    )
    context_builder = ContextBuilder(
        system_prompt=system_prompt,
        tool_registry=tool_registry,
        llm_model_id=args.llm_model_id,
    )
    return ReActAgent(
        llm_model_id=args.llm_model_id,
        tool_registry=tool_registry,
        compressor=compressor,
        context_builder=context_builder,
        config=config,
        mode=AgentMode(args.agent_mode),
        mcts_controller=mcts_controller,
    )


# ── Per-episode worker ────────────────────────────────────────────────────────

def _run_one_episode(
    req,
    args: argparse.Namespace,
    compressor,
    system_prompt: str,
    mcts_controller,
    total: int,
    counter: dict,
    lock: threading.Lock,
    log,
) -> None:
    """Run a single episode and save it.  Safe to call from multiple threads.

    Each call creates its own SimulatorAdapter — the world's in-memory booking
    state must not be shared across concurrent episodes.
    """
    from optimized_llm_planning_memory.simulator.adapter import SimulatorAdapter
    from optimized_llm_planning_memory.utils.episode_io import save_episode
    from optimized_llm_planning_memory.utils.live_writer import LiveEpisodeWriter

    world_id = req.world_id
    worlds_dir = _find_worlds_dir(world_id, args.worlds_root)

    try:
        sim = SimulatorAdapter(
            seed=args.seed,
            worlds_dir=str(worlds_dir),
            world_id=world_id,
        )
    except Exception as exc:
        with lock:
            counter["failed"] += 1
        log.error("simulator.init_failed", world_id=world_id, error=str(exc))
        print(f"  SKIP {req.request_id}: world init failed — {exc}", flush=True)
        return

    try:
        agent = _build_agent_for_episode(args, sim, compressor, system_prompt, mcts_controller)
        episode_id = str(uuid.uuid4())
        with LiveEpisodeWriter(episode_id=episode_id, output_dir=args.output_dir) as lw:
            episode_log = agent.run_episode(
                request=req,
                simulator=sim,
                live_writer=lw,
                episode_id=episode_id,
            )
        save_episode(episode_log, args.output_dir)
        with lock:
            counter["done"] += 1
            n = counter["done"]
        print(f"EPISODE_DONE: {n}/{total} {episode_id}", flush=True)
        log.info("episode.saved", episode_id=episode_id, n_done=n, total=total)
    except Exception as exc:
        with lock:
            counter["failed"] += 1
        log.error("episode.failed", request_id=req.request_id, error=str(exc))
        print(f"  FAIL {req.request_id}: {exc}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    configure_logging(level="INFO")
    log = get_logger(__name__)
    set_seed(args.seed)

    requests = _load_requests(args, log)
    if not requests:
        print("ERROR: No requests loaded. Check your --collection / --request_ids / --from_eval_run args.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = len(requests)
    print(f"Batch run: {total} requests · agent_mode={args.agent_mode} · concurrency={args.concurrency}", flush=True)

    # Build expensive components once; shared read-only across all worker threads
    compressor, system_prompt, mcts_controller = _build_shared_components(args, log)

    counter: dict = {"done": 0, "failed": 0}
    lock = threading.Lock()

    worker_kwargs = dict(
        args=args,
        compressor=compressor,
        system_prompt=system_prompt,
        mcts_controller=mcts_controller,
        total=total,
        counter=counter,
        lock=lock,
        log=log,
    )

    if args.concurrency == 1:
        for req in requests:
            _run_one_episode(req, **worker_kwargs)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(_run_one_episode, req, **worker_kwargs) for req in requests]
            concurrent.futures.wait(futures)

    n_done = counter["done"]
    n_failed = counter["failed"]
    print(f"\n{'-' * 52}")
    print(f"  Completed : {n_done} / {total} episodes")
    print(f"  Failed    : {n_failed} / {total} episodes")
    print(f"  Output    : {args.output_dir.resolve()}")
    print(f"{'-' * 52}\n")


if __name__ == "__main__":
    main()
