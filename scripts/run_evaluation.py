"""
scripts/run_evaluation.py
==========================
Evaluate a trained compressor (or baseline) on the test split.

Usage
-----
    # Deterministic only (fast — no LLM judge)
    python scripts/run_evaluation.py eval.deterministic_only=true agent.mode=raw

    # Full evaluation with LLM judge
    python scripts/run_evaluation.py agent.mode=compressor

    # With a specific checkpoint
    python scripts/run_evaluation.py \\
        agent.mode=compressor \\
        training.resume_from=outputs/checkpoints/final/ppo_model.zip

    # Load checkpoint automatically from a training run_id (reads manifest.json)
    python scripts/run_evaluation.py \\
        +run_id=20260501_120000 \\
        eval.deterministic_only=true

Outputs
-------
- JSON file at ``outputs/eval_results/<run_name>.json`` with all EvalResult objects.
- Aggregated metrics printed to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

import hydra
from omegaconf import DictConfig, OmegaConf

from optimized_llm_planning_memory.utils.logging import configure_logging, get_logger
from optimized_llm_planning_memory.utils.seed import set_seed


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    import datetime as _dt
    _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file = OmegaConf.select(cfg, "logging.log_file") or str(
        Path(cfg.project.output_dir) / "logs" / f"run_evaluation_{_ts}.log"
    )
    configure_logging(level=cfg.logging.level, log_file=_log_file)
    log = get_logger(__name__)
    set_seed(cfg.project.seed)

    log.info("evaluation.start", mode=cfg.agent.mode, deterministic_only=cfg.eval.deterministic_only)

    # ── Auto-resolve checkpoint from run_id (--run-id / +run_id=...) ─────────
    run_id = OmegaConf.select(cfg, "run_id")
    if run_id:
        from optimized_llm_planning_memory.training.run_manifest import (
            load_manifest,
            resolve_checkpoint,
        )
        manifest = load_manifest(run_id, training_dir=Path(cfg.project.output_dir) / "training")
        if manifest:
            log.info(
                "eval.run_id.manifest_found",
                run_id=run_id,
                compressor_type=manifest.compressor_type,
                n_train_requests=manifest.n_train_requests,
            )
            # Override compressor type from manifest if not explicitly set
            if not OmegaConf.select(cfg, "compressor.type"):
                log.info("eval.run_id.inferring_compressor", type=manifest.compressor_type)
        ckpt = resolve_checkpoint(run_id, output_dir=cfg.project.output_dir)
        if ckpt:
            log.info("eval.run_id.checkpoint_resolved", path=str(ckpt))
            # Inject as training.resume_from so the checkpoint-loading code below picks it up
            OmegaConf.update(cfg, "training.resume_from", str(ckpt), merge=True)
        else:
            log.warning("eval.run_id.no_checkpoint_found", run_id=run_id)

    # ── Load test requests ────────────────────────────────────────────────────
    from optimized_llm_planning_memory.core.models import UserRequest

    test_dir = Path("data/user_requests/test")
    if not test_dir.exists() or not list(test_dir.glob("*.json")):
        log.warning("no_test_requests", path=str(test_dir),
                    hint="Run scripts/generate_user_requests.py first.")
        template = Path("data/user_requests/templates/request_template.json")
        user_requests = [UserRequest.model_validate(json.loads(template.read_text()))]
    else:
        user_requests = [
            UserRequest.model_validate(json.loads(f.read_text()))
            for f in sorted(test_dir.glob("*.json"))
        ]
    log.info("loaded_test_requests", n=len(user_requests))

    # ── Build agent + run episodes ────────────────────────────────────────────
    from optimized_llm_planning_memory.simulator.adapter import SimulatorAdapter
    from optimized_llm_planning_memory.tools.registry import ToolRegistry
    from optimized_llm_planning_memory.tools.tracker import ToolCallTracker
    from optimized_llm_planning_memory.tools.events import EventBus
    from optimized_llm_planning_memory.agent.react_agent import ReActAgent
    from optimized_llm_planning_memory.agent.modes import AgentMode
    from optimized_llm_planning_memory.agent.context_builder import ContextBuilder
    from optimized_llm_planning_memory.agent.prompts import get_system_prompt
    from optimized_llm_planning_memory.core.config import AgentConfig
    from optimized_llm_planning_memory.utils.episode_io import save_episode

    compressor_type = cfg.compressor.type
    if compressor_type == "identity":
        from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
        compressor = IdentityCompressor()
    elif compressor_type == "llm":
        from optimized_llm_planning_memory.compressor.llm_compressor import LLMCompressor
        compressor = LLMCompressor(
            model_id=OmegaConf.select(cfg, "compressor.llm_model_id",
                                       default=OmegaConf.select(cfg, "compressor.model_name_or_path",
                                                                  default="openai/gpt-4o-mini"))
        )
    elif compressor_type == "llm_mcts":
        from optimized_llm_planning_memory.compressor.llm_mcts_compressor import LLMMCTSCompressor
        compressor = LLMMCTSCompressor(
            llm_model_id=OmegaConf.select(cfg, "compressor.llm_model_id", default="openai/gpt-4o-mini"),
            max_output_tokens=OmegaConf.select(cfg, "compressor.max_output_tokens", default=1024),
        )
    elif compressor_type == "transformer":
        from optimized_llm_planning_memory.compressor.transformer_compressor import TransformerCompressor
        compressor = TransformerCompressor(
            model_name_or_path=cfg.compressor.model_name_or_path,
            device=OmegaConf.select(cfg, "compressor.device", default="auto"),
        )
    elif compressor_type == "dummy":
        from optimized_llm_planning_memory.compressor.dummy_compressor import DummyCompressor
        compressor = DummyCompressor()
    else:
        from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
        compressor = IdentityCompressor()

    # Warn on inconsistent compressor/agent-mode combinations.
    if compressor_type == "llm_mcts" and cfg.agent.mode != "mcts_compressor":
        log.warning(
            "config.mismatch",
            detail=(
                f"compressor=llm_mcts requires agent mode 'mcts_compressor', "
                f"but agent.mode='{cfg.agent.mode}'. "
                f"Use: agent=react_mcts compressor=llm_mcts"
            ),
        )

    # Build MCTSController when running in mcts_compressor mode.
    mcts_controller = None
    mcts_cfg_node = OmegaConf.select(cfg, "agent.mcts")
    if cfg.agent.mode == "mcts_compressor" and mcts_cfg_node is not None:
        from optimized_llm_planning_memory.mcts.config import MCTSConfig
        from optimized_llm_planning_memory.mcts.controller import MCTSController
        from optimized_llm_planning_memory.mcts.node_evaluator import NodeEvaluator
        mcts_cfg = MCTSConfig(**OmegaConf.to_container(mcts_cfg_node, resolve=True))
        evaluator = NodeEvaluator(model_id=mcts_cfg.evaluator_model_id, config=mcts_cfg)
        mcts_controller = MCTSController(
            evaluator=evaluator,
            llm_model_id=cfg.agent.llm_model_id,
            config=mcts_cfg,
        )

    agent_config = AgentConfig(
        mode=cfg.agent.mode,
        llm_model_id=cfg.agent.llm_model_id,
        max_steps=cfg.agent.max_steps,
        compress_every_n_steps=cfg.agent.compress_every_n_steps,
    )
    system_prompt = get_system_prompt(
        OmegaConf.select(cfg, "agent.system_prompt_version", default="v1")
    )

    worlds_dir = OmegaConf.select(cfg, "simulator.worlds_dir", default="./worlds")
    world_params = (
        OmegaConf.to_container(cfg.simulator.world_params, resolve=True)
        if OmegaConf.select(cfg, "simulator.world_params") else None
    )

    episodes_dir = Path(cfg.project.output_dir) / "episodes"
    episode_logs = []

    for i, user_request in enumerate(user_requests):
        sim = SimulatorAdapter(seed=cfg.project.seed + i, worlds_dir=worlds_dir,
                               world_config=world_params)
        tracker = ToolCallTracker()
        event_bus = EventBus()
        registry = ToolRegistry.from_config(simulator=sim, tracker=tracker, event_bus=event_bus)
        agent = ReActAgent(
            llm_model_id=cfg.agent.llm_model_id,
            tool_registry=registry,
            compressor=compressor,
            context_builder=ContextBuilder(
                system_prompt=system_prompt,
                tool_registry=registry,
                llm_model_id=cfg.agent.llm_model_id,
            ),
            config=agent_config,
            mode=AgentMode(cfg.agent.mode),
            mcts_controller=mcts_controller,
        )
        episode_log = agent.run_episode(request=user_request, simulator=sim)
        episode_logs.append(episode_log)
        save_episode(episode_log, episodes_dir)
        log.info("episode.complete", i=i + 1, total=len(user_requests),
                 success=episode_log.success, steps=episode_log.total_steps)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    from optimized_llm_planning_memory.evaluation.evaluator import Evaluator
    from optimized_llm_planning_memory.evaluation.deterministic import DeterministicEvaluator
    from optimized_llm_planning_memory.evaluation.llm_judge import LLMJudge
    from optimized_llm_planning_memory.core.config import EvalConfig

    eval_cfg = EvalConfig(**OmegaConf.to_container(cfg.eval, resolve=True))
    det_eval = DeterministicEvaluator()
    judge = None if eval_cfg.deterministic_only else LLMJudge(
        judge_model_id=eval_cfg.judge_model_id,
        rubric_dimensions=eval_cfg.rubric_dimensions,
    )

    evaluator = Evaluator(deterministic_eval=det_eval, llm_judge=judge, config=eval_cfg)
    results = evaluator.evaluate_dataset(episode_logs, user_requests)
    agg = evaluator.aggregate(results)

    # ── Save + print ──────────────────────────────────────────────────────────
    output_dir = Path(cfg.project.output_dir) / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{cfg.project.run_name}.json"
    out_path.write_text(
        json.dumps([r.model_dump() for r in results], indent=2),
        encoding="utf-8",
    )

    print("\n=== Aggregated Evaluation Results ===")
    for key, val in sorted(agg.items()):
        print(f"  {key:<45} {val:.4f}")

    log.info("evaluation.complete", n_episodes=len(results), output=str(out_path))


if __name__ == "__main__":
    main()
