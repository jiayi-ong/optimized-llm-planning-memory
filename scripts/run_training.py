"""
scripts/run_training.py
========================
Launch PPO training of the context compressor.

Usage
-----
    python scripts/run_training.py
    python scripts/run_training.py training=ppo_colab
    python scripts/run_training.py training.num_timesteps=100000 compressor=transformer
    python scripts/run_training.py training.resume_from=outputs/checkpoints/ppo_compressor_100000_steps.zip
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

import hydra
from omegaconf import DictConfig, OmegaConf

from optimized_llm_planning_memory.utils.logging import configure_logging, get_logger
from optimized_llm_planning_memory.utils.seed import set_seed
from optimized_llm_planning_memory.utils.episode_io import list_episodes


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    import datetime as _dt
    _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file = OmegaConf.select(cfg, "logging.log_file") or str(
        Path(cfg.project.output_dir) / "logs" / f"run_training_{_ts}.log"
    )
    configure_logging(level=cfg.logging.level, log_file=_log_file)
    log = get_logger(__name__)
    set_seed(cfg.project.seed)

    log.info("training.start",
             compressor=cfg.compressor.type,
             n_envs=cfg.training.n_envs,
             num_timesteps=cfg.training.num_timesteps)

    # ── Load user requests ────────────────────────────────────────────────────
    import json
    from optimized_llm_planning_memory.core.models import UserRequest

    train_dir = Path("data/user_requests/train")
    if not train_dir.exists() or not list(train_dir.glob("*.json")):
        log.warning("no_training_requests", path=str(train_dir),
                    hint="Run scripts/generate_user_requests.py first, "
                         "or place request JSON files in data/user_requests/train/")
        # Fallback: use template request
        template = Path("data/user_requests/templates/request_template.json")
        user_requests = [UserRequest.model_validate(json.loads(template.read_text()))]
    else:
        user_requests = [
            UserRequest.model_validate(json.loads(f.read_text()))
            for f in sorted(train_dir.glob("*.json"))
        ]
    log.info("loaded_requests", n=len(user_requests))

    # ── Build compressor ──────────────────────────────────────────────────────
    compressor_type = cfg.compressor.type
    reward_predictor = None

    if compressor_type == "identity":
        from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
        use_reward_predictor = OmegaConf.select(cfg, "compressor.use_reward_predictor", default=False)
        if use_reward_predictor:
            from optimized_llm_planning_memory.compressor.reward_predictor import RewardPredictorComponent
            reward_predictor = RewardPredictorComponent()
        compressor = IdentityCompressor(reward_predictor=reward_predictor)
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
            device=cfg.compressor.device or "auto",
        )
        if OmegaConf.select(cfg, "compressor.use_lora", default=False):
            from optimized_llm_planning_memory.core.config import LoRAConfig
            lora_cfg = LoRAConfig(**cfg.compressor.lora)
            compressor.apply_lora(lora_cfg)
        if OmegaConf.select(cfg, "compressor.freeze_base_layers", default=False):
            compressor.freeze_base_layers(freeze=True)
    elif compressor_type == "structured_selective":
        from optimized_llm_planning_memory.compressor.structured_selective_distiller import (
            StructuredSelectiveDistiller,
        )
        from optimized_llm_planning_memory.core.config import LoRAConfig
        lora_cfg = LoRAConfig(**OmegaConf.to_container(cfg.compressor.lora, resolve=True))
        compressor = StructuredSelectiveDistiller(
            model_name_or_path=cfg.compressor.model_name_or_path,
            max_step_tokens=OmegaConf.select(cfg, "compressor.max_step_tokens", default=128),
            max_output_tokens=cfg.compressor.max_output_tokens,
            device=cfg.compressor.device or "auto",
            use_lora=OmegaConf.select(cfg, "compressor.use_lora", default=True),
            lora_config=lora_cfg,
        )
    elif compressor_type == "mcts_gat":
        from optimized_llm_planning_memory.compressor.mcts_gat_distiller import (
            MCTSGraphAttentionDistiller,
        )
        from optimized_llm_planning_memory.core.config import LoRAConfig
        lora_cfg = LoRAConfig(**OmegaConf.to_container(cfg.compressor.lora, resolve=True))
        compressor = MCTSGraphAttentionDistiller(
            model_name_or_path=cfg.compressor.model_name_or_path,
            max_path_tokens=OmegaConf.select(cfg, "compressor.max_input_tokens", default=256),
            max_output_tokens=cfg.compressor.max_output_tokens,
            device=cfg.compressor.device or "auto",
            use_lora=OmegaConf.select(cfg, "compressor.use_lora", default=True),
            lora_config=lora_cfg,
            top_k_paths=OmegaConf.select(cfg, "compressor.top_k_paths", default=3),
        )
    else:
        from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
        compressor = IdentityCompressor()

    # ── Build factories ───────────────────────────────────────────────────────
    from optimized_llm_planning_memory.simulator.adapter import SimulatorAdapter
    from optimized_llm_planning_memory.tools.registry import ToolRegistry
    from optimized_llm_planning_memory.tools.tracker import ToolCallTracker
    from optimized_llm_planning_memory.tools.events import EventBus
    from optimized_llm_planning_memory.agent.react_agent import ReActAgent
    from optimized_llm_planning_memory.agent.modes import AgentMode
    from optimized_llm_planning_memory.agent.context_builder import ContextBuilder
    from optimized_llm_planning_memory.agent.prompts import get_system_prompt
    from optimized_llm_planning_memory.core.config import AgentConfig

    agent_config = AgentConfig(
        mode=cfg.agent.mode,
        llm_model_id=cfg.agent.llm_model_id,
        max_steps=cfg.agent.max_steps,
        compress_every_n_steps=cfg.agent.compress_every_n_steps,
    )

    worlds_dir = OmegaConf.select(cfg, "simulator.worlds_dir", default="./worlds")
    world_params = OmegaConf.to_container(cfg.simulator.world_params, resolve=True) if OmegaConf.select(cfg, "simulator.world_params") else None

    def simulator_factory(seed: int) -> SimulatorAdapter:
        return SimulatorAdapter(seed=seed, worlds_dir=worlds_dir, world_config=world_params)

    def agent_factory() -> ReActAgent:
        sim = SimulatorAdapter(seed=0, world_config=world_params)  # placeholder; env overrides with fresh sim
        tracker = ToolCallTracker()
        event_bus = EventBus()
        registry = ToolRegistry.from_config(simulator=sim, tracker=tracker, event_bus=event_bus)
        system_prompt = get_system_prompt(
            OmegaConf.select(cfg, "agent.system_prompt_version", default="v1")
        )
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

        return ReActAgent(
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

    # ── Build trainer + run ───────────────────────────────────────────────────
    from optimized_llm_planning_memory.training.trainer import RLTrainer
    from optimized_llm_planning_memory.core.config import TrainingConfig, EnvConfig, RewardConfig

    training_cfg = TrainingConfig(**OmegaConf.to_container(cfg.training, resolve=True))
    env_cfg = training_cfg.env  # populated from training YAML's `env:` section
    reward_cfg = RewardConfig(**OmegaConf.to_container(cfg.reward, resolve=True))

    output_dir = Path(cfg.project.output_dir)
    rp_fit_every = int(OmegaConf.select(cfg, "compressor.reward_predictor_fit_every", default=50))
    trainer = RLTrainer(
        compressor=compressor,
        agent_factory=agent_factory,
        simulator_factory=simulator_factory,
        user_requests=user_requests,
        config=training_cfg,
        env_config=env_cfg,
        reward_config=reward_cfg,
        tensorboard_log=output_dir / "logs" / cfg.project.run_name,
        checkpoint_dir=output_dir / "checkpoints",
        reward_predictor=reward_predictor,
        reward_predictor_fit_every=rp_fit_every,
    )

    trainer.train()
    trainer.save_checkpoint(output_dir / "checkpoints" / "final")
    log.info("training.complete")


if __name__ == "__main__":
    main()
