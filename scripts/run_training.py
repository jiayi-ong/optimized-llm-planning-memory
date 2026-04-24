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

import hydra
from omegaconf import DictConfig

from optimized_llm_planning_memory.utils.logging import configure_logging, get_logger
from optimized_llm_planning_memory.utils.seed import set_seed
from optimized_llm_planning_memory.utils.episode_io import list_episodes


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    configure_logging(level=cfg.logging.level)
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
    from omegaconf import OmegaConf

    compressor_type = cfg.compressor.type
    reward_predictor = None

    if compressor_type == "identity":
        from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
        use_reward_predictor = OmegaConf.select(cfg, "compressor.use_reward_predictor", default=False)
        if use_reward_predictor:
            from optimized_llm_planning_memory.compressor.reward_predictor import RewardPredictorComponent
            reward_predictor = RewardPredictorComponent()
        compressor = IdentityCompressor(reward_predictor=reward_predictor)
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
    from optimized_llm_planning_memory.core.config import AgentConfig

    agent_config = AgentConfig(
        mode=cfg.agent.mode,
        llm_model_id=cfg.agent.llm_model_id,
        max_steps=cfg.agent.max_steps,
        compress_every_n_steps=cfg.agent.compress_every_n_steps,
    )

    worlds_dir = OmegaConf.select(cfg, "simulator.worlds_dir", default="./worlds")

    def simulator_factory(seed: int) -> SimulatorAdapter:
        return SimulatorAdapter(seed=seed, worlds_dir=worlds_dir)

    def agent_factory() -> ReActAgent:
        sim = SimulatorAdapter(seed=0)  # placeholder; env overrides with fresh sim
        tracker = ToolCallTracker()
        event_bus = EventBus()
        registry = ToolRegistry.from_config(simulator=sim, tracker=tracker, event_bus=event_bus)
        return ReActAgent(
            llm_model_id=cfg.agent.llm_model_id,
            tool_registry=registry,
            compressor=compressor,
            context_builder=ContextBuilder(),
            config=agent_config,
            mode=AgentMode(cfg.agent.mode),
        )

    # ── Build trainer + run ───────────────────────────────────────────────────
    from optimized_llm_planning_memory.training.trainer import RLTrainer
    from optimized_llm_planning_memory.core.config import TrainingConfig, EnvConfig, RewardConfig
    from omegaconf import OmegaConf

    training_cfg = TrainingConfig(**OmegaConf.to_container(cfg.training, resolve=True))
    env_cfg = EnvConfig()
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
