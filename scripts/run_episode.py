"""
scripts/run_episode.py
=======================
Debug script: run a single planning episode and print the EpisodeLog.

Usage
-----
    python scripts/run_episode.py
    python scripts/run_episode.py agent.mode=raw
    python scripts/run_episode.py agent.mode=llm_summary compressor=llm_prompt
    python scripts/run_episode.py agent.mode=compressor compressor=transformer

Outputs
-------
- Prints a human-readable episode summary to stdout (via visualization.py).
- Saves the full EpisodeLog JSON to ``outputs/episodes/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from omegaconf import DictConfig

from optimized_llm_planning_memory.utils.logging import configure_logging, get_logger
from optimized_llm_planning_memory.utils.seed import set_seed
from optimized_llm_planning_memory.utils.visualization import print_episode
from optimized_llm_planning_memory.utils.episode_io import save_episode


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    configure_logging(level=cfg.logging.level)
    log = get_logger(__name__)
    set_seed(cfg.project.seed)

    log.info("run_episode.start", mode=cfg.agent.mode, seed=cfg.project.seed)

    # ── Build components ──────────────────────────────────────────────────────
    from optimized_llm_planning_memory.simulator.adapter import SimulatorAdapter
    from optimized_llm_planning_memory.tools.registry import ToolRegistry
    from optimized_llm_planning_memory.tools.tracker import ToolCallTracker
    from optimized_llm_planning_memory.tools.events import EventBus
    from optimized_llm_planning_memory.agent.react_agent import ReActAgent
    from optimized_llm_planning_memory.agent.modes import AgentMode
    from optimized_llm_planning_memory.agent.context_builder import ContextBuilder
    from optimized_llm_planning_memory.agent.prompts import get_system_prompt
    from optimized_llm_planning_memory.core.config import AgentConfig
    from omegaconf import OmegaConf

    worlds_dir = OmegaConf.select(cfg, "simulator.worlds_dir", default="./worlds")
    simulator = SimulatorAdapter(seed=cfg.project.seed, worlds_dir=worlds_dir)
    tracker = ToolCallTracker()
    event_bus = EventBus()

    tool_registry = ToolRegistry.from_config(
        simulator=simulator,
        tracker=tracker,
        event_bus=event_bus,
    )

    compressor_type = cfg.compressor.type
    if compressor_type == "identity":
        from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
        use_reward_predictor = OmegaConf.select(cfg, "compressor.use_reward_predictor", default=False)
        reward_predictor = None
        if use_reward_predictor:
            from optimized_llm_planning_memory.compressor.reward_predictor import RewardPredictorComponent
            reward_predictor = RewardPredictorComponent()
        compressor = IdentityCompressor(reward_predictor=reward_predictor)
    elif compressor_type == "llm":
        from optimized_llm_planning_memory.compressor.llm_compressor import LLMCompressor
        compressor = LLMCompressor(model_id=cfg.compressor.model_name_or_path)
    elif compressor_type == "transformer":
        from optimized_llm_planning_memory.compressor.transformer_compressor import TransformerCompressor
        compressor = TransformerCompressor(
            model_name_or_path=cfg.compressor.model_name_or_path,
            device=cfg.compressor.device or "auto",
        )
    elif compressor_type == "dummy":
        from optimized_llm_planning_memory.compressor.dummy_compressor import DummyCompressor
        compressor = DummyCompressor()
    else:
        from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
        compressor = IdentityCompressor()

    agent_config = AgentConfig(
        mode=cfg.agent.mode,
        llm_model_id=cfg.agent.llm_model_id,
        max_steps=cfg.agent.max_steps,
        compress_every_n_steps=cfg.agent.compress_every_n_steps,
    )

    system_prompt = get_system_prompt(cfg.agent.get("system_prompt_version", "v1"))
    context_builder = ContextBuilder(
        system_prompt=system_prompt,
        tool_registry=tool_registry,
        llm_model_id=cfg.agent.llm_model_id,
    )
    agent = ReActAgent(
        llm_model_id=cfg.agent.llm_model_id,
        tool_registry=tool_registry,
        compressor=compressor,
        context_builder=context_builder,
        config=agent_config,
        mode=AgentMode(cfg.agent.mode),
    )

    # ── Load a user request ───────────────────────────────────────────────────
    import json
    from optimized_llm_planning_memory.core.models import UserRequest

    # Prefer real generated requests over the placeholder template
    train_dir = Path("data/user_requests/train")
    train_files = sorted(train_dir.glob("*.json")) if train_dir.exists() else []
    if train_files:
        user_request = UserRequest.model_validate(json.loads(train_files[0].read_text()))
    else:
        template_path = Path("data/user_requests/templates/request_template.json")
        user_request = UserRequest.model_validate(json.loads(template_path.read_text()))

    # ── Run episode ───────────────────────────────────────────────────────────
    import uuid as _uuid
    from optimized_llm_planning_memory.utils.live_writer import LiveEpisodeWriter

    episodes_dir = Path(cfg.project.output_dir) / "episodes"

    # Pre-generate episode_id so LiveEpisodeWriter and run_episode() share the
    # same ID — the live JSONL file is named <episode_id>.jsonl.
    episode_id = str(_uuid.uuid4())
    log.info("episode.start", request_id=user_request.request_id, episode_id=episode_id)

    # LiveEpisodeWriter streams step events to outputs/episodes/live/<id>.jsonl
    # so the Streamlit UI can show live progress while this script runs.
    with LiveEpisodeWriter(episode_id=episode_id, output_dir=episodes_dir) as live_writer:
        episode_log = agent.run_episode(
            request=user_request,
            simulator=simulator,
            live_writer=live_writer,
            episode_id=episode_id,
        )

    # ── Print + save ──────────────────────────────────────────────────────────
    print_episode(episode_log)

    out_path = save_episode(episode_log, episodes_dir)
    log.info("episode.saved", path=str(out_path))


if __name__ == "__main__":
    main()
