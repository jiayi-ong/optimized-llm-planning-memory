"""
core/config.py
==============
Project-wide configuration schema using pydantic-settings.

Design notes
------------
* All config is defined here as Pydantic models. Hydra (via OmegaConf) populates
  the YAML values; pydantic-settings handles environment variable overrides and
  type coercion.

* The ``ProjectConfig`` root model composes all sub-configs. Entry-point scripts
  call ``load_config()`` which instantiates this model from the Hydra config dict.

* Each sub-config corresponds to a YAML file under ``configs/``. The one-to-one
  mapping makes it easy to find which file controls which parameter.

Usage (in scripts/run_training.py)
-----------------------------------
    import hydra
    from optimized_llm_planning_memory.core.config import ProjectConfig

    @hydra.main(config_path="../configs", config_name="config", version_base="1.3")
    def main(cfg: DictConfig) -> None:
        config = ProjectConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
        trainer = RLTrainer(config=config)
        trainer.train()
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Simulator ─────────────────────────────────────────────────────────────────

class SimulatorConfig(BaseModel):
    seed_range: tuple[int, int] = Field(
        default=(0, 9999),
        description="Inclusive range [min, max] from which episode seeds are drawn.",
    )
    world_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs forwarded to the simulator constructor.",
    )


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentConfig(BaseModel):
    mode: str = Field(
        default="compressor",
        description="'raw' | 'llm_summary' | 'compressor'",
    )
    llm_model_id: str = Field(
        default="openai/gpt-4o-mini",
        description="litellm model string. Swap provider by changing this value only.",
    )
    max_steps: int = Field(default=30, ge=1)
    max_retries_per_action: int = Field(
        default=3, ge=1,
        description="How many times the agent may retry a failed tool call.",
    )
    compress_every_n_steps: int = Field(
        default=5, ge=1,
        description="Trigger a compression event after every N ReAct steps.",
    )
    compress_on_token_threshold: int = Field(
        default=3000, ge=100,
        description="Also trigger compression when trajectory token count exceeds this.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens_per_response: int = Field(default=1024, ge=64)
    system_prompt_version: str = Field(default="v1")
    few_shot_examples_path: str = Field(
        default="data/few_shot_examples/react_tool_use.json",
    )


# ── Compressor ────────────────────────────────────────────────────────────────

class LoRAConfig(BaseModel):
    r: int = Field(default=8, ge=1, description="LoRA rank.")
    alpha: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    target_modules: list[str] = Field(default_factory=lambda: ["q", "v"])


class GenerationConfig(BaseModel):
    num_beams: int = Field(default=4, ge=1)
    early_stopping: bool = True
    no_repeat_ngram_size: int = Field(default=3, ge=0)
    max_new_tokens: int = Field(default=512, ge=16)


class CompressorConfig(BaseModel):
    type: str = Field(
        default="transformer",
        description="'llm' | 'transformer' | 'hybrid'",
    )
    model_name_or_path: str = Field(
        default="google/flan-t5-small",
        description="HuggingFace model ID or local path. Only used when type='transformer'.",
    )
    llm_model_id: str = Field(
        default="openai/gpt-4o-mini",
        description="litellm model string. Only used when type='llm' or 'hybrid'.",
    )
    max_input_tokens: int = Field(default=2048, ge=64)
    max_output_tokens: int = Field(default=512, ge=32)
    device: str = Field(default="auto", description="'cpu' | 'cuda' | 'auto'")
    use_lora: bool = False
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    freeze_base_layers: bool = False
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


# ── Training ──────────────────────────────────────────────────────────────────

class PPOHyperparams(BaseModel):
    learning_rate: float = Field(default=3e-5, gt=0.0)
    n_steps: int = Field(default=2048, ge=1,
                         description="Steps per env per rollout collection (SB3 n_steps).")
    clip_epsilon: float = Field(default=0.2, gt=0.0, le=1.0)
    n_epochs: int = Field(default=4, ge=1)
    batch_size: int = Field(default=64, ge=1)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    ent_coef: float = Field(default=0.01, ge=0.0)
    vf_coef: float = Field(default=0.5, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)


class EnvConfig(BaseModel):
    max_obs_tokens: int = Field(
        default=2048,
        description="Max trajectory token IDs in observation vector (padded/truncated).",
    )
    max_action_tokens: int = Field(
        default=512,
        description="Max compressed state token IDs in action vector.",
    )
    steps_per_compression: int = Field(
        default=5,
        description="Number of ReAct steps the agent runs between compression events.",
    )


class TrainingConfig(BaseModel):
    num_timesteps: int = Field(default=500_000, ge=1)
    n_envs: int = Field(default=4, ge=1, description="Parallel gymnasium environments.")
    ppo: PPOHyperparams = Field(default_factory=PPOHyperparams)
    env: EnvConfig = Field(default_factory=EnvConfig)
    checkpoint_every_n_steps: int = Field(default=10_000, ge=1)
    resume_from: str | None = Field(
        default=None,
        description="Path to a checkpoint .zip file to resume training from.",
    )


# ── Reward ────────────────────────────────────────────────────────────────────

class RewardWeights(BaseModel):
    hard_constraint: float = Field(default=2.0)
    soft_constraint: float = Field(default=1.0)
    tool_efficiency: float = Field(default=0.3)
    tool_failure_penalty: float = Field(default=-0.5, le=0.0)
    logical_consistency: float = Field(default=0.5)
    terminal_itinerary: float = Field(default=3.0)


class RewardConfig(BaseModel):
    weights: RewardWeights = Field(default_factory=RewardWeights)
    terminal_bonus: float = Field(
        default=5.0,
        description="Extra bonus if ALL hard constraints are satisfied at episode end.",
    )
    step_penalty: float = Field(
        default=-0.01, le=0.0,
        description="Small per-step penalty to encourage efficiency.",
    )
    normalize: bool = Field(
        default=True,
        description="Normalize total reward to [-1, 1] range.",
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

class LLMJudgeConfig(BaseModel):
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0.0)


class EvalConfig(BaseModel):
    judge_model_id: str = Field(
        default="openai/gpt-4o",
        description="Fixed across all evaluation runs for fair comparison.",
    )
    rubric_path: str = Field(default="data/rubrics/itinerary_rubric_v1.md")
    rubric_dimensions: list[str] = Field(
        default_factory=lambda: [
            "constraint_adherence",
            "logical_coherence",
            "activity_diversity",
            "budget_efficiency",
            "feasibility",
            "creativity",
        ]
    )
    deterministic_only: bool = Field(
        default=False,
        description="Set True to skip the LLM judge (faster CI runs).",
    )
    output_format: str = Field(default="json", description="'json' | 'csv'")
    ablation_components: list[str] = Field(
        default_factory=lambda: ["compressor_type", "agent_mode", "reward_weights"],
    )


# ── Logging ───────────────────────────────────────────────────────────────────

class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = Field(default="optllm-planning")
    wandb_entity: str | None = None
    log_every_n_steps: int = Field(default=10, ge=1)


# ── Root ──────────────────────────────────────────────────────────────────────

class ProjectConfig(BaseModel):
    """
    Root configuration model. Composed from all sub-configs.

    Hydra populates this via OmegaConf at runtime::

        @hydra.main(config_path="../configs", config_name="config", version_base="1.3")
        def main(cfg: DictConfig) -> None:
            config = ProjectConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    """
    name: str = "optimized-llm-planning-memory"
    version: str = "0.1.0"
    seed: int = Field(default=42)
    output_dir: str = Field(default="outputs/")
    run_name: str | None = None

    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    compressor: CompressorConfig = Field(default_factory=CompressorConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
