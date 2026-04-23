"""
training/trainer.py
====================
RLTrainer — wires SB3 PPO + custom CompressorPolicy + CompressionEnv.

Responsibilities
----------------
1. Build ``CompressionEnv`` (vectorised via ``make_vec_env``).
2. Instantiate SB3 ``PPO`` with ``CompressorPolicy`` as the policy class.
3. Attach TensorBoard + custom callbacks for episode logging and checkpointing.
4. Expose ``train()``, ``save_checkpoint()``, ``load_checkpoint()``.

Why a thin wrapper instead of calling SB3 directly?
----------------------------------------------------
- Encapsulates the tedious ``policy_kwargs`` plumbing (passing the compressor
  and value_hidden_dim into CompressorPolicy.__init__).
- Provides a single place for project-specific callbacks (episode-level reward
  logging, compressed-state quality metrics).
- Makes checkpoint/resume logic reusable from ``scripts/run_training.py``.

SB3 version note
----------------
This code targets ``stable-baselines3 >= 2.3``. The ``PPO`` constructor
accepts ``policy_kwargs`` which are forwarded to the policy ``__init__``.
Custom policies must be passed as a class (not an instance).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from optimized_llm_planning_memory.compressor.trainable_base import TrainableCompressorBase
from optimized_llm_planning_memory.core.config import EnvConfig, PPOHyperparams, RewardConfig, TrainingConfig
from optimized_llm_planning_memory.core.models import UserRequest
from optimized_llm_planning_memory.simulator.protocol import SimulatorProtocol
from optimized_llm_planning_memory.training.env import CompressionEnv
from optimized_llm_planning_memory.training.policy import CompressorPolicy
from optimized_llm_planning_memory.training.reward import RewardFunction


# ── Custom callbacks ───────────────────────────────────────────────────────────


class EpisodeLogCallback(BaseCallback):
    """
    SB3 callback that extracts per-episode reward components from the ``info``
    dict returned by ``CompressionEnv.step()`` and writes them to TensorBoard.

    SB3 calls ``on_step()`` after every environment step. When ``terminated``
    is True, the ``info`` dict contains an ``EpisodeLog`` snapshot with full
    ``RewardComponents``. We extract and log these here.
    """

    def __init__(self, tb_log_prefix: str = "episode", verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._prefix = tb_log_prefix
        self._episode_count = 0

    def _on_step(self) -> bool:
        """Called after each env step. Returns True to continue training."""
        infos = self.locals.get("infos", [])
        for info in infos:
            if "reward_components" not in info:
                continue
            rc = info["reward_components"]
            ep = self._episode_count
            self.logger.record(f"{self._prefix}/hard_constraint_score", rc.hard_constraint_score)
            self.logger.record(f"{self._prefix}/soft_constraint_score", rc.soft_constraint_score)
            self.logger.record(f"{self._prefix}/tool_efficiency_score", rc.tool_efficiency_score)
            self.logger.record(f"{self._prefix}/tool_failure_penalty", rc.tool_failure_penalty)
            self.logger.record(f"{self._prefix}/logical_consistency_score", rc.logical_consistency_score)
            self.logger.record(f"{self._prefix}/total_reward", rc.total_reward)
            if rc.terminal_itinerary_score is not None:
                self.logger.record(
                    f"{self._prefix}/terminal_itinerary_score",
                    rc.terminal_itinerary_score,
                )
            self._episode_count += 1
        return True


class CompressorCheckpointCallback(BaseCallback):
    """
    Checkpoints the compressor model (HF save_pretrained format) alongside
    the SB3 PPO zip checkpoint.

    SB3's built-in ``CheckpointCallback`` saves the full policy; this callback
    additionally saves just the compressor weights via
    ``TrainableCompressorBase.save_checkpoint()``, which is lighter-weight
    and loadable without SB3.
    """

    def __init__(
        self,
        compressor: TrainableCompressorBase,
        save_dir: str | Path,
        save_every_n_steps: int = 10_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._compressor = compressor
        self._save_dir = Path(save_dir)
        self._save_every = save_every_n_steps
        self._last_saved = 0

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step - self._last_saved >= self._save_every:
            ckpt_path = self._save_dir / f"compressor_step_{step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            self._compressor.save_checkpoint(str(ckpt_path))
            if self.verbose >= 1:
                print(f"[CompressorCheckpointCallback] Saved compressor to {ckpt_path}")
            self._last_saved = step
        return True


# ── RLTrainer ─────────────────────────────────────────────────────────────────


class SparkWeightCallback(BaseCallback):
    """
    SB3 callback that feeds completed-episode data into a SparkWeightComponent.

    After each step, it scans the ``infos`` list for terminated episodes.
    For each terminated episode, it extracts five scalar features from the
    ``reward_components`` dict and calls ``spark_component.add_episode()``.
    Every ``fit_every_n_episodes`` episodes it triggers ``spark_component.fit()``.

    Parameters
    ----------
    spark_component      : SparkWeightComponent to update.
    fit_every_n_episodes : How often (in completed episodes) to call fit().
    verbose              : SB3 verbosity level.
    """

    def __init__(
        self,
        spark_component: Any,
        fit_every_n_episodes: int = 50,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._spark = spark_component
        self._fit_every = fit_every_n_episodes
        self._episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if not done:
                continue
            rc = info.get("reward_components")
            if rc is None:
                continue

            # Build feature dict from RewardComponents fields
            features = {
                "hard_constraint_score": float(getattr(rc, "hard_constraint_score", 0.0)),
                "soft_constraint_score": float(getattr(rc, "soft_constraint_score", 0.0)),
                "tool_efficiency_score": float(getattr(rc, "tool_efficiency_score", 0.0)),
                "steps_per_episode": float(info.get("steps_per_episode_normalized", 0.0)),
                "budget_adherence": float(info.get("budget_adherence", 0.0)),
            }
            reward = float(getattr(rc, "total_reward", 0.0))
            self._spark.add_episode(features=features, reward=reward)
            self._episode_count += 1

            if self._episode_count % self._fit_every == 0:
                fitted = self._spark.fit()
                if self.verbose >= 1:
                    status = "OK" if fitted else "skipped (insufficient data)"
                    print(
                        f"[SparkWeightCallback] fit() at episode "
                        f"{self._episode_count}: {status}"
                    )

        return True


class RLTrainer:
    """
    Wires SB3 PPO with the custom ``CompressorPolicy`` and ``CompressionEnv``.

    Parameters
    ----------
    compressor        : Trainable compressor model (HF or custom).
    agent_factory     : Callable[[], ReActAgent] — creates a fresh agent per episode.
    simulator_factory : Callable[[int], SimulatorProtocol] — creates a fresh sim per episode.
    user_requests     : Training set of UserRequest objects.
    config            : Full training configuration (n_envs, PPO hyperparams, paths).
    env_config        : Environment configuration (max token dims).
    reward_config     : Reward component weights.
    tokenizer         : HF tokenizer (or None to use char-level fallback in CompressionEnv).
    tensorboard_log   : Directory for TensorBoard logs. Defaults to ``outputs/logs``.
    checkpoint_dir    : Directory for SB3 + compressor checkpoints. Defaults to ``outputs/checkpoints``.
    spark_component   : Optional SparkWeightComponent. When provided, a
                        SparkWeightCallback is added to the callback list so that
                        PySpark MLlib trains on episode rewards during PPO.
    spark_fit_every   : How often (in episodes) the SparkWeightCallback triggers fit().
    """

    def __init__(
        self,
        compressor: TrainableCompressorBase,
        agent_factory: Callable,
        simulator_factory: Callable[[int], SimulatorProtocol],
        user_requests: list[UserRequest],
        config: TrainingConfig | None = None,
        env_config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
        tokenizer: Any = None,
        tensorboard_log: str | Path = "outputs/logs",
        checkpoint_dir: str | Path = "outputs/checkpoints",
        spark_component: Any = None,
        spark_fit_every: int = 50,
    ) -> None:
        self._compressor = compressor
        self._agent_factory = agent_factory
        self._simulator_factory = simulator_factory
        self._user_requests = user_requests
        self._config = config or TrainingConfig()
        self._env_config = env_config or EnvConfig()
        self._reward_config = reward_config or RewardConfig()
        self._tokenizer = tokenizer
        self._tensorboard_log = Path(tensorboard_log)
        self._checkpoint_dir = Path(checkpoint_dir)
        self._spark_component = spark_component
        self._spark_fit_every = spark_fit_every

        self._ppo: PPO | None = None

    # ── Public interface ──────────────────────────────────────────────────────

    def train(self, num_timesteps: int | None = None) -> None:
        """
        Run PPO training for ``num_timesteps`` environment steps.

        If ``config.resume_from`` is set, the SB3 model is loaded from that
        checkpoint before continuing. Compressor weights are loaded separately
        via ``load_checkpoint()``.

        Parameters
        ----------
        num_timesteps : Override for the number of training steps.
                        Defaults to ``config.num_timesteps``.
        """
        n_steps = num_timesteps or self._config.num_timesteps

        # Build vectorised env
        vec_env = self._make_vec_env()

        # Build or restore PPO model
        if self._config.resume_from:
            self._ppo = self._load_ppo(self._config.resume_from, vec_env)
        else:
            self._ppo = self._build_ppo(vec_env)

        # Build callbacks
        callbacks = self._build_callbacks()

        # Run training
        self._ppo.learn(
            total_timesteps=n_steps,
            callback=callbacks,
            reset_num_timesteps=not bool(self._config.resume_from),
        )

    def save_checkpoint(self, path: str | Path | None = None) -> None:
        """
        Save SB3 PPO model + compressor weights.

        Parameters
        ----------
        path : Directory to save into. Defaults to ``checkpoint_dir/final``.
        """
        if self._ppo is None:
            raise RuntimeError("No trained model to save. Call train() first.")

        save_dir = Path(path) if path else self._checkpoint_dir / "final"
        save_dir.mkdir(parents=True, exist_ok=True)

        # SB3 saves policy + optimizer state into a single zip
        ppo_path = save_dir / "ppo_model"
        self._ppo.save(str(ppo_path))

        # Compressor weights (HF format or custom)
        compressor_path = save_dir / "compressor"
        compressor_path.mkdir(parents=True, exist_ok=True)
        self._compressor.save_checkpoint(str(compressor_path))

    def load_checkpoint(self, path: str | Path) -> None:
        """
        Load compressor weights from a checkpoint directory.

        Note: To resume SB3 PPO training state (optimizer, step count), call
        ``train()`` with ``config.resume_from`` set. This method only restores
        the compressor model weights.

        Parameters
        ----------
        path : Directory containing the ``compressor/`` sub-directory.
        """
        compressor_path = Path(path) / "compressor"
        if not compressor_path.exists():
            # Allow passing the compressor directory directly
            compressor_path = Path(path)
        self._compressor.load_checkpoint(str(compressor_path))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _make_vec_env(self) -> VecEnv:
        """
        Create a vectorised ``CompressionEnv`` using SB3's ``make_vec_env``.

        Each parallel environment gets a fresh ``SimulatorAdapter`` and
        ``ReActAgent`` on every ``reset()``, so there is no shared state.
        """
        reward_fn = RewardFunction(config=self._reward_config)
        env_config = self._env_config
        tokenizer = self._tokenizer
        agent_factory = self._agent_factory
        simulator_factory = self._simulator_factory
        user_requests = self._user_requests

        def env_factory() -> CompressionEnv:
            return CompressionEnv(
                agent_factory=agent_factory,
                simulator_factory=simulator_factory,
                reward_fn=reward_fn,
                user_requests=user_requests,
                config=env_config,
                tokenizer=tokenizer,
            )

        vec_env = make_vec_env(
            env_factory,
            n_envs=self._config.n_envs,
            seed=self._config.seed if hasattr(self._config, "seed") else None,
        )
        return vec_env

    def _build_ppo(self, vec_env: VecEnv) -> PPO:
        """
        Instantiate SB3 PPO with ``CompressorPolicy``.

        ``policy_kwargs`` are forwarded to ``CompressorPolicy.__init__()``
        as additional keyword arguments beyond what SB3 injects automatically
        (observation_space, action_space, lr_schedule).
        """
        hp: PPOHyperparams = self._config.ppo

        policy_kwargs: dict[str, Any] = {
            "compressor": self._compressor,
            "value_hidden_dim": 256,
        }

        ppo = PPO(
            policy=CompressorPolicy,
            env=vec_env,
            learning_rate=hp.learning_rate,
            n_steps=hp.n_steps,
            batch_size=hp.batch_size,
            n_epochs=hp.n_epochs,
            gamma=hp.gamma,
            gae_lambda=hp.gae_lambda,
            clip_range=hp.clip_epsilon,
            ent_coef=hp.ent_coef,
            vf_coef=hp.vf_coef,
            max_grad_norm=hp.max_grad_norm,
            tensorboard_log=str(self._tensorboard_log),
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
        return ppo

    def _load_ppo(self, checkpoint_path: str | Path, vec_env: VecEnv) -> PPO:
        """
        Load SB3 PPO from a checkpoint zip and set the new vec_env.

        The compressor weights inside the policy are NOT automatically restored
        by SB3 — ``load_checkpoint()`` must be called separately.
        """
        ppo_zip = Path(checkpoint_path) / "ppo_model.zip"
        if not ppo_zip.exists():
            ppo_zip = Path(checkpoint_path)  # allow passing the zip directly

        policy_kwargs: dict[str, Any] = {
            "compressor": self._compressor,
            "value_hidden_dim": 256,
        }

        ppo = PPO.load(
            str(ppo_zip),
            env=vec_env,
            policy_kwargs=policy_kwargs,
        )
        return ppo

    def _build_callbacks(self) -> CallbackList:
        """
        Build the list of SB3 callbacks used during ``ppo.learn()``.

        Includes:
        - SB3 built-in ``CheckpointCallback`` — saves full PPO model as zip.
        - ``CompressorCheckpointCallback`` — saves only the compressor weights.
        - ``EpisodeLogCallback`` — writes reward components to TensorBoard.
        """
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._tensorboard_log.mkdir(parents=True, exist_ok=True)

        checkpoint_every = getattr(self._config, "checkpoint_every_n_steps", 10_000)

        sb3_ckpt = CheckpointCallback(
            save_freq=checkpoint_every,
            save_path=str(self._checkpoint_dir),
            name_prefix="ppo_compressor",
            verbose=1,
        )

        compressor_ckpt = CompressorCheckpointCallback(
            compressor=self._compressor,
            save_dir=self._checkpoint_dir,
            save_every_n_steps=checkpoint_every,
            verbose=1,
        )

        episode_log = EpisodeLogCallback(verbose=0)

        callbacks: list[BaseCallback] = [sb3_ckpt, compressor_ckpt, episode_log]

        if self._spark_component is not None:
            spark_cb = SparkWeightCallback(
                spark_component=self._spark_component,
                fit_every_n_episodes=self._spark_fit_every,
                verbose=1,
            )
            callbacks.append(spark_cb)

        return CallbackList(callbacks)
