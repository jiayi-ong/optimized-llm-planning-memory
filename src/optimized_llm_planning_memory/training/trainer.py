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
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from optimized_llm_planning_memory.compressor.trainable_base import TrainableCompressorBase
from optimized_llm_planning_memory.core.config import EnvConfig, PPOHyperparams, RewardConfig, SimulatorConfig, TrainingConfig
from optimized_llm_planning_memory.core.models import UserRequest
from optimized_llm_planning_memory.simulator.protocol import SimulatorProtocol
from optimized_llm_planning_memory.simulator.world_pool import WorldPool
from optimized_llm_planning_memory.training.env import CompressionEnv
from optimized_llm_planning_memory.training.policy import CompressorPolicy
from optimized_llm_planning_memory.training.reward import RewardFunction
from optimized_llm_planning_memory.training.run_logger import (
    EpisodeMetricsSummary,
    PPOUpdateMetrics,
    RLRunLogger,
)
from optimized_llm_planning_memory.training.run_manifest import (
    TrainingRunManifest,
    save_manifest,
)
from optimized_llm_planning_memory.utils.logging import get_logger

log = get_logger(__name__)


# ── Custom callbacks ───────────────────────────────────────────────────────────


class EpisodeLogCallback(BaseCallback):
    """
    SB3 callback that extracts per-episode reward components from the ``info``
    dict returned by ``CompressionEnv.step()`` and writes them to TensorBoard
    and (optionally) to the ``RLRunLogger`` JSONL store.

    Enhanced over the original to also log:
    - ``episode/total_steps`` — episode length (diagnoses over/under-planning)
    - ``episode/tool_calls_total`` — total tool invocations
    - ``episode/tool_success_rate`` — fraction of successful tool calls
    - ``episode/reward_mean_20`` — rolling mean of total reward (last 20 episodes)
    - ``episode/num_compressions`` — number of compression events in the episode

    Parameters
    ----------
    run_logger        : Optional ``RLRunLogger``.  When provided, each completed
                        episode is also written to ``episode_metrics.jsonl``.
    tb_log_prefix     : TensorBoard tag prefix.  Defaults to ``"episode"``.
    rolling_window    : Window size for the rolling reward mean.
    episode_save_freq : Save a full EpisodeLog JSON every N episodes.
                        0 = never (recommended for Colab to conserve storage).
    episodes_dir      : Directory for full EpisodeLog JSON files.
    verbose           : SB3 verbosity level.
    """

    def __init__(
        self,
        run_logger: "RLRunLogger | None" = None,
        tb_log_prefix: str = "episode",
        rolling_window: int = 20,
        episode_save_freq: int = 0,
        episodes_dir: str | Path = "outputs/episodes",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._prefix = tb_log_prefix
        self._episode_count = 0
        self._run_logger = run_logger
        self._reward_window: deque[float] = deque(maxlen=rolling_window)
        self._episode_save_freq = episode_save_freq
        self._episodes_dir = Path(episodes_dir)

    def _on_step(self) -> bool:
        """Called after each env step. Returns True to continue training."""
        infos = self.locals.get("infos", [])
        for info in infos:
            if "reward_components" not in info:
                continue
            rc = info["reward_components"]
            episode_log = info.get("episode_log")

            # ── Reward components ──────────────────────────────────────────────
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

            # ── Rolling reward mean ────────────────────────────────────────────
            self._reward_window.append(rc.total_reward)
            reward_mean = sum(self._reward_window) / len(self._reward_window)
            self.logger.record(f"{self._prefix}/reward_mean_20", reward_mean)

            # ── Episode-level metrics from EpisodeLog ─────────────────────────
            total_steps = 0
            tool_calls_total = 0
            tool_success_rate = 0.0
            num_compressions = 0
            request_id = info.get("request_id", "")
            agent_mode = ""

            if episode_log is not None:
                total_steps = getattr(episode_log, "total_steps", 0)
                num_compressions = len(getattr(episode_log, "compressed_states", ()))
                agent_mode = getattr(episode_log, "agent_mode", "")

                tool_stats = getattr(episode_log, "tool_stats", ())
                if tool_stats:
                    total_calls = sum(getattr(ts, "call_count", 0) for ts in tool_stats)
                    total_success = sum(getattr(ts, "success_count", 0) for ts in tool_stats)
                    tool_calls_total = total_calls
                    tool_success_rate = total_success / total_calls if total_calls > 0 else 0.0

                request_id = getattr(episode_log, "request_id", request_id)

            self.logger.record(f"{self._prefix}/total_steps", total_steps)
            self.logger.record(f"{self._prefix}/tool_calls_total", tool_calls_total)
            self.logger.record(f"{self._prefix}/tool_success_rate", tool_success_rate)
            self.logger.record(f"{self._prefix}/num_compressions", num_compressions)

            # ── Persist to JSONL ───────────────────────────────────────────────
            if self._run_logger is not None:
                summary = EpisodeMetricsSummary(
                    episode_id=getattr(episode_log, "episode_id", "") if episode_log else "",
                    request_id=request_id,
                    agent_mode=agent_mode,
                    total_steps=total_steps,
                    success=getattr(episode_log, "success", False) if episode_log else False,
                    total_reward=rc.total_reward,
                    hard_constraint_score=rc.hard_constraint_score,
                    soft_constraint_score=rc.soft_constraint_score,
                    tool_efficiency_score=rc.tool_efficiency_score,
                    tool_failure_penalty=rc.tool_failure_penalty,
                    logical_consistency_score=rc.logical_consistency_score,
                    terminal_itinerary_score=rc.terminal_itinerary_score,
                    tool_calls_total=tool_calls_total,
                    tool_success_rate=tool_success_rate,
                    num_compressions=num_compressions,
                    reward_mean_20=reward_mean,
                )
                self._run_logger.log_episode_summary(summary)

            # ── Optionally persist full EpisodeLog (storage-intensive) ─────────
            if (
                self._episode_save_freq > 0
                and episode_log is not None
                and self._episode_count % self._episode_save_freq == 0
            ):
                try:
                    from optimized_llm_planning_memory.utils.episode_io import save_episode
                    self._episodes_dir.mkdir(parents=True, exist_ok=True)
                    save_episode(episode_log, self._episodes_dir)
                except Exception:
                    pass  # never crash training over a logging failure

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


class RewardPredictorCallback(BaseCallback):
    """
    SB3 callback that feeds completed-episode data into a RewardPredictorComponent.

    After each step, it scans the ``infos`` list for terminated episodes.
    For each terminated episode, it extracts five scalar features from the
    ``reward_components`` dict and calls ``reward_predictor.add_episode()``.
    Every ``fit_every_n_episodes`` episodes it triggers ``reward_predictor.fit()``.

    Parameters
    ----------
    reward_predictor     : RewardPredictorComponent to update.
    fit_every_n_episodes : How often (in completed episodes) to call fit().
    verbose              : SB3 verbosity level.
    """

    def __init__(
        self,
        reward_predictor: Any,
        fit_every_n_episodes: int = 50,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._reward_predictor = reward_predictor
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
            self._reward_predictor.add_episode(features=features, reward=reward)
            self._episode_count += 1

            if self._episode_count % self._fit_every == 0:
                fitted = self._reward_predictor.fit()
                if self.verbose >= 1:
                    status = "OK" if fitted else "skipped (insufficient data)"
                    print(
                        f"[RewardPredictorCallback] fit() at episode "
                        f"{self._episode_count}: {status}"
                    )

        return True


class MCTSMetricsCallback(BaseCallback):
    """
    SB3 callback that logs MCTS-specific metrics to TensorBoard.

    Reads ``mcts_stats`` from the ``episode_log`` embedded in the ``info``
    dict returned by ``CompressionEnv.step()``. Silently no-ops for non-MCTS
    episodes where ``mcts_stats`` is None, so this callback is safe to attach
    unconditionally regardless of agent mode.

    TensorBoard keys written (prefixed by ``tb_log_prefix``):
      - ``mcts/nodes_explored``
      - ``mcts/max_depth_reached``
      - ``mcts/num_simulations``
      - ``mcts/root_value``
      - ``mcts/avg_branching_factor``
    """

    def __init__(self, tb_log_prefix: str = "mcts", verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._prefix = tb_log_prefix

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode_log = info.get("episode_log")
            if episode_log is None:
                continue
            stats = getattr(episode_log, "mcts_stats", None)
            if stats is None:
                continue
            self.logger.record(f"{self._prefix}/nodes_explored", stats.nodes_explored)
            self.logger.record(f"{self._prefix}/max_depth_reached", stats.max_depth_reached)
            self.logger.record(f"{self._prefix}/num_simulations", stats.num_simulations)
            self.logger.record(f"{self._prefix}/root_value", stats.root_value)
            self.logger.record(f"{self._prefix}/avg_branching_factor", stats.avg_branching_factor)
        return True


class PPOUpdateMetricsCallback(BaseCallback):
    """
    Captures per-PPO-update diagnostics from SB3's internal logger and writes
    them to both TensorBoard and the ``RLRunLogger`` JSONL store.

    Fires on ``_on_rollout_end()``, which SB3 calls after each complete
    PPO update cycle (after the n_steps rollout has been collected and all
    n_epochs gradient steps have run).

    Why capture here instead of relying on SB3's default TensorBoard output?
    -------------------------------------------------------------------------
    SB3 computes ``approx_kl``, ``clip_fraction``, and ``explained_variance``
    internally but does not always expose them as standalone TensorBoard
    scalars in all versions.  This callback explicitly re-records them so they
    are always visible, and also writes them to JSONL for offline analysis
    without a running TensorBoard server.

    Parameters
    ----------
    run_logger : ``RLRunLogger`` instance for JSONL persistence.
    verbose    : SB3 verbosity level.
    """

    def __init__(self, run_logger: "RLRunLogger", verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._run_logger = run_logger
        self._update_count = 0

    def _on_rollout_end(self) -> None:
        """Called after each complete PPO update cycle."""
        log_values = self.model.logger.name_to_value  # type: ignore[attr-defined]

        def _get(key: str, default: float = 0.0) -> float:
            val = log_values.get(key, default)
            return float(val) if val is not None else default

        def _get_opt(key: str) -> float | None:
            val = log_values.get(key)
            return float(val) if val is not None else None

        # Advantages mean/std from SB3's rollout buffer (available after collect_rollouts)
        adv_mean: float | None = None
        adv_std: float | None = None
        try:
            rollout_buf = self.model.rollout_buffer  # type: ignore[attr-defined]
            advs = rollout_buf.advantages.flatten()
            if len(advs) > 0:
                adv_mean = float(advs.mean())
                adv_std = float(advs.std())
        except Exception:
            pass

        metrics = PPOUpdateMetrics(
            update_step=self._update_count,
            policy_loss=_get("train/policy_gradient_loss"),
            value_loss=_get("train/value_loss"),
            entropy_loss=_get("train/entropy_loss"),
            total_loss=_get("train/loss"),
            clip_fraction=_get("train/clip_fraction"),
            approx_kl=_get("train/approx_kl"),
            explained_variance=_get("train/explained_variance"),
            learning_rate=_get("train/learning_rate"),
            # grad_norm: SB3 logs this when max_grad_norm > 0
            grad_norm=_get_opt("train/grad_norm"),
            advantages_mean=adv_mean,
            advantages_std=adv_std,
            num_timesteps=self.num_timesteps,
        )

        # Ensure these appear in TensorBoard even if SB3 doesn't record them by default
        self.logger.record("train/approx_kl", metrics.approx_kl)
        self.logger.record("train/clip_fraction", metrics.clip_fraction)
        self.logger.record("train/explained_variance", metrics.explained_variance)
        if metrics.grad_norm is not None:
            self.logger.record("train/grad_norm", metrics.grad_norm)
        if adv_mean is not None:
            self.logger.record("train/advantages_mean", adv_mean)
            self.logger.record("train/advantages_std", adv_std or 0.0)

        self._run_logger.log_ppo_update(metrics)
        self._update_count += 1

    def _on_step(self) -> bool:
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
    training_log_dir  : Root directory for training-only JSONL artifacts (PPO update metrics,
                        episode summaries). Each ``train()`` call creates a timestamped
                        subdirectory under this path. Defaults to ``outputs/training``.
    reward_predictor  : Optional RewardPredictorComponent. When provided, a
                        RewardPredictorCallback is added to the callback list so
                        that a PyTorch linear model trains on episode rewards.
    reward_predictor_fit_every : How often (in episodes) the callback triggers fit().
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
        simulator_config: SimulatorConfig | None = None,
        tokenizer: Any = None,
        tensorboard_log: str | Path = "outputs/logs",
        checkpoint_dir: str | Path = "outputs/checkpoints",
        training_log_dir: str | Path = "outputs/training",
        reward_predictor: Any = None,
        reward_predictor_fit_every: int = 50,
    ) -> None:
        self._compressor = compressor
        self._agent_factory = agent_factory
        self._simulator_factory = simulator_factory
        self._user_requests = user_requests
        self._config = config or TrainingConfig()
        self._env_config = env_config or EnvConfig()
        self._reward_config = reward_config or RewardConfig()
        self._simulator_config = simulator_config or SimulatorConfig()
        self._tokenizer = tokenizer
        self._tensorboard_log = Path(tensorboard_log)
        self._checkpoint_dir = Path(checkpoint_dir)
        self._training_log_dir = Path(training_log_dir)
        self._reward_predictor = reward_predictor
        self._reward_predictor_fit_every = reward_predictor_fit_every

        self._ppo: PPO | None = None
        self._device = self._resolve_device(self._config.device)
        log.info("trainer.device", device=self._device)

    @staticmethod
    def _resolve_device(device_cfg: str) -> str:
        """Resolve 'auto' to 'cuda' or 'cpu' based on availability."""
        if device_cfg == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_cfg

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

        # Create a timestamped run directory for training-only JSONL artifacts.
        # Each train() call gets its own directory so runs don't overwrite each other.
        run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_logger = RLRunLogger(run_id=run_id, training_dir=self._training_log_dir)

        # Save manifest immediately so post-processing tools can find this run.
        run_dir = self._training_log_dir / run_id
        manifest = TrainingRunManifest.create(
            run_id=run_id,
            compressor_type=type(self._compressor).__name__,
            n_train_requests=len(self._user_requests),
            checkpoint_dir=self._checkpoint_dir,
            run_name=getattr(self._config, "run_name", "") or "",
            reward_weights=self._reward_config.weights.model_dump(),
            ppo_hyperparams=self._config.ppo.model_dump(),
            n_envs=self._config.n_envs,
            num_timesteps=n_steps,
            extra={"device": self._device},
        )
        save_manifest(manifest, run_dir)
        log.info("trainer.manifest.saved", run_id=run_id, path=str(run_dir / "manifest.json"))

        # Build vectorised env (with WorldPool if configured)
        vec_env = self._make_vec_env()

        # Build or restore PPO model and move to device
        if self._config.resume_from:
            self._ppo = self._load_ppo(self._config.resume_from, vec_env)
        else:
            self._ppo = self._build_ppo(vec_env)

        # Explicitly move the compressor to the training device.
        # SB3 moves standard policy layers (token_embed, value_net) automatically,
        # but the compressor's HF model is a nested object that needs explicit placement.
        try:
            self._compressor.to(self._device)  # type: ignore[attr-defined]
        except AttributeError:
            pass  # non-HF compressors (IdentityCompressor) may not expose .to()

        # Build callbacks
        callbacks = self._build_callbacks(run_logger=run_logger)

        # Run training
        try:
            self._ppo.learn(
                total_timesteps=n_steps,
                callback=callbacks,
                reset_num_timesteps=not bool(self._config.resume_from),
            )
        finally:
            run_logger.close()

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

        Uses WorldPool when ``simulator_config.unique_per_episode=False`` (default)
        to avoid regenerating a world on every reset.  Falls back to the raw
        simulator_factory when unique_per_episode=True.

        Each parallel environment gets its own ReActAgent on every reset(),
        so there is no shared policy state across workers.
        """
        reward_fn = RewardFunction(config=self._reward_config)
        env_config = self._env_config
        tokenizer = self._tokenizer
        agent_factory = self._agent_factory
        user_requests = self._user_requests

        # Choose simulator source: pool or fresh-per-episode
        if self._simulator_config.unique_per_episode:
            effective_factory = self._simulator_factory
        else:
            pool = WorldPool(
                simulator_factory=self._simulator_factory,
                pool_size=self._simulator_config.pool_size,
                seed_range=tuple(self._simulator_config.seed_range),  # type: ignore[arg-type]
                rng_seed=getattr(self._config, "seed", 42),
            )
            pool.build()
            log.info("world_pool.ready", pool_size=len(pool))
            effective_factory = pool.sample

        def env_factory() -> CompressionEnv:
            return CompressionEnv(
                agent_factory=agent_factory,
                simulator_factory=effective_factory,
                reward_fn=reward_fn,
                user_requests=user_requests,
                config=env_config,
                tokenizer=tokenizer,
            )

        vec_env = make_vec_env(
            env_factory,
            n_envs=self._config.n_envs,
            seed=getattr(self._config, "seed", None),
        )
        return vec_env

    @staticmethod
    def _make_lr_schedule(base_lr: float, schedule: str, num_timesteps: int) -> Any:
        """
        Build an SB3-compatible learning rate schedule callable.

        Parameters
        ----------
        base_lr        : Starting learning rate.
        schedule       : 'constant' | 'linear' | 'cosine'
        num_timesteps  : Total training steps (used for decay schedules).

        Returns
        -------
        Callable(progress_remaining: float) → float
            SB3 passes ``progress_remaining`` which goes from 1.0 → 0.0.
        """
        import math as _math

        if schedule == "linear":
            def lr_fn(progress_remaining: float) -> float:
                # Decays from base_lr to base_lr/10 as progress goes 1.0 → 0.0
                return base_lr * (0.1 + 0.9 * progress_remaining)
            return lr_fn

        if schedule == "cosine":
            def lr_fn(progress_remaining: float) -> float:
                # Cosine annealing from base_lr to base_lr/10
                cos_val = (1.0 + _math.cos(_math.pi * (1.0 - progress_remaining))) / 2.0
                return base_lr * (0.1 + 0.9 * cos_val)
            return lr_fn

        # Default: constant
        return base_lr

    def _build_ppo(self, vec_env: VecEnv) -> PPO:
        """
        Instantiate SB3 PPO with ``CompressorPolicy``.

        ``policy_kwargs`` are forwarded to ``CompressorPolicy.__init__()``
        as additional keyword arguments beyond what SB3 injects automatically
        (observation_space, action_space, lr_schedule).
        """
        hp: PPOHyperparams = self._config.ppo

        lr_schedule = self._make_lr_schedule(
            base_lr=hp.learning_rate,
            schedule=getattr(hp, "lr_schedule", "constant"),
            num_timesteps=self._config.num_timesteps,
        )

        policy_kwargs: dict[str, Any] = {
            "compressor": self._compressor,
            "value_hidden_dim": 256,
        }

        ppo = PPO(
            policy=CompressorPolicy,
            env=vec_env,
            learning_rate=lr_schedule,
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
            device=self._device,
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

    def _build_callbacks(self, run_logger: "RLRunLogger") -> CallbackList:
        """
        Build the list of SB3 callbacks used during ``ppo.learn()``.

        Includes:
        - SB3 built-in ``CheckpointCallback`` — saves full PPO model as zip.
        - ``CompressorCheckpointCallback`` — saves only the compressor weights.
        - ``EpisodeLogCallback`` — reward components, episode stats → TensorBoard + JSONL.
        - ``PPOUpdateMetricsCallback`` — per-update PPO diagnostics → TensorBoard + JSONL.
        - ``MCTSMetricsCallback`` — MCTS tree stats → TensorBoard (no-op for non-MCTS runs).

        Parameters
        ----------
        run_logger : Active ``RLRunLogger`` for this training run.
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

        episode_log_cb = EpisodeLogCallback(
            run_logger=run_logger,
            episode_save_freq=getattr(self._config, "episode_save_freq", 0),
            episodes_dir=Path(self._tensorboard_log).parent / "episodes",
            verbose=0,
        )
        ppo_metrics_cb = PPOUpdateMetricsCallback(run_logger=run_logger, verbose=0)
        mcts_metrics = MCTSMetricsCallback(verbose=0)  # no-op for non-MCTS runs

        callbacks: list[BaseCallback] = [
            sb3_ckpt,
            compressor_ckpt,
            episode_log_cb,
            ppo_metrics_cb,
            mcts_metrics,
        ]

        if self._reward_predictor is not None:
            rp_cb = RewardPredictorCallback(
                reward_predictor=self._reward_predictor,
                fit_every_n_episodes=self._reward_predictor_fit_every,
                verbose=1,
            )
            callbacks.append(rp_cb)

        return CallbackList(callbacks)
