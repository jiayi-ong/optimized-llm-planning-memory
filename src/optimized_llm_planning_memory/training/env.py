"""
training/env.py
===============
CompressionEnv — Gymnasium environment wrapping a planning episode.

Episode framing
---------------
One gymnasium episode = one complete planning trajectory for a UserRequest.
One gymnasium step = one compression event (after N ReAct steps).

Observation space : Box(max_obs_tokens,) int32 — padded token IDs of trajectory
                    text since the last compression.
Action space      : Box(max_action_tokens,) int32 — padded token IDs of the
                    compressed state produced by CompressorPolicy.
Reward            : float from RewardFunction.compute().
Termination       : Episode ends when agent signals DONE or hits max_steps.

Design: No global state
------------------------
``reset()`` creates a fresh ``SimulatorAdapter(seed=N)`` every time, so
multiple ``CompressionEnv`` instances (via ``make_vec_env(n_envs=N)``) are
completely independent. This is safe because the simulator is an in-memory
Python library with no shared global state.

Integration with SB3
---------------------
SB3's ``PPO`` class expects:
  - ``observation_space`` and ``action_space`` as gymnasium spaces.
  - ``reset()`` returning ``(obs, info)``.
  - ``step()`` returning ``(obs, reward, terminated, truncated, info)``.

This class satisfies all of these.
"""

from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np
import gymnasium

from optimized_llm_planning_memory.core.config import EnvConfig
from optimized_llm_planning_memory.core.models import UserRequest
from optimized_llm_planning_memory.simulator.protocol import SimulatorProtocol
from optimized_llm_planning_memory.training.reward import RewardFunction


class CompressionEnv(gymnasium.Env):
    """
    Gymnasium environment for RL training of the context compressor.

    Parameters
    ----------
    agent_factory     : Callable returning a fresh ``ReActAgent`` for each episode.
    simulator_factory : Callable(seed) → ``SimulatorProtocol``.
                        Called on every ``reset()``.
    reward_fn         : Multi-component reward function.
    user_requests     : Training set of ``UserRequest`` objects.
                        One is sampled per episode.
    config            : Environment configuration (max token dimensions).
    tokenizer         : Tokenizer for encoding trajectories.
                        Must have ``encode()`` / ``decode()`` methods.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        agent_factory: Callable,
        simulator_factory: Callable[[int], SimulatorProtocol],
        reward_fn: RewardFunction,
        user_requests: list[UserRequest],
        config: EnvConfig | None = None,
        tokenizer: Any = None,
    ) -> None:
        super().__init__()
        self._agent_factory = agent_factory
        self._simulator_factory = simulator_factory
        self._reward_fn = reward_fn
        self._user_requests = user_requests
        self._config = config or EnvConfig()
        self._tokenizer = tokenizer

        # ── Gymnasium spaces ──────────────────────────────────────────────────
        max_obs = self._config.max_obs_tokens
        max_act = self._config.max_action_tokens

        # Observations: padded token IDs of trajectory text
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=32768,  # typical HF vocab size; override if using larger model
            shape=(max_obs,),
            dtype=np.int32,
        )
        # Actions: padded token IDs of compressed state text
        self.action_space = gymnasium.spaces.Box(
            low=0,
            high=32768,
            shape=(max_act,),
            dtype=np.int32,
        )

        # Per-episode state (initialised in reset())
        self._agent = None
        self._simulator: SimulatorProtocol | None = None
        self._current_request: UserRequest | None = None
        self._episode_log = None
        self._step_count: int = 0
        self._done: bool = False

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Start a new episode.

        1. Sample a random UserRequest from the training set.
        2. Create a fresh SimulatorAdapter with the given seed.
        3. Create a fresh ReActAgent.
        4. Return the initial observation (empty trajectory encoding).
        """
        super().reset(seed=seed)

        # Use gymnasium's np_random for reproducible seeding
        episode_seed = int(self.np_random.integers(0, 10000)) if seed is None else seed

        self._current_request = random.choice(self._user_requests)
        self._simulator = self._simulator_factory(episode_seed)
        self._agent = self._agent_factory()
        self._step_count = 0
        self._done = False
        self._episode_log = None

        # Initial observation: zero-padded (no trajectory yet)
        obs = np.zeros(self._config.max_obs_tokens, dtype=np.int32)
        info: dict = {
            "episode_seed": episode_seed,
            "request_id": self._current_request.request_id,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one compression step.

        The action (compressed state token IDs) is decoded to text, used as
        the current CompressedState context for the agent, and the agent runs
        ``steps_per_compression`` ReAct steps before the next compression.

        Parameters
        ----------
        action : np.ndarray(max_action_tokens,) int32 — compressed state tokens
                 from CompressorPolicy.

        Returns
        -------
        obs         : np.ndarray — trajectory token IDs since this compression.
        reward      : float — shaped reward.
        terminated  : bool — True if episode is done.
        truncated   : bool — True if truncated (max_steps exceeded).
        info        : dict — EpisodeLog snapshot for logging callbacks.
        """
        assert self._agent is not None, "Call reset() before step()."
        assert self._current_request is not None
        assert self._simulator is not None

        self._step_count += 1

        # Decode action tokens → compressed state text (for agent context injection)
        # In this simplified implementation, the agent runs a full episode on reset
        # and compression events are managed internally. The action is the compressor
        # output for the CURRENT compression step.
        # TODO: Integrate with ReActAgent to inject compressed_state mid-episode.

        # Run the full episode (simplified: one step per episode for initial scaffold)
        if self._episode_log is None:
            self._episode_log = self._agent.run_episode(
                request=self._current_request,
                simulator=self._simulator,
            )

        terminated = True  # episode ends after one full run
        truncated = False

        # Compute reward
        reward_components = self._reward_fn.compute(
            episode_log=self._episode_log,
            user_request=self._current_request,
            is_terminal=terminated,
        )
        reward = float(reward_components.total_reward)

        # Observation: encode trajectory text → padded token IDs
        obs = self._encode_trajectory(self._episode_log.trajectory.to_text())

        info: dict = {
            "episode_log": self._episode_log,
            "reward_components": reward_components,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Not implemented (no visual rendering needed)."""
        pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode_trajectory(self, text: str) -> np.ndarray:
        """Encode trajectory text → padded int32 array of max_obs_tokens length."""
        max_len = self._config.max_obs_tokens
        if self._tokenizer is None:
            # Fallback: character-level encoding (for testing without a tokenizer)
            char_ids = [ord(c) % 32768 for c in text[:max_len]]
        else:
            char_ids = self._tokenizer.encode(text, max_length=max_len, truncation=True)

        padded = np.zeros(max_len, dtype=np.int32)
        padded[:len(char_ids)] = char_ids[:max_len]
        return padded
