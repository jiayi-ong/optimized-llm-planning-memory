"""
training/env.py
===============
CompressionEnv — Gymnasium environment wrapping a planning episode.

Episode framing
---------------
One gymnasium episode = one complete planning trajectory for a UserRequest.
One gymnasium step = one compression event (after N ReAct steps).

Observation space : Box(max_obs_tokens,) int32 — padded token IDs of the
                    trajectory text accumulated since episode start.
Action space      : Box(max_action_tokens,) int32 — padded token IDs of the
                    compressed state produced by CompressorPolicy.
Reward            : float from RewardFunction.compute().
Termination       : Episode ends when agent signals DONE or hits max_agent_steps.

Design: No global state
------------------------
``reset()`` creates a fresh ``SimulatorAdapter(seed=N)`` every time, so
multiple ``CompressionEnv`` instances (via ``make_vec_env(n_envs=N)``) are
completely independent. This is safe because the simulator is an in-memory
Python library with no shared global state.

Multi-step episode structure
-----------------------------
``reset()`` runs the first ``steps_per_compression`` ReAct steps with no
compressed context (raw mode for the first window). Each subsequent ``step()``
call:
  1. Decodes the action (compressed state token IDs) to text.
  2. Parses the text as a CompressedState (falls back to prior state on failure).
  3. Runs the next ``steps_per_compression`` ReAct steps using this context.
  4. Computes a shaped reward from the current itinerary and trajectory state.
  5. Returns the updated trajectory encoding as the next observation.

This correctly presents the compressor with a multi-step problem: each action
directly influences the agent's context for the NEXT window of ReAct steps,
creating a meaningful learning signal for PPO.

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
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import numpy as np
import gymnasium

from optimized_llm_planning_memory.agent.trajectory import Trajectory
from optimized_llm_planning_memory.core.config import EnvConfig
from optimized_llm_planning_memory.core.models import (
    CompressedState,
    EpisodeLog,
    Itinerary,
    RewardComponents,
    UserRequest,
)
from optimized_llm_planning_memory.simulator.protocol import SimulatorProtocol
from optimized_llm_planning_memory.tools.events import EventBus
from optimized_llm_planning_memory.tools.registry import ToolRegistry
from optimized_llm_planning_memory.tools.tracker import ToolCallTracker
from optimized_llm_planning_memory.training.reward import RewardFunction
from optimized_llm_planning_memory.utils.logging import get_logger

log = get_logger(__name__)


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
    config            : Environment configuration (token dims, step counts).
    tokenizer         : Tokenizer for encoding/decoding trajectories and actions.
                        Must have ``encode()`` / ``decode()`` methods.
                        When None, falls back to character-level encoding (testing only).
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

        # Derive vocab size from tokenizer when available (L6: no hard-coded 32768)
        vocab_size = (
            getattr(tokenizer, "vocab_size", None)
            or getattr(tokenizer, "n_vocab", None)
            or self._config.vocab_size
        )

        max_obs = self._config.max_obs_tokens
        max_act = self._config.max_action_tokens

        # ── Gymnasium spaces ──────────────────────────────────────────────────
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=vocab_size - 1, shape=(max_obs,), dtype=np.int32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=0, high=vocab_size - 1, shape=(max_act,), dtype=np.int32,
        )

        # Per-episode state (initialised in reset())
        self._agent = None
        self._simulator: SimulatorProtocol | None = None
        self._current_request: UserRequest | None = None
        self._trajectory: Trajectory | None = None
        self._registry: ToolRegistry | None = None
        self._current_compressed: CompressedState | None = None
        self._final_itinerary: Itinerary | None = None
        self._agent_step_count: int = 0
        self._episode_done: bool = False
        self._episode_id: str = ""

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Start a new episode.

        1. Sample a random UserRequest from the training set.
        2. Create a fresh SimulatorAdapter with the given seed.
        3. Create a fresh ReActAgent, Trajectory, and ToolRegistry.
        4. Run the first ``steps_per_compression`` ReAct steps (no compressed context).
        5. Return the encoded trajectory as the initial observation.
        """
        super().reset(seed=seed)

        episode_seed = int(self.np_random.integers(0, 10000)) if seed is None else seed
        self._episode_id = str(uuid.uuid4())

        self._current_request = random.choice(self._user_requests)
        self._simulator = self._simulator_factory(episode_seed)
        self._agent = self._agent_factory()

        # Fresh per-episode state; trajectory is owned by the env, not the agent
        self._trajectory = Trajectory(request_id=self._current_request.request_id)
        tracker = ToolCallTracker()
        event_bus = EventBus()
        self._registry = ToolRegistry.from_config(
            simulator=self._simulator,
            tracker=tracker,
            event_bus=event_bus,
        )
        self._current_compressed = None
        self._final_itinerary = None
        self._agent_step_count = 0
        self._episode_done = False

        log.info(
            "env.episode.start",
            episode_id=self._episode_id,
            request_id=self._current_request.request_id,
            seed=episode_seed,
        )

        # Run first window with no compressed context
        self._final_itinerary, self._episode_done, error_msg = self._agent.run_steps(
            n=self._config.steps_per_compression,
            trajectory=self._trajectory,
            registry=self._registry,
            compressed_state=None,
            request=self._current_request,
            start_step_index=0,
            final_itinerary=None,
        )
        self._agent_step_count = self._trajectory.total_steps
        if error_msg:
            log.warning("env.reset.error", error=error_msg, episode_id=self._episode_id)

        obs = self._encode_trajectory(self._trajectory.to_text())
        info: dict = {
            "episode_id": self._episode_id,
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

        The action (compressed state token IDs) is decoded to text and parsed
        as a CompressedState. The agent then runs ``steps_per_compression`` more
        ReAct steps with this compressed context injected into its context window.

        Parameters
        ----------
        action : np.ndarray(max_action_tokens,) int32 — compressed state tokens
                 produced by CompressorPolicy.

        Returns
        -------
        obs         : np.ndarray — encoded trajectory token IDs (full episode so far).
        reward      : float — shaped reward for this compression window.
        terminated  : bool — True if episode is done (agent DONE signal).
        truncated   : bool — True if max_agent_steps was exceeded.
        info        : dict — diagnostic info for logging callbacks.
        """
        assert self._agent is not None, "Call reset() before step()."
        assert self._current_request is not None
        assert self._simulator is not None
        assert self._trajectory is not None
        assert self._registry is not None

        # Decode action tokens → CompressedState
        action_text = self._decode_action(action)
        self._current_compressed = _parse_compressed_state(
            action_text, self._current_compressed
        )

        # Run next window of ReAct steps with the injected compressed context
        if not self._episode_done:
            itinerary, done, error_msg = self._agent.run_steps(
                n=self._config.steps_per_compression,
                trajectory=self._trajectory,
                registry=self._registry,
                compressed_state=self._current_compressed,
                request=self._current_request,
                start_step_index=self._agent_step_count,
                final_itinerary=self._final_itinerary,
            )
            if itinerary is not None:
                self._final_itinerary = itinerary
            self._episode_done = done
            self._agent_step_count = self._trajectory.total_steps
            if error_msg:
                log.warning("env.step.error", error=error_msg, episode_id=self._episode_id)

        terminated = self._episode_done
        truncated = self._agent_step_count >= self._config.max_agent_steps

        episode_log = _build_episode_log(
            episode_id=self._episode_id,
            request_id=self._current_request.request_id,
            agent_mode=getattr(self._agent, "_mode", "compressor"),
            trajectory=self._trajectory,
            final_itinerary=self._final_itinerary,
            total_steps=self._agent_step_count,
        )

        reward_components = self._reward_fn.compute(
            episode_log=episode_log,
            user_request=self._current_request,
            is_terminal=terminated or truncated,
        )
        reward = float(reward_components.total_reward)

        obs = self._encode_trajectory(self._trajectory.to_text())
        info: dict = {
            "episode_log": episode_log,
            "reward_components": reward_components,
            "agent_steps": self._agent_step_count,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode_trajectory(self, text: str) -> np.ndarray:
        """Encode trajectory text → padded int32 array of max_obs_tokens length."""
        max_len = self._config.max_obs_tokens
        if self._tokenizer is None:
            char_ids = [ord(c) % self._config.vocab_size for c in text[:max_len]]
        else:
            char_ids = self._tokenizer.encode(text, max_length=max_len, truncation=True)

        padded = np.zeros(max_len, dtype=np.int32)
        padded[:len(char_ids)] = char_ids[:max_len]
        return padded

    def _decode_action(self, action_tokens: np.ndarray) -> str:
        """Decode action token IDs back to compressed state text."""
        tokens = action_tokens[action_tokens != 0].tolist()
        if self._tokenizer is not None:
            return self._tokenizer.decode(tokens, skip_special_tokens=True)
        return "".join(chr(min(t, 127)) for t in tokens)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _parse_compressed_state(
    text: str,
    fallback: CompressedState | None,
) -> CompressedState | None:
    """
    Try to parse action text as a CompressedState JSON; return fallback on failure.

    The policy network may produce malformed JSON early in training. Falling back
    to the previous state prevents hard crashes while still penalising low-quality
    compressions via the reward signal.
    """
    if not text.strip():
        return fallback
    try:
        return CompressedState.model_validate_json(text)
    except Exception:
        return fallback


def _build_episode_log(
    episode_id: str,
    request_id: str,
    agent_mode: Any,
    trajectory: Trajectory,
    final_itinerary: Itinerary | None,
    total_steps: int,
) -> EpisodeLog:
    """Build a minimal EpisodeLog from live episode state for reward computation."""
    agent_mode_str = agent_mode.value if isinstance(agent_mode, Enum) else str(agent_mode)
    placeholder_reward = RewardComponents(
        hard_constraint_score=0.0,
        soft_constraint_score=0.0,
        tool_efficiency_score=0.0,
        tool_failure_penalty=0.0,
        logical_consistency_score=0.0,
        terminal_itinerary_score=None,
        total_reward=0.0,
    )
    return EpisodeLog(
        episode_id=episode_id,
        request_id=request_id,
        agent_mode=agent_mode_str,
        trajectory=trajectory.to_model(),
        compressed_states=(),
        final_itinerary=final_itinerary,
        reward_components=placeholder_reward,
        tool_stats=(),
        total_steps=total_steps,
        mcts_stats=None,
        success=True,
        error=None,
        config_hash="",
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )
