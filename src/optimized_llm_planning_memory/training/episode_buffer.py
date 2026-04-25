"""
training/episode_buffer.py
===========================
EpisodeBuffer — transition storage for PPO mini-batch updates.

Stores ``PPOTransition`` objects collected across rollout episodes. After
Generalised Advantage Estimation (GAE) is applied by ``RLTrainer``, the
``advantage`` field of each transition is filled in. The buffer then
supports sampling of mini-batches for PPO policy/value network updates.

Architecture note — SB3 integration path
------------------------------------------
When using ``stable_baselines3.PPO`` via ``RLTrainer``, SB3 manages its own
internal rollout buffer and runs GAE/advantage computation automatically
before each policy update. In that path, ``EpisodeBuffer`` is NOT used —
SB3's ``RolloutBuffer`` is the active data structure.

``EpisodeBuffer`` is retained for two use cases:
  (a) A custom training loop that bypasses SB3 (requires implementing GAE
      externally via ``fill_advantages()``, since SB3 is not called).
  (b) Logging/inspection — collecting completed episode transitions for
      offline analysis or replay without triggering a training step.

If you are running a pure SB3 loop, you do not need to instantiate
``EpisodeBuffer`` at all.
"""

from __future__ import annotations

import random
from typing import Iterator

from optimized_llm_planning_memory.core.models import PPOTransition


class EpisodeBuffer:
    """
    Stores PPO transitions from completed episodes.

    Lifecycle
    ---------
    1. ``add()`` — called once per compression event (RL step).
    2. ``fill_advantages()`` — called after GAE computation over the whole buffer.
    3. ``minibatches()`` — iterate over shuffled mini-batches for PPO updates.
    4. ``clear()`` — reset at the start of each PPO iteration.
    """

    def __init__(self) -> None:
        self._transitions: list[PPOTransition] = []

    def add(self, transition: PPOTransition) -> None:
        """Add a single transition to the buffer."""
        self._transitions.append(transition)

    def fill_advantages(self, advantages: list[float]) -> None:
        """
        Fill in the ``advantage`` field for all transitions.

        Parameters
        ----------
        advantages : List of floats of length ``len(self)``.
                     Computed via GAE by ``RLTrainer._compute_advantages()``.
        """
        if len(advantages) != len(self._transitions):
            raise ValueError(
                f"Expected {len(self._transitions)} advantages, got {len(advantages)}."
            )
        for transition, adv in zip(self._transitions, advantages):
            transition.advantage = adv

    def minibatches(
        self, batch_size: int, shuffle: bool = True
    ) -> Iterator[list[PPOTransition]]:
        """
        Yield mini-batches of transitions for PPO updates.

        Parameters
        ----------
        batch_size : Desired size of each mini-batch.
        shuffle    : Whether to shuffle the buffer before batching.

        Yields
        ------
        list[PPOTransition]
            Each list has at most ``batch_size`` elements.
        """
        indices = list(range(len(self._transitions)))
        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield [self._transitions[i] for i in batch_indices]

    def clear(self) -> None:
        """Remove all stored transitions."""
        self._transitions.clear()

    def __len__(self) -> int:
        return len(self._transitions)

    def is_empty(self) -> bool:
        return len(self._transitions) == 0
