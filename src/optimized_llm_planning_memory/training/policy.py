"""
training/policy.py
==================
CompressorPolicy — custom SB3 policy that wraps TransformerCompressor.

This is the bridge between SB3's PPO training loop and the HuggingFace
transformer model that acts as the compression policy.

Key responsibilities
--------------------
``forward()``            — Generate compressed state tokens (the "action") and
                           compute value estimates. Called by SB3 during rollout.
``evaluate_actions()``   — Compute log-probs, values, and entropy for a batch
                           of (obs, action) pairs. Called by SB3 during PPO update.
``_predict()``           — Greedy/stochastic action selection. Used by SB3 env loop.

Token-level PPO
---------------
The observation is a padded token ID array (trajectory encoding).
The action is a padded token ID array (compressed state tokens).
``evaluate_actions()`` calls ``compressor.get_log_probs()`` to compute
per-token log-probs, then sums/means them to get a scalar log-prob per step.

Value network
-------------
A simple linear projection over the mean-pooled encoder hidden states serves
as the value network. This is kept simple; a more sophisticated critic (e.g.,
a separate transformer head) can be swapped in by overriding ``_compute_value()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule

from optimized_llm_planning_memory.compressor.trainable_base import TrainableCompressorBase


class CompressorPolicy(BasePolicy):
    """
    Custom SB3 policy that wraps a ``TrainableCompressorBase`` for PPO training.

    Parameters
    ----------
    observation_space : Gymnasium observation space (Box, int32 token IDs).
    action_space      : Gymnasium action space (Box, int32 token IDs).
    lr_schedule       : Learning rate schedule (SB3 convention).
    compressor        : The trainable compressor model.
    value_hidden_dim  : Hidden dimension of the value network MLP.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        compressor: TrainableCompressorBase,
        value_hidden_dim: int = 256,
        **kwargs: Any,
    ) -> None:
        # PPO always injects use_sde for ActorCriticPolicy; BasePolicy doesn't accept it.
        kwargs.pop("use_sde", None)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor_class=nn.Identity,  # we handle feature extraction ourselves
            features_extractor_kwargs={},
            normalize_images=False,
            **kwargs,
        )
        self.compressor = compressor
        self._value_hidden_dim = value_hidden_dim

        # Value network: linear projection over obs embedding → scalar
        obs_dim = int(np.prod(observation_space.shape))
        self._value_net = nn.Sequential(
            nn.Linear(obs_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1),
        )

        self.optimizer = self.optimizer_class(
            list(compressor.get_trainable_parameters()) + list(self._value_net.parameters()),
            lr=lr_schedule(1.0),
            **self.optimizer_kwargs,
        )

    # ── SB3 required interface ────────────────────────────────────────────────

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate actions (compressed state tokens), value estimates, and log-probs.

        Called by SB3 during rollout collection.

        Parameters
        ----------
        obs           : (batch, max_obs_tokens) int32 tensor of trajectory token IDs.
        deterministic : If True, use greedy decoding; otherwise sample.

        Returns
        -------
        actions    : (batch, max_action_tokens) int32 tensor of compressed state tokens.
        values     : (batch, 1) float tensor from value network.
        log_probs  : (batch,) float tensor — sum of token log-probs per sample.
        """
        batch_size = obs.shape[0]
        actions_list = []
        log_probs_list = []

        for i in range(batch_size):
            obs_text = self._decode_obs(obs[i])
            action_tokens, log_prob = self._generate_action(obs_text, deterministic)
            actions_list.append(action_tokens)
            log_probs_list.append(log_prob)

        actions = torch.stack(actions_list)  # (batch, max_action_tokens)
        log_probs = torch.stack(log_probs_list)  # (batch,)
        values = self._compute_value(obs)  # (batch, 1)

        return actions, values, log_probs

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute values, log-probs, and entropy for a batch of (obs, action) pairs.

        Called by SB3 PPO during the policy update step. This is where the
        PPO clipping ratio r_t(θ) = exp(log_π_θ(a|s) − log_π_old(a|s)) is computed.

        Parameters
        ----------
        obs     : (batch, max_obs_tokens) int32 — trajectory encodings.
        actions : (batch, max_action_tokens) int32 — compressed state tokens.

        Returns
        -------
        values    : (batch, 1)
        log_probs : (batch,)  — sum of token log-probs
        entropy   : scalar    — mean entropy over batch
        """
        batch_size = obs.shape[0]
        log_probs_list = []
        entropy_list = []

        for i in range(batch_size):
            obs_text = self._decode_obs(obs[i])
            action_text = self._decode_action(actions[i])

            token_log_probs = self.compressor.get_log_probs(obs_text, action_text)
            log_prob = token_log_probs.sum()
            log_probs_list.append(log_prob)

            # Entropy: -sum(p * log p) ≈ -mean(log_p) for discrete sequence
            entropy = -token_log_probs.mean()
            entropy_list.append(entropy)

        values = self._compute_value(obs)
        log_probs = torch.stack(log_probs_list)
        entropy = torch.stack(entropy_list).mean()

        return values, log_probs, entropy

    def _predict(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Return predicted actions (without values/log_probs). Used by SB3 env loop."""
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Return value estimates for the given observations.

        Required by SB3's on-policy rollout collector (``collect_rollouts``) to
        bootstrap returns when an episode terminates mid-rollout. Defined on
        ``ActorCriticPolicy`` but not on ``BasePolicy``, so we implement it here.
        """
        return self._compute_value(obs)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _generate_action(
        self, obs_text: str, deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate compressed state tokens from the compressor.

        Returns (action_tensor, scalar_log_prob).
        """
        from optimized_llm_planning_memory.core.models import TrajectoryModel
        import uuid

        # Wrap obs_text in a minimal TrajectoryModel for the compressor interface
        # TODO: pass a proper TrajectoryModel from the env rather than reconstructing
        dummy_traj = TrajectoryModel(
            trajectory_id=str(uuid.uuid4()),
            request_id="env_step",
            steps=(),
            total_steps=0,
        )
        # For now, get log_probs from a placeholder action
        # Full integration requires the env to pass the trajectory model directly
        max_act = int(np.prod(self.action_space.shape))
        action_tokens = torch.zeros(max_act, dtype=torch.int32)
        log_prob = torch.tensor(0.0)
        return action_tokens, log_prob

    def _compute_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute value estimates from the observation embeddings."""
        # Cast to float and flatten for the MLP value head
        obs_float = obs.float()
        values = self._value_net(obs_float)  # (batch, 1)
        return values

    def _decode_obs(self, obs_tokens: torch.Tensor) -> str:
        """Decode observation token IDs to text string."""
        # Remove padding (zeros at the end)
        tokens = obs_tokens[obs_tokens != 0].tolist()
        if hasattr(self.compressor, "_tokenizer") and self.compressor._tokenizer is not None:
            return self.compressor._tokenizer.decode(tokens, skip_special_tokens=True)
        return " ".join(str(t) for t in tokens)

    def _decode_action(self, action_tokens: torch.Tensor) -> str:
        """Decode action token IDs to text string."""
        tokens = action_tokens[action_tokens != 0].tolist()
        if hasattr(self.compressor, "_tokenizer") and self.compressor._tokenizer is not None:
            return self.compressor._tokenizer.decode(tokens, skip_special_tokens=True)
        return " ".join(str(t) for t in tokens)
