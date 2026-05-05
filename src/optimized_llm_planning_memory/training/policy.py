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

import traceback
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule

from optimized_llm_planning_memory.utils.logging import get_logger

_log = get_logger(__name__)

from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate
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
        self._cs_template = CompressedStateTemplate()

        # Value network: embed token IDs → mean-pool → MLP → scalar
        # Token IDs are categorical, not ordinal; a plain Linear over raw IDs
        # would be meaningless (M1 fix).
        _vocab_size = (
            getattr(getattr(compressor, "_tokenizer", None), "vocab_size", None) or 32768
        )
        _embed_dim = 64
        self._token_embed = nn.Embedding(_vocab_size, _embed_dim)
        self._value_net = nn.Sequential(
            nn.Linear(_embed_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1),
        )

        self.optimizer = self.optimizer_class(
            list(compressor.get_trainable_parameters())
            + list(self._token_embed.parameters())
            + list(self._value_net.parameters()),
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

            # Entropy proxy: we approximate H(π) ≈ -mean(log_prob) per token,
            # which equals per-token cross-entropy, not the true distribution
            # entropy H = -Σ p·log(p).  This is numerically close for well-trained
            # policies (where action distribution is sharp) but will overestimate
            # entropy for flat distributions early in training.  The PPO entropy
            # bonus coefficient (ent_coef) is applied to this proxy, so calibrate
            # ent_coef accordingly.  To use true token entropy, compute
            # F.log_softmax(logits, dim=-1) and apply -Σ exp(lp)*lp per token.
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
        Generate compressed state tokens by calling the compressor on the observation.

        Wraps obs_text in a minimal single-step TrajectoryModel so it can be passed
        to ``compressor.compress()``. The resulting CompressedState is rendered to
        template text (``## SECTION_NAME ##`` format) — matching the exact token
        sequence the decoder generates — and tokenized into the action tensor.
        ``get_log_probs()`` receives the same template text as its target, ensuring
        PPO's ratio r_t(θ) = exp(log_π_new - log_π_old) is computed on the correct
        action representation.

        Returns (action_tensor, scalar_log_prob).
        """
        import uuid
        from datetime import datetime, timezone

        from optimized_llm_planning_memory.core.models import ReActStep, TrajectoryModel

        max_act = int(np.prod(self.action_space.shape))

        try:
            # Build a minimal single-step trajectory from the observation text
            step = ReActStep(
                step_index=0,
                thought=obs_text,
                action=None,
                observation=None,
                itinerary_snapshot=None,
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
            )
            traj = TrajectoryModel(
                trajectory_id=str(uuid.uuid4()),
                request_id="env_step",
                steps=(step,),
                total_steps=1,
            )

            # Generate the compressed state (inference mode inside compress())
            compressed_state = self.compressor.compress(traj, previous_state=None)
            # Render to template text — the exact token sequence the decoder produced.
            # Using model_dump_json() here would create a mismatch: the decoder generates
            # ## SECTION_NAME ## format but get_log_probs() would receive JSON, making
            # PPO's log-prob ratio meaningless.
            compressed_text = self._cs_template.render(compressed_state)

            # Tokenize compressed state text → action token IDs
            tokenizer = getattr(self.compressor, "_tokenizer", None)
            if tokenizer is not None:
                token_ids = tokenizer.encode(
                    compressed_text, max_length=max_act, truncation=True
                )
                # Decode the (possibly truncated) token IDs back to text.
                # evaluate_actions() will decode the stored action tokens the same way,
                # so computing log_prob on this decoded text ensures the old/new
                # ratio = exp(log_prob_new - log_prob_old) doesn't blow up from
                # a text/token mismatch when compressed_text > max_act tokens.
                action_text_for_log_prob = tokenizer.decode(
                    token_ids, skip_special_tokens=True
                )
            else:
                token_ids = [ord(c) % 32768 for c in compressed_text[:max_act]]
                action_text_for_log_prob = "".join(chr(min(t, 127)) for t in token_ids)

            action_arr = np.zeros(max_act, dtype=np.int32)
            action_arr[:len(token_ids)] = token_ids[:max_act]
            action_tensor = torch.tensor(action_arr, dtype=torch.int32)

            # Scalar log-prob computed on the same text evaluate_actions() will see.
            token_log_probs = self.compressor.get_log_probs(obs_text, action_text_for_log_prob)
            log_prob = token_log_probs.sum()

        except Exception as _exc:
            _log.warning(
                "policy.generate_action.failed",
                error=str(_exc),
                traceback=traceback.format_exc(),
            )
            action_tensor = torch.zeros(max_act, dtype=torch.int32)
            # Use a consistent fallback log_prob so the old/new ratio in PPO
            # doesn't blow up.  log(1/32768) ≈ -10.4 is the uniform-distribution
            # baseline for a 32K-vocab model; this is a safe lower bound.
            try:
                token_log_probs = self.compressor.get_log_probs(obs_text, "")
                log_prob = token_log_probs.sum().detach()
            except Exception:
                log_prob = torch.tensor(-10.0)

        return action_tensor, log_prob

    def _compute_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute value estimates from mean-pooled token embeddings.

        Masks padding zeros so the pool is over real tokens only.  Without
        masking, 480+ zero-embedding positions overwhelm the 30-token signal,
        driving value estimates toward a constant regardless of the observation.
        """
        mask = (obs != 0).float().unsqueeze(-1)    # (batch, obs_len, 1)
        embedded = self._token_embed(obs.long())   # (batch, obs_len, embed_dim)
        counts = mask.sum(dim=1).clamp(min=1)      # (batch, 1) — avoid div-by-zero
        pooled = (embedded * mask).sum(dim=1) / counts  # (batch, embed_dim)
        return self._value_net(pooled)             # (batch, 1)

    def _decode_obs(self, obs_tokens: torch.Tensor) -> str:
        """Decode observation token IDs to text string."""
        # Box obs space stores floats; cast to int before tokenizer decode.
        int_tokens = obs_tokens.long()
        tokens = int_tokens[int_tokens != 0].tolist()
        if hasattr(self.compressor, "_tokenizer") and self.compressor._tokenizer is not None:
            return self.compressor._tokenizer.decode(tokens, skip_special_tokens=True)
        return " ".join(str(t) for t in tokens)

    def _decode_action(self, action_tokens: torch.Tensor) -> str:
        """Decode action token IDs to text string."""
        # Box action space stores floats; cast to int before tokenizer decode.
        int_tokens = action_tokens.long()
        tokens = int_tokens[int_tokens != 0].tolist()
        if hasattr(self.compressor, "_tokenizer") and self.compressor._tokenizer is not None:
            return self.compressor._tokenizer.decode(tokens, skip_special_tokens=True)
        return " ".join(str(t) for t in tokens)
