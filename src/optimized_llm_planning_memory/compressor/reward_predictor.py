"""
compressor/reward_predictor.py
================================
RewardPredictorComponent — a small PyTorch linear regression model that learns
to predict episode rewards from episode-level scalar features.

Design rationale
----------------
During PPO training the RL trainer accumulates (feature_vector, reward) pairs
from completed episodes. After every ``fit_every_n_episodes`` episodes it fits
a ``torch.nn.Linear(n_features, 1)`` model on the accumulated data using MSE
loss and Adam. The fitted weights give a fast, inspectable proxy for "which
episode characteristics correlate with higher reward?" — a useful sanity-check
during training without any external infrastructure.

The component is intentionally decoupled from the compressor output: the
IdentityCompressor (and any TrainableCompressorBase subclass) can optionally
hold a reference to this object; the RL trainer's RewardPredictorCallback
populates and fits it, but ``compress()`` never calls ``predict()``.

Features (5 scalars per episode)
---------------------------------
- ``hard_constraint_score``  : fraction of hard constraints satisfied (0–1)
- ``soft_constraint_score``  : weighted soft-constraint score (0–1)
- ``tool_efficiency_score``  : fraction of tool calls that were non-redundant (0–1)
- ``steps_per_episode``      : normalised step count (actual / max_steps → 0–1)
- ``budget_adherence``       : 1.0 if within budget, linearly decaying if over

Usage
-----
    from optimized_llm_planning_memory.compressor.reward_predictor import RewardPredictorComponent

    rp = RewardPredictorComponent()
    rp.add_episode({"hard_constraint_score": 0.8, ...}, reward=3.5)
    if rp.fit():
        pred = rp.predict({"hard_constraint_score": 1.0, ...})
    rp.save("/tmp/reward_predictor_ckpt")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Feature order is fixed — must be consistent across add_episode / predict / save / load
_FEATURE_NAMES: list[str] = [
    "hard_constraint_score",
    "soft_constraint_score",
    "tool_efficiency_score",
    "steps_per_episode",
    "budget_adherence",
]
_N_FEATURES = len(_FEATURE_NAMES)


class RewardPredictorComponent:
    """
    PyTorch linear model that predicts episode reward from scalar episode features.

    Parameters
    ----------
    lr              : Adam learning rate for the linear model.
    fit_n_steps     : Number of gradient steps per call to fit().
    weight_decay    : L2 regularisation (Adam weight_decay).
    device          : Torch device for model and tensors (default ``"cpu"``).
    """

    def __init__(
        self,
        lr: float = 1e-3,
        fit_n_steps: int = 30,
        weight_decay: float = 0.01,
        device: str = "cpu",
    ) -> None:
        self._device = torch.device(device)
        self._fit_n_steps = fit_n_steps

        # Single linear layer: y = W·x + b  (W shape: [1, n_features])
        self._model = nn.Linear(_N_FEATURES, 1, bias=True).to(self._device)
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Episode buffer: list of (feature_vector, scalar_reward)
        self._buffer: list[tuple[list[float], float]] = []
        self._fitted: bool = False

    # ── Public interface ───────────────────────────────────────────────────────

    def add_episode(self, features: dict[str, float], reward: float) -> None:
        """
        Buffer one episode's feature vector and observed reward.

        Missing feature keys default to 0.0.

        Parameters
        ----------
        features : Dict with keys matching _FEATURE_NAMES (extras are ignored).
        reward   : Scalar episode reward returned by the RL environment.
        """
        vec = [float(features.get(name, 0.0)) for name in _FEATURE_NAMES]
        self._buffer.append((vec, float(reward)))

    def fit(self, min_samples: int = 20) -> bool:
        """
        Fit the linear model on all buffered episodes using MSE + Adam.

        Parameters
        ----------
        min_samples : Minimum buffer size before fitting starts.

        Returns
        -------
        True if fitting ran, False if skipped (insufficient data).
        """
        if len(self._buffer) < min_samples:
            logger.debug(
                "RewardPredictorComponent.fit skipped: %d samples < min_samples=%d",
                len(self._buffer),
                min_samples,
            )
            return False

        # Build tensors from buffer — shape (N, n_features) and (N, 1)
        X = torch.tensor(
            [vec for vec, _ in self._buffer], dtype=torch.float32, device=self._device
        )
        y = torch.tensor(
            [[r] for _, r in self._buffer], dtype=torch.float32, device=self._device
        )

        self._model.train()
        final_loss: float = float("inf")
        for _ in range(self._fit_n_steps):
            self._optimizer.zero_grad()
            pred = self._model(X)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            self._optimizer.step()
            final_loss = loss.item()

        self._fitted = True
        weights = self._model.weight.data.squeeze().tolist()
        logger.info(
            "RewardPredictorComponent fitted on %d episodes. "
            "Weights: %s  bias: %.4f  MSE: %.4f",
            len(self._buffer),
            [f"{w:.4f}" for w in weights],
            float(self._model.bias.data.item()),
            final_loss,
        )
        return True

    def predict(self, features: dict[str, float]) -> float:
        """
        Predict reward for a feature dict using the fitted model.

        Returns 0.0 if the model has not been fitted yet.

        Parameters
        ----------
        features : Dict with keys matching _FEATURE_NAMES.

        Returns
        -------
        Predicted reward as a scalar float.
        """
        if not self._fitted:
            return 0.0
        self._model.eval()
        with torch.no_grad():
            x = torch.tensor(
                [[features.get(name, 0.0) for name in _FEATURE_NAMES]],
                dtype=torch.float32,
                device=self._device,
            )
            return float(self._model(x).item())

    def get_weights(self) -> dict[str, Any] | None:
        """
        Return the fitted weight vector as a feature-name → weight mapping.

        Returns None if the model has not been fitted yet.
        """
        if not self._fitted:
            return None
        w = self._model.weight.data.squeeze().tolist()
        b = float(self._model.bias.data.item())
        return {name: float(w[i]) for i, name in enumerate(_FEATURE_NAMES)} | {"bias": b}

    def save(self, path: str) -> None:
        """
        Save model state to ``{path}/reward_predictor.pt``.

        No-op if the model has not been fitted yet.

        Parameters
        ----------
        path : Directory path (created if absent).
        """
        if not self._fitted:
            logger.debug("RewardPredictorComponent.save: model not fitted — nothing to save.")
            return

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state": self._model.state_dict(),
                "feature_names": _FEATURE_NAMES,
                "n_episodes_trained": len(self._buffer),
            },
            save_dir / "reward_predictor.pt",
        )
        logger.info("RewardPredictorComponent saved to %s", save_dir / "reward_predictor.pt")

    def load(self, path: str) -> None:
        """
        Load model state from ``{path}/reward_predictor.pt``.

        Parameters
        ----------
        path : Directory (or direct path to the .pt file).

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.
        ValueError
            If the saved feature list does not match _FEATURE_NAMES.
        """
        p = Path(path)
        weights_file = p / "reward_predictor.pt" if p.is_dir() else p

        if not weights_file.exists():
            raise FileNotFoundError(
                f"RewardPredictorComponent: checkpoint not found at {weights_file}"
            )

        ckpt = torch.load(weights_file, map_location=self._device)

        saved_names = ckpt.get("feature_names", [])
        if saved_names and saved_names != _FEATURE_NAMES:
            raise ValueError(
                f"Feature mismatch: saved {saved_names}, expected {_FEATURE_NAMES}"
            )

        self._model.load_state_dict(ckpt["model_state"])
        self._fitted = True
        logger.info("RewardPredictorComponent loaded from %s", weights_file)
