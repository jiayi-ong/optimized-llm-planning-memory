"""
compressor/spark_component.py
==============================
SparkWeightComponent — PySpark MLlib linear-regression model that learns to
predict episode rewards from episode-level feature vectors.

Design rationale
----------------
The RL training loop needs a distributed-training artefact that (a) is
genuinely trainable from episode data, (b) runs in Colab without a cluster,
and (c) does NOT interfere with the identity compressor's output.

SparkWeightComponent satisfies all three:
- Uses PySpark MLlib ``LinearRegression`` in ``local[*]`` mode (one machine,
  all available cores).
- Buffers (features, reward) tuples from completed episodes; fits the model
  after every ``fit_every_n_episodes`` episodes.
- Exposes ``predict()`` so RL reward shaping could optionally query it, but
  the IdentityCompressor never calls ``predict()`` — no output coupling.

Features (5 scalars per episode)
---------------------------------
- ``hard_constraint_score``  : fraction of hard constraints satisfied (0–1)
- ``soft_constraint_score``  : weighted soft-constraint satisfaction (0–1)
- ``tool_efficiency_score``  : fraction of tool calls that succeeded (0–1)
- ``steps_per_episode``      : normalised step count (actual / max_steps → 0–1)
- ``budget_adherence``       : 1 if within budget, linearly decaying below 0

PySpark session
---------------
Lazy-initialised on first call to ``fit()`` or ``predict()``. Reused for the
lifetime of the component. Call ``stop()`` for explicit cleanup.

Usage
-----
    spark = SparkWeightComponent(master="local[*]")
    spark.add_episode(features={"hard_constraint_score": 0.8, ...}, reward=3.5)
    if spark.fit():
        pred = spark.predict({"hard_constraint_score": 1.0, ...})
    spark.save("/tmp/spark_ckpt")
    spark.stop()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Feature order must be consistent across add_episode / predict / save / load
_FEATURE_NAMES = [
    "hard_constraint_score",
    "soft_constraint_score",
    "tool_efficiency_score",
    "steps_per_episode",
    "budget_adherence",
]


class SparkWeightComponent:
    """
    PySpark MLlib linear-regression model that predicts episode reward.

    Parameters
    ----------
    master      : Spark master URL (default ``"local[*]"``).
    app_name    : SparkSession application name.
    max_iter    : MLlib LinearRegression max iterations.
    reg_param   : L2 regularisation parameter.
    """

    def __init__(
        self,
        master: str = "local[*]",
        app_name: str = "SparkWeightComponent",
        max_iter: int = 100,
        reg_param: float = 0.01,
    ) -> None:
        self._master = master
        self._app_name = app_name
        self._max_iter = max_iter
        self._reg_param = reg_param

        # Episode buffer: list of (feature_vector, reward)
        self._buffer: list[tuple[list[float], float]] = []

        # Fitted model weights (coefficients + intercept), None until first fit
        self._weights: np.ndarray | None = None   # shape (n_features,)
        self._intercept: float = 0.0

        # Lazy Spark session
        self._spark: Any = None  # pyspark.sql.SparkSession

    # ── Public interface ───────────────────────────────────────────────────────

    def add_episode(self, features: dict[str, float], reward: float) -> None:
        """
        Buffer one episode's feature vector and observed reward.

        Missing keys default to 0.0.

        Parameters
        ----------
        features : Dict with keys from _FEATURE_NAMES.
        reward   : Scalar episode reward from the RL environment.
        """
        vec = [float(features.get(name, 0.0)) for name in _FEATURE_NAMES]
        self._buffer.append((vec, float(reward)))

    def fit(self, min_samples: int = 20) -> bool:
        """
        Fit MLlib LinearRegression on the buffered episodes.

        Parameters
        ----------
        min_samples : Minimum buffer size before fitting (avoids degenerate fit).

        Returns
        -------
        True if fitting succeeded, False if skipped (insufficient data).
        """
        if len(self._buffer) < min_samples:
            logger.debug(
                "SparkWeightComponent.fit skipped: %d samples < min_samples=%d",
                len(self._buffer),
                min_samples,
            )
            return False

        try:
            spark = self._get_or_create_spark()
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.regression import LinearRegression
            from pyspark.sql import Row

            # Build Spark DataFrame from buffer
            rows = [
                Row(**{name: vec[i] for i, name in enumerate(_FEATURE_NAMES)}, label=reward)
                for vec, reward in self._buffer
            ]
            df = spark.createDataFrame(rows)

            assembler = VectorAssembler(inputCols=_FEATURE_NAMES, outputCol="features")
            df_assembled = assembler.transform(df).select("features", "label")

            lr = LinearRegression(
                featuresCol="features",
                labelCol="label",
                maxIter=self._max_iter,
                regParam=self._reg_param,
                elasticNetParam=0.0,
            )
            model = lr.fit(df_assembled)

            # Extract weights (numpy array) and intercept
            self._weights = np.array(model.coefficients.toArray(), dtype=np.float64)
            self._intercept = float(model.intercept)

            logger.info(
                "SparkWeightComponent fitted on %d episodes. "
                "Weights: %s  intercept: %.4f",
                len(self._buffer),
                self._weights,
                self._intercept,
            )
            return True

        except Exception as exc:  # noqa: BLE001
            logger.warning("SparkWeightComponent.fit failed: %s", exc)
            return False

    def predict(self, features: dict[str, float]) -> float:
        """
        Predict reward for a feature vector using the fitted weights.

        If the model has not been fitted yet, returns 0.0.

        Parameters
        ----------
        features : Dict with keys from _FEATURE_NAMES.

        Returns
        -------
        Predicted reward (scalar float).
        """
        if self._weights is None:
            return 0.0
        vec = np.array([float(features.get(name, 0.0)) for name in _FEATURE_NAMES])
        return float(np.dot(self._weights, vec) + self._intercept)

    def get_weights(self) -> np.ndarray | None:
        """Return the current coefficient vector, or None if not yet fitted."""
        return self._weights.copy() if self._weights is not None else None

    def save(self, path: str) -> None:
        """
        Persist weights to ``{path}/spark_weights.json``.

        Creates the directory if it does not exist. No-op if model not fitted.

        Parameters
        ----------
        path : Directory path (created if absent).
        """
        if self._weights is None:
            logger.debug("SparkWeightComponent.save: model not fitted — nothing to save.")
            return

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "weights": self._weights.tolist(),
            "intercept": self._intercept,
            "feature_names": _FEATURE_NAMES,
            "n_episodes_trained": len(self._buffer),
        }
        out = save_dir / "spark_weights.json"
        out.write_text(json.dumps(payload, indent=2))
        logger.info("SparkWeightComponent saved weights to %s", out)

    def load(self, path: str) -> None:
        """
        Load weights from ``{path}/spark_weights.json``.

        Parameters
        ----------
        path : Directory containing ``spark_weights.json``.

        Raises
        ------
        FileNotFoundError
            If the weights file does not exist at the given path.
        ValueError
            If the feature list in the file does not match _FEATURE_NAMES.
        """
        p = Path(path)
        weights_file = p / "spark_weights.json" if p.is_dir() else p

        if not weights_file.exists():
            raise FileNotFoundError(
                f"SparkWeightComponent: weights file not found at {weights_file}"
            )

        payload = json.loads(weights_file.read_text())

        saved_names = payload.get("feature_names", [])
        if saved_names and saved_names != _FEATURE_NAMES:
            raise ValueError(
                f"Feature mismatch: saved {saved_names}, expected {_FEATURE_NAMES}"
            )

        self._weights = np.array(payload["weights"], dtype=np.float64)
        self._intercept = float(payload["intercept"])
        logger.info("SparkWeightComponent loaded weights from %s", weights_file)

    def stop(self) -> None:
        """Stop the Spark session if one is running."""
        if self._spark is not None:
            try:
                self._spark.stop()
            except Exception:  # noqa: BLE001
                pass
            self._spark = None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_or_create_spark(self) -> Any:
        """Lazy-init the Spark session."""
        if self._spark is None:
            from pyspark.sql import SparkSession

            self._spark = (
                SparkSession.builder
                .master(self._master)
                .appName(self._app_name)
                .config("spark.ui.showConsoleProgress", "false")
                .config("spark.driver.host", "localhost")
                .getOrCreate()
            )
            self._spark.sparkContext.setLogLevel("WARN")
        return self._spark
