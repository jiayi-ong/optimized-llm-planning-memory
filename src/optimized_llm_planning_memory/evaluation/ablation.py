"""
evaluation/ablation.py
=======================
AblationRunner — sweeps configuration axes and reruns Evaluator.

Usage
-----
    runner = AblationRunner(
        base_config=config,
        evaluator=evaluator,
        episode_generator=run_episodes_fn,
    )
    results = runner.run(axes={
        "compressor_type": ["llm", "transformer"],
        "agent_mode":      ["raw", "llm_summary", "compressor"],
    })
    runner.print_summary(results)

Design
------
The AblationRunner generates the Cartesian product of all axis values, runs
one evaluation per configuration, and collects aggregated metrics. It does not
modify the base config in place — it passes override dicts to the episode
generator callback, which is responsible for applying them (typically via
Hydra's ``OmegaConf.merge``).

``episode_generator`` signature
---------------------------------
    def generate(overrides: dict) -> tuple[list[EpisodeLog], list[UserRequest]]: ...

The caller is responsible for running episodes under the given config overrides
and returning the resulting logs paired with their requests.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

from optimized_llm_planning_memory.core.models import EpisodeLog, UserRequest
from optimized_llm_planning_memory.evaluation.evaluator import Evaluator


@dataclass
class AblationResult:
    """Holds aggregated metrics for one ablation configuration."""
    overrides: dict[str, Any]
    aggregated_scores: dict[str, float]
    n_episodes: int
    label: str = field(default="")

    def __post_init__(self) -> None:
        if not self.label:
            self.label = " | ".join(f"{k}={v}" for k, v in self.overrides.items())


class AblationRunner:
    """
    Sweeps ablation axes and evaluates each configuration.

    Parameters
    ----------
    evaluator          : Evaluator instance used for all runs.
    episode_generator  : Callable(overrides: dict) → (list[EpisodeLog], list[UserRequest]).
                         Responsible for running episodes under the given config overrides.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        episode_generator: Callable[[dict[str, Any]], tuple[list[EpisodeLog], list[UserRequest]]],
    ) -> None:
        self._evaluator = evaluator
        self._generator = episode_generator

    def run(
        self,
        axes: dict[str, list[Any]],
    ) -> list[AblationResult]:
        """
        Run the full ablation sweep.

        Parameters
        ----------
        axes : Dict mapping axis name → list of values to sweep.
               Example: {"compressor_type": ["llm", "transformer"]}

        Returns
        -------
        List of AblationResult, one per (axis, value) combination.
        """
        axis_names = list(axes.keys())
        axis_values = list(axes.values())
        results: list[AblationResult] = []

        for combo in itertools.product(*axis_values):
            overrides = dict(zip(axis_names, combo))
            episodes, requests = self._generator(overrides)

            eval_results = self._evaluator.evaluate_dataset(episodes, requests)
            agg = self._evaluator.aggregate(eval_results)

            results.append(AblationResult(
                overrides=overrides,
                aggregated_scores=agg,
                n_episodes=len(episodes),
            ))

        return results

    @staticmethod
    def print_summary(results: list[AblationResult], metric: str = "overall_score_mean") -> None:
        """
        Print a ranked summary table of ablation results.

        Parameters
        ----------
        results : Output of ``run()``.
        metric  : Metric to rank by. Defaults to ``"overall_score_mean"``.
        """
        sorted_results = sorted(
            results,
            key=lambda r: r.aggregated_scores.get(metric, 0.0),
            reverse=True,
        )
        header = f"{'Config':<50} {'n_ep':>5} {metric:>25}"
        print(header)
        print("-" * len(header))
        for r in sorted_results:
            score = r.aggregated_scores.get(metric, float("nan"))
            print(f"{r.label:<50} {r.n_episodes:>5} {score:>25.4f}")
