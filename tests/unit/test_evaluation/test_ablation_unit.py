"""Unit tests for evaluation/ablation.py — AblationRunner."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from optimized_llm_planning_memory.evaluation.ablation import AblationResult, AblationRunner


def _make_evaluator(score: float = 0.8) -> MagicMock:
    evaluator = MagicMock()
    evaluator.evaluate_dataset.return_value = [MagicMock()]
    evaluator.aggregate.return_value = {"overall_score_mean": score, "overall_score_std": 0.1}
    return evaluator


def _make_generator(n_episodes: int = 2):
    def generator(overrides):
        episodes = [MagicMock() for _ in range(n_episodes)]
        requests = [MagicMock() for _ in range(n_episodes)]
        return episodes, requests
    return generator


@pytest.mark.unit
class TestAblationRunnerSingleAxis:
    def test_single_axis_two_values_returns_two_results(self):
        runner = AblationRunner(
            evaluator=_make_evaluator(),
            episode_generator=_make_generator(),
        )
        results = runner.run(axes={"compressor_type": ["llm", "transformer"]})
        assert len(results) == 2

    def test_single_axis_results_are_ablation_result(self):
        runner = AblationRunner(
            evaluator=_make_evaluator(),
            episode_generator=_make_generator(),
        )
        results = runner.run(axes={"compressor_type": ["llm"]})
        assert all(isinstance(r, AblationResult) for r in results)

    def test_result_overrides_contain_axis_key(self):
        runner = AblationRunner(
            evaluator=_make_evaluator(),
            episode_generator=_make_generator(),
        )
        results = runner.run(axes={"compressor_type": ["llm", "transformer"]})
        keys = [r.overrides.get("compressor_type") for r in results]
        assert set(keys) == {"llm", "transformer"}


@pytest.mark.unit
class TestAblationRunnerTwoAxes:
    def test_two_axes_cartesian_product(self):
        runner = AblationRunner(
            evaluator=_make_evaluator(),
            episode_generator=_make_generator(),
        )
        results = runner.run(axes={
            "compressor_type": ["llm", "transformer"],
            "agent_mode": ["raw", "compressor"],
        })
        assert len(results) == 4

    def test_all_combinations_present(self):
        runner = AblationRunner(
            evaluator=_make_evaluator(),
            episode_generator=_make_generator(),
        )
        results = runner.run(axes={
            "a": [1, 2],
            "b": ["x", "y"],
        })
        combos = [(r.overrides["a"], r.overrides["b"]) for r in results]
        assert set(combos) == {(1, "x"), (1, "y"), (2, "x"), (2, "y")}


@pytest.mark.unit
class TestAblationRunnerEmpty:
    def test_axis_with_no_values_returns_empty(self):
        runner = AblationRunner(
            evaluator=_make_evaluator(),
            episode_generator=_make_generator(),
        )
        results = runner.run(axes={"a": []})
        assert results == []

    def test_empty_axes_runs_single_default_combo(self):
        # itertools.product() with no args yields one empty tuple -> one result
        runner = AblationRunner(
            evaluator=_make_evaluator(),
            episode_generator=_make_generator(),
        )
        results = runner.run(axes={})
        assert len(results) == 1


@pytest.mark.unit
class TestAblationResultLabel:
    def test_label_auto_generated(self):
        result = AblationResult(
            overrides={"compressor_type": "llm", "agent_mode": "raw"},
            aggregated_scores={"overall_score_mean": 0.8},
            n_episodes=5,
        )
        assert "compressor_type=llm" in result.label
        assert "agent_mode=raw" in result.label

    def test_custom_label_preserved(self):
        result = AblationResult(
            overrides={},
            aggregated_scores={},
            n_episodes=0,
            label="my_custom_label",
        )
        assert result.label == "my_custom_label"
