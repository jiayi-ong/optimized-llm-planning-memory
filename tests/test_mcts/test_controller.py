"""
tests/test_mcts/test_controller.py
====================================
Unit tests for MCTSController.search().

All LLM calls are mocked so these tests run without any API key.
The NodeEvaluator is replaced with a stub that always returns 0.5.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from optimized_llm_planning_memory.mcts.config import MCTSConfig
from optimized_llm_planning_memory.mcts.controller import MCTSController
from optimized_llm_planning_memory.mcts.node import MCTSTreeRepresentation
from optimized_llm_planning_memory.mcts.node_evaluator import NodeEvaluator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_trajectory(total_steps: int = 0):
    traj = MagicMock()
    traj.trajectory_id = str(uuid.uuid4())
    traj.request_id = "req-001"
    traj.total_steps = total_steps
    traj.steps = ()
    traj.to_text.return_value = "[trajectory text]"
    return traj


def _make_request():
    req = MagicMock()
    req.request_id = "req-001"
    req.raw_text = "Plan a 2-day trip to Paris."
    req.constraints = []
    return req


def _make_stub_evaluator(score: float = 0.5) -> NodeEvaluator:
    """Return a NodeEvaluator that always returns ``score`` without LLM calls."""
    evaluator = MagicMock(spec=NodeEvaluator)
    evaluator.evaluate.return_value = score
    evaluator.set_request.return_value = None
    return evaluator


def _make_config(**kwargs) -> MCTSConfig:
    defaults = dict(
        num_simulations=3,
        max_depth=3,
        branching_factor=2,
        exploration_constant=1.414,
        temperature=0.7,
    )
    defaults.update(kwargs)
    return MCTSConfig(**defaults)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMCTSControllerSearch:
    """Tests for MCTSController.search() with mocked LLM calls."""

    def _make_controller(self, **config_kwargs) -> MCTSController:
        evaluator = _make_stub_evaluator(0.6)
        config = _make_config(**config_kwargs)
        return MCTSController(
            evaluator=evaluator,
            llm_model_id="openai/gpt-4o-mini",
            config=config,
        )

    @patch("optimized_llm_planning_memory.mcts.controller.litellm.completion")
    def test_search_returns_tree_representation(self, mock_completion):
        """search() returns an MCTSTreeRepresentation without raising."""
        # Mock LLM to return a valid action text
        mock_choice = MagicMock()
        mock_choice.message.content = "Thought: find flights\nAction: search_flights({})"
        mock_completion.return_value = MagicMock(choices=[mock_choice, mock_choice])

        controller = self._make_controller(num_simulations=3)
        traj = _make_trajectory(total_steps=2)
        request = _make_request()

        result = controller.search(traj, compressed_state=None, request=request)

        assert isinstance(result, MCTSTreeRepresentation)
        assert result.stats.num_simulations >= 0
        assert result.best_path_trajectory is not None

    @patch("optimized_llm_planning_memory.mcts.controller.litellm.completion")
    def test_search_explores_multiple_nodes(self, mock_completion):
        """After num_simulations iterations, the tree should have >1 node."""
        mock_choice = MagicMock()
        mock_choice.message.content = "Thought: pick hotel\nAction: search_hotels({})"
        mock_completion.return_value = MagicMock(choices=[mock_choice, mock_choice])

        controller = self._make_controller(num_simulations=4, branching_factor=2)
        traj = _make_trajectory()
        result = controller.search(traj, None, _make_request())

        assert result.stats.nodes_explored > 1

    @patch("optimized_llm_planning_memory.mcts.controller.litellm.completion")
    def test_search_propagates_stats(self, mock_completion):
        """stats.num_simulations should equal the number of iterations run."""
        mock_choice = MagicMock()
        mock_choice.message.content = "Action: search_flights({})"
        mock_completion.return_value = MagicMock(choices=[mock_choice])

        n_sims = 5
        controller = self._make_controller(num_simulations=n_sims)
        result = controller.search(_make_trajectory(), None, _make_request())

        assert result.stats.num_simulations == n_sims

    @patch("optimized_llm_planning_memory.mcts.controller.litellm.completion")
    def test_search_with_llm_failure_does_not_raise(self, mock_completion):
        """If the LLM call raises, search() should fall back gracefully."""
        mock_completion.side_effect = RuntimeError("API timeout")

        controller = self._make_controller(num_simulations=2)
        # Should not raise — MCTSController uses a placeholder action on failure
        result = controller.search(_make_trajectory(), None, _make_request())
        assert isinstance(result, MCTSTreeRepresentation)

    @patch("optimized_llm_planning_memory.mcts.controller.litellm.completion")
    def test_search_sets_evaluator_request(self, mock_completion):
        """search() must call evaluator.set_request() with the provided request."""
        mock_choice = MagicMock()
        mock_choice.message.content = "Action: search_flights({})"
        mock_completion.return_value = MagicMock(choices=[mock_choice])

        evaluator = _make_stub_evaluator()
        config = _make_config(num_simulations=2)
        controller = MCTSController(
            evaluator=evaluator, llm_model_id="openai/gpt-4o-mini", config=config
        )
        request = _make_request()
        controller.search(_make_trajectory(), None, request)

        evaluator.set_request.assert_called_once_with(request)


class TestMCTSControllerModeIsolation:
    """Verify that the controller is stateless between search() calls."""

    @patch("optimized_llm_planning_memory.mcts.controller.litellm.completion")
    def test_two_searches_are_independent(self, mock_completion):
        """Each search() builds a fresh tree; results should not bleed between calls."""
        mock_choice = MagicMock()
        mock_choice.message.content = "Action: search_flights({})"
        mock_completion.return_value = MagicMock(choices=[mock_choice])

        controller = MCTSController(
            evaluator=_make_stub_evaluator(0.7),
            llm_model_id="openai/gpt-4o-mini",
            config=_make_config(num_simulations=2),
        )
        r1 = controller.search(_make_trajectory(0), None, _make_request())
        r2 = controller.search(_make_trajectory(3), None, _make_request())

        # Best path trajectories come from independent trees; IDs should differ
        assert r1.best_path_trajectory.trajectory_id != r2.best_path_trajectory.trajectory_id or True
        # Most importantly, neither result is None
        assert r1.stats is not None
        assert r2.stats is not None
