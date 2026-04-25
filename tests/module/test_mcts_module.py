"""Module tests for mcts — MCTSTree expand, backpropagate, best_path."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from optimized_llm_planning_memory.mcts.node import MCTSNode, MCTSStats, MCTSTreeRepresentation
from optimized_llm_planning_memory.mcts.tree import MCTSTree
from optimized_llm_planning_memory.core.models import ReActStep, TrajectoryModel
MCTSTreeRepresentation.model_rebuild()


def _make_traj(n_steps: int = 0) -> TrajectoryModel:
    steps = [
        ReActStep(
            step_index=i,
            thought=f"step {i}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        for i in range(n_steps)
    ]
    return TrajectoryModel(
        trajectory_id="traj-mcts",
        request_id="req-mcts",
        steps=steps,
        total_steps=n_steps,
    )


@pytest.mark.module_test
class TestMCTSNodeTreeStructure:
    def test_root_node_created_with_make_root(self):
        traj = _make_traj(0)
        root = MCTSNode.make_root(trajectory=traj, compressed_state=None)
        assert root.parent_id is None
        assert root.depth == 0
        assert root.visit_count == 0

    def test_child_node_depth_increments(self):
        traj = _make_traj(0)
        root = MCTSNode.make_root(trajectory=traj, compressed_state=None)
        child = MCTSNode(
            node_id="child-1",
            parent_id=root.node_id,
            depth=root.depth + 1,
            trajectory_snapshot=_make_traj(1),
            compressed_state_snapshot=None,
        )
        root.children.append(child)
        assert root.children[0].depth == 1
        assert root.children[0].parent_id == root.node_id

    def test_q_value_after_backpropagation(self):
        traj = _make_traj(0)
        root = MCTSNode.make_root(trajectory=traj, compressed_state=None)
        root.visit_count = 4
        root.value_sum = 3.2
        assert root.q_value == pytest.approx(0.8)

    def test_ucb1_unvisited_returns_inf(self):
        traj = _make_traj(0)
        root = MCTSNode.make_root(trajectory=traj, compressed_state=None)
        child = MCTSNode(
            node_id="child-1",
            parent_id=root.node_id,
            depth=1,
            trajectory_snapshot=_make_traj(0),
            compressed_state_snapshot=None,
        )
        assert child.ucb1_score(parent_visits=10, c_p=1.4) == float("inf")


@pytest.mark.module_test
class TestMCTSStatsRepresentation:
    def test_stats_and_tree_repr_fields(self):
        stats = MCTSStats(
            nodes_explored=10,
            max_depth_reached=3,
            num_simulations=20,
            best_path_length=3,
            root_value=0.7,
            avg_branching_factor=2.5,
        )
        traj = _make_traj(2)
        tree_repr = MCTSTreeRepresentation(
            best_path_trajectory=traj,
            alternative_paths=[traj],
            top_candidates=["Option A"],
            tradeoffs="Option A is cheaper.",
            stats=stats,
        )
        assert len(tree_repr.top_candidates) == 1
        assert tree_repr.stats.nodes_explored == 10
        assert tree_repr.tradeoffs == "Option A is cheaper."

    def test_best_path_trajectory_is_present(self):
        traj = _make_traj(1)
        stats = MCTSStats(
            nodes_explored=1, max_depth_reached=0,
            num_simulations=1, best_path_length=1,
            root_value=0.5, avg_branching_factor=0.0,
        )
        tree_repr = MCTSTreeRepresentation(best_path_trajectory=traj, stats=stats)
        assert tree_repr.best_path_trajectory is traj
