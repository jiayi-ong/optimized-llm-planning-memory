"""Unit tests for mcts/node.py — MCTSNode properties and frozen summary models."""

from __future__ import annotations

import pytest

from optimized_llm_planning_memory.mcts.node import MCTSNode, MCTSStats, MCTSTreeRepresentation
from optimized_llm_planning_memory.core.models import TrajectoryModel  # triggers forward-ref resolution
MCTSTreeRepresentation.model_rebuild()


def _make_traj_model() -> TrajectoryModel:
    return TrajectoryModel(
        trajectory_id="traj-001",
        request_id="req-001",
        steps=(),
        total_steps=0,
    )


def _make_node(
    node_id: str = "node-root",
    parent_id: str | None = None,
    depth: int = 0,
    visit_count: int = 0,
    value_sum: float = 0.0,
) -> MCTSNode:
    return MCTSNode(
        node_id=node_id,
        parent_id=parent_id,
        depth=depth,
        trajectory_snapshot=_make_traj_model(),
        compressed_state_snapshot=None,
        visit_count=visit_count,
        value_sum=value_sum,
    )


def _make_stats(**kwargs) -> MCTSStats:
    defaults = dict(
        nodes_explored=5,
        max_depth_reached=2,
        num_simulations=10,
        best_path_length=3,
        root_value=0.6,
        avg_branching_factor=2.0,
    )
    defaults.update(kwargs)
    return MCTSStats(**defaults)


@pytest.mark.unit
class TestMCTSNodeQValue:
    def test_unvisited_node_q_value_zero(self):
        node = _make_node(visit_count=0, value_sum=0.0)
        assert node.q_value == 0.0

    def test_q_value_is_mean(self):
        node = _make_node(visit_count=4, value_sum=8.0)
        assert node.q_value == pytest.approx(2.0)

    def test_q_value_single_visit(self):
        node = _make_node(visit_count=1, value_sum=0.75)
        assert node.q_value == pytest.approx(0.75)


@pytest.mark.unit
class TestMCTSNodeUCB1:
    def test_unvisited_node_ucb1_is_inf(self):
        node = _make_node(visit_count=0)
        assert node.ucb1_score(parent_visits=10, c_p=1.4) == float("inf")

    def test_visited_node_ucb1_is_finite(self):
        node = _make_node(visit_count=5, value_sum=3.0)
        score = node.ucb1_score(parent_visits=20, c_p=1.4)
        assert score < float("inf")
        assert score > 0.0

    def test_higher_q_value_gives_higher_ucb(self):
        n1 = _make_node("n1", visit_count=5, value_sum=5.0)   # q=1.0
        n2 = _make_node("n2", visit_count=5, value_sum=2.0)   # q=0.4
        s1 = n1.ucb1_score(parent_visits=20, c_p=1.4)
        s2 = n2.ucb1_score(parent_visits=20, c_p=1.4)
        assert s1 > s2


@pytest.mark.unit
class TestMCTSNodeChildren:
    def test_children_defaults_to_empty(self):
        node = _make_node()
        assert node.children == []

    def test_children_can_be_appended(self):
        root = _make_node("root")
        child = _make_node("child", parent_id="root", depth=1)
        root.children.append(child)
        assert len(root.children) == 1

    def test_make_root_classmethod_creates_root_node(self):
        traj = _make_traj_model()
        root = MCTSNode.make_root(trajectory=traj, compressed_state=None)
        assert root.parent_id is None
        assert root.depth == 0


@pytest.mark.unit
class TestMCTSStatsFrozen:
    def test_mcts_stats_is_frozen(self):
        stats = _make_stats()
        with pytest.raises(Exception):
            stats.root_value = 9.9  # type: ignore[misc]

    def test_mcts_stats_fields_correct(self):
        stats = _make_stats(nodes_explored=7, best_path_length=4)
        assert stats.nodes_explored == 7
        assert stats.best_path_length == 4

    def test_nodes_explored_must_be_nonneg(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MCTSStats(
                nodes_explored=-1,
                max_depth_reached=0,
                num_simulations=0,
                best_path_length=0,
                root_value=0.0,
                avg_branching_factor=0.0,
            )


@pytest.mark.unit
class TestMCTSTreeRepresentation:
    def test_tree_repr_is_frozen(self):
        traj = _make_traj_model()
        stats = _make_stats()
        tree = MCTSTreeRepresentation(best_path_trajectory=traj, stats=stats)
        with pytest.raises(Exception):
            tree.tradeoffs = "new"  # type: ignore[misc]

    def test_tree_repr_defaults(self):
        traj = _make_traj_model()
        stats = _make_stats()
        tree = MCTSTreeRepresentation(best_path_trajectory=traj, stats=stats)
        assert tree.alternative_paths == []
        assert tree.top_candidates == []
        assert tree.tradeoffs == ""
        assert tree.node_values == {}
