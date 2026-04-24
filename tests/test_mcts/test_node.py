"""
tests/test_mcts/test_node.py
============================
Unit tests for MCTSNode, MCTSStats, and MCTSTreeRepresentation.
"""

from __future__ import annotations

import math
import uuid
from unittest.mock import MagicMock

import pytest

from optimized_llm_planning_memory.mcts.node import MCTSNode, MCTSStats, MCTSTreeRepresentation


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_traj():
    """Return a minimal TrajectoryModel mock."""
    traj = MagicMock()
    traj.trajectory_id = str(uuid.uuid4())
    traj.request_id = "req-001"
    traj.total_steps = 0
    traj.steps = ()
    traj.to_text.return_value = ""
    return traj


def _make_node(depth: int = 0, visit_count: int = 0, value_sum: float = 0.0) -> MCTSNode:
    return MCTSNode(
        node_id=str(uuid.uuid4()),
        parent_id=None,
        depth=depth,
        trajectory_snapshot=_make_traj(),
        compressed_state_snapshot=None,
        visit_count=visit_count,
        value_sum=value_sum,
    )


# ── MCTSNode tests ────────────────────────────────────────────────────────────

class TestMCTSNodeQValue:
    def test_q_value_unvisited_is_zero(self):
        node = _make_node(visit_count=0, value_sum=0.0)
        assert node.q_value == 0.0

    def test_q_value_single_visit(self):
        node = _make_node(visit_count=1, value_sum=0.8)
        assert node.q_value == pytest.approx(0.8)

    def test_q_value_multiple_visits(self):
        node = _make_node(visit_count=4, value_sum=2.4)
        assert node.q_value == pytest.approx(0.6)


class TestMCTSNodeUCB1:
    def test_unvisited_returns_inf(self):
        node = _make_node(visit_count=0)
        score = node.ucb1_score(parent_visits=10, c_p=1.414)
        assert score == float("inf")

    def test_ucb1_formula(self):
        node = _make_node(visit_count=4, value_sum=2.0)
        parent_visits = 10
        c_p = 1.414
        expected_q = 2.0 / 4.0
        expected_exploration = c_p * math.sqrt(math.log(parent_visits) / 4.0)
        expected = expected_q + expected_exploration
        assert node.ucb1_score(parent_visits, c_p) == pytest.approx(expected)

    def test_higher_visit_count_reduces_exploration_bonus(self):
        low_visit = _make_node(visit_count=2, value_sum=1.0)   # Q=0.5
        high_visit = _make_node(visit_count=10, value_sum=5.0)  # Q=0.5
        parent = 20
        assert low_visit.ucb1_score(parent, 1.414) > high_visit.ucb1_score(parent, 1.414)

    def test_parent_visits_zero_does_not_crash(self):
        node = _make_node(visit_count=1, value_sum=0.5)
        # Should not raise even with parent_visits=0
        score = node.ucb1_score(parent_visits=0, c_p=1.414)
        assert isinstance(score, float)


class TestMCTSNodeMakeRoot:
    def test_root_has_no_parent(self):
        traj = _make_traj()
        root = MCTSNode.make_root(traj, None)
        assert root.parent_id is None
        assert root.depth == 0
        assert root.visit_count == 0
        assert root.children == []

    def test_root_node_id_is_unique(self):
        traj = _make_traj()
        r1 = MCTSNode.make_root(traj, None)
        r2 = MCTSNode.make_root(traj, None)
        assert r1.node_id != r2.node_id


# ── MCTSStats tests ───────────────────────────────────────────────────────────

class TestMCTSStats:
    def test_construction(self):
        stats = MCTSStats(
            nodes_explored=15,
            max_depth_reached=4,
            num_simulations=10,
            best_path_length=3,
            root_value=0.72,
            avg_branching_factor=2.5,
        )
        assert stats.nodes_explored == 15
        assert stats.root_value == pytest.approx(0.72)

    def test_frozen(self):
        stats = MCTSStats(
            nodes_explored=1, max_depth_reached=1, num_simulations=1,
            best_path_length=1, root_value=0.5, avg_branching_factor=1.0,
        )
        with pytest.raises(Exception):
            stats.nodes_explored = 99  # type: ignore[misc]

    def test_json_round_trip(self):
        stats = MCTSStats(
            nodes_explored=7, max_depth_reached=3, num_simulations=5,
            best_path_length=2, root_value=0.6, avg_branching_factor=2.0,
        )
        json_str = stats.model_dump_json()
        loaded = MCTSStats.model_validate_json(json_str)
        assert loaded == stats


# ── MCTSTreeRepresentation tests ──────────────────────────────────────────────

class TestMCTSTreeRepresentation:
    def test_construction_defaults(self):
        stats = MCTSStats(
            nodes_explored=1, max_depth_reached=0, num_simulations=0,
            best_path_length=0, root_value=0.0, avg_branching_factor=0.0,
        )
        traj = _make_traj()
        repr_ = MCTSTreeRepresentation(best_path_trajectory=traj, stats=stats)
        assert repr_.alternative_paths == []
        assert repr_.top_candidates == []
        assert repr_.tradeoffs == ""

    def test_top_candidates_preserved(self):
        stats = MCTSStats(
            nodes_explored=3, max_depth_reached=1, num_simulations=2,
            best_path_length=1, root_value=0.7, avg_branching_factor=2.0,
        )
        traj = _make_traj()
        candidates = ["Option A: direct flight (Q=0.8)", "Option B: connect (Q=0.6)"]
        repr_ = MCTSTreeRepresentation(
            best_path_trajectory=traj,
            top_candidates=candidates,
            tradeoffs="Direct is faster but costs more.",
            stats=stats,
        )
        assert repr_.top_candidates == candidates
        assert "Direct" in repr_.tradeoffs
