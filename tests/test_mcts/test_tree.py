"""
tests/test_mcts/test_tree.py
============================
Unit tests for MCTSTree — select, expand, backpropagate, best_path, collect_stats.
All tests use mock TrajectoryModel objects; no LLM calls are made.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from optimized_llm_planning_memory.core.models import ReActStep, TrajectoryModel
from optimized_llm_planning_memory.mcts.config import MCTSConfig
from optimized_llm_planning_memory.mcts.node import MCTSNode, MCTSTreeRepresentation
from optimized_llm_planning_memory.mcts.tree import MCTSTree

# Resolve TYPE_CHECKING forward reference so MCTSTreeRepresentation validates fields.
MCTSTreeRepresentation.model_rebuild()

_TS = "2026-05-01T12:00:00+00:00"


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_config(**kwargs) -> MCTSConfig:
    defaults = dict(
        num_simulations=10,
        max_depth=5,
        exploration_constant=1.414,
        branching_factor=2,
        rollout_steps=1,
    )
    defaults.update(kwargs)
    return MCTSConfig(**defaults)


def _make_traj(total_steps: int = 0) -> TrajectoryModel:
    """Return a real TrajectoryModel with ``total_steps`` stub ReActStep objects."""
    steps = tuple(
        ReActStep(step_index=i, thought=f"t{i}", action=None, observation=None, timestamp=_TS)
        for i in range(total_steps)
    )
    return TrajectoryModel(
        trajectory_id=str(uuid.uuid4()),
        request_id="req-001",
        steps=steps,
        total_steps=total_steps,
    )


# ── build_root ────────────────────────────────────────────────────────────────

class TestBuildRoot:
    def test_root_is_stored(self):
        tree = MCTSTree(_make_config())
        traj = _make_traj()
        root = tree.build_root(traj, None)
        assert tree.root.node_id == root.node_id

    def test_root_depth_zero(self):
        tree = MCTSTree(_make_config())
        root = tree.build_root(_make_traj(), None)
        assert root.depth == 0

    def test_select_without_root_raises(self):
        tree = MCTSTree(_make_config())
        with pytest.raises(AssertionError):
            tree.select()


# ── select ────────────────────────────────────────────────────────────────────

class TestSelect:
    def test_select_returns_root_when_no_children(self):
        tree = MCTSTree(_make_config())
        tree.build_root(_make_traj(), None)
        selected = tree.select()
        assert selected.node_id == tree.root.node_id

    def test_select_favours_unvisited_child(self):
        """An unvisited child has UCB1 = +inf and must always be selected."""
        tree = MCTSTree(_make_config())
        tree.build_root(_make_traj(), None)

        # Manually add two children: one visited, one not
        visited = MCTSNode(
            node_id=str(uuid.uuid4()),
            parent_id=tree.root.node_id,
            depth=1,
            trajectory_snapshot=_make_traj(),
            compressed_state_snapshot=None,
            visit_count=5,
            value_sum=3.0,
        )
        unvisited = MCTSNode(
            node_id=str(uuid.uuid4()),
            parent_id=tree.root.node_id,
            depth=1,
            trajectory_snapshot=_make_traj(),
            compressed_state_snapshot=None,
            visit_count=0,
        )
        tree._nodes[visited.node_id] = visited
        tree._nodes[unvisited.node_id] = unvisited
        tree.root.children = [visited, unvisited]
        tree.root.visit_count = 5

        selected = tree.select()
        assert selected.node_id == unvisited.node_id

    def test_select_stops_at_terminal(self):
        tree = MCTSTree(_make_config())
        tree.build_root(_make_traj(), None)
        terminal = MCTSNode(
            node_id=str(uuid.uuid4()),
            parent_id=tree.root.node_id,
            depth=1,
            trajectory_snapshot=_make_traj(),
            compressed_state_snapshot=None,
            is_terminal=True,
            visit_count=3,
            value_sum=1.5,
        )
        tree._nodes[terminal.node_id] = terminal
        tree.root.children = [terminal]
        tree.root.visit_count = 3

        selected = tree.select()
        assert selected.node_id == terminal.node_id


# ── expand ────────────────────────────────────────────────────────────────────

class TestExpand:
    def test_expand_adds_children(self):
        tree = MCTSTree(_make_config())
        root = tree.build_root(_make_traj(), None)
        actions = ["search_flights({})", "search_hotels({})"]
        trajs = [_make_traj(i + 1) for i in range(2)]
        children = tree.expand(root, actions, trajs)
        assert len(children) == 2
        assert len(root.children) == 2

    def test_expand_sets_depth(self):
        tree = MCTSTree(_make_config())
        root = tree.build_root(_make_traj(), None)
        children = tree.expand(root, ["act1"], [_make_traj(1)])
        assert children[0].depth == 1

    def test_expand_sets_parent_id(self):
        tree = MCTSTree(_make_config())
        root = tree.build_root(_make_traj(), None)
        children = tree.expand(root, ["act1"], [_make_traj(1)])
        assert children[0].parent_id == root.node_id

    def test_expand_marks_terminal_at_max_depth(self):
        config = _make_config(max_depth=1)
        tree = MCTSTree(config)
        root = tree.build_root(_make_traj(), None)
        children = tree.expand(root, ["act1"], [_make_traj(1)])
        assert children[0].is_terminal

    def test_expand_mismatched_lengths_raises(self):
        tree = MCTSTree(_make_config())
        root = tree.build_root(_make_traj(), None)
        with pytest.raises(AssertionError):
            tree.expand(root, ["act1", "act2"], [_make_traj()])


# ── backpropagate ─────────────────────────────────────────────────────────────

class TestBackpropagate:
    def test_backprop_updates_root(self):
        tree = MCTSTree(_make_config())
        root = tree.build_root(_make_traj(), None)
        # Single node — backprop should update root
        tree.backpropagate(root, 0.9)
        assert root.visit_count == 1
        assert root.value_sum == pytest.approx(0.9)

    def test_backprop_updates_path(self):
        tree = MCTSTree(_make_config())
        root = tree.build_root(_make_traj(), None)
        children = tree.expand(root, ["act1"], [_make_traj(1)])
        child = children[0]
        tree.backpropagate(child, 0.7)
        # Both child and root should be updated
        assert child.visit_count == 1
        assert child.value_sum == pytest.approx(0.7)
        assert root.visit_count == 1
        assert root.value_sum == pytest.approx(0.7)

    def test_backprop_accumulates_multiple_runs(self):
        tree = MCTSTree(_make_config())
        root = tree.build_root(_make_traj(), None)
        children = tree.expand(root, ["act1"], [_make_traj(1)])
        child = children[0]
        tree.backpropagate(child, 0.5)
        tree.backpropagate(child, 0.9)
        assert child.visit_count == 2
        assert child.value_sum == pytest.approx(1.4)
        assert child.q_value == pytest.approx(0.7)


# ── best_path / collect_stats ─────────────────────────────────────────────────

class TestBestPath:
    def test_best_path_is_root_only_when_no_children(self):
        tree = MCTSTree(_make_config())
        tree.build_root(_make_traj(), None)
        path = tree.best_path()
        assert len(path) == 1
        assert path[0].node_id == tree.root.node_id

    def test_best_path_follows_highest_q(self):
        tree = MCTSTree(_make_config())
        root = tree.build_root(_make_traj(), None)
        c1, c2 = tree.expand(root, ["a1", "a2"], [_make_traj(1), _make_traj(1)])
        # Give c2 a higher Q
        tree.backpropagate(c1, 0.3)
        tree.backpropagate(c2, 0.8)
        path = tree.best_path()
        assert path[-1].node_id == c2.node_id


class TestCollectStats:
    def test_stats_empty_tree_returns_zeros(self):
        tree = MCTSTree(_make_config())
        stats = tree.collect_stats(0)
        assert stats.nodes_explored == 0

    def test_stats_after_search(self):
        tree = MCTSTree(_make_config())
        root = tree.build_root(_make_traj(), None)
        children = tree.expand(root, ["a1", "a2"], [_make_traj(1), _make_traj(1)])
        tree.backpropagate(children[0], 0.5)
        stats = tree.collect_stats(num_simulations=1)
        assert stats.nodes_explored == 3  # root + 2 children
        assert stats.max_depth_reached == 1
        assert stats.num_simulations == 1


# ── to_representation ─────────────────────────────────────────────────────────

class TestToRepresentation:
    def test_representation_has_stats(self):
        tree = MCTSTree(_make_config())
        tree.build_root(_make_traj(), None)
        repr_ = tree.to_representation(num_simulations=0)
        assert repr_.stats.nodes_explored >= 1

    def test_representation_best_path_trajectory(self):
        tree = MCTSTree(_make_config())
        root = tree.build_root(_make_traj(), None)
        children = tree.expand(root, ["a1", "a2"], [_make_traj(1), _make_traj(2)])
        tree.backpropagate(children[1], 0.9)
        tree.backpropagate(children[0], 0.3)
        repr_ = tree.to_representation(num_simulations=2)
        # Best child is children[1] (Q=0.9); its trajectory has total_steps=2
        assert repr_.best_path_trajectory.total_steps == 2
