"""Unit tests for agent/trajectory.py — Trajectory accumulator."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from optimized_llm_planning_memory.agent.trajectory import Trajectory
from optimized_llm_planning_memory.core.models import ReActStep, TrajectoryModel


def _make_step(idx: int) -> ReActStep:
    return ReActStep(
        step_index=idx,
        thought=f"Thought at step {idx}",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@pytest.mark.unit
class TestTrajectoryAccumulation:
    def test_empty_trajectory_total_steps_zero(self):
        traj = Trajectory(request_id="r1")
        assert traj.total_steps == 0

    def test_add_step_increments_total_steps(self):
        traj = Trajectory(request_id="r1")
        traj.add_step(_make_step(0))
        assert traj.total_steps == 1
        traj.add_step(_make_step(1))
        assert traj.total_steps == 2

    def test_len_matches_total_steps(self):
        traj = Trajectory(request_id="r1")
        traj.add_step(_make_step(0))
        assert len(traj) == 1


@pytest.mark.unit
class TestTrajectorySnapshot:
    def test_to_model_step_count_matches(self):
        traj = Trajectory(request_id="r1")
        traj.add_step(_make_step(0))
        traj.add_step(_make_step(1))
        model = traj.to_model()
        assert model.total_steps == 2
        assert len(model.steps) == 2

    def test_to_model_is_frozen_trajectory_model(self):
        traj = Trajectory(request_id="r1")
        model = traj.to_model()
        assert isinstance(model, TrajectoryModel)
        with pytest.raises(Exception):
            model.total_steps = 99  # type: ignore[misc]

    def test_to_model_does_not_mutate_trajectory(self):
        traj = Trajectory(request_id="r1")
        traj.add_step(_make_step(0))
        traj.to_model()
        traj.add_step(_make_step(1))
        assert traj.total_steps == 2  # original still mutable


@pytest.mark.unit
class TestTrajectoryCompression:
    def test_mark_compression_sets_last_compressed_step(self):
        traj = Trajectory(request_id="r1")
        traj.add_step(_make_step(0))
        traj.add_step(_make_step(1))
        traj.mark_compression(at_step=2)
        assert traj.last_compressed_step == 2

    def test_steps_since_last_compression_returns_recent(self):
        traj = Trajectory(request_id="r1")
        for i in range(5):
            traj.add_step(_make_step(i))
        traj.mark_compression(at_step=3)
        recent = traj.steps_since_last_compression()
        assert all(s.step_index >= 3 for s in recent)

    def test_mark_compression_defaults_to_current_length(self):
        traj = Trajectory(request_id="r1")
        traj.add_step(_make_step(0))
        traj.add_step(_make_step(1))
        traj.mark_compression()
        assert traj.last_compressed_step == 2


@pytest.mark.unit
class TestTrajectoryToText:
    def test_to_text_empty_returns_empty_string(self):
        traj = Trajectory(request_id="r1")
        assert traj.to_text() == ""

    def test_to_text_with_thought_contains_thought(self):
        traj = Trajectory(request_id="r1")
        traj.add_step(_make_step(0))
        text = traj.to_text()
        assert "Thought at step 0" in text
