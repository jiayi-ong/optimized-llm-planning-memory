"""Module tests for compressor — DummyCompressor multi-call workflows."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from optimized_llm_planning_memory.core.models import CompressedState, ReActStep, TrajectoryModel
from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate


def _make_trajectory(n_steps: int = 3) -> TrajectoryModel:
    steps = [
        ReActStep(
            step_index=i,
            thought=f"Thought at step {i}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        for i in range(n_steps)
    ]
    return TrajectoryModel(
        trajectory_id="traj-module",
        request_id="req-module",
        steps=steps,
        total_steps=n_steps,
    )


@pytest.mark.module_test
class TestDummyCompressorCompress:
    def test_compress_returns_compressed_state(self, dummy_compressor):
        traj = _make_trajectory(3)
        result = dummy_compressor.compress(traj)
        assert isinstance(result, CompressedState)

    def test_compress_state_id_is_nonempty(self, dummy_compressor):
        traj = _make_trajectory(3)
        result = dummy_compressor.compress(traj)
        assert len(result.state_id) > 0

    def test_multiple_compressions_have_unique_state_ids(self, dummy_compressor):
        traj = _make_trajectory(3)
        ids = {dummy_compressor.compress(traj).state_id for _ in range(5)}
        assert len(ids) == 5

    def test_compress_with_previous_state(self, dummy_compressor):
        traj = _make_trajectory(3)
        first = dummy_compressor.compress(traj)
        second = dummy_compressor.compress(traj, previous_state=first)
        assert isinstance(second, CompressedState)
        assert second.state_id != first.state_id


@pytest.mark.module_test
class TestDummyCompressorLogProbs:
    def test_get_log_probs_correct_shape(self, dummy_compressor):
        import torch
        traj = _make_trajectory(2)
        result = dummy_compressor.compress(traj)
        compressed_text = result.current_itinerary_sketch or "dummy output text"
        trajectory_text = traj.to_text() or "input text"
        lp = dummy_compressor.get_log_probs(trajectory_text, compressed_text)
        assert isinstance(lp, torch.Tensor)
        assert lp.dim() == 1

    def test_get_log_probs_requires_grad(self, dummy_compressor):
        traj = _make_trajectory(2)
        result = dummy_compressor.compress(traj)
        trajectory_text = traj.to_text() or "input"
        compressed_text = result.current_itinerary_sketch or "output"
        lp = dummy_compressor.get_log_probs(trajectory_text, compressed_text)
        assert lp.requires_grad or lp.grad_fn is not None


@pytest.mark.module_test
class TestCompressedStateTemplate:
    def test_template_render_produces_string(self, dummy_compressor):
        template = CompressedStateTemplate()
        traj = _make_trajectory(2)
        state = dummy_compressor.compress(traj)
        rendered = template.render(state)
        assert isinstance(rendered, str)
        assert len(rendered) > 0

    def test_template_validate_passes_on_valid_state(self, dummy_compressor):
        template = CompressedStateTemplate()
        traj = _make_trajectory(2)
        state = dummy_compressor.compress(traj)
        rendered = template.render(state)
        assert rendered is not None
