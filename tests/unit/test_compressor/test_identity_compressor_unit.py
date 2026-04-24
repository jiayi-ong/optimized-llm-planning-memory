"""Unit tests for compressor/identity_compressor.py — IdentityCompressor."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone

import pytest
import torch

from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
from optimized_llm_planning_memory.core.models import CompressedState, ReActStep, TrajectoryModel


def _make_trajectory(n_steps: int = 2) -> TrajectoryModel:
    steps = [
        ReActStep(
            step_index=i,
            thought=f"Thought {i}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        for i in range(n_steps)
    ]
    return TrajectoryModel(
        trajectory_id="traj-unit",
        request_id="req-unit",
        steps=steps,
        total_steps=n_steps,
    )


@pytest.mark.unit
class TestIdentityCompressorCompress:
    def setup_method(self):
        self.comp = IdentityCompressor()

    def test_compress_returns_compressed_state(self):
        traj = _make_trajectory(2)
        result = self.comp.compress(traj)
        assert isinstance(result, CompressedState)

    def test_compress_trajectory_id_matches(self):
        traj = _make_trajectory(2)
        result = self.comp.compress(traj)
        assert result.trajectory_id == "traj-unit"

    def test_compress_state_id_is_string(self):
        traj = _make_trajectory(2)
        result = self.comp.compress(traj)
        assert isinstance(result.state_id, str) and len(result.state_id) > 0

    def test_compress_method_is_identity(self):
        traj = _make_trajectory(2)
        result = self.comp.compress(traj)
        assert result.compression_method == "identity"

    def test_compress_different_calls_have_unique_state_ids(self):
        traj = _make_trajectory(2)
        r1 = self.comp.compress(traj)
        r2 = self.comp.compress(traj)
        assert r1.state_id != r2.state_id

    def test_compress_empty_trajectory_ok(self):
        traj = _make_trajectory(0)
        result = self.comp.compress(traj)
        assert isinstance(result, CompressedState)


@pytest.mark.unit
class TestIdentityCompressorLogProbs:
    def setup_method(self):
        self.comp = IdentityCompressor()

    def test_get_log_probs_returns_tensor(self):
        lp = self.comp.get_log_probs("input text", "output text")
        assert isinstance(lp, torch.Tensor)

    def test_get_log_probs_shape_matches_token_count(self):
        compressed = "word1 word2 word3"
        lp = self.comp.get_log_probs("input", compressed)
        assert lp.shape == (3,)

    def test_get_log_probs_has_grad_fn(self):
        lp = self.comp.get_log_probs("input text", "a b c d")
        assert lp.requires_grad or lp.grad_fn is not None

    def test_get_log_probs_single_token(self):
        lp = self.comp.get_log_probs("x", "onetoken")
        assert lp.shape == (1,)


@pytest.mark.unit
class TestIdentityCompressorTrainability:
    def setup_method(self):
        self.comp = IdentityCompressor()

    def test_is_trainable_true(self):
        assert self.comp.is_trainable() is True

    def test_get_trainable_parameters_has_one_param(self):
        params = self.comp.get_trainable_parameters()
        assert len(params) == 1
        assert isinstance(params[0], torch.nn.Parameter)

    def test_get_metadata_trainable_true(self):
        meta = self.comp.get_metadata()
        assert meta["trainable"] is True
        assert meta["type"] == "identity"


@pytest.mark.unit
class TestIdentityCompressorCheckpoint:
    def test_save_and_load_roundtrip(self):
        comp = IdentityCompressor()
        comp._dummy_param.data = torch.tensor([3.14])

        with tempfile.TemporaryDirectory() as tmpdir:
            comp.save_checkpoint(tmpdir)

            loaded = IdentityCompressor()
            loaded.load_checkpoint(tmpdir)

            assert torch.allclose(loaded._dummy_param.data, torch.tensor([3.14]))

    def test_load_missing_checkpoint_raises(self):
        from optimized_llm_planning_memory.core.exceptions import CompressorCheckpointError
        comp = IdentityCompressor()
        with pytest.raises(CompressorCheckpointError):
            comp.load_checkpoint("/nonexistent/path/to/checkpoint")
