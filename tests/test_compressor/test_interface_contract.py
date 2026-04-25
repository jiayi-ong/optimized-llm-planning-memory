"""
tests/test_compressor/test_interface_contract.py
=================================================
T4 — Verify that the CompressorBase / TrainableCompressorBase interface is
enforced at class-definition time, not silently at runtime.

A new compressor that forgets to implement compress() should fail at class
definition, not survive until a training run discovers the missing method.

We also verify that TrainableCompressorBase imposes get_log_probs() as an
abstract method, so trainable compressors can't skip it.
"""

from __future__ import annotations

import pytest

from optimized_llm_planning_memory.compressor.base import CompressorBase
from optimized_llm_planning_memory.compressor.trainable_base import TrainableCompressorBase
from optimized_llm_planning_memory.core.models import CompressedState, TrajectoryModel


@pytest.mark.unit
class TestCompressorBaseContract:
    def test_cannot_instantiate_without_compress(self):
        """Concrete subclass of CompressorBase missing compress() must raise TypeError."""
        with pytest.raises(TypeError):
            class MissingCompress(CompressorBase):
                pass
            MissingCompress()  # instantiation should fail

    def test_can_instantiate_with_compress_implemented(self):
        """Subclass providing compress() can be instantiated."""
        class MinimalCompressor(CompressorBase):
            def compress(self, trajectory, previous_state=None):
                return CompressedState(
                    compressed_id="test",
                    request_id=trajectory.request_id,
                    step_index=0,
                    summary="",
                    key_decisions=[],
                    active_constraints=[],
                    explored_options=[],
                    next_actions=[],
                )

        comp = MinimalCompressor()
        assert comp is not None

    def test_default_get_log_probs_raises(self):
        """Non-trainable compressor should raise on get_log_probs()."""
        from optimized_llm_planning_memory.core.exceptions import LogProbsNotSupportedError

        class NonTrainableCompressor(CompressorBase):
            def compress(self, trajectory, previous_state=None):
                return CompressedState(
                    compressed_id="x", request_id="r", step_index=0,
                    summary="", key_decisions=[], active_constraints=[],
                    explored_options=[], next_actions=[],
                )

        comp = NonTrainableCompressor()
        with pytest.raises(LogProbsNotSupportedError):
            comp.get_log_probs("trajectory text", "compressed text")

    def test_default_is_trainable_returns_false(self):
        class NonTrainableCompressor(CompressorBase):
            def compress(self, trajectory, previous_state=None):
                return CompressedState(
                    compressed_id="x", request_id="r", step_index=0,
                    summary="", key_decisions=[], active_constraints=[],
                    explored_options=[], next_actions=[],
                )

        assert NonTrainableCompressor().is_trainable() is False

    def test_default_get_trainable_parameters_empty(self):
        class NonTrainableCompressor(CompressorBase):
            def compress(self, trajectory, previous_state=None):
                return CompressedState(
                    compressed_id="x", request_id="r", step_index=0,
                    summary="", key_decisions=[], active_constraints=[],
                    explored_options=[], next_actions=[],
                )

        assert NonTrainableCompressor().get_trainable_parameters() == []


@pytest.mark.unit
class TestTrainableCompressorBaseContract:
    def test_cannot_instantiate_without_get_log_probs(self):
        """TrainableCompressorBase without get_log_probs() must raise TypeError."""
        with pytest.raises(TypeError):
            class MissingLogProbs(TrainableCompressorBase):
                def compress(self, trajectory, previous_state=None):
                    pass
                # Missing: get_log_probs, get_trainable_parameters, save_checkpoint, load_checkpoint
            MissingLogProbs()

    def test_is_trainable_returns_true(self):
        """TrainableCompressorBase returns is_trainable()==True when it has parameters."""
        import torch
        import torch.nn as nn

        class StubTrainableCompressor(TrainableCompressorBase):
            def __init__(self):
                self._param = nn.Parameter(torch.zeros(1))

            def compress(self, trajectory, previous_state=None):
                return CompressedState(
                    compressed_id="t", request_id="r", step_index=0,
                    summary="", key_decisions=[], active_constraints=[],
                    explored_options=[], next_actions=[],
                )

            def get_log_probs(self, trajectory_text, compressed_text):
                return torch.zeros(1)

            def get_trainable_parameters(self):
                return [self._param]

            def save_checkpoint(self, path):
                pass

            def load_checkpoint(self, path):
                pass

        comp = StubTrainableCompressor()
        assert comp.is_trainable() is True
