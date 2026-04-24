"""Unit tests for compressor/base.py — CompressorBase ABC."""

from __future__ import annotations

import pytest

from optimized_llm_planning_memory.compressor.base import CompressorBase
from optimized_llm_planning_memory.core.exceptions import LogProbsNotSupportedError
from optimized_llm_planning_memory.core.models import CompressedState, TrajectoryModel


# ── Minimal concrete stub (not trainable) ────────────────────────────────────

class _StubCompressor(CompressorBase):
    def compress(
        self,
        trajectory: TrajectoryModel,
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        from datetime import datetime, timezone
        from optimized_llm_planning_memory.core.models import HardConstraintLedger
        ledger = HardConstraintLedger(
            constraints=(), satisfied_ids=(), violated_ids=(), unknown_ids=()
        )
        return CompressedState(
            state_id="stub-cs",
            trajectory_id=trajectory.trajectory_id,
            step_index=trajectory.total_steps,
            hard_constraint_ledger=ledger,
            soft_constraints_summary="stub",
            decisions_made=[],
            open_questions=[],
            key_discoveries=[],
            current_itinerary_sketch="stub sketch",
            compression_method="stub",
            token_count=2,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


@pytest.mark.unit
class TestCompressorBaseAbstract:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            CompressorBase()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        comp = _StubCompressor()
        assert comp is not None


@pytest.mark.unit
class TestCompressorBaseDefaults:
    def setup_method(self):
        self.comp = _StubCompressor()

    def test_get_log_probs_raises_not_supported(self):
        with pytest.raises(LogProbsNotSupportedError):
            self.comp.get_log_probs("trajectory text", "compressed text")

    def test_get_trainable_parameters_returns_empty_list(self):
        params = self.comp.get_trainable_parameters()
        assert params == []

    def test_is_trainable_returns_false(self):
        assert self.comp.is_trainable() is False

    def test_get_metadata_has_type_key(self):
        meta = self.comp.get_metadata()
        assert "type" in meta
        assert isinstance(meta["type"], str)

    def test_get_metadata_has_param_count(self):
        meta = self.comp.get_metadata()
        assert "param_count" in meta
        assert meta["param_count"] == 0

    def test_get_metadata_has_trainable_key(self):
        meta = self.comp.get_metadata()
        assert "trainable" in meta
        assert meta["trainable"] is False
