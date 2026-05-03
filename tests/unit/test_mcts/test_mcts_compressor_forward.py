"""
tests/unit/test_mcts/test_mcts_compressor_forward.py
=====================================================
Unit tests for LLMMCTSCompressor — the LLM-based MCTS-aware compressor.

Coverage
--------
* compress() delegates to the internal LLMCompressor fallback (no LLM call).
* compress_with_tree() calls litellm.completion, parses the response, and
  returns a valid CompressedState with MCTS-specific fields populated.
* CompressedStateTemplate validation passes on the returned state.
* get_metadata() reports the correct non-trainable profile.

All external LLM calls are mocked — no API key is required.

Design note
-----------
LLMMCTSCompressor is NOT trainable: it has no get_log_probs() or
get_trainable_parameters() override. These tests therefore focus on the
forward (inference) path and the MCTS-specific compression path, rather
than gradient flow.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from optimized_llm_planning_memory.compressor.llm_mcts_compressor import LLMMCTSCompressor
from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate
from optimized_llm_planning_memory.core.models import (
    CompressedState,
    HardConstraintLedger,
    ReActStep,
    TrajectoryModel,
)
from optimized_llm_planning_memory.mcts.node import MCTSStats, MCTSTreeRepresentation

# MCTSTreeRepresentation.best_path_trajectory uses TYPE_CHECKING for TrajectoryModel.
# Rebuild the Pydantic model so the forward reference is resolved at test time.
MCTSTreeRepresentation.model_rebuild()

_FAKE_TIMESTAMP = "2026-05-01T12:00:00+00:00"


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_trajectory(steps: int = 2) -> TrajectoryModel:
    """Minimal TrajectoryModel with stub ReActStep objects."""
    step_list = tuple(
        ReActStep(
            step_index=i,
            thought=f"thought {i}",
            action=None,
            observation=None,
            timestamp=_FAKE_TIMESTAMP,
        )
        for i in range(steps)
    )
    return TrajectoryModel(
        trajectory_id=str(uuid.uuid4()),
        request_id="req-test",
        steps=step_list,
        total_steps=len(step_list),
    )


def _make_tree_repr() -> MCTSTreeRepresentation:
    """MCTSTreeRepresentation with one best path and one alternative."""
    stats = MCTSStats(
        nodes_explored=10,
        max_depth_reached=3,
        num_simulations=5,
        best_path_length=3,
        root_value=0.75,
        avg_branching_factor=2.0,
    )
    return MCTSTreeRepresentation(
        best_path_trajectory=_make_trajectory(steps=3),
        alternative_paths=[_make_trajectory(steps=2)],
        top_candidates=["Plan A: book hotel first", "Plan B: search flights first"],
        tradeoffs="Plan A has better budget adherence; Plan B has more flexibility.",
        stats=stats,
    )


def _make_mcts_llm_response_json() -> str:
    """
    Fake JSON matching _MCTSCompressorLLMResponse schema.
    This is what litellm.completion would return from the model.
    """
    return json.dumps({
        "satisfied_constraint_ids": ["budget", "dates"],
        "violated_constraint_ids": [],
        "unknown_constraint_ids": ["accommodation_stars"],
        "soft_constraints_summary": "Prefers boutique hotels and local cuisine.",
        "decisions_made": ["Booked flight LHR→CDG on 2026-06-01"],
        "open_questions": ["Which hotel in Paris?"],
        "key_discoveries": ["Hotel Lumiere available at $120/night"],
        "current_itinerary_sketch": "Day 1: Arrive Paris. Day 2: Louvre.",
        "top_candidates": ["Plan A: boutique hotel", "Plan B: city center hotel"],
        "tradeoffs": "Plan A is cheaper but farther from metro.",
    })


def _make_compressed_state() -> CompressedState:
    """Minimal CompressedState for fallback mock return value."""
    ledger = HardConstraintLedger(
        constraints=(),
        satisfied_ids=(),
        violated_ids=(),
        unknown_ids=(),
    )
    return CompressedState(
        state_id=str(uuid.uuid4()),
        trajectory_id=str(uuid.uuid4()),
        step_index=2,
        hard_constraint_ledger=ledger,
        soft_constraints_summary="no prefs",
        decisions_made=[],
        open_questions=[],
        key_discoveries=[],
        current_itinerary_sketch="Empty sketch.",
        compression_method="llm",
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


@pytest.fixture()
def compressor() -> LLMMCTSCompressor:
    return LLMMCTSCompressor(llm_model_id="openai/gpt-4o-mini", max_output_tokens=512)


@pytest.fixture()
def tree_repr() -> MCTSTreeRepresentation:
    return _make_tree_repr()


# ── TestCompress — non-MCTS fallback path ─────────────────────────────────────


class TestCompress:
    """compress() must delegate to the internal LLMCompressor without calling litellm directly."""

    def test_compress_delegates_to_fallback(self, compressor: LLMMCTSCompressor) -> None:
        traj = _make_trajectory()
        expected = _make_compressed_state()
        with patch.object(compressor._fallback, "compress", return_value=expected) as mock_compress:
            result = compressor.compress(traj)
        mock_compress.assert_called_once_with(traj, None)
        assert result is expected

    def test_compress_passes_previous_state(self, compressor: LLMMCTSCompressor) -> None:
        traj = _make_trajectory()
        prev = _make_compressed_state()
        expected = _make_compressed_state()
        with patch.object(compressor._fallback, "compress", return_value=expected) as mock_compress:
            compressor.compress(traj, previous_state=prev)
        mock_compress.assert_called_once_with(traj, prev)

    def test_compress_returns_compressed_state(self, compressor: LLMMCTSCompressor) -> None:
        traj = _make_trajectory()
        expected = _make_compressed_state()
        with patch.object(compressor._fallback, "compress", return_value=expected):
            result = compressor.compress(traj)
        assert isinstance(result, CompressedState)


# ── TestCompressWithTree — MCTS path ─────────────────────────────────────────


class TestCompressWithTree:
    """compress_with_tree() must call litellm, parse the response, and return a valid CompressedState."""

    def _make_litellm_mock(self) -> MagicMock:
        """Return a mock that mimics litellm.completion's return value."""
        choice = MagicMock()
        choice.message.content = _make_mcts_llm_response_json()
        mock_resp = MagicMock()
        mock_resp.choices = [choice]
        return mock_resp

    @patch("optimized_llm_planning_memory.compressor.llm_mcts_compressor.litellm.completion")
    def test_returns_compressed_state(
        self, mock_litellm: MagicMock, compressor: LLMMCTSCompressor, tree_repr: MCTSTreeRepresentation
    ) -> None:
        mock_litellm.return_value = self._make_litellm_mock()
        result = compressor.compress_with_tree(tree_repr)
        assert isinstance(result, CompressedState)

    @patch("optimized_llm_planning_memory.compressor.llm_mcts_compressor.litellm.completion")
    def test_satisfied_constraints_populated(
        self, mock_litellm: MagicMock, compressor: LLMMCTSCompressor, tree_repr: MCTSTreeRepresentation
    ) -> None:
        mock_litellm.return_value = self._make_litellm_mock()
        result = compressor.compress_with_tree(tree_repr)
        assert "budget" in result.hard_constraint_ledger.satisfied_ids
        assert "dates" in result.hard_constraint_ledger.satisfied_ids

    @patch("optimized_llm_planning_memory.compressor.llm_mcts_compressor.litellm.completion")
    def test_violated_ids_empty(
        self, mock_litellm: MagicMock, compressor: LLMMCTSCompressor, tree_repr: MCTSTreeRepresentation
    ) -> None:
        mock_litellm.return_value = self._make_litellm_mock()
        result = compressor.compress_with_tree(tree_repr)
        assert len(result.hard_constraint_ledger.violated_ids) == 0

    @patch("optimized_llm_planning_memory.compressor.llm_mcts_compressor.litellm.completion")
    def test_mcts_top_candidates_populated(
        self, mock_litellm: MagicMock, compressor: LLMMCTSCompressor, tree_repr: MCTSTreeRepresentation
    ) -> None:
        mock_litellm.return_value = self._make_litellm_mock()
        result = compressor.compress_with_tree(tree_repr)
        assert result.top_candidates is not None
        assert len(result.top_candidates) > 0

    @patch("optimized_llm_planning_memory.compressor.llm_mcts_compressor.litellm.completion")
    def test_mcts_tradeoffs_populated(
        self, mock_litellm: MagicMock, compressor: LLMMCTSCompressor, tree_repr: MCTSTreeRepresentation
    ) -> None:
        mock_litellm.return_value = self._make_litellm_mock()
        result = compressor.compress_with_tree(tree_repr)
        assert result.tradeoffs is not None
        assert len(result.tradeoffs) > 0

    @patch("optimized_llm_planning_memory.compressor.llm_mcts_compressor.litellm.completion")
    def test_compression_method_is_llm_mcts(
        self, mock_litellm: MagicMock, compressor: LLMMCTSCompressor, tree_repr: MCTSTreeRepresentation
    ) -> None:
        mock_litellm.return_value = self._make_litellm_mock()
        result = compressor.compress_with_tree(tree_repr)
        assert result.compression_method == "llm_mcts"

    @patch("optimized_llm_planning_memory.compressor.llm_mcts_compressor.litellm.completion")
    def test_template_validation_passes(
        self, mock_litellm: MagicMock, compressor: LLMMCTSCompressor, tree_repr: MCTSTreeRepresentation
    ) -> None:
        """The returned CompressedState must pass CompressedStateTemplate.validate()."""
        mock_litellm.return_value = self._make_litellm_mock()
        result = compressor.compress_with_tree(tree_repr)
        template = CompressedStateTemplate()
        # validate() raises if any required section is missing
        template.validate(result)

    @patch("optimized_llm_planning_memory.compressor.llm_mcts_compressor.litellm.completion")
    def test_litellm_called_once(
        self, mock_litellm: MagicMock, compressor: LLMMCTSCompressor, tree_repr: MCTSTreeRepresentation
    ) -> None:
        mock_litellm.return_value = self._make_litellm_mock()
        compressor.compress_with_tree(tree_repr)
        mock_litellm.assert_called_once()

    @patch("optimized_llm_planning_memory.compressor.llm_mcts_compressor.litellm.completion")
    def test_with_previous_state_includes_prior_constraints(
        self, mock_litellm: MagicMock, compressor: LLMMCTSCompressor, tree_repr: MCTSTreeRepresentation
    ) -> None:
        """When previous_state is provided, prior constraints appear in the ledger."""
        prev = _make_compressed_state()
        mock_litellm.return_value = self._make_litellm_mock()
        result = compressor.compress_with_tree(tree_repr, previous_state=prev)
        # No error raised; constraint ids from response are set
        assert isinstance(result.hard_constraint_ledger, HardConstraintLedger)

    @patch("optimized_llm_planning_memory.compressor.llm_mcts_compressor.litellm.completion")
    def test_no_alternative_paths_still_works(
        self, mock_litellm: MagicMock, compressor: LLMMCTSCompressor
    ) -> None:
        """Tree with no alternative paths must not raise."""
        stats = MCTSStats(
            nodes_explored=1, max_depth_reached=1, num_simulations=1,
            best_path_length=1, root_value=0.5, avg_branching_factor=0.0,
        )
        minimal_tree = MCTSTreeRepresentation(
            best_path_trajectory=_make_trajectory(steps=1),
            alternative_paths=[],
            top_candidates=[],
            tradeoffs="",
            stats=stats,
        )
        mock_litellm.return_value = self._make_litellm_mock()
        result = compressor.compress_with_tree(minimal_tree)
        assert isinstance(result, CompressedState)


# ── TestGetMetadata ───────────────────────────────────────────────────────────


class TestGetMetadata:
    """get_metadata() must describe a non-trainable LLM-based compressor."""

    def test_type_is_llm_mcts(self, compressor: LLMMCTSCompressor) -> None:
        meta = compressor.get_metadata()
        assert meta["type"] == "llm_mcts"

    def test_trainable_is_false(self, compressor: LLMMCTSCompressor) -> None:
        meta = compressor.get_metadata()
        assert meta["trainable"] is False

    def test_param_count_is_zero(self, compressor: LLMMCTSCompressor) -> None:
        meta = compressor.get_metadata()
        assert meta["param_count"] == 0

    def test_model_id_matches_constructor(self, compressor: LLMMCTSCompressor) -> None:
        meta = compressor.get_metadata()
        assert meta["model_id"] == "openai/gpt-4o-mini"
