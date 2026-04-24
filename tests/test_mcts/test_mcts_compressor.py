"""
tests/test_mcts/test_mcts_compressor.py
=========================================
Integration tests for MCTSAwareCompressor and LLMMCTSCompressor.

All LLM calls are mocked. The tests verify:
  1. compress_with_tree() returns a valid CompressedState.
  2. All 6 standard template sections are populated.
  3. top_candidates and tradeoffs optional fields are populated.
  4. compress() (non-MCTS fallback) works as a standard LLMCompressor.
  5. Template validation passes.
  6. Mode isolation: RAW/COMPRESSOR episodes produce mcts_stats=None.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from optimized_llm_planning_memory.compressor.mcts_aware import MCTSAwareCompressor
from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate
from optimized_llm_planning_memory.core.models import (
    CompressedState,
    HardConstraintLedger,
    TrajectoryModel,
)
from optimized_llm_planning_memory.mcts.node import MCTSStats, MCTSTreeRepresentation


# ── Helpers ───────────────────────────────────────────────────────────────────

TEMPLATE = CompressedStateTemplate()


def _make_trajectory() -> TrajectoryModel:
    return TrajectoryModel(
        trajectory_id=str(uuid.uuid4()),
        request_id="req-001",
        steps=(),
        total_steps=0,
    )


def _make_stats() -> MCTSStats:
    return MCTSStats(
        nodes_explored=5,
        max_depth_reached=2,
        num_simulations=3,
        best_path_length=2,
        root_value=0.65,
        avg_branching_factor=2.0,
    )


def _make_tree_repr() -> MCTSTreeRepresentation:
    return MCTSTreeRepresentation(
        best_path_trajectory=_make_trajectory(),
        alternative_paths=[_make_trajectory()],
        top_candidates=[
            "[Best] search_hotels({...}) (Q=0.700, depth=1)",
            "[Alt]  search_flights({...}) (Q=0.500)",
        ],
        tradeoffs="Top branch Q-values after 5 simulations:\n  1. Q=0.700 | hotel\n  2. Q=0.500 | flights",
        stats=_make_stats(),
    )


def _make_valid_llm_response():
    """Simulated structured response from instructor."""
    resp = MagicMock()
    resp.satisfied_constraint_ids = []
    resp.violated_constraint_ids = []
    resp.unknown_constraint_ids = ["hc_budget"]
    resp.soft_constraints_summary = "Prefers boutique hotels in city centre."
    resp.decisions_made = ["Searched flights Paris 2025-06-01"]
    resp.open_questions = ["Which hotel to book?"]
    resp.key_discoveries = ["Flights from $450 one-way"]
    resp.current_itinerary_sketch = "Day 1: arrive Paris, check in. Day 2: museums."
    resp.top_candidates = [
        "Option 1: Book Hotel Lumiere (Q=0.70)",
        "Option 2: Book Hotel Etoile (Q=0.55)",
    ]
    resp.tradeoffs = "Hotel Lumiere is pricier but closer to Louvre."
    return resp


# ── MCTSAwareCompressor abstract interface ────────────────────────────────────

class TestMCTSAwareCompressorABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            MCTSAwareCompressor()  # type: ignore[abstract]

    def test_subclass_must_implement_both_methods(self):
        """A subclass that only implements compress() should still raise on abstract."""

        class IncompleteCompressor(MCTSAwareCompressor):
            def compress(self, trajectory, previous_state=None):
                pass
            # Missing compress_with_tree() — should raise TypeError

        with pytest.raises(TypeError):
            IncompleteCompressor()  # type: ignore[abstract]


# ── LLMMCTSCompressor ─────────────────────────────────────────────────────────

class TestLLMMCTSCompressorCompressWithTree:
    """Tests for the compress_with_tree() path."""

    def _get_compressor(self):
        from optimized_llm_planning_memory.compressor.llm_mcts_compressor import LLMMCTSCompressor
        return LLMMCTSCompressor(llm_model_id="openai/gpt-4o-mini")

    @patch("optimized_llm_planning_memory.compressor.llm_mcts_compressor.instructor")
    def test_compress_with_tree_returns_compressed_state(self, mock_instructor):
        """compress_with_tree() should return a valid CompressedState."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_valid_llm_response()
        mock_instructor.from_litellm.return_value = mock_client

        compressor = self._get_compressor()
        compressor._client = mock_client  # inject mock directly

        tree_repr = _make_tree_repr()
        result = compressor.compress_with_tree(tree_repr, previous_state=None)

        assert isinstance(result, CompressedState)
        assert result.compression_method == "llm_mcts"

    @patch("optimized_llm_planning_memory.compressor.llm_mcts_compressor.instructor")
    def test_compress_with_tree_passes_template_validation(self, mock_instructor):
        """All 6 standard template sections must be non-empty."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_valid_llm_response()
        mock_instructor.from_litellm.return_value = mock_client

        compressor = self._get_compressor()
        compressor._client = mock_client

        result = compressor.compress_with_tree(_make_tree_repr())

        # Must not raise
        TEMPLATE.validate(result)

    @patch("optimized_llm_planning_memory.compressor.llm_mcts_compressor.instructor")
    def test_compress_with_tree_populates_mcts_fields(self, mock_instructor):
        """top_candidates and tradeoffs should be populated from LLM response."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_valid_llm_response()
        mock_instructor.from_litellm.return_value = mock_client

        compressor = self._get_compressor()
        compressor._client = mock_client

        result = compressor.compress_with_tree(_make_tree_repr())

        assert result.top_candidates is not None
        assert len(result.top_candidates) > 0
        assert result.tradeoffs is not None
        assert len(result.tradeoffs) > 0

    @patch("optimized_llm_planning_memory.compressor.llm_mcts_compressor.instructor")
    def test_compress_with_tree_sets_trajectory_id(self, mock_instructor):
        """state_id and trajectory_id should be populated."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_valid_llm_response()
        mock_instructor.from_litellm.return_value = mock_client

        compressor = self._get_compressor()
        compressor._client = mock_client

        tree_repr = _make_tree_repr()
        result = compressor.compress_with_tree(tree_repr)

        assert result.trajectory_id == tree_repr.best_path_trajectory.trajectory_id
        assert result.state_id  # non-empty UUID


# ── Non-MCTS fallback ─────────────────────────────────────────────────────────

class TestLLMMCTSCompressorFallback:
    """Tests for the compress() non-MCTS path."""

    @patch("optimized_llm_planning_memory.compressor.llm_compressor.instructor")
    def test_compress_delegates_to_llm_compressor(self, mock_instructor):
        """compress() without a tree should behave like LLMCompressor."""
        from optimized_llm_planning_memory.compressor.llm_mcts_compressor import LLMMCTSCompressor

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.satisfied_constraint_ids = []
        mock_response.violated_constraint_ids = []
        mock_response.unknown_constraint_ids = []
        mock_response.soft_constraints_summary = "No soft constraints."
        mock_response.decisions_made = ["Booked flight."]
        mock_response.open_questions = []
        mock_response.key_discoveries = ["Price: $500"]
        mock_response.current_itinerary_sketch = "Day 1: fly."
        mock_client.chat.completions.create.return_value = mock_response
        mock_instructor.from_litellm.return_value = mock_client

        compressor = LLMMCTSCompressor(llm_model_id="openai/gpt-4o-mini")
        traj = _make_trajectory()
        result = compressor.compress(traj, previous_state=None)

        assert isinstance(result, CompressedState)
        # Non-MCTS compression should leave top_candidates as None
        assert result.top_candidates is None
        assert result.tradeoffs is None


# ── Mode isolation ─────────────────────────────────────────────────────────────

class TestModeIsolation:
    """Verify that non-MCTS modes produce mcts_stats=None in EpisodeLog."""

    def test_episode_log_mcts_stats_none_for_compressor_mode(self, sample_episode_log):
        """Sample episode log (compressor mode) should have mcts_stats=None."""
        assert sample_episode_log.mcts_stats is None

    def test_episode_log_allows_mcts_stats(self):
        """EpisodeLog accepts mcts_stats when provided."""
        from optimized_llm_planning_memory.core.models import (
            EpisodeLog, RewardComponents, TrajectoryModel
        )

        stats = MCTSStats(
            nodes_explored=10, max_depth_reached=3, num_simulations=5,
            best_path_length=3, root_value=0.75, avg_branching_factor=2.0,
        )
        traj = TrajectoryModel(
            trajectory_id=str(uuid.uuid4()),
            request_id="req-001",
            steps=(),
            total_steps=0,
        )
        reward = RewardComponents(
            hard_constraint_score=0.0,
            soft_constraint_score=0.0,
            tool_efficiency_score=0.0,
            tool_failure_penalty=0.0,
            logical_consistency_score=0.0,
            total_reward=0.0,
        )
        log = EpisodeLog(
            episode_id=str(uuid.uuid4()),
            request_id="req-001",
            agent_mode="mcts_compressor",
            trajectory=traj,
            reward_components=reward,
            total_steps=0,
            success=True,
            config_hash="abc",
            created_at=datetime.now(timezone.utc).isoformat(),
            mcts_stats=stats,
        )
        assert log.mcts_stats is not None
        assert log.mcts_stats.nodes_explored == 10
        assert log.agent_mode == "mcts_compressor"
