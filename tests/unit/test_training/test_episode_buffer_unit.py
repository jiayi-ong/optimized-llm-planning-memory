"""Unit tests for training/episode_buffer.py — EpisodeBuffer."""

from __future__ import annotations

import pytest

from optimized_llm_planning_memory.core.models import PPOTransition
from optimized_llm_planning_memory.training.episode_buffer import EpisodeBuffer


def _make_transition(idx: int = 0) -> PPOTransition:
    return PPOTransition(
        trajectory_text=f"trajectory {idx}",
        compressed_state_text=f"compressed {idx}",
        reward=float(idx),
        value_estimate=0.5,
        log_prob=-0.3,
        advantage=None,
    )


@pytest.mark.unit
class TestEpisodeBufferAddAndLen:
    def test_empty_buffer_len_zero(self):
        buf = EpisodeBuffer()
        assert len(buf) == 0

    def test_empty_buffer_is_empty(self):
        buf = EpisodeBuffer()
        assert buf.is_empty()

    def test_add_increments_len(self):
        buf = EpisodeBuffer()
        buf.add(_make_transition(0))
        assert len(buf) == 1
        buf.add(_make_transition(1))
        assert len(buf) == 2

    def test_not_empty_after_add(self):
        buf = EpisodeBuffer()
        buf.add(_make_transition(0))
        assert not buf.is_empty()


@pytest.mark.unit
class TestEpisodeBufferFillAdvantages:
    def test_fill_advantages_wrong_length_raises(self):
        buf = EpisodeBuffer()
        buf.add(_make_transition(0))
        with pytest.raises(ValueError):
            buf.fill_advantages([1.0, 2.0])

    def test_fill_advantages_empty_list_for_empty_buffer(self):
        buf = EpisodeBuffer()
        buf.fill_advantages([])  # should not raise

    def test_fill_advantages_sets_correctly(self):
        buf = EpisodeBuffer()
        buf.add(_make_transition(0))
        buf.add(_make_transition(1))
        buf.fill_advantages([1.5, 2.5])
        batches = list(buf.minibatches(batch_size=10, shuffle=False))
        advantages = [t.advantage for t in batches[0]]
        assert set(advantages) == {1.5, 2.5}


@pytest.mark.unit
class TestEpisodeBufferMinibatches:
    def test_minibatches_covers_all_transitions(self):
        buf = EpisodeBuffer()
        for i in range(10):
            buf.add(_make_transition(i))
        buf.fill_advantages([float(i) for i in range(10)])

        total = sum(len(b) for b in buf.minibatches(batch_size=3, shuffle=False))
        assert total == 10

    def test_minibatches_batch_size_respected(self):
        buf = EpisodeBuffer()
        for i in range(20):
            buf.add(_make_transition(i))
        buf.fill_advantages([0.0] * 20)

        batch_sizes = [len(b) for b in buf.minibatches(batch_size=4, shuffle=False)]
        for size in batch_sizes[:-1]:
            assert size == 4

    def test_minibatches_yields_all_items(self):
        buf = EpisodeBuffer()
        for i in range(7):
            buf.add(_make_transition(i))
        buf.fill_advantages([0.0] * 7)

        total = sum(len(b) for b in buf.minibatches(batch_size=3, shuffle=False))
        assert total == 7


@pytest.mark.unit
class TestEpisodeBufferClear:
    def test_clear_empties_buffer(self):
        buf = EpisodeBuffer()
        for i in range(5):
            buf.add(_make_transition(i))
        buf.clear()
        assert len(buf) == 0
        assert buf.is_empty()

    def test_can_add_after_clear(self):
        buf = EpisodeBuffer()
        buf.add(_make_transition(0))
        buf.clear()
        buf.add(_make_transition(1))
        assert len(buf) == 1
