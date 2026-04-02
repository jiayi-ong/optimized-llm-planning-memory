"""
tests/test_tools/test_tracker.py
==================================
Unit tests for ToolCallTracker: redundancy detection, thread safety.
"""

from __future__ import annotations

import threading

import pytest

from optimized_llm_planning_memory.tools.tracker import ToolCallTracker


@pytest.fixture
def tracker() -> ToolCallTracker:
    return ToolCallTracker()


def test_record_increments_call_count(tracker):
    tracker.record("search_flights", success=True, latency_ms=50.0, arguments_hash="abc")
    stats = {s.tool_name: s for s in tracker.get_stats()}
    assert stats["search_flights"].call_count == 1
    assert stats["search_flights"].success_count == 1
    assert stats["search_flights"].failure_count == 0


def test_duplicate_call_increments_redundancy(tracker):
    tracker.record("search_flights", success=True, latency_ms=50.0, arguments_hash="abc")
    tracker.record("search_flights", success=True, latency_ms=48.0, arguments_hash="abc")
    stats = {s.tool_name: s for s in tracker.get_stats()}
    assert stats["search_flights"].call_count == 2
    assert stats["search_flights"].redundant_call_count == 1


def test_different_args_not_redundant(tracker):
    tracker.record("search_flights", success=True, latency_ms=50.0, arguments_hash="abc")
    tracker.record("search_flights", success=True, latency_ms=50.0, arguments_hash="xyz")
    stats = {s.tool_name: s for s in tracker.get_stats()}
    assert stats["search_flights"].redundant_call_count == 0


def test_failure_count(tracker):
    tracker.record("book_hotel", success=False, latency_ms=10.0, arguments_hash="def")
    stats = {s.tool_name: s for s in tracker.get_stats()}
    assert stats["book_hotel"].failure_count == 1


def test_reset_clears_all(tracker):
    tracker.record("search_flights", success=True, latency_ms=50.0, arguments_hash="abc")
    tracker.reset()
    assert tracker.get_stats() == []


def test_thread_safety(tracker):
    """Recording from multiple threads should not raise or lose data."""
    def record_many():
        for _ in range(100):
            tracker.record("search_hotels", success=True, latency_ms=1.0, arguments_hash="x")

    threads = [threading.Thread(target=record_many) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = {s.tool_name: s for s in tracker.get_stats()}
    assert stats["search_hotels"].call_count == 400
