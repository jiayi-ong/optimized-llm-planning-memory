"""Unit tests for tools/events.py — EventBus and ToolEvent."""

from __future__ import annotations

import pytest

from optimized_llm_planning_memory.tools.events import EventBus, ToolEvent


def _make_event(**kwargs) -> ToolEvent:
    defaults = dict(
        tool_name="search_flights",
        success=True,
        arguments_hash="abc123",
        result={"flights": []},
        error=None,
        latency_ms=10.0,
    )
    defaults.update(kwargs)
    return ToolEvent(**defaults)


@pytest.mark.unit
class TestToolEvent:
    def test_frozen(self):
        event = _make_event()
        with pytest.raises(Exception):
            event.tool_name = "other"  # type: ignore[misc]

    def test_timestamp_auto_set(self):
        event = _make_event()
        assert isinstance(event.timestamp, str)
        assert len(event.timestamp) > 0


@pytest.mark.unit
class TestEventBus:
    def test_subscribe_and_emit(self):
        bus = EventBus()
        received = []
        bus.subscribe("tool_call", received.append)
        bus.emit(_make_event())
        assert len(received) == 1

    def test_wildcard_receives_all(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", received.append)
        bus.emit(_make_event(), event_type="tool_call")
        bus.emit(_make_event(), event_type="other_event")
        assert len(received) == 2

    def test_specific_handler_not_called_for_other_type(self):
        bus = EventBus()
        received = []
        bus.subscribe("tool_call", received.append)
        bus.emit(_make_event(), event_type="other_event")
        assert len(received) == 0

    def test_unsubscribe_removes_handler(self):
        bus = EventBus()
        received = []
        handler = received.append  # store reference so identity is preserved
        bus.subscribe("tool_call", handler)
        bus.unsubscribe("tool_call", handler)
        bus.emit(_make_event())
        assert len(received) == 0

    def test_handler_exception_does_not_propagate(self):
        bus = EventBus()

        def bad_handler(event):
            raise RuntimeError("handler crash")

        bus.subscribe("tool_call", bad_handler)
        # Should not raise
        bus.emit(_make_event())

    def test_clear_removes_all_subscriptions(self):
        bus = EventBus()
        received = []
        bus.subscribe("tool_call", received.append)
        bus.subscribe("*", received.append)
        bus.clear()
        bus.emit(_make_event())
        assert len(received) == 0

    def test_multiple_handlers_all_called(self):
        bus = EventBus()
        counts = [0, 0]
        bus.subscribe("tool_call", lambda e: counts.__setitem__(0, counts[0] + 1))
        bus.subscribe("tool_call", lambda e: counts.__setitem__(1, counts[1] + 1))
        bus.emit(_make_event())
        assert counts == [1, 1]
