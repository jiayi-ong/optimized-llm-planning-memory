"""
tools/events.py
===============
ToolEvent dataclass and EventBus — lightweight in-process pub/sub.

Design pattern: Observer / Event Bus
-------------------------------------
When a tool call completes (success or failure), ``BaseTool.call()`` emits
a ``ToolEvent`` to the ``EventBus``. Multiple subscribers receive the event
independently:

  - ``ToolCallTracker`` records metrics (count, latency, success/failure).
  - Structured logger writes to the episode log.
  - ``RewardFunction`` can subscribe to receive real-time penalty signals.

This decouples the tool execution path from all metric-collection concerns.
Adding a new subscriber (e.g., a live dashboard) requires zero changes to
``BaseTool`` or any concrete tool.

The bus is scoped per-episode: a new ``EventBus`` is created at the start of
``ReActAgent.run_episode()`` and discarded at the end, so there is no
cross-episode state leakage.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolEvent:
    """
    Immutable record of a single tool invocation outcome.

    Emitted by ``BaseTool.call()`` regardless of success or failure.
    Subscribers receive this object via their registered handler callable.
    """
    tool_name: str
    success: bool
    arguments_hash: str = field(
        metadata={"description": "MD5 of (tool_name, sorted_args_json); used for deduplication."}
    )
    result: object | None
    error: str | None
    latency_ms: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


# Type alias for handler callables
EventHandler = Callable[[ToolEvent], None]


class EventBus:
    """
    Simple synchronous in-process publish/subscribe bus.

    Subscribers register handlers for named event types. The special
    wildcard ``"*"`` receives every event regardless of type.

    This is intentionally simple — no threading, no queuing, no persistence.
    For this project, events are consumed synchronously within the episode loop.

    Example
    -------
        bus = EventBus()

        def log_event(event: ToolEvent) -> None:
            print(f"[{event.tool_name}] success={event.success}")

        bus.subscribe("tool_call", log_event)
        bus.subscribe("*", tracker.on_event)   # tracker receives everything

        bus.emit(ToolEvent(tool_name="search_flights", ...))
    """

    WILDCARD = "*"

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Register ``handler`` to be called whenever an event of ``event_type`` is emitted.

        Parameters
        ----------
        event_type : Event category string, or ``"*"`` for all events.
        handler    : Callable that accepts a single ``ToolEvent`` argument.
        """
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a previously registered handler."""
        self._handlers[event_type] = [
            h for h in self._handlers[event_type] if h is not handler
        ]

    def emit(self, event: ToolEvent, event_type: str = "tool_call") -> None:
        """
        Deliver ``event`` to all handlers registered for ``event_type`` and to
        all wildcard handlers.

        Handler exceptions are caught and logged but do not interrupt delivery
        to remaining handlers.

        Parameters
        ----------
        event      : The ``ToolEvent`` to deliver.
        event_type : Category string (default ``'tool_call'``).
        """
        handlers = self._handlers.get(event_type, []) + self._handlers.get(self.WILDCARD, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "EventBus handler %s raised an exception for event %s",
                    handler,
                    event_type,
                )

    def clear(self) -> None:
        """Remove all subscriptions. Called at episode teardown."""
        self._handlers.clear()
