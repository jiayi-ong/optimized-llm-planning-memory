"""
tools/tracker.py
================
ToolCallTracker — thread-safe usage recorder for tool middleware.

Design: Observer subscriber + shared state
------------------------------------------
``ToolCallTracker`` subscribes to the ``EventBus`` as a handler and records
per-tool statistics. It is also polled directly by:

  - ``RewardFunction._tool_efficiency_score()`` (at reward computation time)
  - ``EpisodeLog`` construction (at episode end)

Thread safety
-------------
All public methods are protected by a ``threading.Lock``. This is necessary
when ``n_envs > 1`` in the gymnasium VecEnv and workers share a tracker
(they should not; each episode should have its own tracker instance).
The lock is a safety net for unexpected sharing.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict
from time import perf_counter

from optimized_llm_planning_memory.core.models import ToolCallStats
from optimized_llm_planning_memory.tools.events import ToolEvent


class ToolCallTracker:
    """
    Records per-tool usage statistics across a single episode.

    Create one instance per episode; reset between episodes via ``reset()``.
    Subscribe to the ``EventBus`` to receive events automatically::

        tracker = ToolCallTracker()
        bus.subscribe("*", tracker.on_event)

    Or call ``record()`` manually if not using the event bus.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[str, list[_CallRecord]] = defaultdict(list)

    # ── EventBus subscriber ───────────────────────────────────────────────────

    def on_event(self, event: ToolEvent) -> None:
        """EventBus handler. Called automatically on each tool invocation."""
        self.record(
            tool_name=event.tool_name,
            success=event.success,
            latency_ms=event.latency_ms,
            arguments_hash=event.arguments_hash,
        )

    # ── Direct API ───────────────────────────────────────────────────────────

    def record(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float,
        arguments_hash: str,
    ) -> None:
        """
        Record a single tool invocation.

        Parameters
        ----------
        tool_name       : Name of the tool that was called.
        success         : Whether the call succeeded.
        latency_ms      : Wall-clock latency in milliseconds.
        arguments_hash  : MD5 of the serialised arguments (for deduplication).
        """
        with self._lock:
            self._calls[tool_name].append(
                _CallRecord(success=success, latency_ms=latency_ms, args_hash=arguments_hash)
            )

    def get_stats(self) -> list[ToolCallStats]:
        """Return per-tool aggregate statistics as a list of ToolCallStats."""
        with self._lock:
            stats = []
            for tool_name, records in self._calls.items():
                total = len(records)
                successes = sum(1 for r in records if r.success)
                total_latency = sum(r.latency_ms for r in records)
                redundant = self._count_redundant(records)
                stats.append(
                    ToolCallStats(
                        tool_name=tool_name,
                        call_count=total,
                        success_count=successes,
                        failure_count=total - successes,
                        total_latency_ms=total_latency,
                        avg_latency_ms=total_latency / total if total > 0 else 0.0,
                        redundant_call_count=redundant,
                    )
                )
            return sorted(stats, key=lambda s: s.tool_name)

    def call_count_for_hash(self, tool_name: str, args_hash: str) -> int:
        """Return how many times this exact (tool_name, args_hash) pair has been recorded."""
        with self._lock:
            return sum(1 for r in self._calls.get(tool_name, []) if r.args_hash == args_hash)

    def get_redundancy_count(self) -> int:
        """Return total number of calls that duplicated a previous (name, args) pair."""
        with self._lock:
            return sum(
                self._count_redundant(records) for records in self._calls.values()
            )

    def get_total_failures(self) -> int:
        """Return total number of failed tool calls across all tools."""
        with self._lock:
            return sum(
                sum(1 for r in records if not r.success)
                for records in self._calls.values()
            )

    def reset(self) -> None:
        """Clear all recorded data. Call at the start of a new episode."""
        with self._lock:
            self._calls.clear()

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def hash_arguments(tool_name: str, arguments: dict) -> str:
        """Compute a stable MD5 hash of (tool_name, arguments) for deduplication."""
        key = json.dumps({"tool": tool_name, "args": arguments}, sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()

    @staticmethod
    def _count_redundant(records: list["_CallRecord"]) -> int:
        """Count records whose args_hash appeared in a previous record."""
        seen: set[str] = set()
        redundant = 0
        for record in records:
            if record.args_hash in seen:
                redundant += 1
            else:
                seen.add(record.args_hash)
        return redundant


class _CallRecord:
    """Internal lightweight record; not part of the public API."""
    __slots__ = ("success", "latency_ms", "args_hash")

    def __init__(self, success: bool, latency_ms: float, args_hash: str) -> None:
        self.success = success
        self.latency_ms = latency_ms
        self.args_hash = args_hash


class EpisodeTimer:
    """
    Utility context manager for measuring tool latency.

    Usage::

        with EpisodeTimer() as t:
            result = simulator.search_flights(...)
        latency_ms = t.elapsed_ms
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "EpisodeTimer":
        self._start = perf_counter()
        return self

    def __exit__(self, *_) -> None:  # type: ignore[override]
        self.elapsed_ms = (perf_counter() - self._start) * 1000.0
