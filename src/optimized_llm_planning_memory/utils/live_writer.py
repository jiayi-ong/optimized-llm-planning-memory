"""
utils/live_writer.py
=====================
LiveEpisodeWriter — streams incremental episode events to a JSONL file.

Design
------
The Streamlit developer UI needs to display episode progress in near-real-time
without waiting for the full ``EpisodeLog`` to be written at episode end.
``LiveEpisodeWriter`` solves this by writing one JSON object per line to a
"live" JSONL file inside ``outputs/episodes/live/``, flushing after every
write so the UI can poll the file and see new events within ~1 second.

Two channels, different consumers
----------------------------------
This is intentionally separate from structlog:
- structlog events → operator log aggregation / monitoring pipelines
- LiveEpisodeWriter events → the local developer UI via file polling

Keeping them separate avoids coupling the UI to the logging configuration.

Usage
-----
    from optimized_llm_planning_memory.utils.live_writer import LiveEpisodeWriter

    with LiveEpisodeWriter(episode_id, output_dir="outputs/episodes") as writer:
        for step in episode:
            writer.write_step(step)
            writer.write_itinerary_update(itinerary)  # if changed
        writer.write_episode_complete(episode_id)

Or pass ``live_writer`` into ``ReActAgent.run_episode()``, which calls the
writer at each step and compression event automatically.

File layout
-----------
    outputs/
    └── episodes/
        └── live/
            └── <episode_id>.jsonl   ← one JSON line per event
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optimized_llm_planning_memory.core.models import (
        CompressedState,
        Itinerary,
        ReActStep,
    )
    from optimized_llm_planning_memory.mcts.node import MCTSStats


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class LiveEpisodeWriter:
    """
    Writes incremental episode events to a JSONL file for real-time UI polling.

    Each event is a JSON object with at minimum:
      ``{"type": "<event_type>", "ts": "<ISO-8601>", ...payload...}``

    The file is flushed after every write so the UI sees new data within ~1 s.

    Parameters
    ----------
    episode_id : Unique episode identifier (used as the filename stem).
    output_dir : Root episodes directory.  The file is written to
                 ``<output_dir>/live/<episode_id>.jsonl``.

    Context manager
    ---------------
    Preferred usage is as a context manager so the file is always closed even
    if the episode raises an exception.
    """

    def __init__(self, episode_id: str, output_dir: str | Path = "outputs/episodes") -> None:
        self._episode_id = episode_id
        live_dir = Path(output_dir) / "live"
        live_dir.mkdir(parents=True, exist_ok=True)
        self._path = live_dir / f"{episode_id}.jsonl"
        self._file = self._path.open("w", encoding="utf-8")

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "LiveEpisodeWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Flush and close the underlying file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    @property
    def path(self) -> Path:
        """Path to the live JSONL file being written."""
        return self._path

    # ── Write methods ──────────────────────────────────────────────────────────

    def write_step(self, step: "ReActStep") -> None:
        """Write a completed ReActStep (thought, action, observation)."""
        self._write_event(
            "react_step",
            {
                "step_index": step.step_index,
                "thought": step.thought,
                "action": (
                    {
                        "tool_name": step.action.tool_name,
                        "arguments": step.action.arguments,
                    }
                    if step.action
                    else None
                ),
                "observation": (
                    {
                        "tool_name": step.observation.tool_name,
                        "success": step.observation.success,
                        "result": _safe_serialise(step.observation.result),
                        "error_message": step.observation.error_message,
                        "latency_ms": step.observation.latency_ms,
                    }
                    if step.observation
                    else None
                ),
                "has_itinerary_snapshot": step.itinerary_snapshot is not None,
            },
        )

    def write_itinerary_update(self, itinerary: "Itinerary") -> None:
        """Write the current partial itinerary (called when a booking changes it)."""
        self._write_event(
            "itinerary_update",
            {
                "num_days": len(itinerary.days),
                "total_cost_usd": itinerary.total_cost_usd,
                "is_complete": itinerary.is_complete,
                "days": [
                    {
                        "date": day.date,
                        "city": day.city,
                        "has_accommodation": day.accommodation is not None,
                        "num_activities": len(day.activities),
                        "num_transport_segments": len(day.transport_segments),
                    }
                    for day in itinerary.days
                ],
            },
        )

    def write_compression(self, state: "CompressedState") -> None:
        """Write a compression event with the resulting CompressedState."""
        self._write_event(
            "compression",
            {
                "step_index": state.step_index,
                "compression_method": state.compression_method,
                "token_count": state.token_count,
                "num_decisions": len(state.decisions_made),
                "num_open_questions": len(state.open_questions),
                "num_key_discoveries": len(state.key_discoveries),
                "decisions_made": list(state.decisions_made),
                "open_questions": list(state.open_questions),
                "key_discoveries": list(state.key_discoveries),
                "soft_constraints_summary": state.soft_constraints_summary,
                "current_itinerary_sketch": state.current_itinerary_sketch,
                "hard_constraint_ledger": {
                    "num_satisfied": len(state.hard_constraint_ledger.satisfied_ids),
                    "num_violated": len(state.hard_constraint_ledger.violated_ids),
                    "num_unknown": len(state.hard_constraint_ledger.unknown_ids),
                },
                # MCTS-specific (None for non-MCTS compressors)
                "top_candidates": state.top_candidates,
                "tradeoffs": state.tradeoffs,
            },
        )

    def write_mcts_stats(self, stats: "MCTSStats") -> None:
        """Write MCTS search statistics after a search completes."""
        self._write_event(
            "mcts_stats",
            {
                "nodes_explored": stats.nodes_explored,
                "max_depth_reached": stats.max_depth_reached,
                "num_simulations": stats.num_simulations,
                "best_path_length": stats.best_path_length,
                "root_value": stats.root_value,
                "avg_branching_factor": stats.avg_branching_factor,
            },
        )

    def write_episode_complete(self, episode_id: str) -> None:
        """Write a terminal event marking the episode as finished."""
        self._write_event("episode_complete", {"episode_id": episode_id})

    # ── Internal ───────────────────────────────────────────────────────────────

    def _write_event(self, event_type: str, payload: dict) -> None:
        event: dict[str, Any] = {"type": event_type, "ts": _now(), **payload}
        self._file.write(json.dumps(event, default=str) + "\n")
        self._file.flush()


def _safe_serialise(value: Any) -> Any:
    """Convert a tool result to a JSON-safe form (best-effort)."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return value.model_dump()
    except AttributeError:
        pass
    try:
        return dict(value)
    except (TypeError, ValueError):
        return str(value)
