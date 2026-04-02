"""
agent/trajectory.py
===================
Trajectory — mutable accumulator for in-progress ReAct episodes.

Design: Mutable builder → immutable snapshot
----------------------------------------------
``Trajectory`` is a mutable class used during an active episode.
``Trajectory.to_model()`` freezes the current state into an immutable
``TrajectoryModel`` (Pydantic frozen model) that is safe to pass to the
compressor and serialiser without mutation risk.

This mirrors the builder pattern: the mutable ``Trajectory`` object is
the builder; ``TrajectoryModel`` is the built product.

Why not use ``TrajectoryModel`` directly?
  ``TrajectoryModel`` is frozen (Pydantic ``frozen=True``), so it cannot
  be accumulated step-by-step. Using a mutable wrapper avoids copying the
  full model on every ``add_step()`` call.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from optimized_llm_planning_memory.core.models import ReActStep, TrajectoryModel


class Trajectory:
    """
    Mutable accumulator for ReActStep objects during an active episode.

    Create one instance per episode, add steps as they are produced,
    and call ``to_model()`` when the episode ends or when the compressor
    needs a frozen snapshot.

    Thread safety
    -------------
    Not thread-safe. Each episode (and therefore each parallel env worker)
    should have its own ``Trajectory`` instance.
    """

    def __init__(self, request_id: str, trajectory_id: str | None = None) -> None:
        self._trajectory_id = trajectory_id or str(uuid.uuid4())
        self._request_id = request_id
        self._steps: list[ReActStep] = []
        self._last_compressed_step: int = 0  # index of step at last compression

    # ── Accumulation ─────────────────────────────────────────────────────────

    def add_step(self, step: ReActStep) -> None:
        """Append a completed ReActStep to the trajectory."""
        self._steps.append(step)

    # ── Snapshots ─────────────────────────────────────────────────────────────

    def to_model(self) -> TrajectoryModel:
        """
        Freeze the current trajectory into an immutable ``TrajectoryModel``.

        The returned model is safe to pass to the compressor and serialiser.
        The underlying ``_steps`` list is not modified.
        """
        return TrajectoryModel(
            trajectory_id=self._trajectory_id,
            request_id=self._request_id,
            steps=tuple(self._steps),
            total_steps=len(self._steps),
        )

    def to_text(self, include_itinerary_snapshots: bool = False) -> str:
        """Linearise the current trajectory to text (delegates to TrajectoryModel)."""
        return self.to_model().to_text(
            include_itinerary_snapshots=include_itinerary_snapshots
        )

    def steps_since(self, step_index: int) -> list[ReActStep]:
        """Return steps with ``step_index >= step_index`` (since last compression)."""
        return [s for s in self._steps if s.step_index >= step_index]

    def steps_since_last_compression(self) -> list[ReActStep]:
        """Return steps accumulated since the last ``mark_compression()`` call."""
        return self.steps_since(self._last_compressed_step)

    def mark_compression(self, at_step: int | None = None) -> None:
        """Record that a compression event just occurred at the current step."""
        self._last_compressed_step = at_step if at_step is not None else len(self._steps)

    # ── Token counting ────────────────────────────────────────────────────────

    def token_count(self, tokenizer: Any) -> int:
        """
        Estimate the token count of the full trajectory text.

        Parameters
        ----------
        tokenizer : Any tokenizer with an ``encode()`` method (e.g., HuggingFace).

        Returns
        -------
        int
            Approximate number of tokens in the linearised trajectory.
        """
        text = self.to_text()
        return len(tokenizer.encode(text))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def trajectory_id(self) -> str:
        return self._trajectory_id

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def total_steps(self) -> int:
        return len(self._steps)

    @property
    def last_compressed_step(self) -> int:
        return self._last_compressed_step

    def __len__(self) -> int:
        return len(self._steps)
