"""
compressor/hybrid_compressor.py
================================
HybridCompressor — typed slot extraction via LLM + free-form narrative.

Design motivation
-----------------
The fixed template provides structure but forces all content into predefined
buckets. The fully free-form LLM compressor is flexible but drifts during
training. The hybrid approach combines both:

  - **Typed slot extraction** (via ``LLMCompressor``): extracts structured
    fields (constraint IDs, decision bullets) with high fidelity using the LLM.
  - **Free-form narrative generation** (via ``TransformerCompressor`` or
    another LLM call): generates the ``current_itinerary_sketch`` and
    ``soft_constraints_summary`` sections where nuance matters.

Trainability (M6 fix)
---------------------
``HybridCompressor`` now inherits from ``TrainableCompressorBase`` and
delegates the trainable contract (get_log_probs, get_trainable_parameters,
save/load_checkpoint) to the ``narrative_compressor`` when it is a
``TrainableCompressorBase``. If the narrative compressor is a plain
``CompressorBase`` (e.g., LLMCompressor), these methods raise
``LogProbsNotSupportedError`` as before — the hybrid is only trainable when
a ``TransformerCompressor`` is injected as the narrative pass.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import torch

from optimized_llm_planning_memory.compressor.base import CompressorBase
from optimized_llm_planning_memory.compressor.llm_compressor import LLMCompressor
from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate
from optimized_llm_planning_memory.compressor.trainable_base import TrainableCompressorBase
from optimized_llm_planning_memory.core.exceptions import LogProbsNotSupportedError
from optimized_llm_planning_memory.core.models import (
    CompressedState,
    TrajectoryModel,
)


class HybridCompressor(TrainableCompressorBase):
    """
    Two-pass compressor: typed slot extraction + free-form narrative.

    Pass 1 — Slot extraction (LLMCompressor):
        Extract constraint IDs (satisfied/violated/unknown), decisions,
        open questions, and key discoveries. These are structured and evaluable.

    Pass 2 — Narrative generation (LLMCompressor or TransformerCompressor):
        Generate ``soft_constraints_summary`` and ``current_itinerary_sketch``.
        These are free-form and capture nuance.

    To enable RL training, inject a ``TransformerCompressor`` as
    ``narrative_compressor``. The trainable contract is delegated to it.

    Parameters
    ----------
    slot_compressor      : Compressor for structured slot extraction.
    narrative_compressor : Compressor for free-form narrative sections.
                           Defaults to the same as slot_compressor if None.
    """

    def __init__(
        self,
        slot_compressor: CompressorBase | None = None,
        narrative_compressor: CompressorBase | None = None,
    ) -> None:
        self._slot_compressor = slot_compressor or LLMCompressor()
        self._narrative_compressor = narrative_compressor or self._slot_compressor
        self._template = CompressedStateTemplate()

    def compress(
        self,
        trajectory: TrajectoryModel,
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        """
        Two-pass compression.

        Pass 1 extracts structured slots. Pass 2 generates free-form narrative.
        Results are merged into a single CompressedState.
        """
        slot_state = self._slot_compressor.compress(trajectory, previous_state)
        narrative_state = self._narrative_compressor.compress(trajectory, slot_state)

        merged = CompressedState(
            state_id=str(uuid.uuid4()),
            trajectory_id=trajectory.trajectory_id,
            step_index=trajectory.total_steps,
            hard_constraint_ledger=slot_state.hard_constraint_ledger,
            soft_constraints_summary=narrative_state.soft_constraints_summary,
            decisions_made=slot_state.decisions_made,
            open_questions=slot_state.open_questions,
            key_discoveries=slot_state.key_discoveries,
            current_itinerary_sketch=narrative_state.current_itinerary_sketch,
            compression_method="hybrid",
            token_count=None,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )

        self._template.validate(merged)
        return merged

    # ── TrainableCompressorBase delegation ───────────────────────────────────

    def get_log_probs(self, trajectory_text: str, compressed_text: str) -> torch.Tensor:
        """Delegate to narrative_compressor if trainable; else unsupported."""
        if isinstance(self._narrative_compressor, TrainableCompressorBase):
            return self._narrative_compressor.get_log_probs(trajectory_text, compressed_text)
        raise LogProbsNotSupportedError(
            "HybridCompressor.get_log_probs() requires a TrainableCompressorBase "
            "as narrative_compressor. Inject a TransformerCompressor to enable RL training."
        )

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Delegate to narrative_compressor if trainable; else return empty list."""
        if isinstance(self._narrative_compressor, TrainableCompressorBase):
            return self._narrative_compressor.get_trainable_parameters()
        return []

    def save_checkpoint(self, path: str) -> None:
        """Delegate to narrative_compressor if trainable."""
        if isinstance(self._narrative_compressor, TrainableCompressorBase):
            self._narrative_compressor.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        """Delegate to narrative_compressor if trainable."""
        if isinstance(self._narrative_compressor, TrainableCompressorBase):
            self._narrative_compressor.load_checkpoint(path)
