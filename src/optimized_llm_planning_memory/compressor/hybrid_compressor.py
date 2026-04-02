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

This is the starting architecture recommended for exploration before
committing to full PPO training of the TransformerCompressor.

Note
----
HybridCompressor inherits from ``CompressorBase`` (not ``TrainableCompressorBase``)
because it delegates to an LLM for the trainable parts. If the free-form
generation is handled by a TransformerCompressor, inject that and promote
HybridCompressor to TrainableCompressorBase.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from optimized_llm_planning_memory.compressor.base import CompressorBase
from optimized_llm_planning_memory.compressor.llm_compressor import LLMCompressor
from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate
from optimized_llm_planning_memory.core.models import (
    CompressedState,
    HardConstraintLedger,
    TrajectoryModel,
)


class HybridCompressor(CompressorBase):
    """
    Two-pass compressor: typed slot extraction + free-form narrative.

    Pass 1 — Slot extraction (LLMCompressor):
        Extract constraint IDs (satisfied/violated/unknown), decisions,
        open questions, and key discoveries. These are structured and evaluable.

    Pass 2 — Narrative generation (LLMCompressor or TransformerCompressor):
        Generate ``soft_constraints_summary`` and ``current_itinerary_sketch``.
        These are free-form and capture nuance.

    Currently both passes use ``LLMCompressor`` (fast to implement, no GPU).
    To make Pass 2 trainable, inject a ``TransformerCompressor`` as
    ``narrative_compressor`` and override with ``TrainableCompressorBase``.

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
        # Pass 1: slot extraction
        slot_state = self._slot_compressor.compress(trajectory, previous_state)

        # Pass 2: narrative generation (may reuse slot_state as context)
        # Here we reuse the same trajectory; a more sophisticated implementation
        # could pass the slot_state as guidance to the narrative compressor.
        narrative_state = self._narrative_compressor.compress(trajectory, slot_state)

        # Merge: take structured fields from slot_state, narrative from narrative_state
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
