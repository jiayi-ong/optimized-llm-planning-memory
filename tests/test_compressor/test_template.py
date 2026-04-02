"""
tests/test_compressor/test_template.py
========================================
Unit tests for CompressedStateTemplate: render/parse roundtrip, missing section error.
"""

from __future__ import annotations

import uuid

import pytest

from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate
from optimized_llm_planning_memory.core.exceptions import CompressedStateRenderError
from optimized_llm_planning_memory.core.models import CompressedState, HardConstraintLedger


@pytest.fixture
def template() -> CompressedStateTemplate:
    return CompressedStateTemplate()


@pytest.fixture
def sample_state() -> CompressedState:
    ledger = HardConstraintLedger(
        constraints=[],
        satisfied_ids=[],
        violated_ids=[],
        unknown_ids=[],
    )
    return CompressedState(
        state_id=str(uuid.uuid4()),
        trajectory_id=str(uuid.uuid4()),
        step_index=5,
        hard_constraint_ledger=ledger,
        soft_constraints_summary="Prefers boutique hotels.",
        decisions_made=["Booked flight AF-001", "Booked hotel HTL-MAR-001"],
        open_questions=["Need to book activity for day 2"],
        key_discoveries=["Louvre costs $20", "Budget remaining: $830"],
        current_itinerary_sketch="Day 1: Paris - hotel booked, flight booked.",
        compression_method="transformer",
        token_count=128,
        created_at="2025-06-01T10:00:00Z",
    )


def test_render_contains_all_sections(template, sample_state):
    rendered = template.render(sample_state)
    for section in CompressedStateTemplate.REQUIRED_SECTIONS:
        assert section in rendered, f"Section '{section}' missing from rendered output"


def test_roundtrip_parse_render(template, sample_state):
    rendered = template.render(sample_state)
    parsed = template.parse(rendered)
    assert parsed.decisions_made == sample_state.decisions_made
    assert parsed.soft_constraints_summary == sample_state.soft_constraints_summary
    assert parsed.current_itinerary_sketch == sample_state.current_itinerary_sketch


def test_validate_raises_on_missing_section(template, sample_state):
    # Remove a required section by blanking it
    broken = sample_state.model_copy(update={"decisions_made": []})
    # validate should still pass (empty list is valid)
    template.validate(broken)  # should not raise

    # Missing soft_constraints_summary
    broken2 = sample_state.model_copy(update={"soft_constraints_summary": ""})
    with pytest.raises(CompressedStateRenderError):
        template.validate(broken2)
