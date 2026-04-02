"""
compressor/template.py
======================
CompressedStateTemplate — fixed section schema, renderer, and parser.

Design: Fixed template as a hard constraint on the action space
----------------------------------------------------------------
During RL training, the compressor's action is the text it generates.
Without structural constraints, the compressor can drift toward unstructured
outputs that maximise a proxy metric but become uninterpretable.

The template enforces:
  1. All required sections are present in every CompressedState.
  2. The rendered string has a predictable structure the agent can parse.
  3. The reward function can evaluate individual sections deterministically
     (e.g., constraint_ledger.satisfied_ids vs. violated_ids).

``render()`` → ``str`` : CompressedState → text for LLM context window.
``parse()``  → CompressedState : text → CompressedState (inverse).
``validate()``          : raises CompressedStateRenderError if sections are missing.

The render format uses sentinel headers (``## SECTION_NAME ##``) that are
unlikely to appear in natural text, making the parser robust.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from optimized_llm_planning_memory.core.exceptions import CompressedStateRenderError
from optimized_llm_planning_memory.core.models import (
    CompressedState,
    Constraint,
    HardConstraintLedger,
)


_SENTINEL = "## {section} ##"
_SENTINEL_RE = re.compile(r"## (\w+) ##")

# Ordered list of required sections
REQUIRED_SECTIONS: list[str] = [
    "HARD_CONSTRAINT_LEDGER",
    "SOFT_CONSTRAINTS_SUMMARY",
    "DECISIONS_MADE",
    "OPEN_QUESTIONS",
    "KEY_DISCOVERIES",
    "CURRENT_ITINERARY_SKETCH",
]


class CompressedStateTemplate:
    """
    Defines the fixed section schema for CompressedState and provides
    render / parse / validate utilities.

    The template is stateless. Create one instance and reuse it.

    Example rendered output
    -----------------------
    ::

        ## HARD_CONSTRAINT_LEDGER ##
        Constraints: [...]
        Satisfied: [c1, c2]  Violated: []  Unknown: [c3]

        ## SOFT_CONSTRAINTS_SUMMARY ##
        Traveler prefers window seats and vegetarian meals.

        ## DECISIONS_MADE ##
        - Booked flight FL001 Paris→Rome on 2025-06-01

        ## OPEN_QUESTIONS ##
        - Need to confirm hotel availability in Rome for June 3-4.

        ## KEY_DISCOVERIES ##
        - Round-trip flights Paris↔Rome avg $240 per person.

        ## CURRENT_ITINERARY_SKETCH ##
        Day 1 (2025-06-01): Fly Paris→Rome. Hotel: TBD.
    """

    SECTIONS = REQUIRED_SECTIONS
    REQUIRED_SECTIONS = REQUIRED_SECTIONS

    def render(self, state: CompressedState) -> str:
        """
        Serialise a CompressedState into the fixed-template string.

        Parameters
        ----------
        state : CompressedState — must pass ``validate()`` first.

        Returns
        -------
        str
            Multi-section text ready for injection into the LLM context window.
        """
        self.validate(state)
        ledger = state.hard_constraint_ledger

        # Render constraint ledger as compact JSON summary
        ledger_lines = [
            f"Constraints: {len(ledger.constraints)} total",
            f"Satisfied: {list(ledger.satisfied_ids)}",
            f"Violated: {list(ledger.violated_ids)}",
            f"Unknown: {list(ledger.unknown_ids)}",
        ]
        if ledger.constraints:
            ledger_lines.append("Details:")
            for c in ledger.constraints:
                status = (
                    "✓" if c.constraint_id in ledger.satisfied_ids
                    else "✗" if c.constraint_id in ledger.violated_ids
                    else "?"
                )
                ledger_lines.append(f"  [{status}] {c.constraint_id}: {c.description}")

        sections: dict[str, str] = {
            "HARD_CONSTRAINT_LEDGER": "\n".join(ledger_lines),
            "SOFT_CONSTRAINTS_SUMMARY": state.soft_constraints_summary,
            "DECISIONS_MADE": _render_list(state.decisions_made),
            "OPEN_QUESTIONS": _render_list(state.open_questions),
            "KEY_DISCOVERIES": _render_list(state.key_discoveries),
            "CURRENT_ITINERARY_SKETCH": state.current_itinerary_sketch,
        }

        parts: list[str] = []
        for section_name in REQUIRED_SECTIONS:
            parts.append(_SENTINEL.format(section=section_name))
            parts.append(sections[section_name])
            parts.append("")

        return "\n".join(parts).strip()

    def parse(self, text: str, trajectory_id: str = "", step_index: int = 0,
              compression_method: str = "unknown") -> CompressedState:
        """
        Reconstruct a CompressedState from a rendered template string.

        This is the inverse of ``render()``. Used by the RL trainer to convert
        the compressor's text output back into a structured model for reward
        computation.

        Parameters
        ----------
        text               : The full rendered template string.
        trajectory_id      : To attach to the reconstructed state.
        step_index         : Trajectory step at which this compression occurred.
        compression_method : 'llm' | 'transformer' | 'hybrid'.

        Returns
        -------
        CompressedState

        Raises
        ------
        CompressedStateRenderError
            If required sections are missing from the text.
        """
        sections = _split_sections(text)

        missing = [s for s in REQUIRED_SECTIONS if s not in sections]
        if missing:
            raise CompressedStateRenderError(
                f"Compressed state text is missing required sections: {missing}. "
                f"Found sections: {list(sections.keys())}."
            )

        # Parse hard constraint ledger from its section text
        ledger = _parse_ledger_section(sections["HARD_CONSTRAINT_LEDGER"])

        # Parse bullet lists
        decisions = _parse_list(sections["DECISIONS_MADE"])
        questions = _parse_list(sections["OPEN_QUESTIONS"])
        discoveries = _parse_list(sections["KEY_DISCOVERIES"])

        state = CompressedState(
            state_id=str(uuid.uuid4()),
            trajectory_id=trajectory_id,
            step_index=step_index,
            hard_constraint_ledger=ledger,
            soft_constraints_summary=sections["SOFT_CONSTRAINTS_SUMMARY"].strip(),
            decisions_made=decisions,
            open_questions=questions,
            key_discoveries=discoveries,
            current_itinerary_sketch=sections["CURRENT_ITINERARY_SKETCH"].strip(),
            compression_method=compression_method,
            token_count=None,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        return state

    def validate(self, state: CompressedState) -> None:
        """
        Raise CompressedStateRenderError if any required field is missing or empty.

        Called by ``render()`` before serialisation and can be called by the
        compressor after generating a state to catch failures early.
        """
        errors: list[str] = []
        if not state.hard_constraint_ledger.constraints and not state.hard_constraint_ledger.unknown_ids:
            pass  # empty ledger is valid (no constraints in request)
        if not state.soft_constraints_summary:
            errors.append("soft_constraints_summary is empty.")
        if not state.current_itinerary_sketch:
            errors.append("current_itinerary_sketch is empty.")

        if errors:
            raise CompressedStateRenderError(
                f"CompressedState validation failed: {'; '.join(errors)}"
            )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_list(items: list[str]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {item}" for item in items)


def _split_sections(text: str) -> dict[str, str]:
    """Split rendered template text into a dict of section_name → content."""
    parts: dict[str, str] = {}
    current_section: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        match = _SENTINEL_RE.fullmatch(line.strip())
        if match:
            if current_section is not None:
                parts[current_section] = "\n".join(current_lines).strip()
            current_section = match.group(1)
            current_lines = []
        else:
            if current_section is not None:
                current_lines.append(line)

    if current_section is not None:
        parts[current_section] = "\n".join(current_lines).strip()

    return parts


def _parse_list(text: str) -> list[str]:
    """Parse a bullet-list section into a Python list."""
    if not text or text.strip() == "(none)":
        return []
    items = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("- ").strip()
        if stripped:
            items.append(stripped)
    return items


def _parse_ledger_section(text: str) -> HardConstraintLedger:
    """
    Parse the HARD_CONSTRAINT_LEDGER section back into a HardConstraintLedger.

    This is a best-effort parser; the lists are reconstructed from the
    "Satisfied: [...] / Violated: [...] / Unknown: [...]" lines.
    """
    satisfied_ids: list[str] = []
    violated_ids: list[str] = []
    unknown_ids: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Satisfied:"):
            satisfied_ids = _parse_id_list(line.split("Satisfied:", 1)[1])
        elif line.startswith("Violated:"):
            violated_ids = _parse_id_list(line.split("Violated:", 1)[1])
        elif line.startswith("Unknown:"):
            unknown_ids = _parse_id_list(line.split("Unknown:", 1)[1])

    return HardConstraintLedger(
        constraints=(),
        satisfied_ids=tuple(satisfied_ids),
        violated_ids=tuple(violated_ids),
        unknown_ids=tuple(unknown_ids),
    )


def _parse_id_list(text: str) -> list[str]:
    """Parse a Python list literal string like "['c1', 'c2']" into a list."""
    text = text.strip()
    try:
        result = json.loads(text.replace("'", '"'))
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, ValueError):
        return []
