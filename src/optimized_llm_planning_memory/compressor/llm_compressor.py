"""
compressor/llm_compressor.py
============================
LLMCompressor — off-the-shelf LLM used as the compression baseline.

Role in the project
-------------------
LLMCompressor implements ``compress()`` by prompting a frozen LLM (via
litellm) to produce a structured CompressedState. It is used as:

  1. **Baseline 2** in evaluation (``agent.mode=llm_summary``).
  2. A source of **supervision data** for initialising the trained compressor.
  3. A fallback when GPU is not available.

Because it calls an external API, LLMCompressor does NOT implement
``get_log_probs()`` — it is not trainable via PPO.

Stack: litellm → API call with response_format=json_object → Pydantic parse
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import litellm
from pydantic import BaseModel

from optimized_llm_planning_memory.compressor.base import CompressorBase
from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate
from optimized_llm_planning_memory.core.models import (
    CompressedState,
    HardConstraintLedger,
    TrajectoryModel,
)


# ── Instructor response schema ────────────────────────────────────────────────

class _CompressorLLMResponse(BaseModel):
    """
    Pydantic model that instructor uses to parse the LLM's structured output.
    Field names mirror CompressedState sections.
    """
    satisfied_constraint_ids: list[str]
    violated_constraint_ids: list[str]
    unknown_constraint_ids: list[str]
    soft_constraints_summary: str
    decisions_made: list[str]
    open_questions: list[str]
    key_discoveries: list[str]
    current_itinerary_sketch: str


# ── LLMCompressor ─────────────────────────────────────────────────────────────

class LLMCompressor(CompressorBase):
    """
    Compressor that uses an off-the-shelf LLM via litellm JSON mode.

    Not trainable (does not override get_log_probs / get_trainable_parameters).

    Parameters
    ----------
    model_id : litellm model string, e.g. ``"openai/gpt-4o-mini"``.
    temperature : LLM temperature for compression calls (0.0 recommended).
    max_tokens  : Max tokens for the LLM response.
    """

    def __init__(
        self,
        model_id: str = "openai/gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self._model_id = model_id
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._template = CompressedStateTemplate()

    def compress(
        self,
        trajectory: TrajectoryModel,
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        """
        Call the LLM to compress the trajectory into a structured state.

        The prompt includes:
        - The full trajectory text (linearised via TrajectoryModel.to_text()).
        - The previous CompressedState if available (for continuity).
        - Explicit instructions to fill every required section.
        """
        prompt = self._build_prompt(trajectory, previous_state)

        raw = litellm.completion(
            model=self._model_id,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
        )
        raw_text = raw.choices[0].message.content or "{}"
        response = _CompressorLLMResponse.model_validate(json.loads(raw_text))

        # Reconstruct HardConstraintLedger from the response
        # (constraints list comes from previous_state if available)
        prior_constraints = (
            tuple(previous_state.hard_constraint_ledger.constraints)
            if previous_state else ()
        )
        ledger = HardConstraintLedger(
            constraints=prior_constraints,
            satisfied_ids=tuple(response.satisfied_constraint_ids),
            violated_ids=tuple(response.violated_constraint_ids),
            unknown_ids=tuple(response.unknown_constraint_ids),
        )

        state = CompressedState(
            state_id=str(uuid.uuid4()),
            trajectory_id=trajectory.trajectory_id,
            step_index=trajectory.total_steps,
            hard_constraint_ledger=ledger,
            soft_constraints_summary=response.soft_constraints_summary,
            decisions_made=response.decisions_made,
            open_questions=response.open_questions,
            key_discoveries=response.key_discoveries,
            current_itinerary_sketch=response.current_itinerary_sketch,
            compression_method="llm",
            token_count=None,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )

        self._template.validate(state)
        return state

    def get_metadata(self) -> dict:
        return {
            "type": "llm",
            "model_id": self._model_id,
            "param_count": 0,
            "trainable": False,
        }

    def _build_prompt(
        self,
        trajectory: TrajectoryModel,
        previous_state: CompressedState | None,
    ) -> str:
        parts: list[str] = []

        if previous_state is not None:
            parts.append("== PREVIOUS COMPRESSED STATE ==")
            parts.append(self._template.render(previous_state))
            parts.append("")
            parts.append("== NEW TRAJECTORY STEPS SINCE LAST COMPRESSION ==")
        else:
            parts.append("== FULL TRAJECTORY ==")

        parts.append(trajectory.to_text())
        return "\n".join(parts)


_SYSTEM_PROMPT = """\
You are a travel planning memory compressor. Your job is to distill a growing
ReAct agent trajectory into a compact, structured memory state.

You will receive the agent's reasoning and tool call history. Produce a JSON
object with the following fields:

- satisfied_constraint_ids: IDs of hard constraints that are now satisfied.
- violated_constraint_ids: IDs of hard constraints that are violated.
- unknown_constraint_ids: IDs of hard constraints not yet evaluable.
- soft_constraints_summary: A concise free-form summary of soft constraints
  and traveler preferences. Be specific; do not lose information.
- decisions_made: A list of confirmed decisions (bookings, plans) made so far.
- open_questions: A list of unresolved planning questions or gaps.
- key_discoveries: Important world facts learned (prices, availability, etc.).
- current_itinerary_sketch: A compact text summary of the current partial plan.

Be thorough. Do not omit hard constraints. Preserve all decision-critical information.
"""
