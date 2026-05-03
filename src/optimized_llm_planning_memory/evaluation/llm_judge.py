"""
evaluation/llm_judge.py
========================
LLMJudge — rubric-based itinerary scoring via litellm + pydantic.

Design principles
-----------------
- **Single judge model**: ``judge_model_id`` is fixed per evaluation run and
  shared across all rubric dimensions for fairness. Do not change it
  mid-evaluation.
- **Structured output via JSON mode**: litellm ``response_format=json_object``
  + ``model_validate()`` parses the judge's response into ``JudgeScores``.
- **Per-dimension prompting**: Each rubric dimension is evaluated independently
  to avoid the judge conflating criteria.
- **Score normalisation**: Raw scores are floats in [0.0, 1.0]. The rubric text
  defines the scale; the ``JudgeDimensionScore`` model enforces the range.
"""

from __future__ import annotations

import json
from typing import Any

import litellm
from pydantic import BaseModel, Field, field_validator


from optimized_llm_planning_memory.core.config import LLMJudgeConfig
from optimized_llm_planning_memory.core.models import Itinerary, UserRequest
from optimized_llm_planning_memory.evaluation.rubrics import RUBRIC_DIMENSIONS, DEFAULT_RUBRIC_DIMENSIONS


class JudgeDimensionScore(BaseModel):
    """Structured output from the LLM judge for one rubric dimension."""
    dimension: str
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> float:
        """Clamp to [0, 1] if the model returns a value slightly outside."""
        return max(0.0, min(1.0, float(v)))


class JudgeScores(BaseModel):
    """All dimension scores from a single judge call."""
    scores: list[JudgeDimensionScore]

    def as_dict(self) -> dict[str, float]:
        return {s.dimension: s.score for s in self.scores}


class LLMJudge:
    """
    Scores an itinerary against rubric dimensions using an LLM as judge.

    Parameters
    ----------
    judge_model_id : litellm model string, e.g. ``"openai/gpt-4o"``.
                     Fixed for the entire evaluation run.
    rubric_dimensions : List of dimension names from ``rubrics.RUBRIC_DIMENSIONS``.
                        Defaults to all six dimensions.
    config         : Judge configuration (temperature, max_tokens, etc.).
    """

    def __init__(
        self,
        judge_model_id: str,
        rubric_dimensions: list[str] | None = None,
        config: LLMJudgeConfig | None = None,
    ) -> None:
        self._model = judge_model_id
        self._dimensions = rubric_dimensions or DEFAULT_RUBRIC_DIMENSIONS
        self._config = config or LLMJudgeConfig()

    def score(
        self,
        itinerary: Itinerary,
        user_request: UserRequest,
    ) -> dict[str, float]:
        """
        Score the itinerary on all rubric dimensions.

        Returns
        -------
        dict[str, float] — dimension name → score in [0.0, 1.0].
        """
        flat, _ = self.score_detailed(itinerary, user_request)
        return flat

    def score_detailed(
        self,
        itinerary: Itinerary,
        user_request: UserRequest,
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        """
        Score the itinerary and return both flat scores and per-dimension reasoning.

        Returns
        -------
        (flat_scores, breakdown)
            flat_scores : dimension → score in [0.0, 1.0]
            breakdown   : dimension → {"score": float, "reasoning": str}
        """
        itinerary_text = self._render_itinerary(itinerary)
        request_text = self._render_request(user_request)
        rubric_text = self._build_rubric_text()

        prompt = self._build_prompt(itinerary_text, request_text, rubric_text)

        raw = litellm.completion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            response_format={"type": "json_object"},
        )
        result = JudgeScores.model_validate(
            json.loads(raw.choices[0].message.content)
        )
        flat = result.as_dict()
        breakdown = {
            s.dimension: {"score": s.score, "reasoning": s.reasoning}
            for s in result.scores
        }
        return flat, breakdown

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_rubric_text(self) -> str:
        lines = []
        for dim in self._dimensions:
            if dim in RUBRIC_DIMENSIONS:
                lines.append(RUBRIC_DIMENSIONS[dim])
        return "\n\n---\n\n".join(lines)

    def _render_itinerary(self, itinerary: Itinerary) -> str:
        """Convert Itinerary to a compact text representation for the judge."""
        lines = [f"Itinerary ID: {itinerary.itinerary_id}"]
        lines.append(f"Total cost: ${itinerary.total_cost_usd:.2f}")
        lines.append(f"Complete: {itinerary.is_complete}")
        lines.append("")
        for day in itinerary.days:
            lines.append(f"--- {day.date} | {day.city} ---")
            if day.accommodation:
                acc = day.accommodation
                lines.append(
                    f"  Hotel: {acc.hotel_name} (${acc.total_cost_usd:.0f})"
                    f" ref={acc.booking_ref or 'none'}"
                )
            for seg in day.transport_segments:
                lines.append(
                    f"  Transport: {seg.mode} {seg.from_location} → {seg.to_location}"
                    f" dep={seg.departure_datetime} arr={seg.arrival_datetime}"
                    f" ${seg.cost_usd:.0f}"
                )
            for act in day.activities:
                lines.append(
                    f"  Activity: {act.activity_name} ({act.category})"
                    f" {act.duration_hours}h ${act.cost_usd:.0f}"
                )
            if day.notes:
                lines.append(f"  Notes: {day.notes}")
        return "\n".join(lines)

    def _render_request(self, request: UserRequest) -> str:
        """Render UserRequest to text for the judge prompt."""
        hard = "; ".join(c.description for c in request.hard_constraints) or "none"
        soft = "; ".join(c.description for c in request.soft_constraints) or "none"
        prefs = "; ".join(request.preferences) or "none"
        return (
            f"Request: {request.raw_text}\n"
            f"Origin: {request.origin_city} | "
            f"Destinations: {', '.join(request.destination_cities)}\n"
            f"Dates: {request.start_date} to {request.end_date} | "
            f"Budget: ${request.budget_usd:.0f}\n"
            f"Hard constraints: {hard}\n"
            f"Soft constraints: {soft}\n"
            f"Preferences: {prefs}"
        )

    def _build_prompt(
        self, itinerary_text: str, request_text: str, rubric_text: str
    ) -> str:
        dim_list = ", ".join(f'"{d}"' for d in self._dimensions)
        example_entry = '{"dimension": "constraint_adherence", "score": 0.85, "reasoning": "Brief explanation."}'
        return f"""\
You are an expert travel itinerary evaluator. Score the following itinerary
across the rubric dimensions listed below.

Return ONLY a JSON object in exactly this structure — no markdown, no extra keys:
{{
  "scores": [
    {example_entry},
    ... one entry per dimension ...
  ]
}}

Use these exact dimension name strings: {dim_list}
Each score must be a float in [0.0, 1.0]. Reasoning must be one concise sentence.

=== USER REQUEST ===
{request_text}

=== ITINERARY ===
{itinerary_text}

=== RUBRIC ===
{rubric_text}

Score each dimension independently. Return only the JSON object described above.
"""
