"""
core/models.py
==============
All Pydantic v2 data models used across the project.

Design notes
------------
* Every public model lives here so that module boundaries are defined by
  imports from ``core.models``, making the inter-module dependency graph
  explicit and acyclic.

* Immutability: models that represent completed, logged data (TrajectoryModel,
  CompressedState, EpisodeLog, EvalResult) use ``frozen=True`` so they are
  safe to hash, cache, and pass between processes without mutation risk.

* Models that are built up incrementally (Itinerary, partial states) use
  the default mutable config so they can be updated in place during an episode.

* All datetime/date fields use ISO 8601 strings (``str``) rather than
  ``datetime`` objects to simplify JSON serialisation and cross-process transfer.

Model hierarchy (read top-to-bottom)
--------------------------------------
1. Constraint models
2. User request + traveler profile
3. Itinerary (transport, accommodation, activity, day, itinerary)
4. ReAct trajectory (tool call/result, step, trajectory)
5. Compressed state
6. Episode log (the central data contract)
7. RL transition
8. Evaluation result
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    # Avoid a circular import at runtime: MCTSStats is only referenced in
    # EpisodeLog's type annotation, which Pydantic resolves lazily.
    from optimized_llm_planning_memory.mcts.node import MCTSStats


# ══════════════════════════════════════════════════════════════════════════════
# 1. Constraint Models
# ══════════════════════════════════════════════════════════════════════════════

class ConstraintType(str, Enum):
    """Whether a constraint is non-negotiable (HARD) or a scored preference (SOFT)."""
    HARD = "hard"
    SOFT = "soft"


class ConstraintCategory(str, Enum):
    """Semantic category of a constraint, used by ConstraintSatisfactionEngine."""
    BUDGET = "budget"
    DATE = "date"
    DURATION = "duration"
    CITY = "city"
    ACCOMMODATION = "accommodation"
    ACTIVITY = "activity"
    TRANSPORT = "transport"
    GROUP = "group"
    ACCESSIBILITY = "accessibility"
    PREFERENCE = "preference"


class Constraint(BaseModel):
    """
    A single hard or soft constraint extracted from a UserRequest.

    ``satisfied`` and ``score`` are populated by ConstraintSatisfactionEngine
    after evaluating an itinerary. They are ``None`` before evaluation.

    Fields
    ------
    constraint_id   : Stable unique identifier (e.g., "hard_budget_001").
    constraint_type : HARD = must be satisfied; SOFT = scored preference.
    category        : Semantic grouping; drives the evaluation strategy.
    description     : Human-readable description (also injected into compressed state).
    value           : Parsed constraint value (float for budget, str for city, etc.).
    unit            : Optional unit of the value (e.g., "USD", "days", "people").
    satisfied       : True/False after evaluation; None = not yet evaluated.
    score           : [0.0, 1.0] partial satisfaction score for soft constraints.
    """
    model_config = ConfigDict(frozen=True)

    constraint_id: str
    constraint_type: ConstraintType
    category: ConstraintCategory
    description: str
    value: Any
    unit: str | None = None
    satisfied: bool | None = None
    score: float | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value_type(cls, v: Any, info: Any) -> Any:
        """Reject type-mismatched constraint values early (M8 fix)."""
        # info.data may be partially populated; only validate when category is known
        category = (info.data or {}).get("category")
        if category is None:
            return v
        if category == ConstraintCategory.BUDGET:
            if not isinstance(v, (int, float)):
                raise ValueError(
                    f"BUDGET constraint requires a numeric value, got {type(v).__name__!r}: {v!r}"
                )
        elif category == ConstraintCategory.DURATION:
            if not isinstance(v, (int, float)):
                raise ValueError(
                    f"DURATION constraint requires a numeric value, got {type(v).__name__!r}: {v!r}"
                )
        elif category in (ConstraintCategory.CITY, ConstraintCategory.DATE):
            if not isinstance(v, str):
                raise ValueError(
                    f"{category.value.upper()} constraint requires a string value, got {type(v).__name__!r}: {v!r}"
                )
        return v


class ConstraintSatisfactionResult(BaseModel):
    """Output of ConstraintSatisfactionEngine.evaluate() for a single constraint."""
    model_config = ConfigDict(frozen=True)

    constraint_id: str
    satisfied: bool
    score: float = Field(ge=0.0, le=1.0, description="1.0 = fully satisfied, 0.0 = violated")
    explanation: str


# ══════════════════════════════════════════════════════════════════════════════
# 2. User Request
# ══════════════════════════════════════════════════════════════════════════════

class TravelerProfile(BaseModel):
    """Traveler group demographics and special requirements."""
    model_config = ConfigDict(frozen=True)

    num_adults: int = Field(default=1, ge=1)
    num_children: int = Field(default=0, ge=0)
    accessibility_needs: list[str] = Field(default_factory=list)
    dietary_restrictions: list[str] = Field(default_factory=list)

    @property
    def total_travelers(self) -> int:
        return self.num_adults + self.num_children


class UserRequest(BaseModel):
    """
    A single-turn user travel planning request.

    This is the top-level input to a planning episode. It is loaded from a
    JSON file in ``data/user_requests/`` and passed into ``ReActAgent.run_episode()``.

    ``hard_constraints`` and ``soft_constraints`` are pre-extracted from
    ``raw_text`` either by an LLM extractor at data-generation time or
    manually for evaluation scenarios.

    ``metadata`` stores generation provenance (template_id, generation_model, etc.)
    for debugging and ablation filtering.
    """
    model_config = ConfigDict(frozen=True)

    request_id: str
    raw_text: str = Field(description="Original natural-language planning request.")
    origin_city: str
    destination_cities: list[str] = Field(min_length=1)
    start_date: str = Field(description="ISO 8601 date string, e.g. '2025-06-01'.")
    end_date: str
    budget_usd: float = Field(gt=0)
    traveler_profile: TravelerProfile = Field(default_factory=TravelerProfile)
    hard_constraints: list[Constraint] = Field(default_factory=list)
    soft_constraints: list[Constraint] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list,
                                   description="Free-form preference strings, e.g. 'prefer window seats'.")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Itinerary Models
# ══════════════════════════════════════════════════════════════════════════════

class TransportSegment(BaseModel):
    """A single transport leg (flight, train, taxi, etc.) between two locations."""
    mode: str = Field(description="One of: flight, train, bus, walk, taxi, ferry.")
    from_location: str
    to_location: str
    departure_datetime: str = Field(description="ISO 8601 datetime string.")
    arrival_datetime: str
    cost_usd: float = Field(ge=0.0)
    booking_ref: str | None = None


class AccommodationBooking(BaseModel):
    """A confirmed or proposed hotel/accommodation stay."""
    hotel_id: str
    hotel_name: str
    city: str
    check_in: str = Field(description="ISO 8601 date.")
    check_out: str
    cost_per_night_usd: float = Field(ge=0.0)
    total_cost_usd: float = Field(ge=0.0)
    star_rating: float | None = None
    booking_ref: str | None = None


class ActivityBooking(BaseModel):
    """A confirmed or proposed activity booking."""
    activity_id: str
    activity_name: str
    location: str
    city: str
    start_datetime: str
    duration_hours: float = Field(gt=0.0)
    cost_usd: float = Field(ge=0.0)
    category: str
    booking_ref: str | None = None


class ItineraryDay(BaseModel):
    """All travel elements for a single day."""
    date: str = Field(description="ISO 8601 date.")
    city: str
    transport_segments: list[TransportSegment] = Field(default_factory=list)
    accommodation: AccommodationBooking | None = None
    activities: list[ActivityBooking] = Field(default_factory=list)
    notes: str = ""

    @property
    def total_cost_usd(self) -> float:
        transport = sum(s.cost_usd for s in self.transport_segments)
        hotel = self.accommodation.total_cost_usd if self.accommodation else 0.0
        activities = sum(a.cost_usd for a in self.activities)
        return transport + hotel + activities


class Itinerary(BaseModel):
    """
    A complete or partial travel plan for a UserRequest.

    ``version`` is incremented each time the agent updates the itinerary,
    allowing the compressor to track how the plan has evolved.

    ``is_complete`` is set to True when the agent signals it has finished
    planning and the itinerary covers the full requested travel period.
    """
    itinerary_id: str
    request_id: str
    days: list[ItineraryDay] = Field(default_factory=list)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    is_complete: bool = False
    version: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _sync_total_cost(self) -> "Itinerary":
        """Auto-sync total_cost_usd from day sums on construction (M9 fix).

        Prevents divergence between the stored field and the sum of per-day costs.
        ``recompute_total_cost()`` remains available for explicit mid-episode updates.
        """
        self.total_cost_usd = sum(d.total_cost_usd for d in self.days)
        return self

    def recompute_total_cost(self) -> float:
        """Recompute and store total cost from all days. Returns the updated total."""
        self.total_cost_usd = sum(d.total_cost_usd for d in self.days)
        return self.total_cost_usd

    def cities_visited(self) -> list[str]:
        """Return unique cities in visit order."""
        seen: list[str] = []
        for day in self.days:
            if day.city not in seen:
                seen.append(day.city)
        return seen


# ══════════════════════════════════════════════════════════════════════════════
# 4. ReAct Trajectory Models
# ══════════════════════════════════════════════════════════════════════════════

class ToolCall(BaseModel):
    """
    An action produced by the LLM agent — a parsed tool invocation.

    ``raw_text`` stores the exact text the LLM produced before parsing,
    useful for debugging malformed outputs.
    """
    model_config = ConfigDict(frozen=True)

    tool_name: str
    arguments: dict[str, Any]
    raw_text: str = Field(description="Unparsed LLM output that produced this call.")


class ToolResult(BaseModel):
    """
    The observation returned by the tool middleware after executing a ToolCall.

    This is what the agent sees as its next input. On failure, ``error_message``
    contains a structured hint string generated by BaseTool._generate_error_feedback().
    """
    model_config = ConfigDict(frozen=True)

    tool_name: str
    success: bool
    result: Any | None = Field(default=None,
                               description="Parsed simulator response on success.")
    error_message: str | None = Field(default=None,
                                      description="Actionable hint for the agent on failure.")
    latency_ms: float = Field(default=0.0, ge=0.0)


class ReActStep(BaseModel):
    """
    One think–act–observe cycle in the ReAct loop.

    ``action`` and ``observation`` are None on the final step when the agent
    decides to conclude without calling a tool.

    ``itinerary_snapshot`` captures the agent's partial itinerary state after
    this step. It may be None early in the episode or if the agent hasn't
    yet constructed any itinerary.
    """
    model_config = ConfigDict(frozen=True)

    step_index: int = Field(ge=0)
    thought: str
    action: ToolCall | None = None
    observation: ToolResult | None = None
    itinerary_snapshot: Itinerary | None = None
    timestamp: str = Field(description="ISO 8601 datetime when this step was recorded.")


class TrajectoryModel(BaseModel):
    """
    Immutable, serialisable snapshot of a full ReAct trajectory.

    Produced by ``Trajectory.to_model()`` at compression time or episode end.
    This frozen model is what the compressor receives as input.
    """
    model_config = ConfigDict(frozen=True)

    trajectory_id: str
    request_id: str
    steps: tuple[ReActStep, ...] = Field(default_factory=tuple)
    total_steps: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_total_steps(self) -> "TrajectoryModel":
        if self.total_steps != len(self.steps):
            raise ValueError(
                f"total_steps={self.total_steps} does not match len(steps)={len(self.steps)}"
            )
        return self

    def to_text(self, include_itinerary_snapshots: bool = False) -> str:
        """
        Linearise all steps into the flat text format consumed by the compressor
        and injected into the agent's LLM context.

        Format per step::

            [Step N]
            Thought: <thought>
            Action: <tool_name>(<arguments_json>)
            Observation: <result or error>
            [Itinerary snapshot omitted by default]
        """
        import json as _json
        lines: list[str] = []
        for step in self.steps:
            lines.append(f"[Step {step.step_index}]")
            lines.append(f"Thought: {step.thought}")
            if step.action is not None:
                lines.append(
                    f"Action: {step.action.tool_name}({_json.dumps(step.action.arguments)})"
                )
            if step.observation is not None:
                if step.observation.success:
                    result = step.observation.result
                    # Serialise dict/list as JSON (double quotes, LLM-friendly) rather
                    # than Python repr (single quotes, ambiguous for smaller models).
                    if isinstance(result, (dict, list)):
                        lines.append(f"Observation: {_json.dumps(result)}")
                    else:
                        lines.append(f"Observation: {result}")
                else:
                    lines.append(f"Observation: ERROR — {step.observation.error_message}")
            if include_itinerary_snapshots and step.itinerary_snapshot is not None:
                lines.append(f"[Partial itinerary: {step.itinerary_snapshot.model_dump_json()}]")
            lines.append("")
        return "\n".join(lines)

    def slice_since(self, step_index: int) -> "TrajectoryModel":
        """Return a new TrajectoryModel containing only steps from step_index onward."""
        new_steps = tuple(s for s in self.steps if s.step_index >= step_index)
        return TrajectoryModel(
            trajectory_id=self.trajectory_id,
            request_id=self.request_id,
            steps=new_steps,
            total_steps=len(new_steps),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5. Compressed State
# ══════════════════════════════════════════════════════════════════════════════

class HardConstraintLedger(BaseModel):
    """
    Structured tracking of hard constraint status at compression time.

    The ledger is the most critical section of a CompressedState — it ensures
    the agent never forgets a hard constraint even as old trajectory steps are
    compressed away.
    """
    model_config = ConfigDict(frozen=True)

    constraints: tuple[Constraint, ...] = Field(default_factory=tuple)
    satisfied_ids: tuple[str, ...] = Field(default_factory=tuple)
    violated_ids: tuple[str, ...] = Field(default_factory=tuple)
    unknown_ids: tuple[str, ...] = Field(default_factory=tuple,
                                         description="Constraints not yet evaluable.")

    @property
    def satisfaction_ratio(self) -> float:
        total = len(self.constraints)
        return len(self.satisfied_ids) / total if total > 0 else 0.0


class CompressedState(BaseModel):
    """
    The output of a CompressorBase.compress() call.

    Design: Fixed-template + partial free-form
    ------------------------------------------
    The fixed sections (``hard_constraint_ledger``, ``decisions_made``,
    ``open_questions``, ``key_discoveries``) provide a structured, evaluable
    skeleton that the reward function can interrogate deterministically.

    The free-form sections (``soft_constraints_summary``,
    ``current_itinerary_sketch``) let the compressor express nuance that
    the template designers didn't anticipate.

    ``CompressedStateTemplate.render()`` serialises all sections into the
    string that the agent sees in its context window.
    ``CompressedStateTemplate.parse()`` reconstructs this model from that string.
    """
    model_config = ConfigDict(frozen=True)

    state_id: str
    trajectory_id: str
    step_index: int = Field(ge=0, description="Trajectory step at which compression occurred.")

    # Fixed structured sections
    hard_constraint_ledger: HardConstraintLedger
    soft_constraints_summary: str = Field(description="Free-form soft constraint / preference summary.")
    decisions_made: list[str] = Field(default_factory=list,
                                      description="Confirmed bookings and planning decisions.")
    open_questions: list[str] = Field(default_factory=list,
                                      description="Unresolved planning questions.")
    key_discoveries: list[str] = Field(default_factory=list,
                                       description="Important world facts learned (prices, availability).")
    current_itinerary_sketch: str = Field(description="Compact text summary of the partial itinerary.")

    compression_method: str = Field(description="'llm' | 'transformer' | 'hybrid' | 'llm_mcts'")
    token_count: int | None = None
    created_at: str = Field(description="ISO 8601 datetime.")

    # MCTS-specific fields — None for all non-MCTS compressors.
    # Populated by MCTSAwareCompressor.compress_with_tree() and injected
    # into the agent context by ContextBuilder._history_mcts_compressor().
    top_candidates: list[str] | None = Field(
        default=None,
        description="Top-K candidate plans from MCTS search. None when not using MCTS.",
    )
    tradeoffs: str | None = Field(
        default=None,
        description="Free-form tradeoffs summary from MCTS tree. None when not using MCTS.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Episode Log — the central data contract
# ══════════════════════════════════════════════════════════════════════════════

class ToolCallStats(BaseModel):
    """Aggregate usage statistics for a single tool across an episode."""
    model_config = ConfigDict(frozen=True)

    tool_name: str
    call_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    total_latency_ms: float = Field(ge=0.0)
    avg_latency_ms: float = Field(ge=0.0)
    redundant_call_count: int = Field(default=0, ge=0,
                                      description="Calls with identical (tool, args) hash.")


class RewardComponents(BaseModel):
    """
    Multi-component shaped reward for a planning episode.

    Each component is independently unit-testable and weighted in configs/reward/.
    ``terminal_itinerary_score`` is None for intermediate steps; it is only set
    when RewardFunction.compute() is called with ``is_terminal=True``.
    """
    model_config = ConfigDict(frozen=True)

    hard_constraint_score: float = Field(ge=0.0, le=1.0)
    soft_constraint_score: float = Field(ge=0.0, le=1.0)
    tool_efficiency_score: float = Field(ge=0.0, le=1.0)
    tool_failure_penalty: float = Field(le=0.0)
    logical_consistency_score: float = Field(ge=0.0, le=1.0)
    terminal_itinerary_score: float | None = None
    total_reward: float


class EpisodeLog(BaseModel):
    """
    The universal output of a complete planning episode.

    Design: Central data contract
    ------------------------------
    EpisodeLog is the hand-off between the planning phase and everything
    downstream: RL training, evaluation, logging, and notebooks.

    By capturing everything in one serialisable model, we can:
    - Re-evaluate any saved episode offline without re-running the agent.
    - Track improvement across training checkpoints by re-evaluating logs.
    - Perform dataset-level analysis in notebooks by loading episode JSON files.

    ``config_hash`` is an MD5/SHA of the Hydra config used for this run,
    enabling exact reproducibility verification.
    """
    model_config = ConfigDict(frozen=True)

    episode_id: str
    request_id: str
    agent_mode: str = Field(description="'raw' | 'llm_summary' | 'compressor' | 'mcts_compressor'")
    trajectory: TrajectoryModel
    compressed_states: tuple[CompressedState, ...] = Field(default_factory=tuple)
    final_itinerary: Itinerary | None = None
    reward_components: RewardComponents
    tool_stats: tuple[ToolCallStats, ...] = Field(default_factory=tuple)
    total_steps: int = Field(ge=0)
    total_tokens_used: int | None = None
    mcts_stats: "MCTSStats | None" = Field(
        default=None,
        description="MCTS search statistics per episode. None for non-MCTS episodes.",
    )
    success: bool
    error: str | None = None
    termination_reason: str | None = Field(
        default=None,
        description=(
            "How the episode ended: DONE_ITINERARY | EXIT_<code> | "
            "MAX_STEPS | PARSE_FAILURE | ERROR_<TYPE>. "
            "EXIT codes: CITY_NOT_FOUND, BUDGET_EXCEEDED, DATE_INVALID, "
            "NO_AVAILABILITY, REPEATED_DEAD_END."
        ),
    )
    user_request: "UserRequest | None" = Field(
        default=None,
        description="The UserRequest that prompted this episode. Stored for UI display and offline analysis.",
    )
    config_hash: str
    created_at: str = Field(description="ISO 8601 datetime.")


# ══════════════════════════════════════════════════════════════════════════════
# 7. RL Transition
# ══════════════════════════════════════════════════════════════════════════════

class PPOTransition(BaseModel):
    """
    A single (state, action, reward, value) tuple stored in EpisodeBuffer.

    ``trajectory_text``     — The observation: trajectory text sent to the compressor.
    ``compressed_state_text`` — The action: compressor's full output token sequence.
    ``log_prob``            — log π(a|s) at collection time; used in PPO clipping ratio.
    ``advantage``           — Filled in after Generalised Advantage Estimation (GAE).
    """
    model_config = ConfigDict(frozen=False)  # mutable: advantage is filled in post-hoc

    trajectory_text: str
    compressed_state_text: str
    reward: float
    value_estimate: float
    log_prob: float
    advantage: float | None = None


# ══════════════════════════════════════════════════════════════════════════════
# 8. Evaluation Result
# ══════════════════════════════════════════════════════════════════════════════

class EvalResult(BaseModel):
    """
    Output of ``Evaluator.evaluate_episode()``.

    ``deterministic_scores`` — Keys like 'hard_constraint_ratio', 'tool_efficiency',
                               'budget_adherence', computed without LLM calls.
    ``llm_judge_scores``     — Per-rubric-dimension scores from LLMJudge.score().
    ``judge_model``          — The exact model ID used for the LLM judge,
                               recorded for reproducibility.
    """
    model_config = ConfigDict(frozen=True)

    episode_id: str
    request_id: str
    agent_mode: str
    deterministic_scores: dict[str, float]
    llm_judge_scores: dict[str, float]
    overall_score: float = Field(ge=0.0, le=1.0)
    rubric_breakdown: dict[str, Any] = Field(default_factory=dict)
    judge_model: str
    created_at: str
    metric_version: str = Field(
        default="v1",
        description="Version tag of the metric schema that produced these scores.",
    )

# mcts.node has no dependency on core.models, so this import is safe here.
# Calling model_rebuild() resolves the "MCTSStats | None" forward reference in
# EpisodeLog so Pydantic can fully validate instances at runtime.
from optimized_llm_planning_memory.mcts.node import MCTSStats  # noqa: E402
EpisodeLog.model_rebuild()