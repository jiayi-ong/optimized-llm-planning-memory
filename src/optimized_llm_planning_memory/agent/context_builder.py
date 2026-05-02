"""
agent/context_builder.py
========================
ContextBuilder — the single location for all mode-switching context logic.

Design pattern: Strategy (via AgentMode)
-----------------------------------------
All three evaluation conditions are implemented here. ``ReActAgent`` calls
``ContextBuilder.build()`` on every ReAct step and passes the resulting string
to the LLM. Switching modes requires only changing ``AgentMode`` in config.

The separation of context assembly into its own class ensures that:
  1. The mode-switching logic is testable in isolation.
  2. ``ReActAgent`` has no knowledge of compression internals.
  3. New modes (e.g., hybrid raw+compressed) can be added without touching
     the agent class.

Context format
--------------
All modes produce a string with the structure::

    [SYSTEM]
    <system prompt from prompts.py>

    [USER REQUEST]
    <user_request.raw_text>
    --- Structured trip details (authoritative) ---
    Route / Dates / Budget / Travelers
    Hard constraints / Soft constraints / Preferences

    [CURRENT ITINERARY STATE]
    <confirmed bookings so far, or "No bookings confirmed yet.">

    [CONTEXT]
    <mode-specific history>

    [AVAILABLE TOOLS]
    <tool list from ToolRegistry>
"""

from __future__ import annotations

import litellm

from optimized_llm_planning_memory.agent.modes import AgentMode
from optimized_llm_planning_memory.agent.trajectory import Trajectory
from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate
from optimized_llm_planning_memory.core.models import CompressedState, Itinerary, UserRequest
from optimized_llm_planning_memory.tools.registry import ToolRegistry


class ContextBuilder:
    """
    Assembles the full LLM context string for each ReAct step.

    Parameters
    ----------
    system_prompt   : The agent system prompt string (from ``prompts.py``).
    tool_registry   : Used to list available tools at the bottom of the context.
    llm_model_id    : Only used in LLM_SUMMARY mode; the model that summarises
                      the old trajectory prefix. Same litellm model string format.
    summary_max_tokens : Max tokens for the LLM summary call.
    """

    def __init__(
        self,
        system_prompt: str,
        tool_registry: ToolRegistry,
        llm_model_id: str = "openai/gpt-4o-mini",
        summary_max_tokens: int = 512,
    ) -> None:
        self._system_prompt = system_prompt
        self._tool_registry = tool_registry
        self._llm_model_id = llm_model_id
        self._summary_max_tokens = summary_max_tokens
        self._template = CompressedStateTemplate()
        # Cache LLM summaries keyed by (trajectory_object_id, last_compressed_step).
        # Regenerating the summary on every step is O(max_steps) LLM calls per episode;
        # the cache reduces this to O(num_compressions) since recent steps are appended
        # verbatim anyway (M11 fix).
        self._summary_cache: dict[tuple[int, int], str] = {}

    def build(
        self,
        trajectory: Trajectory,
        compressed_state: CompressedState | None,
        mode: AgentMode,
        request: UserRequest,
        itinerary: Itinerary | None = None,
    ) -> str:
        """
        Assemble the full context string for the current ReAct step.

        Parameters
        ----------
        trajectory       : Current in-progress episode trajectory.
        compressed_state : Latest CompressedState (None if no compression yet).
        mode             : Which context-assembly strategy to use.
        request          : The user's travel request.
        itinerary        : Current partial itinerary (None if no bookings yet).
                           Injected as [CURRENT ITINERARY STATE] so the agent
                           always knows what has been confirmed.

        Returns
        -------
        str
            The complete context string passed as the LLM prompt.
        """
        history = self._build_history(trajectory, compressed_state, mode)
        tools_section = self._build_tools_section()
        itinerary_section = self._build_itinerary_section(itinerary)
        request_section = self._build_request_section(request)

        parts = [
            "[SYSTEM]",
            self._system_prompt,
            "",
            "[USER REQUEST]",
            request_section,
            "",
            "[CURRENT ITINERARY STATE]",
            itinerary_section,
            "",
            "[CONTEXT]",
            history,
            "",
            "[AVAILABLE TOOLS]",
            tools_section,
        ]
        return "\n".join(parts)

    # ── Mode implementations ──────────────────────────────────────────────────

    def _build_history(
        self,
        trajectory: Trajectory,
        compressed_state: CompressedState | None,
        mode: AgentMode,
    ) -> str:
        if mode == AgentMode.RAW:
            return self._history_raw(trajectory)
        elif mode == AgentMode.LLM_SUMMARY:
            return self._history_llm_summary(trajectory, compressed_state)
        elif mode == AgentMode.COMPRESSOR:
            return self._history_compressor(trajectory, compressed_state)
        elif mode == AgentMode.MCTS_COMPRESSOR:
            return self._history_mcts_compressor(trajectory, compressed_state)
        else:
            raise ValueError(f"Unknown AgentMode: {mode}")

    def _history_raw(self, trajectory: Trajectory) -> str:
        """Baseline 1: inject the full raw trajectory."""
        if trajectory.total_steps == 0:
            return "(No steps yet — begin planning.)"
        return trajectory.to_text()

    def _history_llm_summary(
        self,
        trajectory: Trajectory,
        compressed_state: CompressedState | None,
    ) -> str:
        """
        Baseline 2: summarise the old trajectory prefix with an LLM call.

        If ``compressed_state`` is None (no prior compression), falls back
        to RAW mode to avoid an unnecessary LLM call at the first step.
        Recent steps (since last compression) are always appended verbatim.
        """
        if compressed_state is None:
            return self._history_raw(trajectory)

        # Cache by (object identity of trajectory, last compressed step index).
        # The summary content only changes when a new compression fires and
        # last_compressed_step advances; between compressions the same text applies.
        cache_key = (id(trajectory), trajectory.last_compressed_step)
        if cache_key not in self._summary_cache:
            self._summary_cache[cache_key] = self._llm_summarise(trajectory)
        summary = self._summary_cache[cache_key]

        recent_steps = trajectory.steps_since_last_compression()
        recent_text = _steps_to_text(recent_steps) if recent_steps else ""

        parts = ["[SUMMARY OF PRIOR PLANNING]", summary]
        if recent_text:
            parts += ["", "[RECENT STEPS]", recent_text]
        return "\n".join(parts)

    def _history_compressor(
        self,
        trajectory: Trajectory,
        compressed_state: CompressedState | None,
    ) -> str:
        """
        Our method: inject the CompressedState + recent steps since last compression.
        """
        if compressed_state is None:
            return self._history_raw(trajectory)

        compressed_text = self._template.render(compressed_state)
        recent_steps = trajectory.steps_since_last_compression()
        recent_text = _steps_to_text(recent_steps) if recent_steps else ""

        parts = ["[COMPRESSED MEMORY STATE]", compressed_text]
        if recent_text:
            parts += ["", "[RECENT STEPS (NOT YET COMPRESSED)]", recent_text]
        return "\n".join(parts)

    def _history_mcts_compressor(
        self,
        trajectory: Trajectory,
        compressed_state: CompressedState | None,
    ) -> str:
        """
        MCTS_COMPRESSOR mode: inject the CompressedState (same as COMPRESSOR mode)
        plus optional [TOP CANDIDATE PLANS FROM SEARCH] and [TRADEOFFS] sections
        that carry the multi-hypothesis information from the MCTS tree.

        Falls back to raw mode if no compressed state exists yet.
        """
        if compressed_state is None:
            return self._history_raw(trajectory)

        compressed_text = self._template.render(compressed_state)
        recent_steps = trajectory.steps_since_last_compression()
        recent_text = _steps_to_text(recent_steps) if recent_steps else ""

        parts = ["[COMPRESSED MEMORY STATE (MCTS)]", compressed_text]

        if compressed_state.top_candidates:
            candidates_text = "\n".join(
                f"  {i + 1}. {cand}"
                for i, cand in enumerate(compressed_state.top_candidates)
            )
            parts += ["", "[TOP CANDIDATE PLANS FROM SEARCH]", candidates_text]

        if compressed_state.tradeoffs:
            parts += ["", "[TRADEOFFS]", compressed_state.tradeoffs]

        if recent_text:
            parts += ["", "[RECENT STEPS (NOT YET COMPRESSED)]", recent_text]

        return "\n".join(parts)

    def _llm_summarise(self, trajectory: Trajectory) -> str:
        """Call the LLM to produce a plain-text summary of the trajectory."""
        response = litellm.completion(
            model=self._llm_model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a travel planning assistant. Summarise the following "
                        "planning history concisely, preserving all key decisions, "
                        "constraints, and discoveries. Do not lose hard constraints."
                    ),
                },
                {"role": "user", "content": trajectory.to_text()},
            ],
            temperature=0.0,
            max_tokens=self._summary_max_tokens,
        )
        return response.choices[0].message.content or ""

    def _build_request_section(self, request: UserRequest) -> str:
        """
        Render the user request as raw_text PLUS a structured fact block.

        The structured block ensures the agent always sees the exact trip dates
        and budget even when raw_text is vague or omits the year.
        """
        profile = request.traveler_profile
        travelers = f"{profile.num_adults} adult{'s' if profile.num_adults != 1 else ''}"
        if profile.num_children:
            travelers += f", {profile.num_children} child{'ren' if profile.num_children != 1 else ''}"

        route = f"{request.origin_city} → {', '.join(request.destination_cities)}"

        lines = [
            request.raw_text,
            "",
            "--- Structured trip details (authoritative) ---",
            f"Route:   {route}",
            f"Dates:   {request.start_date} to {request.end_date}  [USE THESE EXACT DATES FOR ALL BOOKINGS]",
            f"Budget:  ${request.budget_usd:,.2f} USD (total)",
            f"Travelers: {travelers}",
        ]

        if request.hard_constraints:
            lines.append("Hard constraints (must satisfy all):")
            for c in request.hard_constraints:
                lines.append(f"  - {c.description}")

        if request.soft_constraints:
            lines.append("Soft constraints (satisfy where possible):")
            for c in request.soft_constraints:
                lines.append(f"  - {c.description}")

        if request.preferences:
            lines.append("Preferences: " + "; ".join(request.preferences))

        return "\n".join(lines)

    def _build_itinerary_section(self, itinerary: Itinerary | None) -> str:
        """
        Render the current partial itinerary as readable text for the agent.

        Gives the agent an unambiguous view of what has already been confirmed
        so it does not double-book or forget prior bookings. Each booking shows
        its booking_ref so the agent can use cancel_booking() if needed.
        """
        if itinerary is None or not itinerary.days:
            return "No bookings confirmed yet."

        lines: list[str] = [f"Total confirmed cost: ${itinerary.total_cost_usd:.2f}"]
        for day in sorted(itinerary.days, key=lambda d: d.date):
            lines.append(f"\n[{day.date}] {day.city}")
            for seg in day.transport_segments:
                ref = seg.booking_ref or "no-ref"
                lines.append(
                    f"  FLIGHT: {seg.from_location} → {seg.to_location} "
                    f"dep={seg.departure_datetime} "
                    f"arr={seg.arrival_datetime} "
                    f"${seg.cost_usd:.2f} (ref={ref})"
                )
            if day.accommodation:
                acc = day.accommodation
                ref = acc.booking_ref or "no-ref"
                lines.append(
                    f"  HOTEL: {acc.hotel_name} ({acc.city}) "
                    f"{acc.check_in} to {acc.check_out} "
                    f"${acc.total_cost_usd:.2f} (ref={ref})"
                )
            for act in day.activities:
                ref = act.booking_ref or "no-ref"
                lines.append(
                    f"  ACTIVITY: {act.activity_name} @ {act.location} "
                    f"{act.start_datetime} "
                    f"${act.cost_usd:.2f} (ref={ref})"
                )
        return "\n".join(lines)

    def _build_tools_section(self) -> str:
        """Format the tool list for injection into the system prompt."""
        tools = self._tool_registry.list_tools()
        lines = []
        for tool in tools:
            lines.append(f"- {tool['name']}: {tool['description']}")
        return "\n".join(lines) if lines else "(No tools available)"


def _steps_to_text(steps: list) -> str:
    """Linearise a list of ReActSteps to text (without itinerary snapshots)."""
    import json
    lines: list[str] = []
    for step in steps:
        lines.append(f"[Step {step.step_index}]")
        lines.append(f"Thought: {step.thought}")
        if step.action is not None:
            lines.append(
                f"Action: {step.action.tool_name}({json.dumps(step.action.arguments)})"
            )
        if step.observation is not None:
            if step.observation.success:
                lines.append(f"Observation: {step.observation.result}")
            else:
                lines.append(f"Observation: ERROR — {step.observation.error_message}")
        lines.append("")
    return "\n".join(lines)
