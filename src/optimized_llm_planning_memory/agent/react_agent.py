"""
agent/react_agent.py
====================
ReActAgent — the travel planning agent orchestrated via pydantic-ai.

Design
------
``ReActAgent`` implements the think → act → observe loop using pydantic-ai's
``Agent`` class as the underlying LLM client. It is responsible for:

1. Building the LLM context on each step (via ``ContextBuilder``).
2. Parsing the LLM response into a ``ReActStep``.
3. Executing the tool call via ``ToolRegistry``.
4. Recording the step in ``Trajectory``.
5. Calling the compressor at configured intervals.
6. Assembling the final ``EpisodeLog``.

Compression triggers
--------------------
Compression fires when EITHER condition is true:
  - ``total_steps % compress_every_n_steps == 0``
  - ``trajectory.token_count(tokenizer) > compress_on_token_threshold``

The second trigger requires a tokenizer, which is only available if the
compressor is a ``TransformerCompressor``. In LLM_SUMMARY and RAW modes,
only the step-count trigger is used.

pydantic-ai integration
-----------------------
litellm is used as the backend model provider via::

    from pydantic_ai import Agent
    from pydantic_ai.models.litellm import LiteLLMModel

Tools are passed to pydantic-ai at agent construction time. For the ReAct
loop, we use pydantic-ai in a custom way: instead of letting it drive the
full conversation, we call the LLM step-by-step and parse the response
ourselves. This gives us full control over the ReAct format, compression
injection, and episode logging.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import litellm

from optimized_llm_planning_memory.agent.context_builder import ContextBuilder
from optimized_llm_planning_memory.agent.modes import AgentMode
from optimized_llm_planning_memory.agent.trajectory import Trajectory
from optimized_llm_planning_memory.compressor.base import CompressorBase
from optimized_llm_planning_memory.compressor.mcts_aware import MCTSAwareCompressor
from optimized_llm_planning_memory.core.config import AgentConfig
from optimized_llm_planning_memory.core.exceptions import AgentMaxStepsError, AgentParseError
from optimized_llm_planning_memory.core.models import (
    AccommodationBooking,
    ActivityBooking,
    CompressedState,
    EpisodeLog,
    Itinerary,
    ItineraryDay,
    ReActStep,
    RewardComponents,
    ToolCall,
    ToolResult,
    TrajectoryModel,
    TransportSegment,
    UserRequest,
)
from optimized_llm_planning_memory.mcts.controller import MCTSController
from optimized_llm_planning_memory.mcts.node import MCTSStats
from optimized_llm_planning_memory.simulator.protocol import SimulatorProtocol
from optimized_llm_planning_memory.tools.events import EventBus
from optimized_llm_planning_memory.tools.registry import ToolRegistry
from optimized_llm_planning_memory.tools.tracker import ToolCallTracker
from optimized_llm_planning_memory.utils.logging import get_logger

if TYPE_CHECKING:
    from optimized_llm_planning_memory.utils.live_writer import LiveEpisodeWriter

log = get_logger(__name__)


class ReActAgent:
    """
    Travel planning agent using the ReAct (Reasoning + Acting) loop.

    One instance can be reused across multiple episodes. Each call to
    ``run_episode()`` creates fresh per-episode state (trajectory, tracker,
    event bus).

    Parameters
    ----------
    llm_model_id    : litellm model string for the planning LLM.
    tool_registry   : Pre-built registry of tool middleware instances.
    compressor      : Compressor for context compression (None → RAW mode).
    context_builder : Assembles LLM context strings from trajectory + state.
    config          : Agent configuration (max_steps, compress_every_n_steps, etc.).
    mode            : Which context-assembly strategy to use.
    mcts_controller : Optional MCTSController. Required when mode=MCTS_COMPRESSOR.
                      If None and mode=MCTS_COMPRESSOR, compression silently falls
                      back to the standard linear compress() path.
    """

    def __init__(
        self,
        llm_model_id: str,
        tool_registry: ToolRegistry,
        compressor: CompressorBase | None,
        context_builder: ContextBuilder,
        config: AgentConfig,
        mode: AgentMode = AgentMode.COMPRESSOR,
        mcts_controller: MCTSController | None = None,
    ) -> None:
        self._llm_model_id = llm_model_id
        self._tool_registry = tool_registry
        self._compressor = compressor
        self._context_builder = context_builder
        self._config = config
        self._mode = mode
        self._mcts_controller = mcts_controller
        # Per-episode state reset in run_episode()
        self._current_request: UserRequest | None = None
        self._last_mcts_stats: MCTSStats | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run_episode(
        self,
        request: UserRequest,
        simulator: SimulatorProtocol,
        live_writer: "LiveEpisodeWriter | None" = None,
        episode_id: str | None = None,
    ) -> EpisodeLog:
        """
        Run a complete planning episode for ``request``.

        A new ``Trajectory``, ``ToolCallTracker``, and ``EventBus`` are created
        for each episode. The episode ends when:
        - The agent produces ``Action: DONE``.
        - ``max_steps`` is reached (raises ``AgentMaxStepsError``, logged in EpisodeLog).

        Parameters
        ----------
        request     : The travel planning request to fulfil.
        simulator   : A simulator adapter instance for this episode world.
        live_writer : Optional :class:`~optimized_llm_planning_memory.utils.live_writer.LiveEpisodeWriter`.
                      When provided, incremental events are streamed to a JSONL
                      file so the developer UI can display live progress.
        episode_id  : Optional pre-generated episode ID.  When provided, the
                      caller is responsible for creating a ``LiveEpisodeWriter``
                      with the same ID so the live file is named consistently.
                      If None, a fresh UUID is generated.

        Returns
        -------
        EpisodeLog
            Complete structured log of the episode.
        """
        episode_id = episode_id or str(uuid.uuid4())
        # Store on self so _run_compression() can access them without threading issues
        self._current_request = request
        self._last_mcts_stats = None

        log.info("episode.start", episode_id=episode_id, request_id=request.request_id, mode=self._mode.value)

        trajectory = Trajectory(request_id=request.request_id)
        tracker = ToolCallTracker()
        event_bus = EventBus()

        # Rebuild the tool registry with fresh per-episode tracker + event bus
        fresh_registry = ToolRegistry.from_config(
            simulator=simulator,
            tracker=tracker,
            event_bus=event_bus,
        )

        compressed_states: list[CompressedState] = []
        current_compressed: CompressedState | None = None
        final_itinerary: Itinerary | None = None
        success = True
        error_msg: str | None = None

        try:
            for step_index in range(self._config.max_steps):
                log.info("react.step.start", episode_id=episode_id, step=step_index)

                # Build context for this step
                context = self._context_builder.build(
                    trajectory=trajectory,
                    compressed_state=current_compressed,
                    mode=self._mode,
                    request=request,
                )

                # Call the LLM
                llm_response = self._call_llm(context)

                # Parse thought and action
                thought, tool_call = self._parse_response(llm_response)

                log.info(
                    "react.step.thought",
                    episode_id=episode_id,
                    step=step_index,
                    thought_preview=thought[:200],
                )

                # Check for DONE signal
                if tool_call is None or (
                    tool_call.tool_name.upper() == "DONE"
                ):
                    log.info("react.step.done", episode_id=episode_id, step=step_index)
                    step = ReActStep(
                        step_index=step_index,
                        thought=thought,
                        action=None,
                        observation=None,
                        itinerary_snapshot=final_itinerary,
                        timestamp=_now(),
                    )
                    trajectory.add_step(step)
                    if live_writer is not None:
                        live_writer.write_step(step)
                    break

                log.info(
                    "react.step.action",
                    episode_id=episode_id,
                    step=step_index,
                    tool_name=tool_call.tool_name,
                    args_summary=str(tool_call.arguments)[:120],
                )

                # Execute the tool call
                tool_result = self._execute_tool(fresh_registry, tool_call)

                log.info(
                    "react.step.observation",
                    episode_id=episode_id,
                    step=step_index,
                    success=tool_result.success,
                    latency_ms=tool_result.latency_ms,
                    error=tool_result.error_message,
                )

                # Extract partial itinerary from booking tool results
                itinerary_snapshot = self._try_extract_itinerary(
                    thought, tool_result, request.request_id, final_itinerary, tool_call
                )
                if itinerary_snapshot is not None:
                    final_itinerary = itinerary_snapshot

                step = ReActStep(
                    step_index=step_index,
                    thought=thought,
                    action=tool_call,
                    observation=tool_result,
                    itinerary_snapshot=itinerary_snapshot,
                    timestamp=_now(),
                )
                trajectory.add_step(step)

                if live_writer is not None:
                    live_writer.write_step(step)
                    if itinerary_snapshot is not None:
                        live_writer.write_itinerary_update(itinerary_snapshot)

                # Compression check
                if self._should_compress(trajectory, step_index):
                    steps_since = step_index - trajectory.last_compressed_step
                    log.info(
                        "react.compress.start",
                        episode_id=episode_id,
                        step=step_index,
                        steps_since_last=steps_since,
                    )
                    compressed = self._run_compression(trajectory, current_compressed)
                    if compressed is not None:
                        current_compressed = compressed
                        compressed_states.append(compressed)
                        trajectory.mark_compression()
                        log.info(
                            "react.compress.complete",
                            episode_id=episode_id,
                            step=step_index,
                            method=compressed.compression_method,
                            token_count=compressed.token_count,
                        )
                        if live_writer is not None:
                            live_writer.write_compression(compressed)
            else:
                # Loop exhausted without DONE → max_steps reached
                success = False
                error_msg = (
                    f"Max steps ({self._config.max_steps}) reached without DONE signal."
                )

        except AgentMaxStepsError as exc:
            success = False
            error_msg = str(exc)
        except Exception as exc:
            success = False
            error_msg = f"Unexpected error: {type(exc).__name__}: {exc}"

        # Build placeholder reward (will be overwritten by RewardFunction in training)
        reward_components = _zero_reward()

        episode_log = EpisodeLog(
            episode_id=episode_id,
            request_id=request.request_id,
            agent_mode=self._mode.value,
            trajectory=trajectory.to_model(),
            compressed_states=tuple(compressed_states),
            final_itinerary=final_itinerary,
            reward_components=reward_components,
            tool_stats=tuple(tracker.get_stats()),
            total_steps=trajectory.total_steps,
            mcts_stats=self._last_mcts_stats,
            success=success,
            error=error_msg,
            config_hash=self._compute_config_hash(),
            created_at=_now(),
        )

        log.info(
            "episode.complete",
            episode_id=episode_id,
            total_steps=episode_log.total_steps,
            success=episode_log.success,
            error=episode_log.error,
        )

        if live_writer is not None:
            live_writer.write_episode_complete(episode_id)

        return episode_log

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, context: str) -> str:
        """Call the planning LLM with the assembled context string."""
        response = litellm.completion(
            model=self._llm_model_id,
            messages=[{"role": "user", "content": context}],
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens_per_response,
        )
        return response.choices[0].message.content or ""

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_response(self, text: str) -> tuple[str, ToolCall | None]:
        """
        Parse an LLM response into (thought, ToolCall | None).

        Expected format::

            Thought: <text>
            Action: <tool_name>(<json_args>)

        Returns (thought, None) if the action is DONE or absent.
        """
        thought = ""
        tool_call: ToolCall | None = None

        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|$)", text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text, re.DOTALL)
        if action_match:
            action_text = action_match.group(1).strip()

            if action_text.upper() == "DONE":
                return thought, None

            # Parse tool_name(json_args) format
            call_match = re.match(r"(\w+)\((.+)\)$", action_text, re.DOTALL)
            if call_match:
                tool_name = call_match.group(1)
                args_str = call_match.group(2)
                try:
                    arguments = json.loads(args_str)
                except json.JSONDecodeError:
                    arguments = {"_raw": args_str}

                tool_call = ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    raw_text=action_text,
                )

        return thought, tool_call

    # ── Tool execution ────────────────────────────────────────────────────────

    def _execute_tool(
        self,
        registry: ToolRegistry,
        tool_call: ToolCall,
    ) -> ToolResult:
        """
        Execute the tool call via the middleware registry.

        On ``ToolNotFoundError``, returns a ToolResult with a helpful error
        message instead of raising, so the agent can self-correct.
        """
        from optimized_llm_planning_memory.core.exceptions import ToolNotFoundError

        try:
            tool = registry.get(tool_call.tool_name)
            return tool.call(tool_call.arguments)
        except ToolNotFoundError:
            available = registry.tool_names()
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                result=None,
                error_message=(
                    f"Tool '{tool_call.tool_name}' does not exist. "
                    f"Available tools: {available}."
                ),
                latency_ms=0.0,
            )

    # ── Compression ───────────────────────────────────────────────────────────

    def _should_compress(self, trajectory: Trajectory, step_index: int) -> bool:
        """Return True if a compression event should fire at this step."""
        if self._mode not in (
            AgentMode.LLM_SUMMARY,
            AgentMode.COMPRESSOR,
            AgentMode.MCTS_COMPRESSOR,
        ):
            return False
        if self._compressor is None:
            return False
        steps_since = step_index - trajectory.last_compressed_step
        return steps_since >= self._config.compress_every_n_steps

    def _run_compression(
        self,
        trajectory: Trajectory,
        previous_state: CompressedState | None,
    ) -> CompressedState | None:
        """
        Invoke the compressor; return None on failure (logged but not raised).

        When mode is MCTS_COMPRESSOR and an MCTSController + MCTSAwareCompressor
        are available, runs MCTS search first and distills the resulting tree.
        Otherwise falls back to the standard linear compress() path.
        """
        if self._compressor is None:
            return None
        try:
            if (
                self._mode == AgentMode.MCTS_COMPRESSOR
                and self._mcts_controller is not None
                and isinstance(self._compressor, MCTSAwareCompressor)
                and self._current_request is not None
            ):
                tree_repr = self._mcts_controller.search(
                    trajectory=trajectory.to_model(),
                    compressed_state=previous_state,
                    request=self._current_request,
                )
                # Store most recent MCTS stats for EpisodeLog (last compression wins)
                self._last_mcts_stats = tree_repr.stats
                log.info(
                    "react.mcts.search_complete",
                    nodes_explored=tree_repr.stats.nodes_explored,
                    root_value=tree_repr.stats.root_value,
                    num_simulations=tree_repr.stats.num_simulations,
                    max_depth_reached=tree_repr.stats.max_depth_reached,
                )
                return self._compressor.compress_with_tree(tree_repr, previous_state)
            else:
                return self._compressor.compress(trajectory.to_model(), previous_state)
        except Exception:
            log.warning("react.compress.failed", exc_info=True)
            return None

    # ── Itinerary extraction ──────────────────────────────────────────────────

    def _try_extract_itinerary(
        self,
        thought: str,
        observation: ToolResult,
        request_id: str,
        current_itinerary: Itinerary | None,
        tool_call: ToolCall | None,
    ) -> Itinerary | None:
        """
        Build or update the partial Itinerary from a successful booking tool call.

        Handles three tool names:
        - ``book_hotel``   → creates AccommodationBooking on the check_in day.
        - ``book_event``   → creates ActivityBooking on the event date.
        - ``select_flight``→ creates TransportSegment on the departure day.

        For all other tools (search calls, info tools, etc.) or failed calls,
        returns ``current_itinerary`` unchanged.
        """
        if tool_call is None or not observation.success:
            return current_itinerary

        result = observation.result
        if result is None:
            return current_itinerary

        # Normalise result to dict
        if not isinstance(result, dict):
            try:
                result = result.model_dump()
            except Exception:
                try:
                    result = dict(result)
                except Exception:
                    return current_itinerary

        itinerary = current_itinerary or Itinerary(
            itinerary_id=request_id,
            request_id=request_id,
        )

        tool_name = tool_call.tool_name.lower()

        if tool_name == "book_hotel":
            hotel_id = result.get("hotel_id", "")
            hotel_name = result.get("hotel_name", hotel_id)
            check_in = result.get("check_in", "")
            check_out = result.get("check_out", "")
            price_per_night = float(result.get("price_per_night") or 0.0)
            total_cost = float(result.get("total_cost") or 0.0)
            booking_ref = result.get("booking_id")
            city = tool_call.arguments.get("city_id", "")

            booking = AccommodationBooking(
                hotel_id=hotel_id,
                hotel_name=hotel_name,
                city=city,
                check_in=check_in,
                check_out=check_out,
                cost_per_night_usd=price_per_night,
                total_cost_usd=total_cost,
                booking_ref=booking_ref,
            )
            day = self._find_or_create_day(itinerary, check_in, city)
            day.accommodation = booking

        elif tool_name == "book_event":
            event_id = result.get("event_id", "")
            event_name = result.get("event_name", event_id)
            total_cost = float(result.get("total_cost") or 0.0)
            booking_ref = result.get("booking_id")
            venue = result.get("venue_name", "")
            city = tool_call.arguments.get("city_id", "")
            # Prefer a date embedded in the result; fall back to today
            start_dt = result.get("start_datetime") or _now()[:10]

            activity = ActivityBooking(
                activity_id=event_id,
                activity_name=event_name,
                location=venue,
                city=city,
                start_datetime=start_dt,
                duration_hours=1.0,
                cost_usd=total_cost,
                category="event",
                booking_ref=booking_ref,
            )
            event_date = start_dt[:10] if len(start_dt) >= 10 else start_dt
            day = self._find_or_create_day(itinerary, event_date, city)
            day.activities.append(activity)

        elif tool_name == "select_flight":
            origin = result.get("origin_city_name", "")
            dest = result.get("destination_city_name", "")
            departure_dt = result.get("departure_datetime", "")
            arrival_dt = result.get("arrival_datetime", "")
            total_price = float(result.get("total_price") or 0.0)
            booking_ref = result.get("booking_id")

            segment = TransportSegment(
                mode="flight",
                from_location=origin,
                to_location=dest,
                departure_datetime=departure_dt,
                arrival_datetime=arrival_dt,
                cost_usd=total_price,
                booking_ref=booking_ref,
            )
            dep_date = departure_dt[:10] if len(departure_dt) >= 10 else departure_dt
            day = self._find_or_create_day(itinerary, dep_date, origin)
            day.transport_segments.append(segment)

        else:
            return current_itinerary

        itinerary.recompute_total_cost()
        return itinerary

    def _find_or_create_day(
        self, itinerary: Itinerary, date: str, city: str
    ) -> ItineraryDay:
        """Return the ItineraryDay for ``date``, creating it if absent."""
        for day in itinerary.days:
            if day.date == date:
                return day
        new_day = ItineraryDay(date=date, city=city)
        itinerary.days.append(new_day)
        return new_day

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_config_hash(self) -> str:
        """Compute a short hash of the config for reproducibility tracking."""
        config_str = json.dumps(self._config.model_dump(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _zero_reward() -> RewardComponents:
    """Placeholder reward (overwritten by RewardFunction in training)."""
    return RewardComponents(
        hard_constraint_score=0.0,
        soft_constraint_score=0.0,
        tool_efficiency_score=0.0,
        tool_failure_penalty=0.0,
        logical_consistency_score=0.0,
        terminal_itinerary_score=None,
        total_reward=0.0,
    )
