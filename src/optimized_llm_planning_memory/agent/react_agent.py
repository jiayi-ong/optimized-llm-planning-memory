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
EpisodeLog.model_rebuild()  # resolve MCTSStats forward reference
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
        termination_reason: str | None = None

        try:
            for step_index in range(self._config.max_steps):
                log.info("react.step.start", episode_id=episode_id, step=step_index)

                # Build context for this step (includes current partial itinerary)
                context = self._context_builder.build(
                    trajectory=trajectory,
                    compressed_state=current_compressed,
                    mode=self._mode,
                    request=request,
                    itinerary=final_itinerary,
                )

                # Call the LLM (with format-reminder retries on parse failure)
                thought, tool_call, is_done, exit_reason = self._call_and_parse(
                    context, step_index, episode_id
                )

                log.info(
                    "react.step.thought",
                    episode_id=episode_id,
                    step=step_index,
                    thought_preview=thought[:200],
                )

                # Terminal signal: DONE (itinerary complete) or EXIT (lethal scenario)
                if is_done or (tool_call is not None and tool_call.tool_name.upper() == "DONE"):
                    termination_reason = f"EXIT_{exit_reason}" if exit_reason else "DONE_ITINERARY"
                    log.info(
                        "react.step.terminal",
                        episode_id=episode_id,
                        step=step_index,
                        termination_reason=termination_reason,
                    )
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

                # Parse failure after all retries — record and end with error
                if tool_call is None:
                    success = False
                    termination_reason = "PARSE_FAILURE"
                    error_msg = (
                        f"LLM did not produce a valid Action after "
                        f"{self._config.max_retries_per_action} attempts at step {step_index}."
                    )
                    log.error("react.parse.failed", episode_id=episode_id, step=step_index, error=error_msg)
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
                # Loop exhausted without terminal signal → max_steps reached
                success = False
                termination_reason = "MAX_STEPS"
                error_msg = (
                    f"Max steps ({self._config.max_steps}) reached without terminal signal."
                )

        except AgentMaxStepsError as exc:
            success = False
            termination_reason = "MAX_STEPS"
            error_msg = str(exc)
        except Exception as exc:
            success = False
            termination_reason = f"ERROR_{type(exc).__name__.upper()}"
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
            termination_reason=termination_reason,
            user_request=request,
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

    def run_steps(
        self,
        n: int,
        trajectory: Trajectory,
        registry: ToolRegistry,
        compressed_state: "CompressedState | None",
        request: UserRequest,
        start_step_index: int = 0,
        final_itinerary: "Itinerary | None" = None,
    ) -> "tuple[Itinerary | None, bool, str | None]":
        """
        Execute up to ``n`` ReAct steps, modifying ``trajectory`` in place.

        Used by ``CompressionEnv.step()`` to run a fixed-length window of ReAct
        steps between compression events. Unlike ``run_episode()``, this method
        does not create its own Trajectory or ToolRegistry — the caller owns those
        objects so that state accumulates correctly across multiple calls.

        Parameters
        ----------
        n                : Maximum number of ReAct steps to execute in this window.
        trajectory       : Live Trajectory (modified in place).
        registry         : Tool middleware registry for this episode.
        compressed_state : Current compressed state to inject into context (None → raw).
        request          : The travel planning request for this episode.
        start_step_index : ReAct step index of the first step in this window.
        final_itinerary  : Partial itinerary carried over from prior windows.

        Returns
        -------
        final_itinerary : Updated partial itinerary (None if no bookings yet).
        done            : True if the episode ended (DONE signal or LLM error).
        error_msg       : Non-None string if done due to an error.
        """
        self._current_request = request
        done = False
        error_msg: str | None = None

        for i in range(n):
            step_index = start_step_index + i

            context = self._context_builder.build(
                trajectory=trajectory,
                compressed_state=compressed_state,
                mode=self._mode,
                request=request,
                itinerary=final_itinerary,
            )

            try:
                thought, tool_call, is_done, _exit_reason = self._call_and_parse(context, step_index)
            except Exception as exc:
                error_msg = f"LLM error at step {step_index}: {type(exc).__name__}: {exc}"
                done = True
                break

            if is_done or (tool_call is not None and tool_call.tool_name.upper() == "DONE"):
                step = ReActStep(
                    step_index=step_index,
                    thought=thought,
                    action=None,
                    observation=None,
                    itinerary_snapshot=final_itinerary,
                    timestamp=_now(),
                )
                trajectory.add_step(step)
                done = True
                break

            if tool_call is None:
                error_msg = (
                    f"LLM did not produce a valid Action after "
                    f"{self._config.max_retries_per_action} attempts at step {step_index}."
                )
                done = True
                break

            tool_result = self._execute_tool(registry, tool_call)

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

        return final_itinerary, done, error_msg

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, context: str) -> str:
        """Call the planning LLM with the assembled context string.

        Splits the leading [SYSTEM] block into a proper system message so
        OpenAI models follow the ReAct format instructions reliably.
        """
        system_content = ""
        user_content = context
        if context.startswith("[SYSTEM]\n"):
            split_marker = "\n[USER REQUEST]"
            idx = context.find(split_marker)
            if idx != -1:
                system_content = context[len("[SYSTEM]\n"):idx].strip()
                user_content = context[idx + 1:]  # drop leading \n before [USER REQUEST]

        messages: list[dict[str, str]] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})

        response = litellm.completion(
            model=self._llm_model_id,
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens_per_response,
        )
        return response.choices[0].message.content or ""

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_response(self, text: str) -> tuple[str, ToolCall | None, bool, str | None]:
        """
        Parse an LLM response into (thought, ToolCall | None, is_done, exit_reason).

        Supported terminal signals::

            Action: DONE            → is_done=True,  exit_reason=None
            Action: EXIT(reason=X)  → is_done=True,  exit_reason="X" (uppercased)

        For regular tool calls:  is_done=False, exit_reason=None.
        On parse failure:        tool_call=None, is_done=False, exit_reason=None.

        Tolerates: multi-line JSON args, code fences, extra whitespace, case variation.
        """
        thought = ""
        tool_call: ToolCall | None = None

        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Alternation order matters: EXIT must precede the generic \w+\(...\) branch.
        action_match = re.search(
            r"Action:\s*(DONE|EXIT\([^)]*\)|\w+\(.*?\))",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if action_match:
            action_text = action_match.group(1).strip()

            if action_text.upper() == "DONE":
                return thought, None, True, None

            # EXIT(reason=<code>) — graceful abort for lethal scenarios
            exit_match = re.match(r"EXIT\(reason=([^)]+)\)", action_text, re.IGNORECASE)
            if exit_match:
                exit_reason = exit_match.group(1).strip().upper()
                return thought, None, True, exit_reason

            # Strip surrounding code fences that LLMs sometimes emit
            action_text = re.sub(r"^```(?:\w+)?\s*", "", action_text)
            action_text = re.sub(r"\s*```$", "", action_text).strip()

            # Parse tool_name(json_args) format.
            # Use (.*) not (.+) so that no-arg calls like get_available_routes() match.
            call_match = re.match(r"(\w+)\((.*)\)\s*$", action_text, re.DOTALL)
            if call_match:
                tool_name = call_match.group(1)
                args_str = call_match.group(2).strip()
                # Strip code fences inside the args block too
                args_str = re.sub(r"^```(?:json)?\s*", "", args_str)
                args_str = re.sub(r"\s*```$", "", args_str).strip()
                if not args_str:
                    # LLM omitted the braces entirely: get_available_routes() → {}
                    arguments = {}
                else:
                    try:
                        arguments = json.loads(args_str)
                    except json.JSONDecodeError:
                        log.warning(
                            "react.parse.json_error",
                            action_text=action_text[:120],
                            args_preview=args_str[:80],
                        )
                        arguments = {"_raw": args_str}

                tool_call = ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    raw_text=action_text,
                )

        return thought, tool_call, False, None

    def _call_and_parse(
        self,
        context: str,
        step_index: int,
        episode_id: str = "unknown",
    ) -> tuple[str, ToolCall | None, bool, str | None]:
        """
        Call the LLM and parse, retrying with a format reminder when no Action
        line is produced.  Terminal signals (DONE, EXIT) are never retried.

        Returns (thought, tool_call, is_done, exit_reason).
        ``exit_reason`` is the EXIT reason code or None (see _parse_response).
        ``tool_call`` is None after all retries failed (is_done will be False).
        """
        _FORMAT_REMINDER = (
            "\n\n[FORMAT REMINDER]\n"
            "Your previous response did not include a valid 'Action:' line.\n"
            "Every response MUST end with exactly one of:\n"
            "  Action: tool_name({\"key\": \"value\"})\n"
            "  Action: DONE\n"
            "  Action: EXIT(reason=<code>)\n"
        )
        thought, tool_call, is_done, exit_reason = "", None, False, None
        for attempt in range(self._config.max_retries_per_action):
            prompt = context if attempt == 0 else context + _FORMAT_REMINDER
            llm_response = self._call_llm(prompt)
            thought, tool_call, is_done, exit_reason = self._parse_response(llm_response)
            if tool_call is not None or is_done:
                return thought, tool_call, is_done, exit_reason
            log.warning(
                "react.parse.no_action",
                episode_id=episode_id,
                step=step_index,
                attempt=attempt + 1,
                max_retries=self._config.max_retries_per_action,
                response_preview=llm_response[:300],
            )
        return thought, None, False, None

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

        # Unwrap the redundancy-warning envelope BaseTool.call() injects on 3+ repeat calls.
        # Without this, result.get("booking_id") returns None (keys are nested one level down).
        if "result" in result and "agent_warning" in result:
            inner = result["result"]
            if not isinstance(inner, dict):
                return current_itinerary
            result = inner

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

        elif tool_name == "cancel_booking":
            booking_ref = result.get("cancelled_booking_ref")
            if not booking_ref or current_itinerary is None:
                return current_itinerary  # nothing to remove from
            _remove_booking(itinerary, booking_ref)
            itinerary.recompute_total_cost()
            return itinerary

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


def _remove_booking(itinerary: "Itinerary", booking_ref: str) -> None:
    """
    Remove the first item matching ``booking_ref`` from the itinerary in place.

    Checks transport_segments, accommodation, and activities across all days.
    Days that become empty after removal are retained (they may still be
    meaningful calendar placeholders).
    """
    for day in itinerary.days:
        # Transport segments
        before = len(day.transport_segments)
        day.transport_segments = [
            s for s in day.transport_segments if s.booking_ref != booking_ref
        ]
        if len(day.transport_segments) < before:
            return

        # Accommodation
        if day.accommodation is not None and day.accommodation.booking_ref == booking_ref:
            day.accommodation = None
            return

        # Activities
        before = len(day.activities)
        day.activities = [a for a in day.activities if a.booking_ref != booking_ref]
        if len(day.activities) < before:
            return


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
