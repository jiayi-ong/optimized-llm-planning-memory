"""
tools/base.py
=============
BaseTool ABC — the middleware foundation for all travel planning tools.

Design pattern: Template Method
---------------------------------
``call()`` is the template method. It orchestrates the full lifecycle of a
tool invocation in a fixed order:

    1. Validate raw_arguments against ``input_schema`` (Pydantic)
    2. Call ``_execute(validated_input)`` [subclass responsibility]
    3. Record to ToolCallTracker
    4. Emit ToolEvent to EventBus
    5. Return ToolResult

Subclasses override ONLY ``_execute()``. They never override ``call()``,
which guarantees every tool gets validation, tracking, and feedback for free.

Design pattern: ABC
--------------------
``BaseTool`` is an ABC (not a Protocol) because:
- All tools share concrete behaviour in ``call()`` (template method).
- We want Python to enforce ``_execute()`` implementation at class definition
  time (``@abstractmethod``), not at runtime.

Contrast with ``SimulatorProtocol`` where no shared behaviour exists and
the external library's class can't inherit from our ABC.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError

from optimized_llm_planning_memory.core.exceptions import ToolExecutionError
from optimized_llm_planning_memory.core.models import ToolResult
from optimized_llm_planning_memory.simulator.protocol import SimulatorProtocol
from optimized_llm_planning_memory.tools.events import EventBus, ToolEvent
from optimized_llm_planning_memory.tools.tracker import EpisodeTimer, ToolCallTracker


class BaseTool(ABC):
    """
    Abstract base class for all tool middleware implementations.

    Subclass checklist
    ------------------
    1. Set class attributes ``tool_name``, ``tool_description``, ``input_schema``.
    2. Implement ``_execute(validated_input)``.
    3. Optionally override ``_generate_error_feedback()`` for custom hints.

    Attributes
    ----------
    tool_name        : str   — Unique tool identifier; must match what the LLM produces.
    tool_description : str   — Shown to the agent in its system prompt.
    input_schema     : type  — Pydantic model class; ``call()`` validates against this.
    """

    # Subclasses MUST set these at class level
    tool_name: str
    tool_description: str
    input_schema: type[BaseModel]

    def __init__(
        self,
        simulator: SimulatorProtocol,
        tracker: ToolCallTracker,
        event_bus: EventBus,
    ) -> None:
        """
        Parameters
        ----------
        simulator  : Simulator adapter; subclass calls its methods in ``_execute()``.
        tracker    : Per-episode usage recorder.
        event_bus  : Episode-scoped event bus; tool emits outcome events here.
        """
        self._simulator = simulator
        self._tracker = tracker
        self._event_bus = event_bus

    # ── Public Template Method ────────────────────────────────────────────────

    def call(self, raw_arguments: dict[str, Any]) -> ToolResult:
        """
        Execute a tool call. This is the ONLY public method; agents call this.

        Subclasses must NOT override this method.

        Steps
        -----
        1. Validate ``raw_arguments`` against ``self.input_schema``.
           → On ValidationError: return ToolResult(success=False) with hint.
        2. Call ``self._execute(validated_input)``.
           → On unexpected exception: return ToolResult(success=False) with hint.
        3. Record to tracker and emit event.
        4. Return ToolResult.
        """
        args_hash = ToolCallTracker.hash_arguments(self.tool_name, raw_arguments)

        # Check for redundant repeat before executing — used to inject a warning below.
        # The tracker records AFTER execution, so prior_count reflects only earlier calls.
        prior_count = self._tracker.call_count_for_hash(self.tool_name, args_hash)

        # Step 1: Validate input
        try:
            validated = self.input_schema.model_validate(raw_arguments)
        except ValidationError as exc:
            feedback = self._generate_validation_feedback(exc, raw_arguments)
            self._record_and_emit(
                success=False, latency_ms=0.0, args_hash=args_hash,
                result=None, error=feedback,
            )
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                result=None,
                error_message=feedback,
                latency_ms=0.0,
            )

        # Step 2: Execute
        with EpisodeTimer() as timer:
            try:
                result = self._execute(validated)
                error_msg = None
                success = True
            except ToolExecutionError as exc:
                result = None
                error_msg = str(exc)
                success = False
            except Exception as exc:
                result = None
                error_msg = self._generate_error_feedback(exc, raw_arguments)
                success = False

        # Inject a redundancy warning into the result after the 2nd repeat (3rd+ call).
        # This surfaces in the agent's Observation text so it knows to stop retrying.
        # Only applied when prior_count >= 2 to allow one legitimate retry after an error.
        if success and prior_count >= 2:
            result = {
                "result": result,
                "agent_warning": (
                    f"[REDUNDANT CALL #{prior_count + 1}] '{self.tool_name}' has been called "
                    f"with these identical arguments {prior_count + 1} times. "
                    f"The result will not change. Either change your approach or "
                    f"use Action: EXIT(reason=REPEATED_DEAD_END)."
                ),
            }

        # Steps 3 + 4
        self._record_and_emit(
            success=success,
            latency_ms=timer.elapsed_ms,
            args_hash=args_hash,
            result=result,
            error=error_msg,
        )

        return ToolResult(
            tool_name=self.tool_name,
            success=success,
            result=result,
            error_message=error_msg,
            latency_ms=timer.elapsed_ms,
        )

    # ── Abstract hook ─────────────────────────────────────────────────────────

    @abstractmethod
    def _execute(self, validated_input: BaseModel) -> Any:
        """
        Perform the actual simulator call and return the raw result.

        Parameters
        ----------
        validated_input : An instance of ``self.input_schema`` — already validated.

        Returns
        -------
        Any
            The result to include in ``ToolResult.result``. Typically a ``list[dict]``
            or ``dict`` matching a ``simulator/schemas.py`` model.

        Raises
        ------
        ToolExecutionError
            On expected, handleable failures (e.g., resource not found).
        Exception
            Unexpected failures are caught by ``call()`` and turned into
            structured error feedback.
        """

    # ── Error feedback ────────────────────────────────────────────────────────

    def _generate_error_feedback(self, error: Exception, arguments: dict[str, Any]) -> str:
        """
        Return a structured, actionable hint string for the agent.

        This string is injected into the agent's context as the Observation on
        failure, so it must be written from the agent's perspective.

        Override in subclasses to provide tool-specific hints.

        Default format::

            Tool 'search_flights' failed: <error type>. Try: <generic advice>.
        """
        return (
            f"Tool '{self.tool_name}' failed: {type(error).__name__}: {error}. "
            f"Try: verify argument types and values, then retry with corrected arguments."
        )

    def _generate_validation_feedback(self, error: ValidationError, arguments: dict[str, Any]) -> str:
        """
        Return actionable feedback for Pydantic validation failures.

        Summarises which fields are missing or have wrong types.
        """
        issues = "; ".join(
            f"field '{'.'.join(str(l) for l in e['loc'])}': {e['msg']}"
            for e in error.errors()
        )
        return (
            f"Tool '{self.tool_name}' rejected invalid arguments: {issues}. "
            f"Expected schema: {self.input_schema.model_json_schema()}."
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _record_and_emit(
        self,
        success: bool,
        latency_ms: float,
        args_hash: str,
        result: Any,
        error: str | None,
    ) -> None:
        self._tracker.record(
            tool_name=self.tool_name,
            success=success,
            latency_ms=latency_ms,
            arguments_hash=args_hash,
        )
        self._event_bus.emit(
            ToolEvent(
                tool_name=self.tool_name,
                success=success,
                arguments_hash=args_hash,
                result=result,
                error=error,
                latency_ms=latency_ms,
            )
        )

    def get_schema_for_agent(self) -> dict[str, Any]:
        """Return the JSON schema dict used to describe this tool to the LLM."""
        return {
            "name": self.tool_name,
            "description": self.tool_description,
            "parameters": self.input_schema.model_json_schema(),
        }
