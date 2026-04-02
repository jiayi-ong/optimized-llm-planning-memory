"""
tools/registry.py
=================
ToolRegistry — central map from tool names to BaseTool instances.

Design pattern: Registry
--------------------------
The registry decouples the ``ReActAgent`` from knowing which tools exist.
The agent asks for a tool by name (as the LLM produces it); the registry
handles the lookup and raises ``ToolNotFoundError`` if the name is unknown.

The registry also serves as the single source of truth for:
  - The tool list injected into the agent's system prompt.
  - The pydantic-ai ``Tool`` objects registered with the pydantic-ai ``Agent``.

Design pattern: Factory Method (from_config)
---------------------------------------------
``ToolRegistry.from_config()`` is a factory that constructs and wires all
enabled tools from a config object. This centralises the complex object
graph (tools need a simulator, tracker, event_bus, and config) and makes
it easy to swap implementations in tests.
"""

from __future__ import annotations

from typing import Any

from optimized_llm_planning_memory.core.exceptions import ToolNotFoundError
from optimized_llm_planning_memory.simulator.protocol import SimulatorProtocol
from optimized_llm_planning_memory.tools.base import BaseTool
from optimized_llm_planning_memory.tools.events import EventBus
from optimized_llm_planning_memory.tools.tracker import ToolCallTracker


class ToolRegistry:
    """
    Maintains a mapping of tool_name → BaseTool instance.

    All tools must be registered before the episode starts. The registry is
    then passed to ``ReActAgent`` for the lifetime of the episode.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool instance.

        Raises
        ------
        ValueError
            If a tool with the same ``tool_name`` is already registered.
        """
        if tool.tool_name in self._tools:
            raise ValueError(
                f"A tool named '{tool.tool_name}' is already registered. "
                f"Use a unique tool_name or deregister the existing tool first."
            )
        self._tools[tool.tool_name] = tool

    def deregister(self, tool_name: str) -> None:
        """Remove a tool by name. No-op if the tool is not registered."""
        self._tools.pop(tool_name, None)

    def get(self, tool_name: str) -> BaseTool:
        """
        Look up a tool by name.

        Raises
        ------
        ToolNotFoundError
            If no tool with ``tool_name`` is registered.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ToolNotFoundError(tool_name)
        return tool

    def list_tools(self) -> list[dict[str, Any]]:
        """
        Return a list of tool descriptor dicts for use in the agent system prompt.

        Each dict has keys: ``name``, ``description``, ``parameters`` (JSON schema).
        """
        return [tool.get_schema_for_agent() for tool in self._tools.values()]

    def tool_names(self) -> list[str]:
        """Return sorted list of all registered tool names."""
        return sorted(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    @classmethod
    def from_config(
        cls,
        simulator: SimulatorProtocol,
        tracker: ToolCallTracker,
        event_bus: EventBus,
        enabled_tools: list[str] | None = None,
    ) -> "ToolRegistry":
        """
        Factory: instantiate all (or a subset of) tools and register them.

        Parameters
        ----------
        simulator      : Simulator adapter instance for this episode.
        tracker        : Per-episode ToolCallTracker.
        event_bus      : Per-episode EventBus.
        enabled_tools  : If provided, only tools whose ``tool_name`` is in this
                         list are registered. Defaults to all available tools.

        Returns
        -------
        ToolRegistry
            Populated registry ready for use by ReActAgent.
        """
        # Import here to avoid circular imports at module load time
        from optimized_llm_planning_memory.tools.activity_tools import BookActivity, SearchActivities
        from optimized_llm_planning_memory.tools.flight_tools import BookFlight, SearchFlights
        from optimized_llm_planning_memory.tools.hotel_tools import BookHotel, SearchHotels
        from optimized_llm_planning_memory.tools.info_tools import (
            GetCityInfo,
            GetEvents,
            GetLocationDetails,
        )

        all_tool_classes = [
            SearchFlights, BookFlight,
            SearchHotels, BookHotel,
            SearchActivities, BookActivity,
            GetCityInfo, GetLocationDetails, GetEvents,
        ]

        registry = cls()
        for tool_cls in all_tool_classes:
            instance = tool_cls(simulator=simulator, tracker=tracker, event_bus=event_bus)
            if enabled_tools is None or instance.tool_name in enabled_tools:
                registry.register(instance)

        return registry
