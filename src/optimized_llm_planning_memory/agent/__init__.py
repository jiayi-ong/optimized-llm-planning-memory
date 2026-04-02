"""
agent/
======
ReAct planning agent — the LLM that builds travel itineraries.

Design notes
------------
* ``ReActAgent`` uses ``pydantic-ai`` for the LLM loop. pydantic-ai's
  ``Agent`` class handles: tool registration, function calling, and
  structured message history. We wrap it to add compression events,
  trajectory accumulation, and episode logging.

* ``ContextBuilder`` is the single location for all mode-switching logic.
  All three baseline modes (RAW, LLM_SUMMARY, COMPRESSOR) are implemented
  here. Changing the compression strategy requires only changing the
  ``AgentMode`` — the rest of the agent is identical.

* The agent does NOT import from ``training/`` or ``evaluation/``. It
  produces an ``EpisodeLog`` and hands it upstream. This keeps the agent
  testable in isolation.
"""
