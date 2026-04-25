"""
agent/prompts.py
================
System prompt templates and few-shot tool-use examples for the ReAct agent.

Design notes
------------
* System prompt versions are stored as module-level constants (``SYSTEM_PROMPT_V1``).
  The active version is selected via ``agent.system_prompt_version`` in config.
  This makes it easy to A/B test prompt variants without code changes.

* Few-shot examples are loaded from ``data/few_shot_examples/react_tool_use.json``
  at runtime. They demonstrate the expected Thought / Action / Observation format.

* The prompt is intentionally TOOL-AGNOSTIC. Tool descriptions are injected
  by ``ContextBuilder._build_tools_section()`` so the prompt does not need
  updating when tools are added or removed.
"""

from __future__ import annotations

import json
from pathlib import Path

# ── System prompts (versioned) ────────────────────────────────────────────────

SYSTEM_PROMPT_V1 = """\
You are an expert travel planning agent. Your goal is to build a complete,
constraint-satisfying travel itinerary by reasoning step-by-step and using
the available tools to query a simulated travel world.

PLANNING APPROACH
-----------------
1. Start by understanding the user's hard constraints (budget, dates, cities).
2. Use get_city_info to understand the geography before searching.
3. Search before booking — use search_flights, search_hotels, search_activities
   to find options, then book the best ones.
4. Track your partial itinerary and total cost at every step.
5. Verify hard constraints before concluding.

OUTPUT FORMAT
-------------
At each step, produce:

Thought: <your reasoning about the current state and next action>
Action: <tool_name>(<json_arguments>)

After receiving the Observation, continue with the next Thought.
When planning is complete, produce:

Thought: <final reasoning>
Action: DONE
Itinerary: <structured itinerary summary>

IMPORTANT
---------
- Never call a tool without first thinking about why.
- If a tool call fails, read the error message and adjust your arguments.
- Do not book the same resource twice.
- Stay within the user's budget.
"""

SYSTEM_PROMPT_V2 = SYSTEM_PROMPT_V1 + """
CONSTRAINT TRACKING
-------------------
At each step, mentally track which hard constraints are satisfied and which
are not yet addressed. Prioritise unsatisfied hard constraints.
"""

SYSTEM_PROMPT_V3 = SYSTEM_PROMPT_V2 + """
STRICT FORMAT REQUIREMENT
--------------------------
Every response MUST follow this exact format (no exceptions):

Thought: <your single-paragraph reasoning — what you know, what you need, what you'll do next>
Action: tool_name({"arg1": "value1", "arg2": "value2"})

Or when done:

Thought: <final summary of the itinerary and constraint satisfaction>
Action: DONE

EXAMPLE STEP:
Thought: I have confirmed the outbound flight. Now I need to book a hotel in Paris.
  The user's budget is $3000 total; $1300 is spent on flights, leaving $1700.
  I'll book Hôtel du Marais at $540 for 3 nights.
Action: book_hotel({"hotel_id": "HTL_PAR_001", "check_in": "2025-06-01", "check_out": "2025-06-04"})

ERROR RECOVERY
--------------
If a tool call returns an error, read the error message carefully. Common fixes:
- "city_id not found" → call get_available_routes first to discover valid city IDs.
- "hotel_id not found" → call search_hotels first; do not guess hotel_id values.
- "edge_id not found" → call search_flights first; do not guess edge_id values.
- JSON parse error → check that your arguments are valid JSON (no trailing commas, quoted strings).
After an error, adjust only the failing argument and retry. Do not restart from scratch.

REAL-TIME BUDGET TRACKING
--------------------------
After every booking, compute and state your running total:
  Running total: $<amount> of $<budget> budget used.
Do not proceed to the next booking step without verifying that the new total stays within budget.
"""

_VERSIONS: dict[str, str] = {
    "v1": SYSTEM_PROMPT_V1,
    "v2": SYSTEM_PROMPT_V2,
    "v3": SYSTEM_PROMPT_V3,
}


def get_system_prompt(version: str = "v1") -> str:
    """
    Return the system prompt for the given version string.

    Parameters
    ----------
    version : Version key (e.g., 'v1', 'v2').

    Raises
    ------
    ValueError
        If the version is not registered in ``_VERSIONS``.
    """
    if version not in _VERSIONS:
        raise ValueError(
            f"Unknown system prompt version '{version}'. "
            f"Available: {list(_VERSIONS.keys())}."
        )
    return _VERSIONS[version]


# ── Few-shot examples ─────────────────────────────────────────────────────────

def load_few_shot_examples(path: str) -> list[dict]:
    """
    Load few-shot ReAct tool-use examples from a JSON file.

    Parameters
    ----------
    path : Path to the JSON file (relative to project root or absolute).

    Returns
    -------
    list[dict]
        Each dict has keys: ``user_request``, ``steps`` (list of
        {thought, action, observation} dicts).
    """
    p = Path(path)
    if not p.exists():
        return []  # Graceful fallback; examples are optional
    with p.open() as f:
        return json.load(f)


def format_few_shot_examples(examples: list[dict]) -> str:
    """
    Format few-shot examples into the prompt string injected after the system prompt.

    Returns an empty string if no examples are provided.
    """
    if not examples:
        return ""

    parts = ["", "--- FEW-SHOT EXAMPLES ---", ""]
    for i, example in enumerate(examples, start=1):
        parts.append(f"Example {i}:")
        parts.append(f"User request: {example.get('user_request', '')}")
        for step in example.get("steps", []):
            parts.append(f"Thought: {step.get('thought', '')}")
            if "action" in step:
                parts.append(f"Action: {step['action']}")
            if "observation" in step:
                parts.append(f"Observation: {step['observation']}")
        parts.append("")
    parts.append("--- END EXAMPLES ---")
    return "\n".join(parts)
