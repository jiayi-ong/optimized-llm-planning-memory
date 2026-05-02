# Agent — ReAct Planning Agent

The ReAct agent is the core planning loop. It interleaves **Thought** (reasoning) and **Action** (tool call) steps until it produces a complete itinerary or exhausts `max_steps`. Three context-building modes let you swap in different memory strategies without touching agent logic.

---

## Files

| File | Role |
|---|---|
| `agent/react_agent.py` | `ReActAgent` — the main planning loop |
| `agent/context_builder.py` | `ContextBuilder` — builds the LLM prompt per step |
| `agent/modes.py` | `AgentMode` enum |
| `agent/trajectory.py` | `Trajectory` — accumulates steps, tracks token count |
| `agent/prompts.py` | System prompt constants (`SYSTEM_PROMPT_V1/V2/V3`) and few-shot loader |

---

## AgentMode

```python
class AgentMode(str, Enum):
    RAW              = "raw"               # append full trajectory verbatim
    LLM_SUMMARY      = "llm_summary"       # LLM-summarized trajectory
    COMPRESSOR       = "compressor"        # PPO-trained compressed state
    MCTS_COMPRESSOR  = "mcts_compressor"   # compressed state + MCTS tree info
```

Mode is set in `configs/agent/*.yaml` and read into `AgentConfig.mode`.

---

## Prompt Versions

Three prompt versions are stored in `agent/prompts.py`. The active version is selected by `AgentConfig.system_prompt_version` (default `"v2"`).

| Version | Config key | What it adds over the previous |
|---|---|---|
| `v1` | `"v1"` | Base ReAct instructions plus: WORLD CONTEXT (synthetic city names, no `get_city_info`), PLANNING PHASE (required execution order), BOOKING RULE (commit before proceeding), THOUGHT DISCIPLINE (`"The last observation showed..."` opener), LETHAL SCENARIOS (EXIT codes for unresolvable episodes) |
| `v2` | `"v2"` | + explicit **constraint tracking** guidance (track satisfied vs. open constraints at each step); + **ITINERARY STATE** section instructing the agent to use `[CURRENT ITINERARY STATE]` as source of truth and to call `cancel_booking` to remove an incorrect item before re-booking |
| `v3` | `"v3"` | + strict format requirement with inline example; error-recovery and budget-tracking guidance; city-not-found → `EXIT(reason=CITY_NOT_FOUND)` |

`v2` is the default for all configs (including `react_default.yaml`). Do not change an existing config to `v1` — it regresses planning quality.

### V1 base sections (inherited by all versions)

| Section | Purpose |
|---|---|
| WORLD CONTEXT | Reminds the agent it is in a synthetic world; city IDs come only from `get_available_routes`; `get_city_info` does not exist |
| PLANNING PHASE | Required execution order: discover routes → flights → hotels → activities → verify budget → DONE |
| BOOKING RULE | Never search the next category until the current booking is confirmed (prevents deferring commitments) |
| THOUGHT DISCIPLINE | Every `Thought:` must open with `"The last observation showed..."` to prevent verbatim thought repetition |
| LETHAL SCENARIOS — IMMEDIATE EXIT | Conditions that require `Action: EXIT(reason=<code>)` rather than continued search (see Response Parsing below) |

### Selecting a prompt in config

```yaml
# configs/agent/react_default.yaml
agent:
  system_prompt_version: v2
```

```python
# programmatically
from optimized_llm_planning_memory.agent.prompts import get_system_prompt
prompt = get_system_prompt("v3")
```

---

## Few-Shot Examples

Few-shot tool-use examples are loaded at runtime from `data/few_shot_examples/react_tool_use.json`. They teach the agent the exact `Thought: ... Action: tool_name({...})` format and demonstrate correct argument schemas.

### File format

Each step is a flat JSON object with three string fields: `thought`, `action`, `observation`.

```json
[
  {
    "thought": "The last observation showed nothing yet — this is the start. I need to call get_available_routes to discover which cities exist in this world.",
    "action": "get_available_routes({})",
    "observation": "[{'city_id': 'city_synth_001_0000', 'city_name': 'Aeloria', 'vibe_summary': '...', 'dominant_cuisines': ['Japanese', 'French'], ...}, ...]"
  },
  {
    "thought": "The last observation showed two cities: Aeloria (city_id: city_synth_001_0000) and Brindor (city_id: city_synth_001_0001). I will search for flights from Aeloria to Brindor.",
    "action": "search_flights({\"origin_city_id\": \"city_synth_001_0000\", \"destination_city_id\": \"city_synth_001_0001\", \"departure_date\": \"2026-06-01\", \"passengers\": 2})",
    "observation": "[{'edge_id': 'city_synth_001_0000-city_synth_001_0001-20260601-AE', 'airline': 'Aeloria Air', 'total_price': 480.0, ...}]"
  }
]
```

**Critical format rules:**
- `action` must be a plain string in the format `tool_name({json_args})` — **not** a dict like `{"tool_name": ..., "arguments": ...}`. The latter is printed as Python repr and confuses the agent.
- `observation` must be a Python repr string matching `TrajectoryModel.to_text()` output (single-quoted keys, not JSON).
- `thought` must open with `"The last observation showed..."` per THOUGHT DISCIPLINE.

Each example must use **real tool names from the registry** and **valid argument schemas**. Never invent tool names or fields that don't exist in the input schema.

### Rules for good few-shot examples

1. Always start with `get_available_routes({})` — the agent must discover city IDs before searching.
2. Show `get_available_routes` returning per-city descriptors (`city_id`, `city_name`, `vibe_summary`, ...) — not origin/destination route pairs.
3. Use realistic synthetic city IDs like `city_synth_001_0000`, not real-world names like `"nyc-001"`.
4. Use ISO 8601 dates (`YYYY-MM-DD`).
5. The final step must use `Action: DONE` followed by an `Itinerary:` block listing all booked items and total cost.

The best way to generate examples is to run a live episode with `agent.mode=raw`, capture the trajectory, and save the successful steps. See `scripts/run_episode.py`.

---

## ContextBuilder

`ContextBuilder` implements the Strategy pattern. For each step it assembles the full prompt that goes to the LLM.

### Prompt structure (all modes)

```
[SYSTEM PROMPT]            ← versioned (v1/v2/v3)
[USER REQUEST]             ← hard constraints, dates, budget, soft constraints
[CURRENT ITINERARY STATE]  ← confirmed bookings with booking_refs and costs
                              (injected by ContextBuilder from the live Itinerary object;
                               shows "No bookings confirmed yet." until a booking succeeds)
[MEMORY SECTION]           ← varies by mode (see below)
[TOOL SCHEMA]              ← auto-generated from ToolRegistry
[FEW-SHOT EXAMPLES]        ← loaded from react_tool_use.json
[CURRENT STEP HEADER]      ← "Step N:"
```

`[CURRENT ITINERARY STATE]` is always present, even on the first step. It is the agent's authoritative view of what has been booked — the agent uses it to avoid re-booking items already confirmed. Each entry shows the `booking_ref` (e.g., `FLT-XXXX`, `HTL-XXXX`) needed to call `cancel_booking`.

### Memory section per mode

| Mode | Memory section content |
|---|---|
| `RAW` | Full trajectory text: every `Thought / Action / Observation` since step 1 |
| `LLM_SUMMARY` | A bullet-point summary generated by a separate LLM call; regenerated only when new steps are added since the last call (cached) |
| `COMPRESSOR` | The last `CompressedState` rendered via `CompressedStateTemplate` (6 sections) |
| `MCTS_COMPRESSOR` | Same as `COMPRESSOR` plus a section listing MCTS candidate actions and their simulated values |

### Tool schema injection

The tool section is generated automatically from `ToolRegistry.get_tool_schemas()`. You do **not** need to edit the prompt when adding or removing tools — the schema section updates automatically.

---

## Response Parsing

The agent expects LLM output in one of three formats:

**Mid-episode:**
```
Thought: <reasoning paragraph>
Action: tool_name({"key": "value"})
```

**Successful completion:**
```
Thought: <final reasoning>
Action: DONE
Itinerary:
- Flight: Aeloria → Brindor, 2026-06-01 (FLT-AE0601, $480.00)
- Hotel: Harbour View Boutique, 2026-06-01 to 2026-06-04 (HTL-001, $525.00)
Total cost: $1005.00 of $4180.00 budget
```

**Lethal scenario exit (no itinerary producible):**
```
Thought: <explanation of why the request cannot be fulfilled>
Action: EXIT(reason=<CODE>)
Reason: <one-sentence explanation>
```

Valid EXIT codes: `CITY_NOT_FOUND`, `BUDGET_EXCEEDED`, `DATE_INVALID`, `NO_AVAILABILITY`, `REPEATED_DEAD_END`.

`_parse_response` returns a 4-tuple `(thought, tool_call, is_done, exit_reason)`. The `exit_reason` is `None` for normal tool calls and `DONE`, or the exit code string (e.g. `"CITY_NOT_FOUND"`) for EXIT signals. The `run_episode` method records the outcome in `EpisodeLog.termination_reason`.

Additional parser properties:
- Case-insensitive `Thought:` / `Action:` / `Itinerary:` matching.
- Strips markdown code fences (` ```json ... ``` `) from arguments before JSON parsing.
- On parse failure: logs a warning and returns structured feedback to the agent so it can self-correct on the next step (up to `max_retries_per_action`).

---

## Trajectory and Token Budget

`Trajectory` tracks all `ReActStep` objects and exposes:
- `token_count()` — approximate token count of the full trajectory text.
- `mark_compression(at_step)` — records when compression last fired.
- `steps_since_last_compression()` — used by `ReActAgent._should_compress()`.

Compression triggers on whichever comes first:
- `steps_since_last_compression() >= config.compress_every_n_steps` (default 5)
- `token_count() >= config.compress_on_token_threshold` (default 3000)

---

## Extending the Agent

### Adding a new prompt version

1. Open `agent/prompts.py`.
2. Define `SYSTEM_PROMPT_V4 = SYSTEM_PROMPT_V3 + "..."`.
3. Register it in `get_system_prompt()`:
   ```python
   def get_system_prompt(version: str) -> str:
       _PROMPTS = {"v1": SYSTEM_PROMPT_V1, "v2": SYSTEM_PROMPT_V2,
                   "v3": SYSTEM_PROMPT_V3, "v4": SYSTEM_PROMPT_V4}
       ...
   ```
4. Update any agent config YAML that should use it:
   ```yaml
   agent:
     system_prompt_version: v4
   ```

### Adding a new AgentMode

1. Add the enum value to `agent/modes.py`:
   ```python
   MY_MODE = "my_mode"
   ```
2. Add a new `_build_memory_section_my_mode()` method in `ContextBuilder`.
3. Wire it into `ContextBuilder.build()`:
   ```python
   elif mode == AgentMode.MY_MODE:
       memory = self._build_memory_section_my_mode(trajectory, compressed_state)
   ```
   The final `else` branch raises `ValueError("Unhandled mode: ...")` — Python will tell you immediately if you forget to wire a new mode.
4. Add a corresponding agent config YAML under `configs/agent/`.
5. Add tests in `tests/test_agent/` to verify the new context structure.

### Switching LLM provider

Change only the model ID in `configs/agent/*.yaml` — no code changes needed. The agent uses `litellm` for transparent provider routing:

```yaml
agent:
  llm_model_id: "anthropic/claude-3-5-haiku-20241022"  # was "openai/gpt-4o-mini"
```

Any `litellm`-supported model ID works here.

---

## Configuration Reference

`configs/agent/react_default.yaml`:

```yaml
agent:
  mode: compressor                        # RAW | llm_summary | compressor | mcts_compressor
  llm_model_id: openai/gpt-4o-mini        # any litellm model string
  max_steps: 30                           # hard stop on ReAct steps
  max_retries_per_action: 3              # retries after failed tool call
  compress_every_n_steps: 5              # compression interval
  compress_on_token_threshold: 3000      # also compress if trajectory exceeds this
  temperature: 0.0                       # set > 0 for diverse sampling
  max_tokens_per_response: 1024
  system_prompt_version: v2
  few_shot_examples_path: data/few_shot_examples/react_tool_use.json
```
