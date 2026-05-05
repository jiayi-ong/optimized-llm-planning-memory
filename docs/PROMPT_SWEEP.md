# Prompt Sweep — Baseline Agent Prompt Design Experiment

Before evaluating the compressor, we need to establish that the **baseline agent** (no compressor, raw trajectory) can reliably produce **non-degenerate itineraries**: episodes where the agent actually commits to booking at least one flight, one hotel, and one activity, rather than spending all its steps in pure exploration.

This document describes the methodology, the 7 prompt variants tested, and how to configure and run the sweep.

---

## Motivation

The current prompts (`v1`–`v5` in `agent/prompts.py`) are additive and progressively more detailed, but they share a common structural risk: they describe *what to do* without enforcing *when* or *how frequently* to commit. An LLM agent following these prompts can satisfy every instruction literally while still spending 30 steps searching and reasoning without ever calling `select_flight`, `book_hotel`, or `book_event`.

This is called the **pure-exploration failure mode**: the agent explores the search space correctly but never transitions to booking. The result is an empty or near-empty itinerary at episode end — a `completion_rate` of 0 — even when all city IDs were correctly discovered and all searches returned valid results.

The prompt sweep tests 7 distinct *commitment strategies* — prompt designs that differ in their core mechanism for forcing booking actions. The goal is to identify which strategy most reliably produces a non-degenerate itinerary, which then becomes the system prompt for the formal baseline evaluation.

---

## Non-Degenerate Itinerary Definition

An itinerary is **non-degenerate** if and only if:

| Condition | Measured by |
|---|---|
| At least 1 booked transport segment | `final_itinerary.days[*].transport_segments` is non-empty |
| At least 1 booked hotel | `accommodation_coverage_ratio > 0` |
| At least 1 booked or recorded activity | `activity_density_score > 0` OR self-guided attractions noted |
| `completion_rate > 0` | `booked_days / required_trip_days > 0` |

A near-empty itinerary that happens to have `budget_adherence = 1.0` (trivially satisfied because nothing was booked) or `schedule_overlap = 1.0` (no overlaps because there are no activities) scores high on individual metrics but fails the non-degeneracy gate.

---

## Experiment Structure

```
experiments/
  prompt_sweep_baseline/
    user_requests/          # 3 control user requests (same world)
    prompts/
      sweep_overview.md     # Summary of each variant's philosophy
    results/
      sweep_A_react/        # Per-variant: EpisodeLog + eval JSON per request
      sweep_B_react/
      sweep_C_react/
      sweep_D_react/
      sweep_E_react/
      sweep_F_stateless/
      sweep_G_stateless/
    run_sweep.py            # Orchestrator script
    summary_results.json    # Aggregated results table (written after run)
```

### Control User Requests

All 3 requests target the same world (`world_42_20260504_075144`) for reproducibility:

| File | Description | Budget | Duration |
|---|---|---|---|
| `req_solo_simple.json` | Solo traveler, minimal constraints | $2,000 | 4 days |
| `req_couple_medium.json` | 2 adults, arts & culture soft preferences | $3,500 | 5 days |
| `req_family_constrained.json` | Family of 4, park activity hard constraint | $6,000 | 6 days |

The requests are intentionally simple: a single destination, a clear budget, and at most one hard activity constraint. Any reasonable prompt should be able to satisfy them. Failure here is a clear signal of the pure-exploration failure mode.

---

## The 7 Prompt Variants

Each variant is a **complete, standalone system prompt** — not additive to `v5`. This isolates the effect of the core philosophy. All variants share: tool-agnostic design (tools are injected at runtime), standard LETHAL SCENARIO exit conditions, and the same `Thought / Action` output format.

Variants are registered in `agent/prompts.py` under `_VERSIONS` and can be selected via `agent.system_prompt_version` in any agent config YAML.

### ReAct Variants (A–E)

These preserve the Thought / Action / Observation loop. The agent sees its full prior reasoning history at each step.

---

#### sweep_A — Grow-then-Prune

**Core mechanism:** The agent is instructed to build a *large* candidate itinerary in Phase 1 by booking greedily after every search (even over-budget), then cancel the most expensive non-essential items in Phase 2 until the plan fits constraints.

**Key instruction:**
> "After EVERY search, you MUST immediately book or select at least one item from the results before calling any other search tool. Keep booking: flights, hotels, events. Overshoot the budget slightly — you will prune later."

**When it works:** When the agent's primary failure mode is hesitating to book due to uncertainty. Grow-then-Prune converts "I'm not sure this is the best option" into "book now, reconsider later."

**When it fails:** If the agent is also bad at using `cancel_booking` correctly, the pruning phase may not execute, leaving an over-budget itinerary.

**Config:** `configs/agent/sweep_A_react.yaml` | `max_steps: 30`

---

#### sweep_B — Ordered Pipeline with Booking Quotas

**Core mechanism:** Explicit phase gating with booking quotas. The agent cannot advance from Phase 2 (Flights) to Phase 3 (Hotels) until at least 1 flight is booked. Each phase transition requires announcing the quota was met.

**Key instruction:**
> "PHASE 2 — FLIGHTS (quota: 1–2 booked flights). Do NOT start Phase 3 until at least 1 flight is booked."
> 
> "The agent must state the current phase and quota status in every Thought."

**When it works:** When the agent's failure mode is jumping between categories (search hotels before booking a flight). The phase announcement creates a cognitive scaffold that forces linear progression.

**When it fails:** If the agent makes up quota counts or skips phase announcements, the gating mechanism breaks down.

**Config:** `configs/agent/sweep_B_react.yaml` | `max_steps: 30`

---

#### sweep_C — Commit-Every-Step

**Core mechanism:** An iron pairing rule. Every search call must be immediately followed by a booking call. No two search calls in a row, ever. The agent is instructed to book the "least bad option" if no perfect result exists.

**Key instruction:**
> "IRON RULE: You operate in strict search–book pairs. No exceptions.
> - `search_flights` → MUST be followed by `select_flight`
> - `search_hotels` → MUST be followed by `book_hotel`
> - `search_events` → MUST be followed by `book_event`"

**When it works:** This is the strictest ReAct commitment mechanism. If the agent can follow the rule, forward progress on the itinerary is guaranteed every 2 steps.

**When it fails:** If the model interprets the rule as a soft guideline rather than a hard constraint, or if it uses search results as evidence to argue against booking rather than to select among options.

**Config:** `configs/agent/sweep_C_react.yaml` | `max_steps: 30`

---

#### sweep_D — Minimal Viable Itinerary First

**Core mechanism:** The agent has a hard 8-step deadline to reach a **Minimal Viable Itinerary (MVI)**: 1 booked flight + 1 booked hotel + 1 booked or recorded activity. Enhancement (more activities, better hotels) is only allowed after MVI is declared.

**Key instruction:**
> "MVI TARGET: Reach MVI within 8 steps. If you have not reached MVI by step 8, stop searching and book whatever is available."
> 
> "A COMPLETE MVI IS ALWAYS BETTER THAN AN INCOMPLETE ENHANCED PLAN."

**When it works:** Directly attacks the pure-exploration failure mode by making "having something booked" the first-priority success criterion. The MVI deadline creates an urgency signal the LLM can reason about.

**When it fails:** If the agent achieves MVI but then uses the enhancement phase to cancel bookings in pursuit of a better option, re-entering the exploration loop.

**Config:** `configs/agent/sweep_D_react.yaml` | `max_steps: 20`

> **Note:** sweep_D produced a non-degenerate itinerary in 11 steps on the first live test (flight + hotel + event booked, `DONE_ITINERARY`), making it the early candidate winner.

---

#### sweep_E — Step-Budget Phased Planning

**Core mechanism:** Hard step-count deadlines per category. If the deadline is missed (e.g., no flight booked by step 6), the agent must *skip immediately* to the next phase rather than spending more steps trying to catch up.

**Key instruction:**
> "Step N of 25. Phase X. Deadline at step Y."
> 
> "If step 6 passes with no flight booked, skip to Phase 3 immediately. Missing a deadline means skip to next phase — never spend extra steps catching up."

**Step budget:**
- Steps 1–2: Discovery
- Steps 3–6: Flight deadline
- Steps 7–12: Hotel deadline
- Steps 13–18: Activity deadline
- Steps 19+: Refinement + DONE

**When it works:** When the agent's failure mode is getting stuck on one category (e.g., running 10 flight searches without booking). The step counter makes the agent aware it is "running out of time" and must act.

**When it fails:** If the agent ignores the step counter or treats deadlines as targets rather than hard cutoffs.

**Config:** `configs/agent/sweep_E_react.yaml` | `max_steps: 25`

---

### Non-ReAct Variants (F–G)

These use `AgentMode.STATELESS` — a new mode where the agent receives **no trajectory history** at each step. The `[CONTEXT]` section is omitted from the prompt entirely. The agent's only memory is the `[CURRENT ITINERARY STATE]` block (the actual confirmed bookings object, rendered as structured text).

This models a fundamentally different agent architecture: instead of reasoning over a growing log of past thoughts and observations, the agent makes each decision fresh by looking at the current state of the world (the itinerary) and asking "what is missing?"

**When stateless agents work:** When the pure-exploration failure mode is *caused* by the agent reasoning itself into analysis paralysis over prior failed attempts. Without the history, there is nothing to second-guess.

**When stateless agents fail:** When the agent needs trajectory history to avoid repeating mistakes (e.g., it already tried hotel X and it was full — without history, it might try again).

---

#### sweep_F — Non-ReAct Stateless / Assess-then-Act

**Core mechanism:** Each call, the agent assesses the current itinerary against a fixed priority order and calls the single tool that addresses the first missing element.

**Priority order:**
1. City IDs unknown → `get_available_routes`
2. No outbound flight → `search_flights`
3. No hotel → `search_hotels`
4. No activity → `search_events` or `search_attractions`
5. Everything present + budget OK → `DONE`

**Key instruction:**
> "When you call a search tool, you COMMIT to booking the first result that fits the constraints on your VERY NEXT call. State in your Thought: 'I will book [item] on my next call.'"

**Config:** `configs/agent/sweep_F_stateless.yaml` | `max_steps: 25`

---

#### sweep_G — Non-ReAct Commitment-Gated

**Core mechanism:** The strictest non-ReAct variant. Each call must be one of three types:
- **Type A (Booking):** `select_flight`, `book_hotel`, `book_event`, `cancel_booking`
- **Type B (Committed search):** any search tool, only if the agent declares "COMMITTED: I will book X next call" in its Thought
- **Type C (Terminal):** `DONE`

Two consecutive Type B calls are forbidden. Pure discovery (searching without intent to book immediately) is explicitly prohibited.

**Key instruction:**
> "COMMITMENT GATE: Every time you are called, you MUST do one of these three things: (A) call a booking tool, (B) call a search tool with COMMITTED booking statement, (C) DONE."

**Config:** `configs/agent/sweep_G_stateless.yaml` | `max_steps: 20`

---

## Design Principles

### 1. Complete Standalone Prompts

Each sweep variant is a *complete, self-contained system prompt* rather than an incremental addition to `v5`. This is intentional: additive prompts produce confounded results because the base behavior is always the same. Standalone prompts let you attribute differences in behavior to the specific mechanism being tested.

### 2. Single Independent Variable

The variants differ in **one dimension**: the commitment strategy. All other factors are held constant:
- Same LLM model (`openai/gpt-4o-mini`, `temperature=0.0`)
- Same world seed (`42`) and world ID (`world_42_20260504_075144`)
- Same 3 user requests per variant
- Same evaluation pipeline (`DeterministicEvaluator`)
- No compressor (all variants use `IdentityCompressor`)

### 3. STATELESS Mode

`AgentMode.STATELESS` (added in `agent/modes.py`) instructs `ContextBuilder.build()` to omit the `[CONTEXT]` section entirely. The context assembled is:

```
[SYSTEM]       ← sweep_F or sweep_G system prompt
[USER REQUEST] ← original travel request + structured constraints
[CURRENT ITINERARY STATE] ← confirmed bookings (flights, hotels, events)
[AVAILABLE TOOLS]          ← runtime tool list
```

No `[CONTEXT]` block means no Thought / Action / Observation history. The agent's decision at step $t$ is conditioned only on $(\text{system prompt}, \text{request}, \text{itinerary}_t, \text{tools})$ — not on any prior reasoning.

This is equivalent to a **stateless reactive policy**: the same context in → the same action out, independent of history. The itinerary object accumulates state externally; the agent need not track it in its own context.

---

## Running the Sweep

### Prerequisites

- Python environment with the project installed (`pip install -e .`)
- API key in `.env` (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`)
- World `world_42_20260504_075144` present in `./worlds/` (already included)

### Quick Reference

```bash
# Dry run: verify imports, world loading, request parsing (no API calls)
python experiments/prompt_sweep_baseline/run_sweep.py --dry-run

# Single variant, single request (fastest meaningful test)
python experiments/prompt_sweep_baseline/run_sweep.py \
    --variants sweep_D \
    --requests req_solo_simple

# Subset of variants (e.g., only ReAct variants)
python experiments/prompt_sweep_baseline/run_sweep.py \
    --variants sweep_A sweep_B sweep_C sweep_D sweep_E

# Full sweep (21 episodes: 7 variants x 3 requests)
python experiments/prompt_sweep_baseline/run_sweep.py
```

### Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--variants` | list | all 7 | Subset of variants to run. Choices: `sweep_A` ... `sweep_G` |
| `--requests` | list | all 3 | Subset of requests. Choices: `req_solo_simple`, `req_couple_medium`, `req_family_constrained` |
| `--dry-run` | flag | False | Skip LLM calls; verify config, imports, and request loading only |

### Outputs

For each `(variant, request)` pair the runner writes:

| File | Contents |
|---|---|
| `results/<variant>/ep_<request_id>.json` | Full `EpisodeLog` (trajectory, final itinerary, tool stats, reward components) |
| `results/<variant>/eval_<request_id>.json` | `DeterministicEvaluator.score()` output — all 15 metrics |
| `summary_results.json` | Aggregated table: variant × request → key metrics + non-degeneracy flag |

The runner prints a live summary table and an **EARLY REPORT** banner if any variant achieves non-degeneracy on every request in its batch.

### Reading the Summary Table

```
variant       request                   non_deg  completion  accommodation  steps  term_reason
sweep_D       req_solo_simple           YES      0.50        1.00           11     DONE_ITINERARY
sweep_A       req_solo_simple           NO       0.00        0.00           30     MAX_STEPS
```

Key columns:
- **non_deg**: `YES` if transport + hotel booked AND `completion_rate > 0`
- **completion**: `completion_rate` = booked days / required trip days
- **accommodation**: `accommodation_coverage_ratio` = days with hotel / total days
- **steps**: total ReAct steps before terminal signal
- **term_reason**: `DONE_ITINERARY` (success), `MAX_STEPS` (ran out), `EXIT_*` (lethal scenario)

---

## Adding a New Variant

1. **Write the prompt** in `agent/prompts.py`:
   ```python
   SYSTEM_PROMPT_SWEEP_H = """\
   You are an expert travel planning agent...
   [describe the new core strategy]
   ...
   """
   ```

2. **Register it** in `_VERSIONS`:
   ```python
   _VERSIONS = {
       ...
       "sweep_H": SYSTEM_PROMPT_SWEEP_H,
   }
   ```

3. **Create a config YAML** in `configs/agent/sweep_H_react.yaml` (or `_stateless.yaml` for STATELESS mode):
   ```yaml
   mode: raw          # or "stateless"
   llm_model_id: openai/gpt-4o-mini
   max_steps: 25
   system_prompt_version: sweep_H
   compress_every_n_steps: 9999
   compress_on_token_threshold: 999999
   temperature: 0.0
   ```

4. **Register it** in `run_sweep.py`:
   ```python
   VARIANTS = [
       ...
       VariantConfig("sweep_H", "sweep_H", "raw", 25, "My New Strategy"),
   ]
   ```

5. **Run it:**
   ```bash
   python experiments/prompt_sweep_baseline/run_sweep.py --variants sweep_H
   ```

---

## Interpreting Results

### Diagnostic Decision Tree

```
completion_rate = 0 AND steps = max_steps
    → Pure-exploration failure: agent searched but never booked.
      Try: sweep_C (Commit-Every-Step) or sweep_D (Minimal Viable First).

completion_rate = 0 AND term_reason = EXIT_CITY_NOT_FOUND
    → World-discovery failure: get_available_routes returned no match.
      Check that the user request's origin/destination names appear in the world.

completion_rate > 0 AND accommodation_coverage_ratio = 0
    → Flight booked but no hotel: agent terminated early or got stuck post-flight.
      Try: sweep_B (Ordered Pipeline) to force hotel phase.

completion_rate > 0 AND accommodation_coverage_ratio > 0 AND hard_constraint_ratio < 1.0
    → Non-degenerate but constraint violations: itinerary exists but misses requirements.
      This is a scoring problem, not a booking problem — the prompt works; refine constraints.
```

### Choosing a Winning Variant

A variant is considered **passing** if it is non-degenerate on all 3 control requests. Among passing variants, prefer the one with the highest `hard_constraint_ratio` and lowest `steps_per_episode` — it satisfies more constraints while using fewer steps (better tool efficiency).

The winning variant's `system_prompt_version` key becomes the value of `system_prompt_version` in the formal baseline eval config (`configs/agent/react_baseline_raw.yaml`).

---

## Relationship to the Full Evaluation Pipeline

The prompt sweep is a **pre-evaluation calibration step**, not part of the formal comparison. Its output — the winning prompt — is frozen before the compressor evaluation begins.

```
Prompt sweep (this doc)
    └── identifies winning prompt variant
            └── used in react_baseline_raw.yaml
                    └── formal baseline evaluation (docs/EVALUATION.md)
                            └── compared against compressor-enhanced agent
```

This separation ensures the baseline is not tuned post-hoc to maximize metrics against the compressor — the prompt is chosen solely on the criterion of non-degeneracy.
