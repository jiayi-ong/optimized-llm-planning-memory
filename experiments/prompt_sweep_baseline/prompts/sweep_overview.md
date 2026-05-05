# Prompt Sweep — Variant Philosophy Summary

Each variant is a **complete standalone system prompt** with a distinct core design principle.
Variants A–E are ReAct agents (Thought/Action/Observation loop).
Variants F–G are non-ReAct stateless agents (no history; only current itinerary as memory).

---

## sweep_A — Grow-then-Prune (ReAct)

**Core idea:** Build a large candidate itinerary first by booking aggressively on every search,
then cancel what violates budget/constraints in a second pass.

**Key rules:**
- Phase 1 (steps 1–15): After every search, MUST book at least one result.
- Phase 2 (steps 16+): Cancel most expensive non-essential items until budget is met.

**Hypothesis:** The agent commits early (reducing pure-exploration risk) and pruning
is a natural second pass that any LLM should be able to do.

---

## sweep_B — Ordered Pipeline with Booking Quotas (ReAct)

**Core idea:** Explicit phase quotas (book N flights → N hotels → N events) before
advancing to the next phase. Phase advancement is gated by quota fulfillment.

**Key rules:**
- Phase 1 → 2 → 3 → 4 → 5 in strict order.
- Each phase has a minimum booking count that MUST be met before advancing.
- Agent must state the current phase and quota status in every Thought.

**Hypothesis:** Forcing the agent to announce its phase and quota creates a cognitive
scaffold that prevents getting stuck in one category indefinitely.

---

## sweep_C — Commit-Every-Step (ReAct)

**Core idea:** An iron rule — every search_* call must be immediately followed by
a booking call. No two search calls in a row. Ever.

**Key rules:**
- search_flights → select_flight (mandatory next action)
- search_hotels → book_hotel (mandatory next action)
- search_events → book_event (mandatory next action)
- "Book the least-bad option" if no result is perfect.

**Hypothesis:** The strictest commitment rule. If the agent follows it, it will
always be making forward progress on the itinerary.

---

## sweep_D — Minimal Viable Itinerary First (ReAct)

**Core idea:** Build the smallest possible complete itinerary first
(1 flight + 1 hotel + 1 activity) within 8 steps. Enhance only after the baseline exists.

**Key rules:**
- First search for each category → immediately book it (no second search before MVI).
- Announce "MVI ACHIEVED" when all 3 components are booked.
- Enhancement is only allowed after MVI.

**Hypothesis:** The agent commits to having "something" before trying to have "the best thing."
This directly counters the failure mode of infinite exploration without commitment.

---

## sweep_E — Step-Budget Phased Planning (ReAct)

**Core idea:** Hard step-count deadlines per phase. If the deadline is missed,
skip immediately to the next phase rather than spending more steps on the current one.

**Key rules:**
- Steps 1–2: Discovery deadline
- Steps 3–6: Flight booking deadline
- Steps 7–12: Hotel booking deadline
- Steps 13–18: Activity deadline
- Steps 19+: Refinement + DONE
- Agent must report "Step N of 25. Phase X. Deadline at step Y." in every Thought.

**Hypothesis:** Making step budgets explicit forces the agent to realize it's "running
out of time" and take action even with imperfect information.

---

## sweep_F — Non-ReAct Stateless / Assess-then-Act

**Core idea:** Agent has NO trajectory history. Each call it sees only:
current itinerary + user request + tools. Decides what's missing and fills it.

**Key rules:**
- Assess the itinerary to find the first missing priority element.
- Call the SINGLE tool that addresses it.
- If calling a search: commits to booking the result on the VERY NEXT call.

**Hypothesis:** Without a growing context window of past reasoning to second-guess,
the agent is forced to make a clean decision each step based only on what's booked.
Eliminates "analysis paralysis" from reviewing prior failed attempts.

---

## sweep_G — Non-ReAct Commitment-Gated

**Core idea:** The strictest non-ReAct variant. Each call must be one of:
(A) a booking action, (B) a committed search (immediately followed by a booking),
or (C) DONE. Pure exploration calls are forbidden.

**Key rules:**
- Type A: select_flight, book_hotel, book_event, cancel_booking
- Type B: any search_* — but ONLY if you declare "COMMITTED: I will book X next call"
- Type C: DONE when itinerary is complete
- Cannot call get_available_routes more than once.
- Cannot do two Type B calls in a row.

**Hypothesis:** The hardest possible commitment gate. If any prompt produces
non-degenerate itineraries, this one should — at the cost of possibly booking
suboptimal options.
