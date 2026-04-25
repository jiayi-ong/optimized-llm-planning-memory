# Itinerary Evaluation Rubric v1

This rubric defines the scoring criteria used by the LLM judge to evaluate
travel itineraries produced by the planning agent.

Each dimension is scored from **0.0** (worst) to **1.0** (best).
The judge scores each dimension independently to avoid criteria conflation.

---

## Constraint Adherence

Evaluate how well the itinerary satisfies ALL hard constraints stated in the
user request (budget, dates, city sequence, accommodation requirements, etc.).

| Score | Description |
|-------|-------------|
| 1.0   | All hard constraints are explicitly met; no violations |
| 0.75  | Minor violation (1 hard constraint slightly missed, e.g., budget overshoot < 5%) |
| 0.5   | Moderate violation (1–2 hard constraints not met, or 1 major miss) |
| 0.25  | Several hard constraints violated; itinerary is substantially wrong |
| 0.0   | Most hard constraints ignored or completely violated |

---

## Logical Coherence

Evaluate whether the itinerary is internally consistent and physically feasible:
dates are in order, flights arrive before check-in, no double-booked accommodation,
travel times between cities are plausible.

| Score | Description |
|-------|-------------|
| 1.0   | Completely coherent; all transitions are feasible |
| 0.75  | One minor incoherence (e.g., tight layover but still plausible) |
| 0.5   | One clearly infeasible transition or date inconsistency |
| 0.25  | Multiple incoherences; plan would be difficult to execute |
| 0.0   | Itinerary is logically broken (reversed dates, overlapping bookings) |

---

## Activity Diversity

Evaluate whether the activities across the trip are varied and appropriate for
the traveler profile (interests, group composition, accessibility needs).

| Score | Description |
|-------|-------------|
| 1.0   | Rich mix of activity categories; clearly tailored to stated preferences |
| 0.75  | Good variety with minor gaps in preference alignment |
| 0.5   | Acceptable but repetitive or generic activity selection |
| 0.25  | Limited variety; activities feel disconnected from preferences |
| 0.0   | No activities, or all identical, or completely mismatched |

---

## Budget Efficiency

Evaluate how well the itinerary uses the available budget: within the stated
budget limit, balanced spend across accommodation/transport/activities, avoids
obvious overpaying.

| Score | Description |
|-------|-------------|
| 1.0   | Within budget; good value; sensible allocation |
| 0.75  | Within budget; minor inefficiency |
| 0.5   | Near budget limit with suboptimal allocation, or modest overshoot |
| 0.25  | Significant overspend (>10%) or highly unbalanced allocation |
| 0.0   | Far over budget or budget not considered at all |

---

## Feasibility

Evaluate whether the itinerary could realistically be executed by the traveler:
booking references exist, sufficient time allocated per activity, reasonable
connection times, no physical impossibilities.

> **Note on booking references:** This system uses a synthetic travel simulator.
> All booking references are auto-generated and prefixed with `SIM-` (e.g.,
> `SIM-HTL-BOUTIQUE-001`). These are valid synthetic confirmations, not real
> bookings. Do **not** penalise for the `SIM-` prefix — treat any `SIM-` booking
> reference as a confirmed booking for scoring purposes.

| Score | Description |
|-------|-------------|
| 1.0   | Fully executable as described; all details present |
| 0.75  | Executable with minor adjustments (e.g., one missing booking ref) |
| 0.5   | Requires moderate revision to execute (some details missing) |
| 0.25  | Several execution gaps; would be hard to follow |
| 0.0   | Not executable; placeholder content or self-contradictory |

---

## Creativity

Evaluate whether the itinerary goes beyond the generic tourist experience to
offer distinctive, memorable, or personalised suggestions while remaining practical.

| Score | Description |
|-------|-------------|
| 1.0   | Imaginative choices clearly shaped by user preferences; includes non-obvious but high-value suggestions |
| 0.75  | Some personalised touches above the baseline |
| 0.5   | Adequate but generic; could apply to any traveler |
| 0.25  | Minimal personalisation; off-the-shelf tourist checklist |
| 0.0   | No personalisation; entirely generic or irrelevant suggestions |
