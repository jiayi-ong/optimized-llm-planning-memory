"""
evaluation/rubrics.py
======================
Rubric text constants for LLM judge scoring.

Each rubric dimension is a string that describes what the judge should evaluate
and what a perfect, acceptable, and poor score looks like.

Design note
-----------
Rubric text is kept in Python constants (not YAML) so it is versioned alongside
the code and importable without file I/O. The ``data/rubrics/`` directory
contains the canonical human-readable version; this module is the programmatic
mirror used at runtime.
"""

from __future__ import annotations

CONSTRAINT_ADHERENCE = """\
Dimension: Constraint Adherence

Evaluate how well the itinerary satisfies ALL hard constraints stated in the
user request (budget, dates, city sequence, accommodation requirements, etc.).

Score 1.0 — All hard constraints are explicitly met; no violations.
Score 0.75 — Minor violation (1 hard constraint slightly missed, e.g., budget
             overshoot < 5%).
Score 0.5  — Moderate violation (1–2 hard constraints not met, or 1 major miss).
Score 0.25 — Several hard constraints violated; itinerary is substantially wrong.
Score 0.0  — Most hard constraints ignored or completely violated.
"""

LOGICAL_COHERENCE = """\
Dimension: Logical Coherence

Evaluate whether the itinerary is internally consistent and physically feasible:
- Dates are in order with no gaps or overlaps.
- Flights arrive before hotel check-in.
- No double-booked accommodation.
- Travel times between cities are plausible.

Score 1.0 — Completely coherent; all transitions are feasible.
Score 0.75 — One minor incoherence (e.g., tight layover but still plausible).
Score 0.5  — One clearly infeasible transition or date inconsistency.
Score 0.25 — Multiple incoherences; plan would be difficult to execute.
Score 0.0  — Itinerary is logically broken (reversed dates, overlapping bookings).
"""

ACTIVITY_DIVERSITY = """\
Dimension: Activity Diversity

Evaluate whether the activities across the trip are varied and appropriate for
the traveler profile (interests, group composition, accessibility needs).

Score 1.0 — Rich mix of activity categories (culture, food, outdoors, etc.);
             clearly tailored to stated preferences.
Score 0.75 — Good variety with minor gaps in preference alignment.
Score 0.5  — Acceptable but repetitive or generic activity selection.
Score 0.25 — Limited variety; activities feel disconnected from preferences.
Score 0.0  — No activities, or all identical, or completely mismatched.
"""

BUDGET_EFFICIENCY = """\
Dimension: Budget Efficiency

Evaluate how well the itinerary uses the available budget:
- Stays within stated budget limit.
- Balances spend across accommodation, transport, and activities.
- Avoids obvious overpaying (e.g., booking expensive hotel when cheaper
  equivalent available in same city).

Score 1.0 — Within budget; good value; sensible allocation.
Score 0.75 — Within budget; minor inefficiency (e.g., slightly over-allocated
             to one category).
Score 0.5  — Near budget limit with suboptimal allocation, or modest overshoot.
Score 0.25 — Significant overspend (>10%) or highly unbalanced allocation.
Score 0.0  — Far over budget or budget not considered at all.
"""

FEASIBILITY = """\
Dimension: Feasibility

Evaluate whether the itinerary could realistically be executed by the traveler:
- Booking references exist (not placeholder text).
- Sufficient time allocated for each activity.
- Reasonable connection times between legs.
- Does not require the traveler to be in two places at once.

Score 1.0 — Fully executable as described; all details present.
Score 0.75 — Executable with minor adjustments (e.g., one missing booking ref).
Score 0.5  — Requires moderate revision to execute (some details missing).
Score 0.25 — Several execution gaps; would be hard to follow.
Score 0.0  — Not executable; placeholder content or self-contradictory.
"""

CREATIVITY = """\
Dimension: Creativity

Evaluate whether the itinerary goes beyond the generic tourist experience to
offer distinctive, memorable, or personalised suggestions while remaining
practical.

Score 1.0 — Imaginative choices clearly shaped by user preferences; includes
             non-obvious but high-value suggestions.
Score 0.75 — Some personalised touches above the baseline.
Score 0.5  — Adequate but generic; could apply to any traveler.
Score 0.25 — Minimal personalisation; off-the-shelf tourist checklist.
Score 0.0  — No personalisation; entirely generic or irrelevant suggestions.
"""

# Master rubric combining all dimensions
ITINERARY_RUBRIC_V1 = "\n\n---\n\n".join([
    CONSTRAINT_ADHERENCE,
    LOGICAL_COHERENCE,
    ACTIVITY_DIVERSITY,
    BUDGET_EFFICIENCY,
    FEASIBILITY,
    CREATIVITY,
])

# Registry: dimension name → rubric text
RUBRIC_DIMENSIONS: dict[str, str] = {
    "constraint_adherence": CONSTRAINT_ADHERENCE,
    "logical_coherence": LOGICAL_COHERENCE,
    "activity_diversity": ACTIVITY_DIVERSITY,
    "budget_efficiency": BUDGET_EFFICIENCY,
    "feasibility": FEASIBILITY,
    "creativity": CREATIVITY,
}

DEFAULT_RUBRIC_DIMENSIONS = list(RUBRIC_DIMENSIONS.keys())
