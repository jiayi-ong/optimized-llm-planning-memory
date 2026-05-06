"""
core/complexity.py
==================
RequestComplexityScorer — compute interpretable difficulty dimensions from a UserRequest.

Design
------
Complexity is computed purely from existing UserRequest fields (no LLM calls, no
simulator access). The scorer assigns five normalized sub-scores and combines them
into a composite score that drives the ``complexity_tier`` label.

The five dimensions:

1. Constraint density    — (n_hard + n_soft) / trip_duration_days
   Dense requests force the agent to satisfy many requirements per day, leaving
   little slack for rescheduling when bookings fail.

2. Geographic complexity — number of distinct destination cities
   More cities require inter-city routing, split accommodation budgets, and
   activity discovery across multiple simulator sub-graphs.

3. Budget tightness      — budget_usd / (n_travelers * trip_duration_days)
   A low per-person-per-day budget narrows the feasible solution space; the agent
   must try more alternatives before all cost constraints are satisfied.

4. Group complexity      — presence of children, accessibility needs, dietary restrictions
   Each special need cross-cuts every booking decision and adds implicit constraints
   that must be discovered from preference strings.

5. Preference specificity — number of soft constraints as a proxy
   More soft constraints = more dimensions on which the agent can fail to satisfy
   the user's unstated expectations.

Composite score:
    complexity_score = weighted_mean(sub-scores, weights=[0.25, 0.25, 0.2, 0.15, 0.15])

Tier thresholds:
    "low"    — complexity_score < 0.35
    "medium" — 0.35 ≤ complexity_score < 0.65
    "high"   — complexity_score ≥ 0.65
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from optimized_llm_planning_memory.core.models import UserRequest


class ComplexityBreakdown(TypedDict):
    n_hard_constraints: int
    n_soft_constraints: int
    n_destinations: int
    trip_duration_days: int
    budget_per_traveler_per_day: float
    constraint_density: float
    geographic_complexity: str          # "single_city" | "multi_city"
    group_type: str                     # "solo" | "couple" | "family" | "group"
    has_special_needs: bool
    preference_specificity: str         # "low" | "medium" | "high"
    # Sub-scores [0.0, 1.0]
    constraint_density_score: float
    geographic_complexity_score: float
    budget_tightness_score: float
    group_complexity_score: float
    preference_specificity_score: float
    # Composite
    complexity_score: float
    complexity_tier: str                # "low" | "medium" | "high"


# Reference budget per traveler per day values that define tightness thresholds.
# Below $50/traveler/day  → very tight (score 1.0)
# Above $300/traveler/day → very loose (score 0.0)
_BUDGET_TIGHT_USD = 50.0
_BUDGET_LOOSE_USD = 300.0


class RequestComplexityScorer:
    """Compute ComplexityBreakdown from a UserRequest instance."""

    WEIGHTS = {
        "constraint_density": 0.25,
        "geographic_complexity": 0.25,
        "budget_tightness": 0.20,
        "group_complexity": 0.15,
        "preference_specificity": 0.15,
    }

    def score(self, req: "UserRequest") -> ComplexityBreakdown:
        from datetime import date as _date

        n_hard = len(req.hard_constraints)
        n_soft = len(req.soft_constraints)
        n_dest = len(req.destination_cities)
        n_travelers = req.traveler_profile.total_travelers

        # Trip duration (fall back to 1 if dates missing/unparseable)
        try:
            start = _date.fromisoformat(req.start_date)
            end = _date.fromisoformat(req.end_date)
            duration_days = max(1, (end - start).days)
        except Exception:
            duration_days = 1

        budget_per_traveler_per_day = req.budget_usd / max(1, n_travelers * duration_days)

        # ── Sub-scores ────────────────────────────────────────────────────────

        # 1. Constraint density: clamp (n_hard+n_soft)/days to [0,1]
        density_raw = (n_hard + n_soft) / duration_days
        constraint_density_score = min(1.0, density_raw / 5.0)  # 5+ constraints/day = max

        # 2. Geographic complexity: 1 city=0, 2=0.5, 3+=1.0
        geo_score = min(1.0, (n_dest - 1) / 2.0) if n_dest > 0 else 0.0

        # 3. Budget tightness: inverse-linear between loose and tight
        if budget_per_traveler_per_day <= _BUDGET_TIGHT_USD:
            budget_score = 1.0
        elif budget_per_traveler_per_day >= _BUDGET_LOOSE_USD:
            budget_score = 0.0
        else:
            budget_score = 1.0 - (budget_per_traveler_per_day - _BUDGET_TIGHT_USD) / (
                _BUDGET_LOOSE_USD - _BUDGET_TIGHT_USD
            )

        # 4. Group complexity: base + bonuses for children and special needs
        profile = req.traveler_profile
        group_score = 0.0
        if profile.num_children > 0:
            group_score += 0.4
        if profile.accessibility_needs:
            group_score += 0.35
        if profile.dietary_restrictions:
            group_score += 0.25
        group_score = min(1.0, group_score)

        # 5. Preference specificity via soft constraint count
        pref_score = min(1.0, n_soft / 6.0)  # 6+ soft constraints = high

        # ── Composite ─────────────────────────────────────────────────────────
        w = self.WEIGHTS
        composite = (
            w["constraint_density"] * constraint_density_score
            + w["geographic_complexity"] * geo_score
            + w["budget_tightness"] * budget_score
            + w["group_complexity"] * group_score
            + w["preference_specificity"] * pref_score
        )

        if composite < 0.35:
            tier = "low"
        elif composite < 0.65:
            tier = "medium"
        else:
            tier = "high"

        # ── Derived labels ────────────────────────────────────────────────────
        n_adults = profile.num_adults
        if n_adults == 1 and profile.num_children == 0:
            group_type = "solo"
        elif n_adults == 2 and profile.num_children == 0:
            group_type = "couple"
        elif profile.num_children > 0:
            group_type = "family"
        else:
            group_type = "group"

        if n_soft <= 2:
            pref_label = "low"
        elif n_soft <= 4:
            pref_label = "medium"
        else:
            pref_label = "high"

        return ComplexityBreakdown(
            n_hard_constraints=n_hard,
            n_soft_constraints=n_soft,
            n_destinations=n_dest,
            trip_duration_days=duration_days,
            budget_per_traveler_per_day=round(budget_per_traveler_per_day, 2),
            constraint_density=round(density_raw, 3),
            geographic_complexity="single_city" if n_dest <= 1 else "multi_city",
            group_type=group_type,
            has_special_needs=bool(profile.accessibility_needs or profile.dietary_restrictions),
            preference_specificity=pref_label,
            constraint_density_score=round(constraint_density_score, 4),
            geographic_complexity_score=round(geo_score, 4),
            budget_tightness_score=round(budget_score, 4),
            group_complexity_score=round(group_score, 4),
            preference_specificity_score=round(pref_score, 4),
            complexity_score=round(composite, 4),
            complexity_tier=tier,
        )
