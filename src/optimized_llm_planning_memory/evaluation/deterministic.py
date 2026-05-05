"""
evaluation/deterministic.py
============================
DeterministicEvaluator — constraint-based scoring without LLM calls.

CRITICAL DESIGN INVARIANT
--------------------------
This module uses ``ConstraintSatisfactionEngine`` from ``core/constraints.py``.
``training/reward.py`` uses the **same class**. This is intentional and must
be preserved. If the imports ever diverge (e.g., one module monkey-patches the
engine), the training reward and evaluation metric will silently diverge, and
the compressor will optimise a proxy.

Metrics produced
----------------
v1 (8 metrics)
  hard_constraint_ratio   — fraction of hard constraints satisfied [0, 1]
  soft_constraint_score   — weighted soft constraint score [0, 1]
  tool_efficiency         — 1 - redundant_calls / total_calls [0, 1]
  tool_failure_rate       — failures / total_calls [0, 1]  (lower is better)
  avg_tool_latency_ms     — mean latency per tool call
  steps_per_episode       — total ReAct steps taken
  budget_adherence        — 1 if within budget, fraction remaining otherwise
  logical_consistency     — date ordering + no double-bookings [0, 1]

v2 (6 trip-quality metrics)
  destination_coverage_ratio      — fraction of required cities visited [0, 1]
  accommodation_coverage_ratio    — fraction of non-departure nights with hotel [0, 1]
  activity_density_score          — mean per-day score vs optimal 2-4 activities [0, 1]
  rest_day_ratio                  — light-day frequency vs target 1-per-4 [0, 1]
  schedule_overlap_score          — fraction of activity pairs with no time overlap [0, 1]
  intra_city_feasibility          — fraction of activity gaps ≥ 15 min [0, 1]

v3 (4 token-efficiency metrics — nan when token data not available)
  tokens_per_episode                      — total LLM tokens in episode
  tokens_per_hard_constraint_satisfied    — tokens / (n_satisfied + 1)
  tokens_per_soft_constraint_satisfied    — tokens / (effective_soft_satisfied + 1)
  compression_ratio                       — mean compressed/raw token ratio [0, 1]
"""

from __future__ import annotations

from datetime import datetime, timedelta

from optimized_llm_planning_memory.core.constraints import ConstraintSatisfactionEngine
from optimized_llm_planning_memory.core.models import EpisodeLog, Itinerary, UserRequest

METRIC_VERSION = "v3"

METRIC_CHANGELOG: dict[str, str] = {
    "v1": (
        "Initial 8 metrics: hard_constraint_ratio, soft_constraint_score, "
        "tool_efficiency, tool_failure_rate, avg_tool_latency_ms, "
        "steps_per_episode, budget_adherence, logical_consistency. "
        "Constraint evaluation via ConstraintSatisfactionEngine."
    ),
    "v2": (
        "Added 6 trip-quality metrics: destination_coverage_ratio, "
        "accommodation_coverage_ratio, activity_density_score, rest_day_ratio, "
        "schedule_overlap_score, intra_city_feasibility. "
        "Fixed multi-hotel star-rating bug in ConstraintSatisfactionEngine."
    ),
    "v3": (
        "Added 4 token-efficiency metrics: tokens_per_episode, "
        "tokens_per_hard_constraint_satisfied, tokens_per_soft_constraint_satisfied, "
        "compression_ratio. Requires EpisodeLog.total_tokens_used (populated by "
        "ReActAgent) and CompressedState.token_count / raw_token_count (populated "
        "by LLMCompressor and TransformerCompressor). Returns nan when data absent."
    ),
}


class DeterministicEvaluator:
    """
    Computes deterministic (rule-based) scores for a completed episode.

    Parameters
    ----------
    constraint_engine : Shared ``ConstraintSatisfactionEngine`` instance.
                        MUST be the same implementation used by RewardFunction.
    """

    def __init__(
        self,
        constraint_engine: ConstraintSatisfactionEngine | None = None,
    ) -> None:
        self._engine = constraint_engine or ConstraintSatisfactionEngine()

    def score(
        self,
        episode_log: EpisodeLog,
        user_request: UserRequest,
    ) -> dict[str, float]:
        """
        Compute all deterministic metrics for the episode.

        Parameters
        ----------
        episode_log  : Completed episode log.
        user_request : Original user request (for constraint evaluation).

        Returns
        -------
        dict[str, float] with keys matching the metric names in the module docstring.
        """
        itinerary = episode_log.final_itinerary
        stats = episode_log.tool_stats

        hard_ratio = self._hard_constraint_ratio(itinerary, user_request)
        soft_score = self._soft_constraint_score(itinerary, user_request)

        return {
            # ── v1 metrics (unchanged) ───────────────────────────────────────
            "hard_constraint_ratio": hard_ratio,
            "soft_constraint_score": soft_score,
            "tool_efficiency": self._tool_efficiency(stats),
            "tool_failure_rate": self._tool_failure_rate(stats),
            "avg_tool_latency_ms": self._avg_tool_latency_ms(stats),
            "steps_per_episode": float(episode_log.total_steps),
            "budget_adherence": self._budget_adherence(itinerary, user_request),
            "logical_consistency": self._logical_consistency(itinerary),
            # ── v2 metrics (trip quality) ────────────────────────────────────
            "destination_coverage_ratio": self._destination_coverage_ratio(itinerary, user_request),
            "accommodation_coverage_ratio": self._accommodation_coverage_ratio(itinerary),
            "activity_density_score": self._activity_density_score(itinerary),
            "rest_day_ratio": self._rest_day_ratio(itinerary),
            "schedule_overlap_score": self._schedule_overlap_score(itinerary),
            "intra_city_feasibility": self._intra_city_feasibility(itinerary),
            # ── v3 metrics (token efficiency) ────────────────────────────────
            "tokens_per_episode": self._tokens_per_episode(episode_log),
            "tokens_per_hard_constraint_satisfied": self._tokens_per_hard_constraint_satisfied(
                episode_log, hard_ratio, user_request
            ),
            "tokens_per_soft_constraint_satisfied": self._tokens_per_soft_constraint_satisfied(
                episode_log, soft_score, user_request
            ),
            "compression_ratio": self._compression_ratio(episode_log),
        }

    # ── Component scorers ─────────────────────────────────────────────────────

    def _hard_constraint_ratio(
        self, itinerary: Itinerary | None, request: UserRequest
    ) -> float:
        if itinerary is None or not request.hard_constraints:
            return 0.0
        results = self._engine.evaluate(itinerary, list(request.hard_constraints))
        return self._engine.hard_satisfaction_ratio(results, list(request.hard_constraints))

    def _soft_constraint_score(
        self, itinerary: Itinerary | None, request: UserRequest
    ) -> float:
        if itinerary is None or not request.soft_constraints:
            # 1.0 matches constraints.py and reward.py: no constraints = fully satisfied (H5 fix).
            return 1.0
        results = self._engine.evaluate(itinerary, list(request.soft_constraints))
        return self._engine.soft_satisfaction_score(results, list(request.soft_constraints))

    def _tool_efficiency(self, stats: tuple) -> float:
        """1 - redundant_call_ratio. Higher is better."""
        if not stats:
            return 1.0
        total = sum(s.call_count for s in stats)
        redundant = sum(s.redundant_call_count for s in stats)
        if total == 0:
            return 1.0
        return max(0.0, 1.0 - redundant / total)

    def _tool_failure_rate(self, stats: tuple) -> float:
        """Fraction of tool calls that failed. Lower is better."""
        if not stats:
            return 0.0
        total = sum(s.call_count for s in stats)
        failures = sum(s.failure_count for s in stats)
        if total == 0:
            return 0.0
        return failures / total

    def _avg_tool_latency_ms(self, stats: tuple) -> float:
        """Mean latency per tool call in milliseconds."""
        if not stats:
            return 0.0
        total_calls = sum(s.call_count for s in stats)
        total_latency = sum(s.total_latency_ms for s in stats)
        if total_calls == 0:
            return 0.0
        return total_latency / total_calls

    def _budget_adherence(
        self, itinerary: Itinerary | None, request: UserRequest
    ) -> float:
        """
        1.0 if itinerary is within budget.
        Fraction of budget used if over budget (clamped to [0, 1]).
        """
        if itinerary is None or request.budget_usd <= 0:
            return 0.0
        if itinerary.total_cost_usd <= request.budget_usd:
            return 1.0
        # Penalise overspend proportionally
        overshoot = itinerary.total_cost_usd - request.budget_usd
        return max(0.0, 1.0 - overshoot / request.budget_usd)

    def _logical_consistency(self, itinerary: Itinerary | None) -> float:
        """
        Date ordering, no duplicate hotel bookings, and temporal feasibility. [0, 1]

        Checks (M3 fix — added temporal feasibility):
        1. Itinerary days are in chronological order.
        2. No hotel_id appears on more than one day.
        3. No two activities on the same day have overlapping time windows
           (requires start_datetime + duration_hours on ActivityBooking).
        4. Flight arrival datetime is on or before the hotel check-in date.
        """
        if itinerary is None or not itinerary.days:
            return 0.0
        issues = 0
        checks = 0

        # Check 1: days are sorted
        dates = [d.date for d in itinerary.days]
        checks += 1
        if dates != sorted(dates):
            issues += 1

        # Check 2: no duplicate hotel bookings
        hotel_ids = [
            d.accommodation.hotel_id
            for d in itinerary.days
            if d.accommodation is not None
        ]
        checks += 1
        if len(hotel_ids) != len(set(hotel_ids)):
            issues += 1

        # Check 3: no overlapping activities within a day
        for day in itinerary.days:
            if not day.activities:
                continue
            slots: list[tuple[str, str]] = []
            for act in day.activities:
                start = getattr(act, "start_datetime", None)
                dur = getattr(act, "duration_hours", None)
                if start and dur:
                    try:
                        from datetime import datetime, timedelta
                        start_dt = datetime.fromisoformat(start)
                        end_dt = start_dt + timedelta(hours=float(dur))
                        slots.append((start_dt.isoformat(), end_dt.isoformat()))
                    except (ValueError, TypeError):
                        pass
            if len(slots) > 1:
                checks += 1
                sorted_slots = sorted(slots)
                overlap = any(
                    sorted_slots[i][1] > sorted_slots[i + 1][0]
                    for i in range(len(sorted_slots) - 1)
                )
                if overlap:
                    issues += 1

        # Check 4: flight arrival ≤ hotel check-in date on the same day
        for day in itinerary.days:
            if day.accommodation is None:
                continue
            hotel_checkin = getattr(day.accommodation, "check_in", None)
            for seg in day.transport_segments:
                arrival = getattr(seg, "arrival_datetime", None)
                if hotel_checkin and arrival:
                    try:
                        arrival_date = arrival[:10]
                        checks += 1
                        if arrival_date > hotel_checkin:
                            issues += 1
                    except (IndexError, TypeError):
                        pass

        return max(0.0, 1.0 - issues / checks) if checks > 0 else 1.0

    # ── v2 metrics ────────────────────────────────────────────────────────────

    def _destination_coverage_ratio(
        self, itinerary: Itinerary | None, request: UserRequest
    ) -> float:
        """Fraction of required destination cities with ≥1 itinerary day planned.

        Uses case-insensitive substring matching between ItineraryDay.city (human
        name) and metadata.dest_names (list[str] injected at generation time).
        Falls back to checking destination_cities IDs directly if dest_names is absent.
        """
        if itinerary is None or not itinerary.days:
            return 0.0
        dest_names: list[str] = request.metadata.get("dest_names", [])
        if not dest_names:
            dest_names = [n.lower() for n in request.destination_cities]
        visited = {d.city.lower() for d in itinerary.days}
        covered = sum(
            1 for name in dest_names
            if any(name.lower() in v or v in name.lower() for v in visited)
        )
        return covered / len(dest_names) if dest_names else 1.0

    def _accommodation_coverage_ratio(self, itinerary: Itinerary | None) -> float:
        """Fraction of non-departure days with an accommodation booking.

        The last day is excluded since travellers depart rather than check in.
        Returns 1.0 for single-day itineraries (no overnight stay expected).
        """
        if itinerary is None or not itinerary.days:
            return 0.0
        nights = itinerary.days[:-1]  # exclude departure day
        if not nights:
            return 1.0
        booked = sum(1 for d in nights if d.accommodation is not None)
        return booked / len(nights)

    def _activity_density_score(self, itinerary: Itinerary | None) -> float:
        """Score activity count per day against the optimal range [2, 4].

        Per-day scoring:
          0 activities  → 0.0  (empty day)
          1 activity    → 0.5  (under-scheduled)
          2–4 activities → 1.0  (optimal)
          5+ activities → max(0, 1.0 - 0.2 × (count − 4))  (over-packed)

        Final score is the mean across all days.
        """
        if itinerary is None or not itinerary.days:
            return 0.0

        def _day_score(n: int) -> float:
            if n == 0:
                return 0.0
            if n == 1:
                return 0.5
            if n <= 4:
                return 1.0
            return max(0.0, 1.0 - 0.2 * (n - 4))

        scores = [_day_score(len(d.activities)) for d in itinerary.days]
        return sum(scores) / len(scores)

    def _rest_day_ratio(self, itinerary: Itinerary | None) -> float:
        """Fraction of days classified as light (≤1 activity) relative to a target.

        Target: 1 rest day per every 4 travel days. Returns 1.0 for trips ≤5 days
        where continuous activity is expected and no rest days are required.
        """
        if itinerary is None or not itinerary.days:
            return 0.0
        total = len(itinerary.days)
        if total <= 5:
            return 1.0
        rest_days = sum(1 for d in itinerary.days if len(d.activities) <= 1)
        target = max(1, total // 4)
        return min(1.0, rest_days / target)

    def _schedule_overlap_score(self, itinerary: Itinerary | None) -> float:
        """Fraction of same-day activity pairs that do NOT have overlapping time windows.

        Complements logical_consistency check 3 by returning a continuous score
        rather than a binary flag. Returns 1.0 if no time data is available.
        """
        if itinerary is None or not itinerary.days:
            return 1.0
        total_pairs = 0
        non_overlapping = 0
        for day in itinerary.days:
            slots: list[tuple[datetime, datetime]] = []
            for act in day.activities:
                start = getattr(act, "start_datetime", None)
                dur = getattr(act, "duration_hours", None)
                if start and dur:
                    try:
                        s = datetime.fromisoformat(start)
                        e = s + timedelta(hours=float(dur))
                        slots.append((s, e))
                    except (ValueError, TypeError):
                        pass
            slots.sort()
            for i in range(len(slots) - 1):
                total_pairs += 1
                if slots[i][1] <= slots[i + 1][0]:
                    non_overlapping += 1
        return non_overlapping / total_pairs if total_pairs > 0 else 1.0

    def _intra_city_feasibility(self, itinerary: Itinerary | None) -> float:
        """Fraction of consecutive same-city activity gaps that are ≥15 minutes.

        A gap shorter than 15 minutes between the end of one activity and the
        start of the next is flagged as infeasible (no time to travel/transition).
        Returns 1.0 if fewer than 2 timed activities exist.
        """
        if itinerary is None or not itinerary.days:
            return 1.0
        _MIN_GAP = timedelta(minutes=15)
        total_gaps = 0
        feasible_gaps = 0
        for day in itinerary.days:
            timed: list[tuple[datetime, datetime]] = []
            for act in day.activities:
                start = getattr(act, "start_datetime", None)
                dur = getattr(act, "duration_hours", None)
                if start and dur:
                    try:
                        s = datetime.fromisoformat(start)
                        e = s + timedelta(hours=float(dur))
                        timed.append((s, e))
                    except (ValueError, TypeError):
                        pass
            timed.sort()
            for i in range(len(timed) - 1):
                total_gaps += 1
                gap = timed[i + 1][0] - timed[i][1]
                if gap >= _MIN_GAP:
                    feasible_gaps += 1
        return feasible_gaps / total_gaps if total_gaps > 0 else 1.0

    # ── v3 metrics (token efficiency) ─────────────────────────────────────────

    def _tokens_per_episode(self, episode_log: EpisodeLog) -> float:
        """Total LLM tokens consumed in the episode. nan when not tracked."""
        t = episode_log.total_tokens_used
        return float(t) if t is not None else float("nan")

    def _tokens_per_hard_constraint_satisfied(
        self,
        episode_log: EpisodeLog,
        hard_ratio: float,
        request: UserRequest,
    ) -> float:
        """Tokens per hard constraint satisfied (lower is more efficient). nan when not tracked.

        Formula: total_tokens / (n_satisfied + 1)
        The +1 prevents division-by-zero when no constraints are satisfied.
        """
        t = episode_log.total_tokens_used
        if t is None:
            return float("nan")
        n_hard = len(request.hard_constraints)
        n_satisfied = hard_ratio * n_hard
        return float(t) / (n_satisfied + 1.0)

    def _tokens_per_soft_constraint_satisfied(
        self,
        episode_log: EpisodeLog,
        soft_score: float,
        request: UserRequest,
    ) -> float:
        """Tokens per soft constraint satisfied (lower is more efficient). nan when not tracked.

        Formula: total_tokens / (n_soft * soft_score + 1)
        Uses soft_score as a fractional satisfaction count across all soft constraints.
        """
        t = episode_log.total_tokens_used
        if t is None:
            return float("nan")
        n_soft = len(request.soft_constraints)
        effective_satisfied = n_soft * soft_score
        return float(t) / (effective_satisfied + 1.0)

    def _compression_ratio(self, episode_log: EpisodeLog) -> float:
        """Mean ratio of compressed-to-raw tokens across all compression events. nan when not tracked.

        A ratio of 0.3 means the compressor reduced trajectory size by 70%.
        Only compression events where both token_count and raw_token_count are
        non-None and non-zero are included. Returns nan if no such events exist
        (e.g. RAW mode, or compressor did not populate token counts).
        """
        ratios = []
        for cs in episode_log.compressed_states:
            if (
                cs.token_count is not None
                and cs.raw_token_count is not None
                and cs.raw_token_count > 0
            ):
                ratios.append(cs.token_count / cs.raw_token_count)
        return sum(ratios) / len(ratios) if ratios else float("nan")
