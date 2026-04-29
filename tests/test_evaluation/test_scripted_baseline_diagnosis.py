"""
tests/test_evaluation/test_scripted_baseline_diagnosis.py
==========================================================
Regression tests that pin the two known scoring failures in the scripted
baseline episode (run_id=dcc621eb, req-aeloria-4day-v001, seed=42):

  hard_constraint_ratio  = 0.667  (hc-dates fails)
  logical_consistency    = 0.667  (overlapping activities)

Each test builds the exact itinerary structure that run_scripted_episode()
currently produces — one ItineraryDay on the check-in date, all attractions
sharing the same start_datetime.  The tests assert the current (broken)
behaviour so that a future fix turns them green.

WHY THESE LIVE IN TESTS (not only in the notebook):
- The notebook cell in Section 8d shows the failure interactively on a live
  episode.  These tests catch regressions automatically in CI and document
  the *exact* structural cause, not just the aggregate score.
"""

from __future__ import annotations

import uuid

import pytest

from optimized_llm_planning_memory.core.constraints import ConstraintSatisfactionEngine
from optimized_llm_planning_memory.core.models import (
    AccommodationBooking,
    ActivityBooking,
    Constraint,
    ConstraintCategory,
    ConstraintType,
    Itinerary,
    ItineraryDay,
)
from optimized_llm_planning_memory.evaluation.deterministic import DeterministicEvaluator


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _hotel() -> AccommodationBooking:
    return AccommodationBooking(
        hotel_id="hotel-scripted-001",
        hotel_name="The Old Town Inn",
        city="Aeloria",
        check_in="2026-06-01",
        check_out="2026-06-05",
        cost_per_night_usd=233.0,
        total_cost_usd=932.0,
        star_rating=3.0,
        booking_ref="BK-001",
    )


def _attraction(i: int, start: str = "2026-06-01T10:00:00", dur: float = 2.0) -> ActivityBooking:
    return ActivityBooking(
        activity_id=f"attr-{i}",
        activity_name=f"Attraction {i}",
        location="Aeloria",
        city="Aeloria",
        start_datetime=start,
        duration_hours=dur,
        cost_usd=10.0,
        category="attraction",
    )


def _event() -> ActivityBooking:
    return ActivityBooking(
        activity_id="event-001",
        activity_name="City Aquarium",
        location="Aeloria",
        city="Aeloria",
        start_datetime="2026-06-01T19:00:00",
        duration_hours=2.0,
        cost_usd=34.0,
        category="event",
    )


def _scripted_baseline_itinerary() -> Itinerary:
    """
    Reproduces what run_scripted_episode() currently builds.

    The scripted agent calls _get_or_create_day(check_in, ...) for every
    booking, so everything lands on a single ItineraryDay ("2026-06-01").
    The four nights June 2–5 are never represented.
    """
    day = ItineraryDay(
        date="2026-06-01",
        city="Aeloria",
        accommodation=_hotel(),
        # Three attractions all at 10:00 + one evening event
        activities=[_attraction(0), _attraction(1), _attraction(2), _event()],
    )
    return Itinerary(
        itinerary_id="it-scripted",
        request_id="req-aeloria-4day-v001",
        days=[day],
    )


def _date_constraint() -> Constraint:
    return Constraint(
        constraint_id="hc-dates",
        constraint_type=ConstraintType.HARD,
        category=ConstraintCategory.DATE,
        description="Stay must cover June 1–5 2026.",
        value="2026-06-01 to 2026-06-05",
        unit="date_range",
    )


# ── hc-dates failure ──────────────────────────────────────────────────────────

class TestDateConstraintOnSingleDayItinerary:
    """
    _evaluate_date checks days[-1].date == expected_end.
    A single-day itinerary has days[-1].date == "2026-06-01", not "2026-06-05".
    """

    def test_date_constraint_fails(self):
        engine = ConstraintSatisfactionEngine()
        itinerary = _scripted_baseline_itinerary()
        result = engine._evaluate_date(itinerary, _date_constraint())

        assert result.satisfied is False, (
            "hc-dates should FAIL: only 1 ItineraryDay in scripted itinerary.\n"
            f"days[-1].date = {itinerary.days[-1].date!r}, expected '2026-06-05'.\n"
            f"Engine explanation: {result.explanation}"
        )

    def test_date_constraint_partial_score(self):
        """Start date matches, end date doesn't → score = 0.5."""
        engine = ConstraintSatisfactionEngine()
        itinerary = _scripted_baseline_itinerary()
        result = engine._evaluate_date(itinerary, _date_constraint())

        assert result.score == pytest.approx(0.5), (
            "Expected score=0.5: start '2026-06-01' correct, "
            "end '2026-06-01' != '2026-06-05'.\n"
            f"Engine explanation: {result.explanation}"
        )

    def test_date_constraint_passes_with_all_days(self):
        """Confirm the fix: 4 days (Jun 1–5) should satisfy hc-dates."""
        engine = ConstraintSatisfactionEngine()
        days = [
            ItineraryDay(date=f"2026-06-0{i}", city="Aeloria",
                         accommodation=_hotel() if i == 1 else None)
            for i in range(1, 5)
        ]
        days.append(ItineraryDay(date="2026-06-05", city="Aeloria"))
        itinerary = Itinerary(
            itinerary_id="it-fixed",
            request_id="req-aeloria-4day-v001",
            days=days,
        )
        result = engine._evaluate_date(itinerary, _date_constraint())
        assert result.satisfied is True, (
            "With days for Jun 1–5, hc-dates should pass.\n"
            f"Engine explanation: {result.explanation}"
        )


# ── hard_constraint_ratio = 2/3 ───────────────────────────────────────────────

class TestHardConstraintRatioOnScriptedBaseline:
    """
    Of the 3 hard constraints (hc-budget, hc-dates, hc-hotel-stars),
    only hc-dates fails on the scripted baseline itinerary.
    """

    def _all_hard_constraints(self) -> list[Constraint]:
        return [
            Constraint(
                constraint_id="hc-budget",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.BUDGET,
                description="Budget <= $1200",
                value=1200.0,
                unit="USD",
            ),
            _date_constraint(),
            Constraint(
                constraint_id="hc-hotel-stars",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.ACCOMMODATION,
                description="Hotel >= 3 stars",
                value=3,
                unit="min_stars",
            ),
        ]

    def test_hard_ratio_is_two_thirds(self):
        evaluator = DeterministicEvaluator()
        itinerary = _scripted_baseline_itinerary()
        constraints = self._all_hard_constraints()
        ratio = evaluator._hard_constraint_ratio(itinerary, _MockRequest(constraints))
        assert ratio == pytest.approx(2 / 3), (
            "Expected 2/3: hc-budget and hc-hotel-stars pass, hc-dates fails "
            "(scripted baseline only creates 1 ItineraryDay for the check-in date)."
        )

    def test_which_constraint_fails(self):
        """Confirm it is specifically hc-dates that fails, not the others."""
        engine = ConstraintSatisfactionEngine()
        itinerary = _scripted_baseline_itinerary()
        constraints = self._all_hard_constraints()
        results = engine.evaluate(itinerary, constraints)
        by_id = {r.constraint_id: r for r in results}

        assert by_id["hc-budget"].satisfied is True
        assert by_id["hc-hotel-stars"].satisfied is True
        assert by_id["hc-dates"].satisfied is False, (
            f"hc-dates should fail. Explanation: {by_id['hc-dates'].explanation}"
        )


# ── logical_consistency = 2/3 ─────────────────────────────────────────────────

class TestLogicalConsistencyOnScriptedBaseline:
    """
    The scripted baseline puts 3 attractions at 10:00 on the same day.
    _logical_consistency fires 3 checks (sort, no-dupe-hotels, overlap-day)
    and finds 1 failure (overlap), giving 1 - 1/3 = 0.667.
    """

    def test_consistency_is_two_thirds(self):
        evaluator = DeterministicEvaluator()
        itinerary = _scripted_baseline_itinerary()
        score = evaluator._logical_consistency(itinerary)
        assert score == pytest.approx(2 / 3), (
            "Expected 2/3: 3 total consistency checks, 1 fails due to "
            "overlapping attraction slots all starting at 10:00."
        )

    def test_overlap_comes_from_same_start_time(self):
        """Isolated: two activities with identical start times always overlap."""
        evaluator = DeterministicEvaluator()
        day = ItineraryDay(
            date="2026-06-01",
            city="Aeloria",
            activities=[_attraction(0, "2026-06-01T10:00:00", 2.0),
                        _attraction(1, "2026-06-01T10:00:00", 2.0)],
        )
        itinerary = Itinerary(
            itinerary_id="it-overlap",
            request_id="req-test",
            days=[day],
        )
        score = evaluator._logical_consistency(itinerary)
        assert score < 1.0, (
            "Two activities sharing start_datetime should trigger the overlap check."
        )

    def test_no_overlap_with_spread_activities(self):
        """Confirm the fix: spreading activities across the day clears the overlap."""
        evaluator = DeterministicEvaluator()
        day = ItineraryDay(
            date="2026-06-01",
            city="Aeloria",
            activities=[
                _attraction(0, "2026-06-01T09:00:00", 2.0),   # 09:00–11:00
                _attraction(1, "2026-06-01T13:00:00", 2.0),   # 13:00–15:00
                _attraction(2, "2026-06-01T16:00:00", 2.0),   # 16:00–18:00
                _event(),                                        # 19:00–21:00
            ],
        )
        itinerary = Itinerary(
            itinerary_id="it-no-overlap",
            request_id="req-test",
            days=[day],
        )
        score = evaluator._logical_consistency(itinerary)
        assert score == pytest.approx(1.0), (
            "Non-overlapping activities should give logical_consistency = 1.0."
        )


# ── Minimal request stub used by DeterministicEvaluator._hard_constraint_ratio ─

class _MockRequest:
    """Minimal stand-in for UserRequest — only hard_constraints is needed."""
    def __init__(self, hard_constraints: list[Constraint]):
        self.hard_constraints = hard_constraints
        self.soft_constraints = []
        self.budget_usd = 1200.0
