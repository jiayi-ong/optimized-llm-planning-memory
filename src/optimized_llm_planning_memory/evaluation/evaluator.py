"""
evaluation/evaluator.py
========================
Evaluator — orchestrates deterministic scoring and LLM judge for a dataset of
episodes.

Data flow
---------
    EpisodeLog + UserRequest
        ├── DeterministicEvaluator.score() → dict[str, float]
        └── LLMJudge.score()              → dict[str, float]
                                           ↓
                                     EvalResult
                                           ↓
                                 Evaluator.aggregate() → dict[str, float]
                                                         (mean ± std per metric)

Design notes
------------
- ``LLMJudge`` calls are optional (``deterministic_only=True`` skips them).
  This is useful for fast inner-loop evaluation during training.
- ``evaluate_dataset()`` processes episodes sequentially. For parallel
  evaluation, wrap with ``concurrent.futures.ThreadPoolExecutor`` — both
  evaluators are stateless between calls.
- ``aggregate()`` returns means across the dataset; individual ``EvalResult``
  objects contain per-episode scores for per-request analysis.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone

from optimized_llm_planning_memory.core.config import EvalConfig
from optimized_llm_planning_memory.core.models import EpisodeLog, EvalResult, UserRequest
from optimized_llm_planning_memory.evaluation.deterministic import DeterministicEvaluator, METRIC_VERSION
from optimized_llm_planning_memory.evaluation.llm_judge import LLMJudge


class Evaluator:
    """
    Orchestrates deterministic + LLM judge evaluation of planning episodes.

    Parameters
    ----------
    deterministic_eval : Pre-built ``DeterministicEvaluator`` instance.
    llm_judge          : Pre-built ``LLMJudge`` instance (or None to skip).
    config             : Evaluation configuration (deterministic_only, etc.).
    """

    def __init__(
        self,
        deterministic_eval: DeterministicEvaluator | None = None,
        llm_judge: LLMJudge | None = None,
        config: EvalConfig | None = None,
    ) -> None:
        self._det = deterministic_eval or DeterministicEvaluator()
        self._judge = llm_judge
        self._config = config or EvalConfig()

    def evaluate_episode(
        self,
        episode_log: EpisodeLog,
        user_request: UserRequest,
    ) -> EvalResult:
        """
        Evaluate a single completed episode.

        Parameters
        ----------
        episode_log  : Completed episode log.
        user_request : Original user request.

        Returns
        -------
        EvalResult with deterministic + LLM judge scores.
        """
        det_scores = self._det.score(episode_log, user_request)

        llm_scores: dict[str, float] = {}
        rubric_breakdown: dict[str, dict] = {}
        judge_model = "none"

        use_judge = (
            not self._config.deterministic_only
            and self._judge is not None
            and episode_log.final_itinerary is not None
        )
        if use_judge:
            llm_scores, rubric_breakdown = self._judge.score_detailed(  # type: ignore[union-attr]
                episode_log.final_itinerary,  # type: ignore[arg-type]
                user_request,
            )
            judge_model = self._judge._model  # type: ignore[union-attr]

        overall = self._compute_overall(det_scores, llm_scores)

        return EvalResult(
            episode_id=episode_log.episode_id,
            request_id=episode_log.request_id,
            agent_mode=episode_log.agent_mode,
            deterministic_scores=det_scores,
            llm_judge_scores=llm_scores,
            overall_score=max(0.0, min(1.0, overall)),
            rubric_breakdown=rubric_breakdown,
            judge_model=judge_model,
            created_at=datetime.now(timezone.utc).isoformat(),
            metric_version=METRIC_VERSION,
            world_seed=episode_log.world_seed,
        )

    def evaluate_dataset(
        self,
        episodes: list[EpisodeLog],
        requests: list[UserRequest],
    ) -> list[EvalResult]:
        """
        Evaluate a list of episodes.

        Parameters
        ----------
        episodes : List of completed episode logs.
        requests : Corresponding list of user requests (same order / same length).

        Returns
        -------
        List of EvalResult, one per episode.
        """
        if len(episodes) != len(requests):
            raise ValueError(
                f"episodes and requests must be the same length, "
                f"got {len(episodes)} and {len(requests)}."
            )
        return [
            self.evaluate_episode(ep, req)
            for ep, req in zip(episodes, requests)
        ]

    def aggregate(self, results: list[EvalResult]) -> dict[str, float]:
        """
        Compute mean (and std) across all EvalResult objects.

        Returns a flat dict with keys like ``"hard_constraint_ratio_mean"``,
        ``"hard_constraint_ratio_std"``, ``"overall_score_mean"``, etc.

        Parameters
        ----------
        results : List of EvalResult objects from ``evaluate_dataset()``.

        Returns
        -------
        dict[str, float]
        """
        if not results:
            return {}

        # Collect all metric keys
        all_det_keys: set[str] = set()
        all_llm_keys: set[str] = set()
        for r in results:
            all_det_keys.update(r.deterministic_scores.keys())
            all_llm_keys.update(r.llm_judge_scores.keys())

        agg: dict[str, float] = {}

        for key in sorted(all_det_keys):
            vals = [r.deterministic_scores.get(key, 0.0) for r in results]
            agg[f"{key}_mean"] = statistics.mean(vals)
            agg[f"{key}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0

        for key in sorted(all_llm_keys):
            vals = [r.llm_judge_scores.get(key, 0.0) for r in results]
            agg[f"judge_{key}_mean"] = statistics.mean(vals)
            agg[f"judge_{key}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0

        overall_vals = [r.overall_score for r in results]
        agg["overall_score_mean"] = statistics.mean(overall_vals)
        agg["overall_score_std"] = statistics.stdev(overall_vals) if len(overall_vals) > 1 else 0.0

        return agg

    # ── Private ───────────────────────────────────────────────────────────────

    def _compute_overall(
        self,
        det_scores: dict[str, float],
        llm_scores: dict[str, float],
    ) -> float:
        """
        Compute a single overall score as a weighted average.

        Hard constraint ratio is given double weight since it is the primary
        success criterion. Other deterministic metrics and LLM judge scores
        contribute equally.
        """
        components: list[float] = []

        hard = det_scores.get("hard_constraint_ratio", 0.0)
        components.extend([hard, hard])  # double weight

        for key in ["soft_constraint_score", "tool_efficiency", "budget_adherence",
                    "logical_consistency"]:
            if key in det_scores:
                components.append(det_scores[key])

        # LLM judge scores (equally weighted)
        for v in llm_scores.values():
            components.append(v)

        if not components:
            return 0.0
        return sum(components) / len(components)
