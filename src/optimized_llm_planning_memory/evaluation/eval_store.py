"""
evaluation/eval_store.py
=========================
EvalStore — centralised read layer for all evaluation-related data.

Design rationale
----------------
Previously each Streamlit page imported ``list_eval_runs``, ``load_eval_run``,
and ``list_episodes`` independently, duplicating data-loading logic across files.
EvalStore centralises those calls behind a single object so:
  - The on-disk layout is described in one place.
  - Callers get a flat pandas DataFrame with manifest join columns pre-applied,
    without knowing the directory structure.
  - The class is trivially mockable for tests.
  - All caching (if used with Streamlit) can be applied at this boundary.

Usage
-----
    store = EvalStore(
        eval_dir=Path("outputs/eval_results"),
        episodes_dir=Path("outputs/episodes"),
        requests_dir=Path("data/user_requests"),
    )

    manifests = store.list_runs()
    manifest, results = store.load_run("20260506_120000_abc12345")
    agg = store.load_aggregate("20260506_120000_abc12345")
    df = store.flat_results(agent_modes=["raw", "compressor"])
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from optimized_llm_planning_memory.core.models import EpisodeLog, EvalResult, UserRequest
from optimized_llm_planning_memory.evaluation.manifest import EvalRunManifest
from optimized_llm_planning_memory.utils.episode_io import (
    load_aggregate,
    load_episode,
    load_eval_run,
    list_eval_runs,
)


class EvalStore:
    """Read-only facade over the eval_results / episodes / user_requests on-disk layout."""

    def __init__(
        self,
        eval_dir: Path,
        episodes_dir: Path,
        requests_dir: Path,
    ) -> None:
        self.eval_dir = Path(eval_dir)
        self.episodes_dir = Path(episodes_dir)
        self.requests_dir = Path(requests_dir)

    # ── Run-level ─────────────────────────────────────────────────────────────

    def list_runs(self) -> list[EvalRunManifest]:
        """Return all eval run manifests, newest first."""
        return list_eval_runs(self.eval_dir)

    def load_run(self, run_id: str) -> tuple[EvalRunManifest, list[EvalResult]]:
        """Load manifest + per-episode results for a single run."""
        return load_eval_run(run_id, self.eval_dir)

    def load_aggregate(self, run_id: str) -> dict | None:
        """Load pre-computed aggregate stats (mean/std/min/max per metric).

        Returns None when aggregate.json has not yet been generated for the run,
        which can happen for runs created before this feature was added.
        """
        return load_aggregate(run_id, self.eval_dir)

    # ── Episode-level ─────────────────────────────────────────────────────────

    def load_episode(self, episode_id: str) -> EpisodeLog:
        """Load a single EpisodeLog by episode_id."""
        return load_episode(self.episodes_dir / f"ep_{episode_id}.json")

    # ── Request-level ─────────────────────────────────────────────────────────

    def load_request(self, request_id: str) -> UserRequest | None:
        """Search across train/val/test splits for a request by ID."""
        for split_dir in self._split_dirs():
            for json_file in split_dir.glob("request_*.json"):
                try:
                    req = UserRequest.model_validate_json(json_file.read_text(encoding="utf-8"))
                    if req.request_id == request_id:
                        return req
                except Exception:
                    continue
        return None

    def list_requests(self, split: str | None = None, collection: str | None = None) -> list[UserRequest]:
        """Return all UserRequest objects, optionally restricted to one collection (or legacy split)."""
        name = collection or split
        dirs = [self.requests_dir / name] if name else self._split_dirs()
        requests: list[UserRequest] = []
        for d in dirs:
            if not d.exists():
                continue
            for json_file in sorted(d.glob("request_*.json")):
                try:
                    requests.append(
                        UserRequest.model_validate_json(json_file.read_text(encoding="utf-8"))
                    )
                except Exception:
                    continue
        return requests

    # ── Flat DataFrame ────────────────────────────────────────────────────────

    def flat_results(
        self,
        run_ids: list[str] | None = None,
        run_names: list[str] | None = None,
        augmentation_ids: list[str] | None = None,
        agent_modes: list[str] | None = None,
        complexity_tiers: list[str] | None = None,
        metric_version: str | None = None,
    ) -> pd.DataFrame:
        """Return all EvalResults as a flat DataFrame with manifest join columns.

        Columns include all deterministic scores (prefixed ``det_``), all LLM judge
        scores (prefixed ``llm_``), and manifest-level columns (``run_id``,
        ``run_name``, ``augmentation_id``, ``prompt_id``, ``episode_source``).

        Filters are applied in-memory after loading. For large stores, prefer
        scoping to specific ``run_ids`` to avoid loading all results files.
        """
        all_manifests = self.list_runs()
        if run_ids is not None:
            run_id_set = set(run_ids)
            all_manifests = [m for m in all_manifests if m.run_id in run_id_set]

        rows: list[dict] = []
        for manifest in all_manifests:
            try:
                _, results = self.load_run(manifest.run_id)
            except Exception:
                continue
            for r in results:
                row: dict = {
                    "run_id": manifest.run_id,
                    "run_name": manifest.run_name,
                    "episode_id": r.episode_id,
                    "request_id": r.request_id,
                    "itinerary_id": r.itinerary_id,
                    "agent_mode": r.agent_mode,
                    "augmentation_id": r.augmentation_id or manifest.augmentation_id,
                    "prompt_id": r.prompt_id or manifest.prompt_id,
                    "episode_source": manifest.episode_source,
                    "metric_version": r.metric_version,
                    "world_seed": r.world_seed,
                    "overall_score": r.overall_score,
                    "judge_model": r.judge_model,
                    "eval_key": r.eval_key,
                    "created_at": r.created_at,
                    "_rubric": r.rubric_breakdown,
                    "_llm_scores": r.llm_judge_scores,
                }
                row.update({f"det_{k}": v for k, v in r.deterministic_scores.items()})
                row.update({f"llm_{k}": v for k, v in r.llm_judge_scores.items()})
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Apply filters
        if run_names is not None:
            df = df[df["run_name"].isin(run_names)]
        if augmentation_ids is not None:
            df = df[df["augmentation_id"].isin(augmentation_ids)]
        if agent_modes is not None:
            df = df[df["agent_mode"].isin(agent_modes)]
        if metric_version is not None:
            df = df[df["metric_version"] == metric_version]
        if complexity_tiers is not None and "complexity_tier" in df.columns:
            df = df[df["complexity_tier"].isin(complexity_tiers)]

        return df.reset_index(drop=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _split_dirs(self) -> list[Path]:
        """Return all immediate sub-directories of requests_dir that contain request files.

        Searches every sub-folder (not just train/val/test) so that any named
        collection (e.g. 'ablation_hard', '20260506_143022') is found automatically.
        Falls back to requests_dir itself if no sub-directories exist.
        """
        if not self.requests_dir.exists():
            return [self.requests_dir]
        subdirs = [
            d for d in sorted(self.requests_dir.iterdir())
            if d.is_dir() and any(d.glob("request_*.json"))
        ]
        return subdirs or [self.requests_dir]
