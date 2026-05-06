"""
evaluation/results_loader.py
=============================
Notebook-facing utilities for loading all evaluation results into a single
flat pandas DataFrame and computing comparison tables.

Intended usage in analysis notebooks::

    from optimized_llm_planning_memory.evaluation.results_loader import (
        load_flat_results,
        list_runs_summary,
        make_comparison_table,
        filter_results,
    )

    df = load_flat_results(EVAL_DIR, EPISODES_DIR, REQUESTS_DIR)
    summary = list_runs_summary(EVAL_DIR)
    table = make_comparison_table(df, group_by="agent_mode")
    subset = filter_results(df, agent_modes=["raw", "compressor"])

DataFrame layout
----------------
The returned DataFrame uses a two-level MultiIndex on columns so that column
groups can be selected with a single key:

    df["ids"]       — episode_id, request_id, run_id, world_seed, itinerary_id
    df["run"]       — run_name, created_at, run_id
    df["config"]    — agent_mode, augmentation_id, prompt_id,
                      judge_model, metric_version, compressor_type
    df["episode"]   — total_steps, total_tokens_used, success, termination_reason
    df["request"]   — world_id, complexity_tier, archetype, n_hard_constraints,
                      n_soft_constraints, n_destinations, duration_days, budget_usd,
                      group_type, geographic_complexity
    df["metrics"]   — overall_score, eval_key (header metrics)
    df["det"]       — one column per deterministic_scores key
    df["llm"]       — one column per llm_judge_scores key (empty when det-only run)

Example selection::

    df["det"]["hard_constraint_ratio"]
    df[["det", "llm"]]          # all metric columns
    df.xs("raw", level=1, axis=1)  # all columns where level-1 == "raw" (won't work — use filter_results)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from optimized_llm_planning_memory.utils.episode_io import list_eval_runs, load_eval_run


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_episodes_index(episodes_dir: Path) -> dict[str, dict]:
    """Return {episode_id: {...flat episode fields...}} without loading trajectories."""
    index: dict[str, dict] = {}
    for ep_path in episodes_dir.glob("ep_*.json"):
        try:
            raw = json.loads(ep_path.read_text(encoding="utf-8"))
            ep_id = raw.get("episode_id", ep_path.stem.removeprefix("ep_"))
            index[ep_id] = {
                "total_steps": raw.get("total_steps"),
                "total_tokens_used": raw.get("total_tokens_used"),
                "success": raw.get("success"),
                "termination_reason": raw.get("termination_reason"),
                "world_seed": raw.get("world_seed"),
                # Pull request metadata embedded in episode (may be None for old episodes)
                "_request": raw.get("user_request"),
            }
        except Exception:
            continue
    return index


def _extract_request_fields(req: dict | None) -> dict:
    """Flatten UserRequest dict into request-group columns."""
    if req is None:
        return {}
    meta = req.get("metadata", {})
    cb = meta.get("complexity_breakdown", {})
    profile = req.get("traveler_profile", {})
    try:
        from datetime import date
        start = date.fromisoformat(req.get("start_date", ""))
        end = date.fromisoformat(req.get("end_date", ""))
        duration_days = (end - start).days
    except Exception:
        duration_days = None
    group_type = "solo"
    n_adults = profile.get("num_adults", 1)
    n_children = profile.get("num_children", 0)
    if n_children > 0:
        group_type = "family"
    elif n_adults == 2:
        group_type = "couple"
    elif n_adults > 2:
        group_type = "group"
    return {
        "world_id": req.get("world_id") or meta.get("world_id"),
        "complexity_tier": cb.get("complexity_tier"),
        "archetype": meta.get("archetype"),
        "n_hard_constraints": len(req.get("hard_constraints", [])),
        "n_soft_constraints": len(req.get("soft_constraints", [])),
        "n_destinations": len(req.get("destination_cities", [])),
        "duration_days": duration_days,
        "budget_usd": req.get("budget_usd"),
        "group_type": group_type,
        "geographic_complexity": cb.get("geographic_complexity"),
    }


def _load_requests_index(requests_dir: Path) -> dict[str, dict]:
    """Return {request_id: {request fields}} by scanning all request JSON files."""
    index: dict[str, dict] = {}
    for req_path in requests_dir.rglob("request_*.json"):
        try:
            raw = json.loads(req_path.read_text(encoding="utf-8"))
            req_id = raw.get("request_id")
            if req_id:
                index[req_id] = _extract_request_fields(raw)
        except Exception:
            continue
    return index


# ── Public API ────────────────────────────────────────────────────────────────

def list_runs_summary(eval_dir: str | Path) -> pd.DataFrame:
    """Return a DataFrame of all eval run manifests, newest first.

    Columns: run_id, run_name, created_at, status, agent_mode, augmentation_id,
             prompt_id, compressor_type, judge_model_id, metric_version,
             n_episodes, n_completed, deterministic_only, notes.
    """
    rows = []
    for m in list_eval_runs(Path(eval_dir)):
        rows.append({
            "run_id": m.run_id,
            "run_name": m.run_name,
            "created_at": m.created_at[:19].replace("T", " "),
            "status": m.status,
            "agent_mode": m.agent_mode,
            "augmentation_id": m.augmentation_id,
            "prompt_id": m.prompt_id,
            "compressor_type": m.compressor_type,
            "judge_model": m.judge_model_id,
            "metric_version": m.metric_version,
            "n_episodes": m.n_episodes,
            "n_completed": m.n_completed,
            "det_only": m.deterministic_only,
            "notes": m.notes or "",
        })
    return pd.DataFrame(rows)


def load_flat_results(
    eval_dir: str | Path,
    episodes_dir: str | Path,
    requests_dir: str | Path,
    run_ids: list[str] | None = None,
    latest_per_eval_key: bool = False,
) -> pd.DataFrame:
    """Load all evaluation results into a flat DataFrame with two-level columns.

    Parameters
    ----------
    eval_dir          : Path to outputs/eval_results/
    episodes_dir      : Path to outputs/episodes/
    requests_dir      : Path to data/user_requests/
    run_ids           : If set, only load results from these run IDs.
    latest_per_eval_key : When True, keep only the most recent result for each
                        (request_id, world_seed, agent_mode, metric_version) tuple,
                        deduplicating re-runs.

    Returns
    -------
    pd.DataFrame with MultiIndex columns (group, field).
    """
    eval_dir = Path(eval_dir)
    episodes_dir = Path(episodes_dir)
    requests_dir = Path(requests_dir)

    manifests = list_eval_runs(eval_dir)
    if run_ids is not None:
        run_ids_set = set(run_ids)
        manifests = [m for m in manifests if m.run_id in run_ids_set]

    # Build lookup indices (loaded once, reused across all runs)
    ep_index = _load_episodes_index(episodes_dir)
    req_index = _load_requests_index(requests_dir)

    rows: list[dict] = []
    for manifest in manifests:
        try:
            _, results = load_eval_run(manifest.run_id, eval_dir)
        except Exception:
            continue
        for r in results:
            ep = ep_index.get(r.episode_id, {})
            # Request: prefer embedded in episode, fall back to index
            req_fields = _extract_request_fields(ep.get("_request")) or req_index.get(r.request_id, {})

            row: dict = {}
            # ids
            row[("ids", "episode_id")]   = r.episode_id
            row[("ids", "request_id")]   = r.request_id
            row[("ids", "run_id")]       = manifest.run_id
            row[("ids", "world_seed")]   = r.world_seed or ep.get("world_seed")
            row[("ids", "itinerary_id")] = r.itinerary_id
            row[("ids", "eval_key")]     = r.eval_key
            # run
            row[("run", "run_name")]   = manifest.run_name
            row[("run", "created_at")] = manifest.created_at[:19].replace("T", " ")
            # config
            row[("config", "agent_mode")]      = r.agent_mode
            row[("config", "augmentation_id")] = r.augmentation_id or manifest.augmentation_id
            row[("config", "prompt_id")]       = r.prompt_id or manifest.prompt_id
            row[("config", "judge_model")]     = r.judge_model
            row[("config", "metric_version")]  = r.metric_version
            row[("config", "compressor_type")] = manifest.compressor_type
            # episode
            row[("episode", "total_steps")]        = ep.get("total_steps")
            row[("episode", "total_tokens_used")]  = ep.get("total_tokens_used")
            row[("episode", "success")]            = ep.get("success")
            row[("episode", "termination_reason")] = ep.get("termination_reason")
            # request
            for field, val in req_fields.items():
                row[("request", field)] = val
            # header metrics
            row[("metrics", "overall_score")] = r.overall_score
            row[("metrics", "has_llm_scores")] = bool(r.llm_judge_scores)
            # deterministic scores
            for k, v in r.deterministic_scores.items():
                row[("det", k)] = v
            # llm judge scores
            for k, v in r.llm_judge_scores.items():
                row[("llm", k)] = v

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.sort_index(axis=1)  # sort column groups alphabetically for consistent display

    if latest_per_eval_key:
        df = (
            df.sort_values(("run", "created_at"), ascending=False)
            .drop_duplicates(subset=[("ids", "eval_key")])
            .reset_index(drop=True)
        )

    return df


def filter_results(
    df: pd.DataFrame,
    agent_modes: list[str] | None = None,
    augmentation_ids: list[str] | None = None,
    prompt_ids: list[str] | None = None,
    run_ids: list[str] | None = None,
    complexity_tiers: list[str] | None = None,
    metric_version: str | None = None,
    has_llm_scores: bool | None = None,
    latest_per_eval_key: bool = False,
) -> pd.DataFrame:
    """Subset the flat results DataFrame by one or more filter criteria.

    All criteria are ANDed together. None means "no filter on this dimension."

    Parameters
    ----------
    latest_per_eval_key : Keep only the most recent result per unique
                          (request_id, world_seed, agent_mode, metric_version).
    """
    mask = pd.Series(True, index=df.index)
    if agent_modes is not None:
        mask &= df[("config", "agent_mode")].isin(agent_modes)
    if augmentation_ids is not None:
        mask &= df[("config", "augmentation_id")].isin(augmentation_ids)
    if prompt_ids is not None:
        mask &= df[("config", "prompt_id")].isin(prompt_ids)
    if run_ids is not None:
        mask &= df[("ids", "run_id")].isin(run_ids)
    if complexity_tiers is not None:
        mask &= df[("request", "complexity_tier")].isin(complexity_tiers)
    if metric_version is not None:
        mask &= df[("config", "metric_version")] == metric_version
    if has_llm_scores is not None:
        mask &= df[("metrics", "has_llm_scores")] == has_llm_scores

    out = df[mask].copy()
    if latest_per_eval_key:
        out = (
            out.sort_values(("run", "created_at"), ascending=False)
            .drop_duplicates(subset=[("ids", "eval_key")])
            .reset_index(drop=True)
        )
    return out


def make_comparison_table(
    df: pd.DataFrame,
    group_by: str | list[str] = "agent_mode",
    metrics: list[str] | None = None,
    metric_group: str = "det",
    include_overall: bool = True,
    stat: str = "mean±std",
) -> pd.DataFrame:
    """Pivot the flat results DataFrame into a metric comparison table.

    Parameters
    ----------
    group_by     : Column name(s) within the "config" group to group by.
                   E.g. "agent_mode", "augmentation_id", or ["agent_mode", "augmentation_id"].
    metrics      : Metric column names within ``metric_group``. None = all columns in group.
    metric_group : "det" or "llm" — which metric group to aggregate.
    include_overall : Prepend overall_score from the "metrics" group.
    stat         : "mean±std" returns "X.XXX ± Y.YYY" strings.
                   "mean" returns float means only.
                   "full" returns a multi-level column with mean/std/min/max.

    Returns
    -------
    DataFrame indexed by ``group_by`` values, columns = selected metrics.
    """
    if isinstance(group_by, str):
        group_by = [group_by]

    group_cols = [("config", g) for g in group_by]
    missing = [g for g in group_cols if g not in df.columns]
    if missing:
        raise ValueError(f"group_by columns not found: {missing}")

    # Build metric columns list
    metric_cols: list[tuple[str, str]] = []
    if include_overall and ("metrics", "overall_score") in df.columns:
        metric_cols.append(("metrics", "overall_score"))
    if metric_group in df.columns.get_level_values(0):
        avail = df[metric_group].columns.tolist()
        selected = metrics if metrics is not None else avail
        metric_cols.extend([(metric_group, m) for m in selected if m in avail])

    if not metric_cols:
        raise ValueError(f"No metric columns found in group '{metric_group}'.")

    grouped = df[group_cols + metric_cols].groupby(group_cols)

    if stat == "mean":
        result = grouped[metric_cols].mean()
        result.index = result.index.droplevel([0] * (len(group_cols) - 1)) if len(group_by) == 1 else result.index
        result.index.name = " × ".join(group_by)
        return result

    if stat == "full":
        agg = grouped[metric_cols].agg(["mean", "std", "min", "max"])
        return agg

    # "mean±std" — formatted strings
    means = grouped[metric_cols].mean()
    stds  = grouped[metric_cols].std()
    result = means.copy().astype(object)
    for col in metric_cols:
        result[col] = [
            f"{m:.3f} ± {s:.3f}" if not pd.isna(m) else "—"
            for m, s in zip(means[col], stds[col])
        ]
    result.index = pd.Index(
        [" / ".join(str(v) for v in (idx if isinstance(idx, tuple) else (idx,)))
         for idx in result.index],
        name=" × ".join(group_by),
    )
    result.columns = pd.Index([c[1] for c in result.columns], name=metric_group)
    return result
