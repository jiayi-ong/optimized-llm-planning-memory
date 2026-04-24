"""
evaluation/manifest.py
=======================
EvalRunManifest — run-level metadata for a batch evaluation.

Design: one manifest per evaluation run
-----------------------------------------
An "eval run" is a single invocation of Evaluator.evaluate_dataset() for a
fixed (compressor, agent_mode, checkpoint, metric_version) configuration.
The manifest is written alongside the per-episode EvalResult objects so that
any future notebook or analysis script can reconstruct what configuration
produced each set of results.

Why a separate manifest instead of embedding metadata in each EvalResult?
- A manifest is written once; per-episode results are written N times.
- The manifest is the index entry; results are the detail records.
- Keeping them separate allows the index to stay small even for large runs.

Persistence layout
------------------
outputs/eval_results/
    {run_id}/
        manifest.json          — EvalRunManifest (this model)
        results.jsonl          — one EvalResult JSON per line

See utils/episode_io.py for save_eval_run / load_eval_run / list_eval_runs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvalRunManifest(BaseModel):
    """
    Run-level metadata for a batch evaluation.

    Stored alongside EvalResult objects to enable ablation analysis,
    version-aware metric comparison, and checkpoint traceability.

    Fields
    ------
    run_id            : Short unique identifier (e.g. first 8 chars of uuid4).
    created_at        : ISO 8601 UTC timestamp.
    compressor_type   : "identity" | "llm" | "transformer" | "dummy" | "hybrid"
    agent_mode        : "raw" | "llm_summary" | "compressor" | "mcts_compressor"
    judge_model_id    : litellm model string used for LLMJudge, or "none".
    checkpoint_path   : Path to the compressor checkpoint loaded, if any.
    config_hash       : MD5/SHA of the Hydra config for exact reproducibility.
                        Use "manual" when running outside Hydra (e.g., notebook).
    metric_version    : Value of METRIC_VERSION from deterministic.py.
    request_ids       : All request IDs included in this run.
    n_episodes        : Total number of episodes evaluated.
    deterministic_only: True if LLMJudge was skipped.
    world_seeds       : List of random seeds used for world generation.
                        Empty list means the default seed was used.
    mcts_enabled      : True when this run used AgentMode.MCTS_COMPRESSOR.
    mcts_config_hash  : Short hash of the MCTSSearchConfig used. None for non-MCTS runs.
    notes             : Free-form developer notes.
    """
    model_config = ConfigDict(frozen=True)

    run_id: str
    created_at: str = Field(description="ISO 8601 UTC timestamp.")
    compressor_type: str
    agent_mode: str
    judge_model_id: str
    checkpoint_path: str | None = None
    config_hash: str
    metric_version: str
    request_ids: list[str]
    n_episodes: int = Field(ge=0)
    deterministic_only: bool
    world_seeds: list[int] = Field(default_factory=list)
    mcts_enabled: bool = Field(
        default=False,
        description="True when this evaluation run used AgentMode.MCTS_COMPRESSOR.",
    )
    mcts_config_hash: str | None = Field(
        default=None,
        description="Short hash of MCTSSearchConfig. None for non-MCTS runs.",
    )
    notes: str | None = None
