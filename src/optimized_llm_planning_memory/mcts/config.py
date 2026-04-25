"""
mcts/config.py
==============
MCTSConfig — hyperparameters that govern the Monte Carlo Tree Search.

Design notes
------------
* Kept separate from ``core/config.py`` to isolate MCTS concerns; ``AgentConfig``
  holds an optional ``mcts: MCTSSearchConfig`` field defined directly in
  ``core/config.py`` to avoid circular imports.
* All fields have safe defaults so the object can be instantiated without any
  arguments during unit testing or notebook prototyping.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MCTSConfig(BaseModel):
    """
    Configuration for one MCTS search run.

    Fields
    ------
    num_simulations       : Total SELECT→EXPAND→SIMULATE→BACKPROPAGATE iterations.
    max_depth             : Maximum depth from root before a node is treated as terminal.
    exploration_constant  : UCB1 C_p term. ``sqrt(2) ≈ 1.414`` is the theoretical
                            optimum; lower values favour exploitation.
    branching_factor      : Number of candidate action texts sampled from the LLM
                            at each expansion (temperature > 0 sampling).
    rollout_steps         : ReAct steps executed during simulation phase before
                            the NodeEvaluator scores the node.
    evaluator_model_id    : litellm model string used for LLM-based node scoring.
                            Only called when heuristic score is ambiguous.
    use_cached_evaluations: When True, cache NodeEvaluator results keyed by
                            trajectory hash to avoid redundant LLM calls.
    temperature           : Sampling temperature for candidate action generation.
                            Must be > 0 to produce diverse branches.
    """

    num_simulations: int = Field(default=50, ge=1,
                                 description="Total MCTS iterations per search call.")
    max_depth: int = Field(default=10, ge=1,
                           description="Max tree depth before forcing terminal evaluation.")
    exploration_constant: float = Field(
        default=1.414, gt=0.0,
        description="UCB1 C_p. sqrt(2) balances exploration/exploitation.",
    )
    branching_factor: int = Field(
        default=3, ge=1,
        description="LLM-sampled candidate actions per expansion.",
    )
    rollout_steps: int = Field(
        default=5, ge=1,
        description="ReAct steps simulated per node before heuristic scoring.",
    )
    evaluator_model_id: str = Field(
        default="openai/gpt-4o-mini",
        description="litellm model for LLM-based node evaluation.",
    )
    use_cached_evaluations: bool = Field(
        default=True,
        description="Cache node scores by trajectory hash to save LLM calls.",
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0,
        description="Sampling temperature for branching action generation.",
    )
    timeout_seconds: float = Field(
        default=30.0, gt=0.0,
        description="Wall-clock budget for one search() call; raises MCTSSearchTimeoutError when exceeded.",
    )
