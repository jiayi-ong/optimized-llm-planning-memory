"""
mcts/
=====
Monte Carlo Tree Search module for MCTS-augmented context compression.

Public API
----------
MCTSConfig            — hyperparameters for one search run
MCTSNode              — mutable search tree node (dataclass)
MCTSStats             — immutable search statistics (Pydantic, frozen)
MCTSTreeRepresentation — immutable tree summary passed to the compressor
MCTSTree              — mutable search tree (select/expand/simulate/backprop)
NodeEvaluator         — heuristic + LLM node scoring
MCTSController        — orchestrates the full MCTS search loop

Usage
-----
    from optimized_llm_planning_memory.mcts import (
        MCTSConfig, MCTSController, MCTSTree, NodeEvaluator
    )

    config = MCTSConfig(num_simulations=20, branching_factor=3)
    evaluator = NodeEvaluator(model_id="openai/gpt-4o-mini", config=config)
    controller = MCTSController(evaluator=evaluator,
                                llm_model_id="openai/gpt-4o-mini",
                                config=config)
    tree_repr = controller.search(trajectory, compressed_state, request)
"""

from optimized_llm_planning_memory.mcts.config import MCTSConfig
from optimized_llm_planning_memory.mcts.controller import MCTSController
from optimized_llm_planning_memory.mcts.node import MCTSNode, MCTSStats, MCTSTreeRepresentation
from optimized_llm_planning_memory.mcts.node_evaluator import NodeEvaluator
from optimized_llm_planning_memory.mcts.tree import MCTSTree

__all__ = [
    "MCTSConfig",
    "MCTSController",
    "MCTSNode",
    "MCTSStats",
    "MCTSTree",
    "MCTSTreeRepresentation",
    "NodeEvaluator",
]
