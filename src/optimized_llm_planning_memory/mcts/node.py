"""
mcts/node.py
============
Core MCTS data structures: MCTSNode, MCTSStats, MCTSTreeRepresentation.

Design notes
------------
* ``MCTSNode`` is a plain Python ``dataclass`` (NOT a Pydantic model and NOT
  frozen) because it is mutated in-place during the search loop — visit counts
  and value sums are incremented on every backpropagation pass.

* ``MCTSStats`` and ``MCTSTreeRepresentation`` are frozen Pydantic models because
  they are outputs consumed by downstream components (compressor, EpisodeLog)
  that expect immutable, serialisable data contracts.

* The separation between the mutable search structure (``MCTSNode``) and the
  immutable summary (``MCTSStats``, ``MCTSTreeRepresentation``) mirrors the
  separation between ``Trajectory`` (mutable) and ``TrajectoryModel`` (frozen)
  in the agent layer.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from optimized_llm_planning_memory.core.models import CompressedState, TrajectoryModel


# ── Mutable search node ───────────────────────────────────────────────────────

@dataclass
class MCTSNode:
    """
    A single node in the MCTS search tree.

    Mutated during search (visit_count, value_sum, children).
    NOT serialisable directly — convert to MCTSStats / MCTSTreeRepresentation
    for logging.

    Parameters
    ----------
    node_id                  : UUID string uniquely identifying this node.
    parent_id                : ``node_id`` of the parent, or None for the root.
    depth                    : Distance from root (root has depth 0).
    trajectory_snapshot      : Frozen TrajectoryModel at this node's state.
    compressed_state_snapshot: CompressedState that was active when this node
                               was created (may be None at root if no prior
                               compression has occurred).
    prior_action_text        : The LLM-generated action string that produced
                               this node from its parent (empty for root).
    is_terminal              : True when the node represents a DONE signal or
                               max_depth has been reached.
    children                 : Child nodes added by MCTSTree.expand().
    visit_count              : Number of times this node has been visited.
    value_sum                : Cumulative value received via backpropagation.
    """

    node_id: str
    parent_id: str | None
    depth: int
    trajectory_snapshot: "TrajectoryModel"
    compressed_state_snapshot: "CompressedState | None"
    prior_action_text: str = ""
    is_terminal: bool = False
    children: list["MCTSNode"] = field(default_factory=list)
    visit_count: int = 0
    value_sum: float = 0.0

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def q_value(self) -> float:
        """Mean value across all visits. Zero for unvisited nodes."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def ucb1_score(self, parent_visits: int, c_p: float) -> float:
        """
        UCB1 = Q(s,a) + C_p * sqrt(ln(N_parent) / N(s,a))

        Unvisited nodes return ``+inf`` so they are always selected first.
        """
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.q_value
        exploration = c_p * math.sqrt(math.log(max(parent_visits, 1)) / self.visit_count)
        return exploitation + exploration

    @classmethod
    def make_root(
        cls,
        trajectory: "TrajectoryModel",
        compressed_state: "CompressedState | None",
    ) -> "MCTSNode":
        """Convenience constructor for the root node."""
        return cls(
            node_id=str(uuid.uuid4()),
            parent_id=None,
            depth=0,
            trajectory_snapshot=trajectory,
            compressed_state_snapshot=compressed_state,
        )


# ── Immutable summary models ──────────────────────────────────────────────────

class MCTSStats(BaseModel):
    """
    Immutable summary of a completed MCTS search run.

    Embedded in ``EpisodeLog.mcts_stats`` (None for non-MCTS episodes).
    Serialisable to JSON without losing information.

    Fields
    ------
    nodes_explored        : Total nodes created (root + all expansions).
    max_depth_reached     : Deepest node depth reached during search.
    num_simulations       : Iterations actually executed (≤ config.num_simulations
                            if an early-termination condition fired).
    best_path_length      : Length of the greedy-best path from root.
    root_value            : Q-value of the root node after search.
    avg_branching_factor  : Mean number of children per expanded node.
    """
    model_config = ConfigDict(frozen=True)

    nodes_explored: int = Field(ge=0)
    max_depth_reached: int = Field(ge=0)
    num_simulations: int = Field(ge=0)
    best_path_length: int = Field(ge=0)
    root_value: float
    avg_branching_factor: float = Field(ge=0.0)


class MCTSTreeRepresentation(BaseModel):
    """
    Compact, serialisable representation of the MCTS tree handed to
    ``MCTSAwareCompressor.compress_with_tree()``.

    Design
    ------
    Rather than passing the full mutable MCTSTree (which contains raw
    TrajectoryModel objects and mutable node state), we project the tree into
    this contract that carries exactly the information the compressor needs:

    * The greedy-best path trajectory (for filling the 6 standard template
      sections as if it were a linear compression).
    * Up to ``branching_factor`` alternative branch trajectories (for the
      TOP_CANDIDATES section).
    * Pre-computed human-readable candidate descriptions (for context injection).
    * A free-form tradeoffs summary (derived from value differences).
    * The search statistics (for EpisodeLog).

    Fields
    ------
    best_path_trajectory  : Trajectory along the highest-Q path root→leaf.
    alternative_paths     : Other top-K branches (by Q-value) for context.
    node_values           : node_id → Q-value mapping for all visited nodes.
    top_candidates        : Human-readable description of each alternative.
                            Length == len(alternative_paths).
    tradeoffs             : Free-form paragraph describing value gaps between
                            the top branches.
    stats                 : MCTSStats for embedding in EpisodeLog.
    """
    model_config = ConfigDict(frozen=True)

    best_path_trajectory: "TrajectoryModel"
    alternative_paths: list["TrajectoryModel"] = Field(default_factory=list)
    node_values: dict[str, float] = Field(default_factory=dict)
    top_candidates: list[str] = Field(default_factory=list)
    tradeoffs: str = ""
    stats: MCTSStats
