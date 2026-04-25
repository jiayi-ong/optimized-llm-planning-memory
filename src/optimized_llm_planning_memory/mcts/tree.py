"""
mcts/tree.py
============
MCTSTree — mutable search tree implementing SELECT → EXPAND → SIMULATE → BACKPROPAGATE.

Design notes
------------
* The four MCTS phases are implemented as separate methods to make each phase
  independently testable. ``MCTSController.search()`` calls them in the standard
  MCTS loop.

* ``select()`` uses UCB1 to traverse the tree to the most promising leaf.
  A leaf is any node with no children OR a node that has been visited but not
  yet expanded (visit_count > 0 and no children).

* ``expand()`` creates child nodes but does NOT execute tool calls. The caller
  (``MCTSController._simulate_action()``) is responsible for advancing the
  trajectory by applying the action to a simulator clone.

* ``backpropagate()`` walks from the expanded/simulated node back to the root
  via parent_id lookups, updating visit_count and value_sum on every node.

* ``to_representation()`` projects the completed tree into the immutable
  ``MCTSTreeRepresentation`` consumed by the compressor.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from optimized_llm_planning_memory.mcts.node import MCTSNode, MCTSStats, MCTSTreeRepresentation

if TYPE_CHECKING:
    from optimized_llm_planning_memory.core.models import CompressedState, TrajectoryModel
    from optimized_llm_planning_memory.mcts.config import MCTSConfig
    from optimized_llm_planning_memory.mcts.node_evaluator import NodeEvaluator


class MCTSTree:
    """
    Mutable MCTS search tree.

    Manages the full node graph and implements all four MCTS phases.
    One ``MCTSTree`` is created per ``MCTSController.search()`` call —
    it is NOT reused across calls.

    Parameters
    ----------
    config : MCTSConfig governing depth, branching factor, and UCB constant.
    """

    def __init__(self, config: "MCTSConfig") -> None:
        self._config = config
        self._nodes: dict[str, MCTSNode] = {}
        self._root_id: str | None = None

    # ── Phase 0: Initialisation ───────────────────────────────────────────────

    def build_root(
        self,
        trajectory: "TrajectoryModel",
        compressed_state: "CompressedState | None",
    ) -> MCTSNode:
        """
        Create the root node from the current live trajectory.

        Must be called before any other method.
        """
        root = MCTSNode.make_root(trajectory, compressed_state)
        self._nodes[root.node_id] = root
        self._root_id = root.node_id
        return root

    # ── Phase 1: Selection ────────────────────────────────────────────────────

    def select(self) -> MCTSNode:
        """
        Traverse from root to the most promising leaf using UCB1.

        A node is considered a leaf if:
        - It has no children (never expanded), OR
        - It is marked as terminal.

        Returns the selected leaf node.
        """
        assert self._root_id is not None, "Call build_root() before select()."
        node = self._nodes[self._root_id]

        while node.children and not node.is_terminal:
            # UCB1 over children.
            # Approximation: we read parent.visit_count at selection time, which
            # is the count AFTER all previous iterations (already incremented).
            # Standard UCB1 uses the parent count at the moment the child was
            # created, but the difference shrinks as N grows and is negligible
            # for the exploration constant values used here (C ≈ 1.4).
            parent_visits = node.visit_count
            node = max(
                node.children,
                key=lambda c: c.ucb1_score(parent_visits, self._config.exploration_constant),
            )

        return node

    # ── Phase 2: Expansion ────────────────────────────────────────────────────

    def expand(
        self,
        parent: MCTSNode,
        candidate_actions: list[str],
        child_trajectories: list["TrajectoryModel"],
    ) -> list[MCTSNode]:
        """
        Add child nodes to ``parent`` for each candidate action.

        Parameters
        ----------
        parent              : The selected leaf node to expand.
        candidate_actions   : LLM-sampled action text strings (one per branch).
        child_trajectories  : Resulting TrajectoryModel for each action
                              (after applying the action to a simulator clone).
                              Must have the same length as ``candidate_actions``.

        Returns
        -------
        list[MCTSNode]
            Newly created child nodes in the same order as ``candidate_actions``.
        """
        assert len(candidate_actions) == len(child_trajectories), (
            "candidate_actions and child_trajectories must have the same length."
        )

        children: list[MCTSNode] = []
        for action_text, trajectory in zip(candidate_actions, child_trajectories):
            is_terminal = (
                parent.depth + 1 >= self._config.max_depth
                or trajectory.total_steps == 0  # degenerate: action produced no step
            )
            child = MCTSNode(
                node_id=str(uuid.uuid4()),
                parent_id=parent.node_id,
                depth=parent.depth + 1,
                trajectory_snapshot=trajectory,
                compressed_state_snapshot=parent.compressed_state_snapshot,
                prior_action_text=action_text,
                is_terminal=is_terminal,
            )
            self._nodes[child.node_id] = child
            parent.children.append(child)
            children.append(child)

        return children

    # ── Phase 3: Simulation (scoring) ─────────────────────────────────────────

    def simulate(self, node: MCTSNode, evaluator: "NodeEvaluator") -> float:
        """
        Estimate the value of ``node`` using the provided evaluator.

        The evaluator combines fast heuristics with an optional LLM tiebreaker.
        Returns a float in [0.0, 1.0].
        """
        # Terminal nodes are scored directly without simulation steps.
        return evaluator.evaluate(node)

    # ── Phase 4: Backpropagation ──────────────────────────────────────────────

    def backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Walk from ``node`` to the root, incrementing visit_count and adding
        ``value`` to value_sum at every node on the path.
        """
        current: MCTSNode | None = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            current = self._nodes.get(current.parent_id) if current.parent_id else None

    # ── Tree queries ──────────────────────────────────────────────────────────

    @property
    def root(self) -> MCTSNode:
        assert self._root_id is not None, "Call build_root() first."
        return self._nodes[self._root_id]

    def best_child(self, node: MCTSNode) -> MCTSNode | None:
        """Return the child with the highest Q-value (greedy, no exploration bonus)."""
        if not node.children:
            return None
        return max(node.children, key=lambda c: c.q_value)

    def best_path(self) -> list[MCTSNode]:
        """
        Return the greedy-best path from root to the deepest visited leaf.
        Each step selects the child with the highest Q-value.
        """
        path = [self.root]
        while path[-1].children:
            nxt = self.best_child(path[-1])
            if nxt is None:
                break
            path.append(nxt)
        return path

    def top_k_children(self, k: int = 3) -> list[MCTSNode]:
        """
        Return the top-K root children by Q-value for building the
        alternative_paths list in MCTSTreeRepresentation.
        """
        children = sorted(self.root.children, key=lambda c: c.q_value, reverse=True)
        return children[:k]

    # ── Statistics ────────────────────────────────────────────────────────────

    def collect_stats(self, num_simulations: int) -> MCTSStats:
        """Traverse the tree and compute a serialisable MCTSStats summary."""
        if not self._nodes:
            return MCTSStats(
                nodes_explored=0,
                max_depth_reached=0,
                num_simulations=0,
                best_path_length=0,
                root_value=0.0,
                avg_branching_factor=0.0,
            )

        all_nodes = list(self._nodes.values())
        max_depth = max(n.depth for n in all_nodes)
        expanded = [n for n in all_nodes if n.children]
        avg_branch = (
            sum(len(n.children) for n in expanded) / len(expanded)
            if expanded else 0.0
        )

        return MCTSStats(
            nodes_explored=len(all_nodes),
            max_depth_reached=max_depth,
            num_simulations=num_simulations,
            best_path_length=len(self.best_path()),
            root_value=self.root.q_value,
            avg_branching_factor=round(avg_branch, 2),
        )

    # ── Export ────────────────────────────────────────────────────────────────

    def to_representation(self, num_simulations: int) -> MCTSTreeRepresentation:
        """
        Convert the completed tree into an immutable ``MCTSTreeRepresentation``
        for the compressor.

        The best-path trajectory is used as the "primary" trajectory for the
        compressor's standard 6 template sections. Alternative root-children
        trajectories are passed as ``alternative_paths`` for multi-hypothesis
        context.
        """
        path = self.best_path()
        best_traj = path[-1].trajectory_snapshot if path else self.root.trajectory_snapshot

        # Collect top-K alternative branches (excluding the best-path root child)
        best_child = self.best_child(self.root)
        alt_nodes = [
            c for c in self.top_k_children(k=self._config.branching_factor)
            if best_child is None or c.node_id != best_child.node_id
        ]
        alternative_paths = [n.trajectory_snapshot for n in alt_nodes]

        # Build human-readable candidate descriptions
        top_candidates = _describe_candidates(path, alt_nodes)

        # Build tradeoffs summary from Q-value gaps
        tradeoffs = _describe_tradeoffs(self.root, best_child)

        # Collect node values for reference
        node_values = {nid: n.q_value for nid, n in self._nodes.items()}

        stats = self.collect_stats(num_simulations)

        return MCTSTreeRepresentation(
            best_path_trajectory=best_traj,
            alternative_paths=alternative_paths,
            node_values=node_values,
            top_candidates=top_candidates,
            tradeoffs=tradeoffs,
            stats=stats,
        )


# ── Private helpers ───────────────────────────────────────────────────────────

def _describe_candidates(
    best_path: list[MCTSNode],
    alt_nodes: list[MCTSNode],
) -> list[str]:
    """
    Build short human-readable descriptions for the best path and alternatives.
    Used as the ``top_candidates`` list in MCTSTreeRepresentation.
    """
    candidates: list[str] = []

    # Best path candidate
    if best_path and len(best_path) > 1:
        first_action = best_path[1].prior_action_text
        q = best_path[-1].q_value
        candidates.append(
            f"[Best] {first_action[:120]} (Q={q:.3f}, depth={len(best_path)-1})"
        )

    # Alternative candidates
    for node in alt_nodes:
        q = node.q_value
        candidates.append(
            f"[Alt]  {node.prior_action_text[:120]} (Q={q:.3f})"
        )

    return candidates


def _describe_tradeoffs(root: MCTSNode, best_child: MCTSNode | None) -> str:
    """
    Produce a one-paragraph tradeoff summary from root children Q-values.
    """
    if not root.children:
        return "No branches explored yet."

    sorted_children = sorted(root.children, key=lambda c: c.q_value, reverse=True)

    lines = [
        f"Top branch Q-values after {root.visit_count} simulations:"
    ]
    for i, child in enumerate(sorted_children[:4]):
        marker = " (selected)" if best_child and child.node_id == best_child.node_id else ""
        lines.append(
            f"  {i+1}. Q={child.q_value:.3f}, visits={child.visit_count}"
            f" | {child.prior_action_text[:80]}{marker}"
        )

    if len(sorted_children) >= 2:
        gap = sorted_children[0].q_value - sorted_children[1].q_value
        lines.append(f"Q-gap between top two branches: {gap:.3f}.")

    return "\n".join(lines)
