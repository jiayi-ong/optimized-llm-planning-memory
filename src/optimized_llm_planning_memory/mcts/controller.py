"""
mcts/controller.py
==================
MCTSController — orchestrates one MCTS search from a live ReAct trajectory.

Design
------
``MCTSController.search()`` implements the standard four-phase MCTS loop:

    for i in range(num_simulations):
        leaf    = tree.select()
        if leaf.visit_count > 0 and not leaf.is_terminal:
            actions   = _sample_candidate_actions(leaf, request)
            traj_list = _simulate_actions(leaf, actions)
            children  = tree.expand(leaf, actions, traj_list)
            leaf      = children[0]
        value  = tree.simulate(leaf, evaluator)
        tree.backpropagate(leaf, value)
    return tree.to_representation(num_simulations)

``MCTSController`` is intentionally stateless between calls: a fresh
``MCTSTree`` is built at the start of each ``search()`` invocation. This
means the controller can be safely shared across parallel ``CompressionEnv``
instances — each call creates its own tree and evaluator cache.

Action simulation
-----------------
In a true MCTS, expanding a node requires actually executing the candidate
action against a simulator to produce a new world state. Here the "world
state" is the ReAct trajectory: we append the candidate action as a synthetic
ReActStep (without executing a real tool call) and let the NodeEvaluator
score the resulting trajectory structure.

This is a pragmatic approximation: the LLM-generated action text is treated
as a "what would happen if" signal, and the heuristic evaluator scores the
plausibility of the resulting trajectory. Full rollout (actually calling tools)
is only triggered when ``MCTSConfig.rollout_steps > 0`` AND the evaluator's
LLM tier is enabled.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import litellm

from optimized_llm_planning_memory.mcts.node import MCTSNode, MCTSTreeRepresentation
from optimized_llm_planning_memory.mcts.tree import MCTSTree
from optimized_llm_planning_memory.core.exceptions import MCTSSearchTimeoutError
from optimized_llm_planning_memory.core.models import ReActStep, ToolCall, TrajectoryModel

if TYPE_CHECKING:
    from optimized_llm_planning_memory.core.models import CompressedState, UserRequest
    from optimized_llm_planning_memory.mcts.config import MCTSConfig
    from optimized_llm_planning_memory.mcts.node_evaluator import NodeEvaluator


class MCTSController:
    """
    Orchestrates MCTS search from a live ReAct trajectory.

    Called by ``ReActAgent._run_compression()`` when mode is MCTS_COMPRESSOR.
    Returns an ``MCTSTreeRepresentation`` which is then passed to
    ``MCTSAwareCompressor.compress_with_tree()``.

    Parameters
    ----------
    evaluator    : NodeEvaluator instance (stateful; holds evaluation cache).
    llm_model_id : litellm model for generating candidate branch actions.
    config       : MCTSConfig controlling search depth, breadth, etc.
    """

    def __init__(
        self,
        evaluator: "NodeEvaluator",
        llm_model_id: str,
        config: "MCTSConfig",
    ) -> None:
        self._evaluator = evaluator
        self._llm_model_id = llm_model_id
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        trajectory: TrajectoryModel,
        compressed_state: "CompressedState | None",
        request: "UserRequest",
    ) -> MCTSTreeRepresentation:
        """
        Run the full MCTS search from the current trajectory state.

        Parameters
        ----------
        trajectory       : Current frozen trajectory (from Trajectory.to_model()).
        compressed_state : Active CompressedState (None if no prior compression).
        request          : UserRequest for context in action generation and scoring.

        Returns
        -------
        MCTSTreeRepresentation
            Immutable summary of the search tree for the compressor.
        """
        # Update evaluator with current request
        self._evaluator.set_request(request)

        # Build a fresh tree for this search
        tree = MCTSTree(config=self._config)
        tree.build_root(trajectory, compressed_state)

        simulations_run = 0
        deadline = time.monotonic() + self._config.timeout_seconds
        try:
            for _ in range(self._config.num_simulations):
                if time.monotonic() > deadline:
                    raise MCTSSearchTimeoutError(
                        f"MCTS search exceeded {self._config.timeout_seconds}s wall-clock budget "
                        f"after {simulations_run} simulations."
                    )
                leaf = tree.select()

                # Expand unvisited leaf that is not terminal
                if leaf.visit_count == 0 or (not leaf.is_terminal and not leaf.children):
                    if not leaf.is_terminal:
                        actions = self._sample_candidate_actions(leaf, request)
                        child_trajs = self._build_child_trajectories(leaf, actions)
                        children = tree.expand(leaf, actions, child_trajs)
                        if children:
                            leaf = children[0]

                value = tree.simulate(leaf, self._evaluator)
                tree.backpropagate(leaf, value)
                simulations_run += 1

        except MCTSSearchTimeoutError:
            # Timeout is non-fatal — return best result so far
            pass

        return tree.to_representation(simulations_run)

    # ── Action generation ─────────────────────────────────────────────────────

    def _sample_candidate_actions(
        self,
        node: MCTSNode,
        request: "UserRequest",
    ) -> list[str]:
        """
        Call the planning LLM with temperature > 0 to generate
        ``branching_factor`` diverse candidate action texts.

        Returns a list of raw action strings in the format
        ``tool_name({"key": "value"})``.
        On any failure, returns a single placeholder action.
        """
        context = _build_branching_prompt(node, request)
        try:
            response = litellm.completion(
                model=self._llm_model_id,
                messages=[{"role": "user", "content": context}],
                temperature=self._config.temperature,
                max_tokens=256 * self._config.branching_factor,
                n=self._config.branching_factor,
            )
            actions = []
            for choice in response.choices:
                text = (choice.message.content or "").strip()
                # Extract only the Action: line
                for line in text.splitlines():
                    if line.strip().startswith("Action:"):
                        action_text = line.replace("Action:", "").strip()
                        if action_text and action_text.upper() != "DONE":
                            actions.append(action_text)
                            break
            return actions if actions else [_placeholder_action(node)]
        except Exception:
            return [_placeholder_action(node)]

    # ── Trajectory simulation ─────────────────────────────────────────────────

    def _build_child_trajectories(
        self,
        parent: MCTSNode,
        actions: list[str],
    ) -> list[TrajectoryModel]:
        """
        Build a synthetic child TrajectoryModel for each candidate action.

        This is a lightweight approximation: we append each action as a
        synthetic ReActStep (no real tool execution) to the parent's trajectory.
        The NodeEvaluator then scores the resulting structure.

        For a more faithful simulation (real tool calls), subclass this
        controller and override this method.
        """
        parent_traj = parent.trajectory_snapshot
        child_trajs: list[TrajectoryModel] = []

        for action_text in actions:
            new_step = _make_synthetic_step(
                step_index=parent_traj.total_steps,
                action_text=action_text,
            )
            # Append synthetic step to produce child trajectory
            new_steps = parent_traj.steps + (new_step,)
            child_traj = TrajectoryModel(
                trajectory_id=parent_traj.trajectory_id,
                request_id=parent_traj.request_id,
                steps=new_steps,
                total_steps=parent_traj.total_steps + 1,
            )
            child_trajs.append(child_traj)

        return child_trajs


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_branching_prompt(node: MCTSNode, request: "UserRequest") -> str:
    """Build the prompt used to generate candidate branch actions."""
    request_text = getattr(request, "raw_text", str(request))
    traj_text = node.trajectory_snapshot.to_text()
    last_steps = "\n".join(traj_text.splitlines()[-30:])  # last ~30 lines

    return (
        f"[USER REQUEST]\n{request_text}\n\n"
        f"[RECENT PLANNING STEPS]\n{last_steps}\n\n"
        "Based on the planning progress above, what is the single most important "
        "next tool call? Respond with ONLY:\n"
        "Thought: <one sentence>\n"
        "Action: tool_name({\"key\": \"value\"})\n"
        "Be specific about which tool to call and what arguments to use."
    )


def _make_synthetic_step(step_index: int, action_text: str) -> ReActStep:
    """Create a synthetic ReActStep representing a candidate action (no observation)."""
    tool_call: ToolCall | None = None
    import re
    call_match = re.match(r"(\w+)\((.+)\)$", action_text, re.DOTALL)
    if call_match:
        tool_name = call_match.group(1)
        args_str = call_match.group(2)
        try:
            arguments = json.loads(args_str)
        except (json.JSONDecodeError, ValueError):
            arguments = {"_raw": args_str}
        tool_call = ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            raw_text=action_text,
        )

    return ReActStep(
        step_index=step_index,
        thought="[MCTS simulated step]",
        action=tool_call,
        observation=None,
        itinerary_snapshot=None,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )


def _placeholder_action(node: MCTSNode) -> str:
    """Fallback action when LLM generation fails."""
    return f"search_flights({{\"step\": {node.depth}}})"
