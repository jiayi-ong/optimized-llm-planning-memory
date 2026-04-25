"""
tests/unit/test_agent/test_compression_trigger_unit.py
=======================================================
T3 — Verify that _should_compress fires at exactly the right step.

The invariant: with compress_every_n_steps=N, compression fires at step N,
2N, 3N, ... (counting from the last compression event), not at N-1 or N+1.
"""

from __future__ import annotations

import pytest

from optimized_llm_planning_memory.agent.modes import AgentMode
from optimized_llm_planning_memory.agent.react_agent import ReActAgent
from optimized_llm_planning_memory.agent.trajectory import Trajectory
from optimized_llm_planning_memory.core.config import AgentConfig


def _make_compressor_agent(n: int = 5) -> ReActAgent:
    from unittest.mock import MagicMock
    from optimized_llm_planning_memory.agent.context_builder import ContextBuilder
    from optimized_llm_planning_memory.agent.prompts import get_system_prompt
    from optimized_llm_planning_memory.tools.events import EventBus
    from optimized_llm_planning_memory.tools.registry import ToolRegistry
    from optimized_llm_planning_memory.tools.tracker import ToolCallTracker

    sim = MagicMock()
    tracker = ToolCallTracker()
    event_bus = EventBus()
    registry = ToolRegistry.from_config(simulator=sim, tracker=tracker, event_bus=event_bus)
    config = AgentConfig(
        mode="compressor",
        llm_model_id="groq/llama3-8b-8192",
        max_steps=50,
        compress_every_n_steps=n,
        temperature=0.0,
    )
    context_builder = ContextBuilder(
        system_prompt=get_system_prompt("v1"),
        tool_registry=registry,
        llm_model_id="groq/llama3-8b-8192",
    )
    compressor = MagicMock()
    return ReActAgent(
        llm_model_id="groq/llama3-8b-8192",
        tool_registry=registry,
        compressor=compressor,
        context_builder=context_builder,
        config=config,
        mode=AgentMode.COMPRESSOR,
    )


@pytest.mark.unit
class TestCompressionTriggerTiming:
    def test_fires_exactly_at_n_steps(self):
        """Compression should trigger at step N but not at step N-1."""
        n = 5
        agent = _make_compressor_agent(n=n)
        traj = Trajectory(request_id="r1")

        assert agent._should_compress(traj, n - 1) is False
        assert agent._should_compress(traj, n) is True

    def test_does_not_fire_before_n(self):
        n = 5
        agent = _make_compressor_agent(n=n)
        traj = Trajectory(request_id="r1")
        for step in range(1, n):
            assert agent._should_compress(traj, step) is False, (
                f"Should not compress at step {step} < {n}"
            )

    def test_fires_again_at_2n_after_reset(self):
        """After marking compression at step N, should fire again at 2N."""
        n = 5
        agent = _make_compressor_agent(n=n)
        traj = Trajectory(request_id="r1")

        traj.mark_compression(at_step=n)
        # N+1 through 2N-1 should not compress
        for step in range(n + 1, 2 * n):
            assert agent._should_compress(traj, step) is False
        # Exactly 2N should compress
        assert agent._should_compress(traj, 2 * n) is True

    def test_step_n_plus_one_does_not_double_fire(self):
        """Step N+1 after compression at step N should not trigger again."""
        n = 5
        agent = _make_compressor_agent(n=n)
        traj = Trajectory(request_id="r1")
        traj.mark_compression(at_step=n)
        assert agent._should_compress(traj, n + 1) is False

    def test_n_equals_one_fires_every_step(self):
        """n=1 means compress after every step."""
        agent = _make_compressor_agent(n=1)
        traj = Trajectory(request_id="r1")
        assert agent._should_compress(traj, 1) is True
        traj.mark_compression(at_step=1)
        assert agent._should_compress(traj, 2) is True
