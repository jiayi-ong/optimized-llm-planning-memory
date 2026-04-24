"""System tests — full episode → DeterministicEvaluator → aggregate pipeline."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from optimized_llm_planning_memory.agent.context_builder import ContextBuilder
from optimized_llm_planning_memory.agent.modes import AgentMode
from optimized_llm_planning_memory.agent.prompts import get_system_prompt
from optimized_llm_planning_memory.agent.react_agent import ReActAgent
from optimized_llm_planning_memory.core.config import AgentConfig
from optimized_llm_planning_memory.evaluation.deterministic import DeterministicEvaluator
from optimized_llm_planning_memory.evaluation.evaluator import Evaluator
from optimized_llm_planning_memory.tools.events import EventBus
from optimized_llm_planning_memory.tools.registry import ToolRegistry
from optimized_llm_planning_memory.tools.tracker import ToolCallTracker
from optimized_llm_planning_memory.utils.episode_io import save_episode, load_episode


def _make_llm_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


DONE_RESPONSE = _make_llm_response(
    "Thought: Done. Action: DONE\n{}"
)


def _build_agent(mock_sim, mode: AgentMode = AgentMode.RAW, max_steps: int = 3) -> ReActAgent:
    tracker = ToolCallTracker()
    event_bus = EventBus()
    registry = ToolRegistry.from_config(simulator=mock_sim, tracker=tracker, event_bus=event_bus)
    config = AgentConfig(
        mode=mode.value,
        llm_model_id="mock/model",
        max_steps=max_steps,
        temperature=0.0,
    )
    context_builder = ContextBuilder(
        system_prompt=get_system_prompt("v1"),
        tool_registry=registry,
    )
    return ReActAgent(
        llm_model_id="mock/model",
        tool_registry=registry,
        compressor=None,
        context_builder=context_builder,
        config=config,
        mode=mode,
    )


@pytest.mark.system_test
class TestFullEvaluationPipeline:
    def test_single_episode_eval_result_has_required_keys(self, paris_request, mock_sim):
        agent = _build_agent(mock_sim)

        with patch("litellm.completion", return_value=DONE_RESPONSE):
            log = agent.run_episode(paris_request, mock_sim)

        evaluator = Evaluator(deterministic_eval=DeterministicEvaluator())
        result = evaluator.evaluate_episode(log, paris_request)

        assert result.episode_id == log.episode_id
        assert result.overall_score >= 0.0
        assert len(result.deterministic_scores) > 0

    def test_three_episodes_aggregate_returns_overall_score_mean(self, paris_request, mock_sim):
        agent = _build_agent(mock_sim)
        evaluator = Evaluator(deterministic_eval=DeterministicEvaluator())

        with patch("litellm.completion", return_value=DONE_RESPONSE):
            logs = [agent.run_episode(paris_request, mock_sim) for _ in range(3)]

        results = evaluator.evaluate_dataset(logs, [paris_request] * 3)
        agg = evaluator.aggregate(results)
        assert "overall_score_mean" in agg

    def test_all_deterministic_scores_bounded(self, paris_request, mock_sim):
        agent = _build_agent(mock_sim)

        with patch("litellm.completion", return_value=DONE_RESPONSE):
            log = agent.run_episode(paris_request, mock_sim)

        evaluator = Evaluator(deterministic_eval=DeterministicEvaluator())
        result = evaluator.evaluate_episode(log, paris_request)

        for key, val in result.deterministic_scores.items():
            assert 0.0 <= val <= 1.0, f"Score '{key}' = {val} out of bounds"

    def test_save_reload_episode_gives_same_scores(self, paris_request, mock_sim):
        agent = _build_agent(mock_sim)

        with patch("litellm.completion", return_value=DONE_RESPONSE):
            log = agent.run_episode(paris_request, mock_sim)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_episode(log, tmpdir)
            loaded_log = load_episode(path)

        evaluator = Evaluator(deterministic_eval=DeterministicEvaluator())
        r1 = evaluator.evaluate_episode(log, paris_request)
        r2 = evaluator.evaluate_episode(loaded_log, paris_request)

        assert r1.overall_score == r2.overall_score
