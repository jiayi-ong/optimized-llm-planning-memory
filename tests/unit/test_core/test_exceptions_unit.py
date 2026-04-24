"""Unit tests for core/exceptions.py — exception hierarchy."""

from __future__ import annotations

import pytest

from optimized_llm_planning_memory.core.exceptions import (
    AgentError,
    AgentMaxStepsError,
    AgentParseError,
    CompressorCheckpointError,
    CompressorError,
    CompressedStateRenderError,
    ConfigError,
    EvaluationError,
    LLMJudgeError,
    LogProbsNotSupportedError,
    MCTSError,
    MCTSSearchTimeoutError,
    ProjectError,
    RewardComputationError,
    SimulatorConnectionError,
    SimulatorError,
    SimulatorSeedError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
    TrainingError,
)


@pytest.mark.unit
class TestExceptionHierarchy:
    def test_all_inherit_project_error(self):
        leaf_exceptions = [
            SimulatorConnectionError("x"),
            SimulatorSeedError("x"),
            ToolNotFoundError("my_tool"),
            ToolValidationError("x"),
            ToolExecutionError("x"),
            AgentParseError("x"),
            AgentMaxStepsError("x"),
            CompressedStateRenderError("x"),
            CompressorCheckpointError("x"),
            LogProbsNotSupportedError("x"),
            RewardComputationError("x"),
            LLMJudgeError("x"),
            MCTSSearchTimeoutError("x"),
            ConfigError("x"),
        ]
        for exc in leaf_exceptions:
            assert isinstance(exc, ProjectError), f"{type(exc).__name__} must inherit ProjectError"

    def test_tool_not_found_includes_tool_name(self):
        exc = ToolNotFoundError("fly_rocket")
        assert "fly_rocket" in str(exc)

    def test_tool_not_found_has_tool_name_attribute(self):
        exc = ToolNotFoundError("search_flights")
        assert exc.tool_name == "search_flights"

    def test_log_probs_not_supported_is_compressor_error(self):
        exc = LogProbsNotSupportedError("test")
        assert isinstance(exc, CompressorError)

    def test_simulator_errors_inherit_simulator_error(self):
        assert issubclass(SimulatorConnectionError, SimulatorError)
        assert issubclass(SimulatorSeedError, SimulatorError)

    def test_agent_errors_inherit_agent_error(self):
        assert issubclass(AgentParseError, AgentError)
        assert issubclass(AgentMaxStepsError, AgentError)

    def test_training_error_hierarchy(self):
        assert issubclass(RewardComputationError, TrainingError)
        assert issubclass(TrainingError, ProjectError)

    def test_evaluation_error_hierarchy(self):
        assert issubclass(LLMJudgeError, EvaluationError)
        assert issubclass(EvaluationError, ProjectError)

    def test_mcts_error_hierarchy(self):
        assert issubclass(MCTSSearchTimeoutError, MCTSError)
        assert issubclass(MCTSError, ProjectError)
