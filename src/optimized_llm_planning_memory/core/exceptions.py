"""
core/exceptions.py
==================
Project-wide custom exception hierarchy.

Design pattern: Exception hierarchy
------------------------------------
All project exceptions inherit from ``ProjectError``. This lets callers catch
a broad class of errors with a single ``except ProjectError`` while still being
able to distinguish sub-types. Each module defines its own sub-class here rather
than inline, so the full hierarchy is visible in one place.
"""


class ProjectError(Exception):
    """Base class for all project-specific exceptions."""


# ── Simulator ─────────────────────────────────────────────────────────────────

class SimulatorError(ProjectError):
    """Raised when the travel simulator returns an unexpected response."""


class SimulatorConnectionError(SimulatorError):
    """Raised when the adapter cannot reach the simulator backend."""


class SimulatorSeedError(SimulatorError):
    """Raised when an invalid or out-of-range seed is provided."""


# ── Tools ─────────────────────────────────────────────────────────────────────

class ToolError(ProjectError):
    """Base for tool-middleware errors."""


class ToolNotFoundError(ToolError):
    """Raised by ToolRegistry when a requested tool name is not registered."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"Tool '{tool_name}' is not registered in the ToolRegistry.")
        self.tool_name = tool_name


class ToolValidationError(ToolError):
    """Raised when tool input fails Pydantic validation."""


class ToolExecutionError(ToolError):
    """Raised when a tool's _execute() raises an unexpected exception."""


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentError(ProjectError):
    """Base for planning agent errors."""


class AgentParseError(AgentError):
    """Raised when the agent's LLM output cannot be parsed into a valid action."""


class AgentMaxStepsError(AgentError):
    """Raised when an episode exceeds the configured max_steps limit."""


# ── Compressor ────────────────────────────────────────────────────────────────

class CompressorError(ProjectError):
    """Base for context compressor errors."""


class CompressedStateRenderError(CompressorError):
    """Raised when a CompressedState is missing required template sections."""


class CompressorCheckpointError(CompressorError):
    """Raised when a checkpoint file cannot be loaded or saved."""


class LogProbsNotSupportedError(CompressorError):
    """Raised when get_log_probs() is called on a non-trainable compressor."""


# ── Training ──────────────────────────────────────────────────────────────────

class TrainingError(ProjectError):
    """Base for RL training pipeline errors."""


class RewardComputationError(TrainingError):
    """Raised when reward computation fails (e.g., malformed itinerary)."""


# ── Evaluation ────────────────────────────────────────────────────────────────

class EvaluationError(ProjectError):
    """Base for evaluation pipeline errors."""


class LLMJudgeError(EvaluationError):
    """Raised when the LLM judge fails to produce a parseable score."""


# ── Configuration ─────────────────────────────────────────────────────────────

class ConfigError(ProjectError):
    """Raised when the loaded configuration is invalid or incomplete."""
