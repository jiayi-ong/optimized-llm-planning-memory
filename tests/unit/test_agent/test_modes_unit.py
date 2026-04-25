"""Unit tests for agent/modes.py — AgentMode enum."""

from __future__ import annotations

import pytest

from optimized_llm_planning_memory.agent.modes import AgentMode


@pytest.mark.unit
class TestAgentModeValues:
    def test_raw_value(self):
        assert AgentMode.RAW == "raw"
        assert AgentMode.RAW.value == "raw"

    def test_llm_summary_value(self):
        assert AgentMode.LLM_SUMMARY == "llm_summary"

    def test_compressor_value(self):
        assert AgentMode.COMPRESSOR == "compressor"

    def test_mcts_compressor_value(self):
        assert AgentMode.MCTS_COMPRESSOR == "mcts_compressor"

    def test_all_four_modes_importable(self):
        modes = list(AgentMode)
        assert len(modes) == 4

    def test_from_string_raw(self):
        assert AgentMode("raw") is AgentMode.RAW

    def test_from_string_compressor(self):
        assert AgentMode("compressor") is AgentMode.COMPRESSOR

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError):
            AgentMode("unknown_mode")

    def test_is_str_subclass(self):
        assert isinstance(AgentMode.RAW, str)

    def test_str_representation(self):
        assert AgentMode.RAW.value == "raw"
