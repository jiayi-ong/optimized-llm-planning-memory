"""
agent/modes.py
==============
AgentMode — enum selecting the context-assembly strategy.

Design pattern: Strategy (via enum)
-------------------------------------
All three evaluation conditions use the same ``ReActAgent``. The mode enum
selects which strategy ``ContextBuilder.build()`` applies. This ensures
fair comparison between conditions: the only difference is how the context
window is assembled from the trajectory.

RAW          — Full raw trajectory injected as-is. No compression.
               Baseline 1: demonstrates the problem (context bloat).

LLM_SUMMARY  — A frozen off-the-shelf LLM summarises the old trajectory
               prefix. Baseline 2: tests whether simple LLM summarisation helps.

COMPRESSOR   — The trained ``TrainableCompressorBase`` produces a structured
               ``CompressedState`` that replaces the old trajectory prefix.
               Our proposed method.
"""

from enum import Enum


class AgentMode(str, Enum):
    """Selects the context-assembly strategy for ReActAgent."""

    RAW = "raw"
    """Baseline 1: inject the full raw trajectory into every LLM call."""

    LLM_SUMMARY = "llm_summary"
    """Baseline 2: use an LLM to summarise the old trajectory prefix."""

    COMPRESSOR = "compressor"
    """Our method: use the trained compressor to produce a CompressedState."""
