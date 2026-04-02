"""
optimized-llm-planning-memory
==============================
Context-compression training for LLM travel-planning agents via PPO.

Package structure
-----------------
core/        — Pydantic data models, constraint engine, config schema. No internal imports.
simulator/   — Protocol contract + thin Python adapter over the external travel simulator lib.
tools/       — Middleware layer: validation, tracking, failure feedback, event bus.
agent/       — ReAct planning agent (pydantic-ai) + trajectory management.
compressor/  — ABC hierarchy for compression strategies; template renderer.
training/    — Gymnasium environment, SB3 policy wrapper, PPO trainer, reward function.
evaluation/  — Deterministic metrics, LLM-judge, ablation runner.
utils/       — Logging, TensorBoard helpers, visualization, reproducibility.
"""

__version__ = "0.1.0"
