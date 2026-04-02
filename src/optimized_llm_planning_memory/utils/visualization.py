"""
utils/visualization.py
=======================
Pretty-print helpers for ReAct steps, CompressedState, and reward components.

All functions write to stdout (or an optional ``file`` argument). No external
dependencies beyond the standard library — this module is intentionally
import-safe so it can be used in debugging contexts without the full stack.

Usage
-----
    from optimized_llm_planning_memory.utils.visualization import (
        print_episode, print_step, print_compressed_state, print_reward_components
    )

    print_episode(episode_log)
    print_step(step, index=3)
    print_compressed_state(compressed_state)
    print_reward_components(reward_components)
"""

from __future__ import annotations

import sys
import textwrap
from io import StringIO
from typing import TextIO

from optimized_llm_planning_memory.core.models import (
    CompressedState,
    EpisodeLog,
    ReActStep,
    RewardComponents,
)

_SEP = "─" * 72
_THIN = "·" * 72


def print_episode(
    episode_log: EpisodeLog,
    show_steps: bool = True,
    show_compressed_states: bool = True,
    file: TextIO = sys.stdout,
) -> None:
    """Print a full EpisodeLog in a human-readable format."""
    print(_SEP, file=file)
    print(f"  EPISODE: {episode_log.episode_id}", file=file)
    print(f"  Request: {episode_log.request_id}  |  Mode: {episode_log.agent_mode}", file=file)
    print(f"  Steps: {episode_log.total_steps}  |  Success: {episode_log.success}", file=file)
    print(_SEP, file=file)

    if show_steps:
        for step in episode_log.trajectory.steps:
            print_step(step, file=file)

    if show_compressed_states and episode_log.compressed_states:
        print(f"\n  COMPRESSED STATES ({len(episode_log.compressed_states)})", file=file)
        for cs in episode_log.compressed_states:
            print_compressed_state(cs, file=file)

    print("\n  REWARD COMPONENTS", file=file)
    print_reward_components(episode_log.reward_components, file=file)

    if episode_log.final_itinerary:
        it = episode_log.final_itinerary
        print(f"\n  FINAL ITINERARY: {len(it.days)} days  |  "
              f"Total cost: ${it.total_cost_usd:.2f}  |  Complete: {it.is_complete}", file=file)
    print(_SEP, file=file)


def print_step(step: ReActStep, file: TextIO = sys.stdout) -> None:
    """Print a single ReAct step."""
    print(f"\n  ── Step {step.step_index} ──────────────────────", file=file)
    if step.thought:
        _print_wrapped("  Thought", step.thought, file=file)
    if step.action:
        print(f"  Action : {step.action.tool_name}({_compact_dict(step.action.arguments)})", file=file)
    if step.observation:
        obs = step.observation
        status = "OK" if obs.success else "FAIL"
        result_summary = _truncate(str(obs.result), 120) if obs.result else ""
        error_summary = f"  ERROR: {_truncate(obs.error_message or '', 120)}" if not obs.success else ""
        print(f"  Obs    : [{status}] {obs.tool_name}  {result_summary}{error_summary}", file=file)


def print_compressed_state(cs: CompressedState, file: TextIO = sys.stdout) -> None:
    """Print a CompressedState in section-labelled format."""
    print(f"\n  {_THIN}", file=file)
    print(f"  CompressedState @ step {cs.step_index}  |  method={cs.compression_method}", file=file)
    if cs.token_count:
        print(f"  tokens={cs.token_count}", file=file)
    _print_wrapped("  decisions_made", " · ".join(cs.decisions_made), file=file)
    _print_wrapped("  open_questions", " · ".join(cs.open_questions), file=file)
    _print_wrapped("  key_discoveries", " · ".join(cs.key_discoveries), file=file)
    _print_wrapped("  sketch", cs.current_itinerary_sketch, file=file)


def print_reward_components(rc: RewardComponents, file: TextIO = sys.stdout) -> None:
    """Print all reward component values as a compact table."""
    rows = [
        ("hard_constraint", rc.hard_constraint_score),
        ("soft_constraint", rc.soft_constraint_score),
        ("tool_efficiency", rc.tool_efficiency_score),
        ("tool_failure_penalty", rc.tool_failure_penalty),
        ("logical_consistency", rc.logical_consistency_score),
    ]
    if rc.terminal_itinerary_score is not None:
        rows.append(("terminal_itinerary", rc.terminal_itinerary_score))

    for name, val in rows:
        bar = _bar(val)
        print(f"  {name:<25} {val:+.3f}  {bar}", file=file)
    print(f"  {'TOTAL':<25} {rc.total_reward:+.3f}", file=file)


def episode_to_string(episode_log: EpisodeLog, **kwargs) -> str:
    """Return episode pretty-print as a string instead of writing to stdout."""
    buf = StringIO()
    print_episode(episode_log, file=buf, **kwargs)
    return buf.getvalue()


# ── Private helpers ────────────────────────────────────────────────────────────

def _print_wrapped(label: str, text: str, width: int = 80, file: TextIO = sys.stdout) -> None:
    if not text:
        return
    prefix = f"{label}: "
    wrapped = textwrap.fill(text, width=width, initial_indent=prefix,
                            subsequent_indent=" " * len(prefix))
    print(wrapped, file=file)


def _truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[:max_len] + "…"


def _compact_dict(d: dict) -> str:
    parts = [f"{k}={_truncate(str(v), 30)}" for k, v in list(d.items())[:4]]
    return ", ".join(parts)


def _bar(value: float, width: int = 20) -> str:
    """Simple ASCII progress bar for [0, 1] values."""
    clamped = max(0.0, min(1.0, value))
    filled = int(round(clamped * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"
