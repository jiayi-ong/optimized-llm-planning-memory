"""
core/registry.py
================
Prompt and Augmentation registries — ID-based catalogs for named agent
configurations and compressor states.

Design notes
------------
Two JSON-backed catalogs live in ``data/registry/``:

  prompts.json
    Maps ``prompt_id`` (e.g. "v2", "sweep_D") to a ``PromptEntry`` — metadata
    plus a ``text_ref`` key that resolves to the actual prompt text in
    ``agent/prompts.py``.  No text is duplicated: ``PromptRegistry.get_prompt_text()``
    calls ``get_system_prompt(entry.text_ref)`` to fetch the live constant.
    Custom prompts not in ``prompts.py`` can store text directly via
    ``inline_text``; that field takes precedence over ``text_ref``.

  augmentations.json
    Maps ``aug_id`` (e.g. "ssd-init-001", "tgad-trained-001") to an
    ``AugmentationEntry`` — type enum, checkpoint path (relative to repo root),
    config snapshot, and lineage fields (parent_init_id, training_run_id).

    The type enum drives two derived properties used by scripts:
      entry.agent_mode     → "compressor" | "mcts_compressor" | "raw" | "llm_summary"
      entry.requires_mcts  → True for *MCTS_ONLY / *TGAD_* types

Why separate init and trained entries?
  Init snapshots capture the *exact* weight state before PPO training.  Default
  init values change whenever the LoRA config, base model, or random seed changes,
  so reproducibility requires an explicit snapshot.  Trained entries then point
  both to their checkpoint *and* to the init they started from (parent_init_id),
  making the full training lineage traceable from a single ID.

Usage
-----
    from optimized_llm_planning_memory.core.registry import (
        PromptRegistry, AugmentationRegistry, AugmentationEntry, AugmentationType
    )

    prompt_text = PromptRegistry.load().get_prompt_text("sweep_D")

    aug_entry = AugmentationRegistry.load().get("ssd-init-001")
    print(aug_entry.agent_mode)      # "compressor"
    print(aug_entry.has_checkpoint)  # True
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ── Default paths ─────────────────────────────────────────────────────────────
# Resolved relative to this file: src/…/core/registry.py → repo root is 4 up.
_REGISTRY_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "registry"
DEFAULT_PROMPTS_PATH = _REGISTRY_DIR / "prompts.json"
DEFAULT_AUGMENTATIONS_PATH = _REGISTRY_DIR / "augmentations.json"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ── Prompt Registry ───────────────────────────────────────────────────────────

class PromptEntry(BaseModel):
    """Metadata record for a named system prompt."""

    prompt_id: str
    description: str
    text_ref: str = Field(
        description=(
            "Key in agent/prompts._VERSIONS (e.g. 'v2', 'sweep_D'). "
            "Defaults to prompt_id when the prompt lives in prompts.py."
        )
    )
    created_at: str = Field(default_factory=_now_iso)
    notes: str = ""
    deprecated: bool = False
    inline_text: str | None = Field(
        default=None,
        description=(
            "Full prompt text for prompts not registered in prompts.py. "
            "Takes precedence over text_ref when set."
        ),
    )


class PromptRegistry:
    """
    Catalog of named system prompts.

    Loading and saving
    ------------------
    The JSON file stores one object per prompt_id.  The prompt_id key itself
    is excluded from the stored value (it is the dict key).  On load, it is
    re-injected so each PromptEntry is self-contained.

    Text resolution
    ---------------
    ``get_prompt_text(prompt_id)`` checks ``inline_text`` first; if absent it
    calls ``agent.prompts.get_system_prompt(entry.text_ref)``.  This means the
    live constant in ``prompts.py`` is always the source of truth for built-in
    prompts.

    Example
    -------
    registry = PromptRegistry.load()
    text = registry.get_prompt_text("sweep_D")

    registry.register(PromptEntry(
        prompt_id="v6",
        text_ref="v6",
        description="Adds budget-aware phase skipping",
    ))
    registry.save()
    """

    def __init__(self, entries: dict[str, PromptEntry], path: Path) -> None:
        self._entries = entries
        self._path = path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PROMPTS_PATH) -> "PromptRegistry":
        """Load from JSON. Returns empty registry if the file does not exist."""
        p = Path(path)
        if not p.exists():
            return cls(entries={}, path=p)
        raw: dict[str, dict[str, Any]] = json.loads(p.read_text(encoding="utf-8"))
        entries = {k: PromptEntry(prompt_id=k, **v) for k, v in raw.items()}
        return cls(entries=entries, path=p)

    def get_prompt_text(self, prompt_id: str) -> str:
        """
        Return the system prompt string for *prompt_id*.

        Resolution order:
          1. entry.inline_text (if set)
          2. get_system_prompt(entry.text_ref)  — resolves via prompts._VERSIONS

        Raises
        ------
        KeyError   : prompt_id not in registry
        ValueError : text_ref not found in prompts._VERSIONS
        """
        entry = self.get_entry(prompt_id)
        if entry.inline_text is not None:
            return entry.inline_text
        from optimized_llm_planning_memory.agent.prompts import get_system_prompt
        return get_system_prompt(entry.text_ref)

    def get_entry(self, prompt_id: str) -> PromptEntry:
        if prompt_id not in self._entries:
            raise KeyError(
                f"Prompt ID '{prompt_id}' not in registry. "
                f"Available: {sorted(self._entries)}."
            )
        return self._entries[prompt_id]

    def list(self) -> list[PromptEntry]:
        return list(self._entries.values())

    def register(self, entry: PromptEntry, overwrite: bool = False) -> None:
        if entry.prompt_id in self._entries and not overwrite:
            raise ValueError(
                f"Prompt ID '{entry.prompt_id}' already registered. "
                "Pass overwrite=True to update."
            )
        self._entries[entry.prompt_id] = entry

    def save(self, path: Path | str | None = None) -> None:
        """Persist the registry to disk (creates parent dirs if needed)."""
        target = Path(path) if path else self._path
        target.parent.mkdir(parents=True, exist_ok=True)
        out: dict[str, Any] = {
            pid: entry.model_dump(mode="json", exclude={"prompt_id"})
            for pid, entry in self._entries.items()
        }
        target.write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


# ── Augmentation Registry ─────────────────────────────────────────────────────

class AugmentationType(str, Enum):
    """
    Type taxonomy for augmentation entries.

    *_INIT  = pre-PPO initialization snapshot (trainable weights, no PPO yet)
    *_TRAINED = post-PPO trained checkpoint

    Non-neural types (MCTS_ONLY, RAW, LLM_SUMMARY) have no checkpoint and
    do not require PPO training.
    """

    MCTS_ONLY           = "mcts_only"           # LLMMCTSCompressor, no trainable weights
    TGAD_INIT           = "tgad_init"           # MCTSGraphAttentionDistiller, pre-PPO
    TGAD_TRAINED        = "tgad_trained"        # MCTSGraphAttentionDistiller, post-PPO
    SSD_INIT            = "ssd_init"            # StructuredSelectiveDistiller, pre-PPO
    SSD_TRAINED         = "ssd_trained"         # StructuredSelectiveDistiller, post-PPO
    TRANSFORMER_INIT    = "transformer_init"    # TransformerCompressor, pre-PPO
    TRANSFORMER_TRAINED = "transformer_trained" # TransformerCompressor, post-PPO
    RAW                 = "raw"                 # IdentityCompressor (no compression)
    LLM_SUMMARY         = "llm_summary"         # LLMCompressor (frozen, non-trainable)


# Agent mode and MCTS requirement implied by each augmentation type
_AUG_TYPE_META: dict[AugmentationType, tuple[str, bool]] = {
    # type                         → (agent_mode,        requires_mcts)
    AugmentationType.MCTS_ONLY:           ("mcts_compressor", True),
    AugmentationType.TGAD_INIT:           ("mcts_compressor", True),
    AugmentationType.TGAD_TRAINED:        ("mcts_compressor", True),
    AugmentationType.SSD_INIT:            ("compressor",      False),
    AugmentationType.SSD_TRAINED:         ("compressor",      False),
    AugmentationType.TRANSFORMER_INIT:    ("compressor",      False),
    AugmentationType.TRANSFORMER_TRAINED: ("compressor",      False),
    AugmentationType.RAW:                 ("raw",             False),
    AugmentationType.LLM_SUMMARY:         ("llm_summary",     False),
}

# Compressor config type string implied by each augmentation type
_AUG_TYPE_TO_COMPRESSOR: dict[AugmentationType, str] = {
    AugmentationType.MCTS_ONLY:           "llm_mcts",
    AugmentationType.TGAD_INIT:           "mcts_gat",
    AugmentationType.TGAD_TRAINED:        "mcts_gat",
    AugmentationType.SSD_INIT:            "structured_selective",
    AugmentationType.SSD_TRAINED:         "structured_selective",
    AugmentationType.TRANSFORMER_INIT:    "transformer",
    AugmentationType.TRANSFORMER_TRAINED: "transformer",
    AugmentationType.RAW:                 "identity",
    AugmentationType.LLM_SUMMARY:         "llm",
}


class AugmentationEntry(BaseModel):
    """
    Metadata + location record for a named augmentation state.

    Fields
    ------
    aug_id          : Unique identifier (e.g. "ssd-init-001", "tgad-trained-001").
    type            : AugmentationType enum value.
    checkpoint_path : Path to compressor weights, relative to repo root.
                      None for MCTS_ONLY / RAW / LLM_SUMMARY.
    parent_init_id  : For *_TRAINED entries, the aug_id of the init snapshot
                      this training run started from.
    training_run_id : TrainingRunManifest.run_id of the PPO run (if applicable).
    training_step   : Which step's checkpoint (None = final checkpoint).
    mcts_config_ref : Agent YAML name for MCTS hyperparams (mcts_only only).
    config_snapshot : Compressor hyperparams frozen at creation time
                      (model_name_or_path, LoRA settings, etc.).
    """

    aug_id: str
    type: AugmentationType
    description: str
    created_at: str = Field(default_factory=_now_iso)
    checkpoint_path: str | None = None
    parent_init_id: str | None = None
    training_run_id: str | None = None
    training_step: int | None = None
    mcts_config_ref: str | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    reward_weights: dict[str, float] = Field(default_factory=dict)
    notes: str = ""
    deprecated: bool = False

    @property
    def agent_mode(self) -> str:
        """Agent mode string implied by this augmentation type."""
        return _AUG_TYPE_META[self.type][0]

    @property
    def requires_mcts(self) -> bool:
        """True when this augmentation requires an MCTSController."""
        return _AUG_TYPE_META[self.type][1]

    @property
    def compressor_type(self) -> str:
        """Compressor config type string implied by this augmentation type."""
        return _AUG_TYPE_TO_COMPRESSOR[self.type]

    @property
    def has_checkpoint(self) -> bool:
        return self.checkpoint_path is not None

    @property
    def is_trainable_type(self) -> bool:
        """True for types that have trainable neural weights."""
        return self.type not in (
            AugmentationType.MCTS_ONLY,
            AugmentationType.RAW,
            AugmentationType.LLM_SUMMARY,
        )


class AugmentationRegistry:
    """
    Catalog of named augmentation states.

    New entries are added via:
      - ``scripts/snapshot_init.py`` — for init snapshots before PPO training
      - Auto-registration at end of ``run_training.py`` — for trained checkpoints

    Example
    -------
    registry = AugmentationRegistry.load()
    entry = registry.get("ssd-init-001")
    print(entry.agent_mode)       # "compressor"
    print(entry.has_checkpoint)   # True

    # Register a trained checkpoint after a PPO run:
    registry.register(AugmentationEntry(
        aug_id="ssd-trained-001",
        type=AugmentationType.SSD_TRAINED,
        description="SSD trained for 50k PPO steps from ssd-init-001",
        checkpoint_path="outputs/augmentations/ssd-trained-001/compressor.pt",
        parent_init_id="ssd-init-001",
        training_run_id="20260505_120000",
    ))
    registry.save()
    """

    def __init__(self, entries: dict[str, AugmentationEntry], path: Path) -> None:
        self._entries = entries
        self._path = path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_AUGMENTATIONS_PATH) -> "AugmentationRegistry":
        """Load from JSON. Returns empty registry if the file does not exist."""
        p = Path(path)
        if not p.exists():
            return cls(entries={}, path=p)
        raw: dict[str, dict[str, Any]] = json.loads(p.read_text(encoding="utf-8"))
        entries = {k: AugmentationEntry(aug_id=k, **v) for k, v in raw.items()}
        return cls(entries=entries, path=p)

    def get(self, aug_id: str) -> AugmentationEntry:
        if aug_id not in self._entries:
            raise KeyError(
                f"Augmentation ID '{aug_id}' not in registry. "
                f"Available: {sorted(self._entries)}."
            )
        return self._entries[aug_id]

    def list(self) -> list[AugmentationEntry]:
        return list(self._entries.values())

    def register(self, entry: AugmentationEntry, overwrite: bool = False) -> None:
        if entry.aug_id in self._entries and not overwrite:
            raise ValueError(
                f"Augmentation ID '{entry.aug_id}' already registered. "
                "Pass overwrite=True to update."
            )
        self._entries[entry.aug_id] = entry

    def save(self, path: Path | str | None = None) -> None:
        """Persist the registry to disk (creates parent dirs if needed)."""
        target = Path(path) if path else self._path
        target.parent.mkdir(parents=True, exist_ok=True)
        out: dict[str, Any] = {
            aid: entry.model_dump(mode="json", exclude={"aug_id"})
            for aid, entry in self._entries.items()
        }
        target.write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
