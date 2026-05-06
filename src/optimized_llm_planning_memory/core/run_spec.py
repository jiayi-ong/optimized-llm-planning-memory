"""
core/run_spec.py
================
RunSpec — a (prompt_id, augmentation_id) pair that fully identifies an
agent × augmentation combination for episode generation or RL training.

Design notes
------------
``resolve_run_spec()`` converts a RunSpec into a ResolvedRunSpec that holds
the concrete system prompt text and AugmentationEntry.  Scripts then call:

  1. ``apply_run_spec_to_cfg(resolved, cfg)`` — merges the augmentation's
     implied agent mode and compressor type into the live Hydra DictConfig
     so the rest of the script can proceed unchanged.

  2. ``build_compressor_from_entry(entry, cfg, repo_root)`` — constructs the
     compressor instance and loads checkpoint weights when present.

Backward compatibility
----------------------
Both functions are no-ops when ``prompt_id`` / ``augmentation_id`` are None
(i.e. the legacy Hydra-only path).  Existing CLI invocations without registry
IDs continue to work without changes.

Usage
-----
From scripts/run_training.py:

    from optimized_llm_planning_memory.core.run_spec import (
        resolve_run_spec, apply_run_spec_to_cfg, build_compressor_from_entry
    )
    from optimized_llm_planning_memory.core.registry import RunSpec

    prompt_id  = OmegaConf.select(cfg, "agent.prompt_id")
    aug_id     = OmegaConf.select(cfg, "project.augmentation_id")

    if prompt_id or aug_id:
        spec     = RunSpec(prompt_id=prompt_id or "v2", augmentation_id=aug_id or "raw-default")
        resolved = resolve_run_spec(spec)
        apply_run_spec_to_cfg(resolved, cfg)
        compressor = build_compressor_from_entry(resolved.augmentation_entry, cfg, _REPO_ROOT, log)
    else:
        compressor = <existing if/elif factory block>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from optimized_llm_planning_memory.core.registry import (
    AugmentationEntry,
    AugmentationRegistry,
    AugmentationType,
    PromptRegistry,
)


# ── Data models ───────────────────────────────────────────────────────────────

class RunSpec(BaseModel):
    """A (prompt_id, augmentation_id) combination for episode generation or training."""

    prompt_id: str
    augmentation_id: str
    description: str | None = None


class ResolvedRunSpec(BaseModel):
    """Fully resolved run spec: prompt text + augmentation entry, ready for use."""

    prompt_id: str
    augmentation_id: str
    system_prompt: str
    augmentation_entry: AugmentationEntry

    model_config = {"arbitrary_types_allowed": True}


# ── Resolution ────────────────────────────────────────────────────────────────

def resolve_run_spec(
    spec: RunSpec,
    prompt_registry: PromptRegistry | None = None,
    aug_registry: AugmentationRegistry | None = None,
) -> ResolvedRunSpec:
    """
    Resolve a RunSpec to its concrete prompt text and AugmentationEntry.

    Loads default registry files if registries are not supplied.

    Parameters
    ----------
    spec            : RunSpec with prompt_id and augmentation_id.
    prompt_registry : Pre-loaded PromptRegistry (loads default if None).
    aug_registry    : Pre-loaded AugmentationRegistry (loads default if None).
    """
    if prompt_registry is None:
        prompt_registry = PromptRegistry.load()
    if aug_registry is None:
        aug_registry = AugmentationRegistry.load()

    system_prompt = prompt_registry.get_prompt_text(spec.prompt_id)
    aug_entry = aug_registry.get(spec.augmentation_id)

    return ResolvedRunSpec(
        prompt_id=spec.prompt_id,
        augmentation_id=spec.augmentation_id,
        system_prompt=system_prompt,
        augmentation_entry=aug_entry,
    )


# ── Cfg integration ───────────────────────────────────────────────────────────

def apply_run_spec_to_cfg(resolved: ResolvedRunSpec, cfg: Any) -> None:
    """
    Merge a resolved RunSpec into a Hydra OmegaConf DictConfig in-place.

    Sets ``agent.mode`` and ``compressor.type`` derived from the augmentation
    type, and applies any compressor hyperparams from the config snapshot.
    The prompt text itself is NOT written to cfg — callers use
    ``resolved.system_prompt`` directly when constructing ContextBuilder.

    This function does NOT load checkpoint weights; that is handled by
    ``build_compressor_from_entry()``.

    Parameters
    ----------
    resolved : ResolvedRunSpec from resolve_run_spec().
    cfg      : OmegaConf DictConfig (modified in-place via OmegaConf.update).
    """
    from omegaconf import OmegaConf

    entry = resolved.augmentation_entry

    OmegaConf.update(cfg, "agent.mode", entry.agent_mode, merge=True)
    OmegaConf.update(cfg, "compressor.type", entry.compressor_type, merge=True)

    # Apply config snapshot to compressor config (best-effort; ignore unknown keys)
    _COMPRESSOR_SCALAR_FIELDS = {
        "model_name_or_path", "llm_model_id", "max_input_tokens", "max_output_tokens",
        "use_lora", "freeze_base_layers", "max_step_tokens", "top_k_paths",
        "path_encoder_dropout", "device",
    }
    for key, value in entry.config_snapshot.items():
        if key in _COMPRESSOR_SCALAR_FIELDS:
            try:
                OmegaConf.update(cfg, f"compressor.{key}", value, merge=True)
            except Exception:
                pass


# ── Compressor factory ────────────────────────────────────────────────────────

def build_compressor_from_entry(
    entry: AugmentationEntry,
    cfg: Any,
    repo_root: Path,
    log: Any | None = None,
) -> Any:
    """
    Build and return the compressor described by an AugmentationEntry.

    Construction order:
      1. Instantiate the compressor class using config_snapshot values,
         falling back to the Hydra cfg for any missing fields.
      2. If entry.checkpoint_path is set, load weights via load_checkpoint().

    Parameters
    ----------
    entry     : AugmentationEntry from the augmentation registry.
    cfg       : Hydra OmegaConf DictConfig (fallback for unsnapshotted fields).
    repo_root : Absolute path to repo root (checkpoint_path is relative to this).
    log       : Optional structlog logger for info messages.

    Raises
    ------
    FileNotFoundError : checkpoint_path is set but the file does not exist.
    """
    from omegaconf import OmegaConf

    snap = entry.config_snapshot
    compressor_type = entry.compressor_type

    if log:
        log.info("registry.build_compressor",
                 aug_id=entry.aug_id,
                 type=compressor_type,
                 has_checkpoint=entry.has_checkpoint)

    def _snap(key: str, fallback_path: str, default: Any = None) -> Any:
        """Return snapshot[key] if present, else OmegaConf.select(cfg, fallback_path)."""
        if key in snap:
            return snap[key]
        return OmegaConf.select(cfg, fallback_path, default=default)

    # ── Construct ─────────────────────────────────────────────────────────────
    compressor: Any

    if compressor_type == "identity":
        from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
        compressor = IdentityCompressor()

    elif compressor_type == "llm":
        from optimized_llm_planning_memory.compressor.llm_compressor import LLMCompressor
        compressor = LLMCompressor(
            model_id=_snap("llm_model_id", "compressor.llm_model_id", "openai/gpt-4o-mini"),
        )

    elif compressor_type == "llm_mcts":
        from optimized_llm_planning_memory.compressor.llm_mcts_compressor import LLMMCTSCompressor
        compressor = LLMMCTSCompressor(
            llm_model_id=_snap("llm_model_id", "compressor.llm_model_id", "openai/gpt-4o-mini"),
            max_output_tokens=_snap("max_output_tokens", "compressor.max_output_tokens", 1024),
        )

    elif compressor_type == "transformer":
        from optimized_llm_planning_memory.compressor.transformer_compressor import TransformerCompressor
        from optimized_llm_planning_memory.core.config import LoRAConfig
        compressor = TransformerCompressor(
            model_name_or_path=_snap(
                "model_name_or_path", "compressor.model_name_or_path", "google/flan-t5-small"
            ),
            device=_snap("device", "compressor.device", "auto"),
        )
        if _snap("use_lora", "compressor.use_lora", False):
            lora_raw = snap.get("lora") or OmegaConf.to_container(cfg.compressor.lora, resolve=True)
            compressor.apply_lora(LoRAConfig(**lora_raw))
        if _snap("freeze_base_layers", "compressor.freeze_base_layers", False):
            compressor.freeze_base_layers(freeze=True)

    elif compressor_type == "structured_selective":
        from optimized_llm_planning_memory.compressor.structured_selective_distiller import (
            StructuredSelectiveDistiller,
        )
        from optimized_llm_planning_memory.core.config import LoRAConfig
        lora_raw = snap.get("lora") or OmegaConf.to_container(cfg.compressor.lora, resolve=True)
        compressor = StructuredSelectiveDistiller(
            model_name_or_path=_snap(
                "model_name_or_path", "compressor.model_name_or_path", "google/flan-t5-small"
            ),
            max_step_tokens=_snap("max_step_tokens", "compressor.max_step_tokens", 128),
            max_output_tokens=_snap("max_output_tokens", "compressor.max_output_tokens", 512),
            device=_snap("device", "compressor.device", "auto"),
            use_lora=_snap("use_lora", "compressor.use_lora", True),
            lora_config=LoRAConfig(**lora_raw),
        )

    elif compressor_type == "mcts_gat":
        from optimized_llm_planning_memory.compressor.mcts_gat_distiller import (
            MCTSGraphAttentionDistiller,
        )
        from optimized_llm_planning_memory.core.config import LoRAConfig
        lora_raw = snap.get("lora") or OmegaConf.to_container(cfg.compressor.lora, resolve=True)
        compressor = MCTSGraphAttentionDistiller(
            model_name_or_path=_snap(
                "model_name_or_path", "compressor.model_name_or_path", "google/flan-t5-small"
            ),
            max_path_tokens=_snap("max_input_tokens", "compressor.max_input_tokens", 256),
            max_output_tokens=_snap("max_output_tokens", "compressor.max_output_tokens", 512),
            device=_snap("device", "compressor.device", "auto"),
            use_lora=_snap("use_lora", "compressor.use_lora", True),
            lora_config=LoRAConfig(**lora_raw),
            top_k_paths=_snap("top_k_paths", "compressor.top_k_paths", 3),
            path_encoder_dropout=_snap("path_encoder_dropout", "compressor.path_encoder_dropout", 0.0),
        )

    else:
        from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
        compressor = IdentityCompressor()

    # ── Load checkpoint ────────────────────────────────────────────────────────
    if entry.checkpoint_path:
        ckpt = repo_root / entry.checkpoint_path
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint for augmentation '{entry.aug_id}' not found at: {ckpt}\n"
                f"Run scripts/snapshot_init.py to create it, or update the registry path."
            )
        compressor.load_checkpoint(ckpt)
        if log:
            log.info("registry.checkpoint_loaded", aug_id=entry.aug_id, path=str(ckpt))

    return compressor
