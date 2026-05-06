"""
scripts/snapshot_init.py
========================
Snapshot a fresh (or pre-trained) compressor and register it in the
augmentation registry under a stable aug_id.

Why snapshot inits?
-------------------
Default init values change whenever LoRA config, base model, or random seed
changes. Snapshotting an explicit init before any PPO training lets you start
multiple training runs from the exact same weights, making ablations and
re-runs reproducible.

Usage
-----
    # Snapshot a fresh random init of SSD:
    python scripts/snapshot_init.py --aug-id ssd-init-001 --compressor structured_selective

    # Snapshot a fresh random init of TGAD:
    python scripts/snapshot_init.py --aug-id tgad-init-001 --compressor mcts_gat

    # Snapshot TGAD after supervised pre-training (warm start):
    python scripts/snapshot_init.py \\
        --aug-id tgad-init-001 \\
        --compressor mcts_gat \\
        --from-checkpoint outputs/supervised/tgad_supervised.pt \\
        --description "TGAD after supervised pretraining on LLMMCTSCompressor outputs"

    # Snapshot a Transformer init with LoRA:
    python scripts/snapshot_init.py --aug-id transformer-init-001 --compressor transformer \\
        compressor.use_lora=true compressor.lora.r=8

All Hydra overrides are accepted after the named flags.

Output
------
- Weights saved to: outputs/augmentations/<aug_id>/compressor.pt
- Registry updated: data/registry/augmentations.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env", override=False)


# ── Augmentation type map ─────────────────────────────────────────────────────

_COMPRESSOR_TO_INIT_TYPE = {
    "structured_selective": "ssd_init",
    "mcts_gat":             "tgad_init",
    "transformer":          "transformer_init",
    "identity":             "raw",
    "llm":                  "llm_summary",
    "llm_mcts":             "mcts_only",
}


def _build_config_snapshot(
    compressor_type: str,
    cfg: object,
    agent_llm_model_id: str | None = None,
) -> dict:
    """Extract the compressor hyperparams that matter for reproducibility.

    agent_llm_model_id is recorded so a trained checkpoint can be traced back
    to the agent model it was trained with (important because the reward signal
    depends on agent behaviour, which differs across LLM models).
    """
    from omegaconf import OmegaConf
    snap: dict = {}

    # Always snapshot the agent LLM model used during this run
    snap["agent_llm_model_id"] = agent_llm_model_id or OmegaConf.select(
        cfg, "agent.llm_model_id", default="openai/gpt-4o-mini"
    )

    # Always include model path and output size
    if compressor_type not in ("identity", "llm", "llm_mcts"):
        snap["model_name_or_path"] = OmegaConf.select(
            cfg, "compressor.model_name_or_path", default="google/flan-t5-small"
        )
        snap["max_output_tokens"] = OmegaConf.select(
            cfg, "compressor.max_output_tokens", default=512
        )
        use_lora = OmegaConf.select(cfg, "compressor.use_lora", default=False)
        snap["use_lora"] = use_lora
        if use_lora:
            lora = OmegaConf.to_container(cfg.compressor.lora, resolve=True)
            snap["lora"] = lora

    if compressor_type == "structured_selective":
        snap["max_step_tokens"] = OmegaConf.select(cfg, "compressor.max_step_tokens", default=128)
        snap["max_input_tokens"] = OmegaConf.select(cfg, "compressor.max_input_tokens", default=2048)

    if compressor_type == "mcts_gat":
        snap["top_k_paths"] = OmegaConf.select(cfg, "compressor.top_k_paths", default=3)
        snap["path_encoder_dropout"] = OmegaConf.select(
            cfg, "compressor.path_encoder_dropout", default=0.0
        )
        snap["max_input_tokens"] = OmegaConf.select(cfg, "compressor.max_input_tokens", default=256)

    if compressor_type in ("llm", "llm_mcts"):
        snap["llm_model_id"] = OmegaConf.select(
            cfg, "compressor.llm_model_id", default="openai/gpt-4o-mini"
        )
        snap["max_output_tokens"] = OmegaConf.select(
            cfg, "compressor.max_output_tokens", default=1024
        )

    return snap


def main() -> None:
    # ── Parse named args (strip them before passing to Hydra) ─────────────────
    parser = argparse.ArgumentParser(
        description="Snapshot a compressor init and register it in the augmentation registry.",
        add_help=True,
    )
    parser.add_argument("--aug-id", required=True,
                        help="Registry ID for this init (e.g. ssd-init-001)")
    parser.add_argument("--compressor", required=True,
                        help="Compressor type key (structured_selective | mcts_gat | transformer | …)")
    parser.add_argument("--from-checkpoint", default=None,
                        help="Optional: load weights from this .pt file before saving "
                             "(e.g. supervised pretraining output)")
    parser.add_argument("--agent-llm-model", default=None,
                        help="litellm model string for the ReAct agent used with this snapshot "
                             "(e.g. openai/gpt-4o-mini, google/gemini-2.0-flash). "
                             "Defaults to agent.llm_model_id from the active agent config.")
    parser.add_argument("--description", default=None,
                        help="Human-readable description for the registry entry")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing registry entry with this aug_id")

    # Extract our named args; leave the rest for Hydra
    known, hydra_overrides = parser.parse_known_args()
    aug_id           = known.aug_id
    comp_type        = known.compressor
    from_ckpt        = known.from_checkpoint
    agent_llm_model  = known.agent_llm_model
    description      = known.description or f"Init snapshot of {comp_type}"
    overwrite        = known.overwrite

    # Inject compressor type as Hydra override so the config loads correctly
    if not any(o.startswith("compressor=") for o in hydra_overrides):
        hydra_overrides = [f"compressor={comp_type}"] + hydra_overrides

    # ── Hydra config ──────────────────────────────────────────────────────────
    # Patch sys.argv for Hydra (it reads sys.argv[1:])
    sys.argv = [sys.argv[0]] + hydra_overrides

    import hydra
    from omegaconf import DictConfig, OmegaConf

    # Use hydra.initialize + hydra.compose to avoid the @hydra.main decorator
    with hydra.initialize(config_path="../configs", version_base="1.3"):
        cfg = hydra.compose(config_name="config", overrides=hydra_overrides)

    from optimized_llm_planning_memory.utils.logging import configure_logging, get_logger
    configure_logging(level="INFO")
    log = get_logger(__name__)

    # ── Build compressor ──────────────────────────────────────────────────────
    log.info("snapshot.build", aug_id=aug_id, compressor_type=comp_type)

    from optimized_llm_planning_memory.core.registry import (
        AugmentationEntry, AugmentationRegistry, AugmentationType,
    )
    from optimized_llm_planning_memory.core.run_spec import build_compressor_from_entry

    aug_type_str = _COMPRESSOR_TO_INIT_TYPE.get(comp_type, "raw")
    aug_type = AugmentationType(aug_type_str)
    snap = _build_config_snapshot(comp_type, cfg, agent_llm_model_id=agent_llm_model)

    # Build a temporary entry with no checkpoint (just to drive the factory)
    temp_entry = AugmentationEntry(
        aug_id=aug_id,
        type=aug_type,
        description=description,
        config_snapshot=snap,
    )
    compressor = build_compressor_from_entry(temp_entry, cfg, _REPO_ROOT, log)

    # Optionally load supervised pre-training weights
    if from_ckpt:
        ckpt_path = Path(from_ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = _REPO_ROOT / ckpt_path
        if not ckpt_path.exists():
            log.error("snapshot.from_checkpoint_missing", path=str(ckpt_path))
            sys.exit(1)
        compressor.load_checkpoint(ckpt_path)
        log.info("snapshot.loaded_from_checkpoint", path=str(ckpt_path))

    # ── Save weights ──────────────────────────────────────────────────────────
    ckpt_rel = f"outputs/augmentations/{aug_id}/compressor.pt"
    ckpt_abs = _REPO_ROOT / ckpt_rel
    ckpt_abs.parent.mkdir(parents=True, exist_ok=True)
    compressor.save_checkpoint(ckpt_abs)
    log.info("snapshot.saved", path=str(ckpt_abs))

    # ── Register ──────────────────────────────────────────────────────────────
    registry = AugmentationRegistry.load()
    entry = AugmentationEntry(
        aug_id=aug_id,
        type=aug_type,
        description=description,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        checkpoint_path=ckpt_rel,
        config_snapshot=snap,
        notes=f"from_checkpoint={from_ckpt}" if from_ckpt else "Random init snapshot",
    )
    registry.register(entry, overwrite=overwrite)
    registry.save()
    log.info("snapshot.registered", aug_id=aug_id, type=aug_type_str)

    print(f"\n✓  Snapshot registered: {aug_id}")
    print(f"   Type      : {aug_type_str}")
    print(f"   Checkpoint: {ckpt_rel}")
    print(f"   Config    : {snap}")


if __name__ == "__main__":
    main()
