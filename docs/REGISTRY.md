# Registry — Prompt and Augmentation Catalogs

The registry system lets you identify any agent × augmentation combination with two short IDs — `prompt_id` and `augmentation_id` — instead of threading YAML paths, checkpoint paths, and version strings across CLI flags and notebooks.

---

## Overview

| Catalog | File | What it stores |
|---|---|---|
| Prompt Registry | `data/registry/prompts.json` | Named system prompts (maps ID → metadata + `text_ref`) |
| Augmentation Registry | `data/registry/augmentations.json` | Named compressor states (maps ID → type + checkpoint path + config snapshot) |

Python API:
```python
from optimized_llm_planning_memory.core.registry import PromptRegistry, AugmentationRegistry

prompt_text = PromptRegistry.load().get_prompt_text("sweep_D")
aug_entry   = AugmentationRegistry.load().get("ssd-init-001")
```

---

## Prompt Registry

### Built-in IDs

| `prompt_id` | What it is |
|---|---|
| `v1` | Base ReAct: WORLD CONTEXT, PLANNING PHASE order, BOOKING RULE, LETHAL SCENARIOS |
| `v2` | **Default.** v1 + constraint-tracking guidance + ITINERARY STATE section |
| `v3` | v2 + strict format requirement + error-recovery + budget tracking |
| `v4` | v3 + CONTEXT SECTION GUIDE |
| `v5` | v4 + STEP-BUDGET PHASED PLANNING with deadlines |
| `sweep_A` | Grow-then-Prune |
| `sweep_B` | Ordered pipeline with booking quotas |
| `sweep_C` | Commit-every-step (search-book pairs) |
| `sweep_D` | Minimal Viable First — confirmed non-degenerate in 11 steps |
| `sweep_E` | Step-budget phased planning |
| `sweep_F` | Stateless (no trajectory history) — requires `agent.mode=stateless` |
| `sweep_G` | Commitment gate — requires `agent.mode=stateless` |

### How text resolution works

`PromptRegistry.get_prompt_text(prompt_id)` resolves in this order:
1. If `entry.inline_text` is set → return it directly (for custom prompts not in `prompts.py`)
2. Otherwise → call `get_system_prompt(entry.text_ref)` (resolves the `text_ref` key against `agent/prompts._VERSIONS`)

The actual prompt text always lives in `agent/prompts.py`. The registry stores only metadata and a reference key — no text duplication.

### Adding a new prompt

1. Define `SYSTEM_PROMPT_VNAME = ...` in `agent/prompts.py`
2. Add it to `_VERSIONS` in `prompts.py`:
   ```python
   _VERSIONS["vname"] = SYSTEM_PROMPT_VNAME
   ```
3. Register it in the catalog:
   ```python
   from optimized_llm_planning_memory.core.registry import PromptRegistry, PromptEntry
   reg = PromptRegistry.load()
   reg.register(PromptEntry(
       prompt_id="vname",
       text_ref="vname",
       description="What this prompt adds",
   ))
   reg.save()
   ```
4. Use it: `agent.prompt_id=vname`

---

## Augmentation Registry

### Type taxonomy

| Type | When to use | Has checkpoint? | Requires MCTS? |
|---|---|---|---|
| `raw` | No compression. Baseline 1. | No | No |
| `llm_summary` | LLM summarization. Baseline 2. | No | No |
| `mcts_only` | MCTS + LLM compressor. No trainable weights. | No | Yes |
| `tgad_init` | TGAD init snapshot (before PPO). | Yes | Yes |
| `tgad_trained` | TGAD post-PPO checkpoint. | Yes | Yes |
| `ssd_init` | SSD init snapshot (before PPO). | Yes | No |
| `ssd_trained` | SSD post-PPO checkpoint. | Yes | No |
| `transformer_init` | TransformerCompressor init snapshot. | Yes | No |
| `transformer_trained` | TransformerCompressor post-PPO checkpoint. | Yes | No |

### Built-in IDs

| `augmentation_id` | Type | Notes |
|---|---|---|
| `raw-default` | `raw` | Full trajectory baseline |
| `llm-summary-default` | `llm_summary` | Frozen LLM summarization baseline |
| `mcts-default` | `mcts_only` | Default MCTS config (react_mcts.yaml) |

New init and trained IDs are created by your team and registered locally.

### Why init IDs are needed

Default init weights change whenever you change:
- LoRA config (`r`, `alpha`, `dropout`, `target_modules`)
- Base model (`model_name_or_path`)
- Random seed at construction time

Without an explicit snapshot, two training runs that both start "from scratch" may not start from the same weights. `snapshot_init.py` freezes the exact state before PPO begins.

### Derived properties (read-only)

Every `AugmentationEntry` exposes:
- `entry.agent_mode` → `"compressor"` | `"mcts_compressor"` | `"raw"` | `"llm_summary"`
- `entry.compressor_type` → `"structured_selective"` | `"mcts_gat"` | `"identity"` | …
- `entry.requires_mcts` → `True` for `mcts_only`, `tgad_*`
- `entry.has_checkpoint` → `True` when `checkpoint_path` is set

Scripts use these to automatically wire the correct agent mode and compressor factory.

---

## Snapshotting an Init

```bash
# SSD init (random weights):
python scripts/snapshot_init.py --aug-id ssd-init-001 --compressor structured_selective

# TGAD init from supervised pre-training:
python scripts/snapshot_init.py \
    --aug-id tgad-init-001 \
    --compressor mcts_gat \
    --from-checkpoint outputs/supervised/tgad_supervised.pt \
    --description "TGAD after supervised pretraining on LLMMCTSCompressor outputs"

# Transformer init with LoRA:
python scripts/snapshot_init.py \
    --aug-id transformer-init-001 \
    --compressor transformer \
    compressor.use_lora=true compressor.lora.r=8
```

Output:
- Weights saved to `outputs/augmentations/{aug_id}/compressor.pt`
- Entry added to `data/registry/augmentations.json`

---

## Using IDs in CLI Scripts

```bash
# Episode generation
python scripts/run_episode.py \
    agent.prompt_id=sweep_D \
    project.augmentation_id=ssd-init-001

# RL training from init snapshot
python scripts/run_training.py \
    agent.prompt_id=v2 \
    project.augmentation_id=tgad-init-001 \
    training=ppo_mcts \
    project.register_as=tgad-trained-001   # auto-registers final checkpoint

# RL training starting from a trained checkpoint (continue training)
python scripts/run_training.py \
    agent.prompt_id=v2 \
    project.augmentation_id=tgad-trained-001 \
    training=ppo_mcts
```

### Backward compatibility

All existing Hydra-only invocations continue to work unchanged. Registry IDs are resolved only when `agent.prompt_id` or `project.augmentation_id` is set; otherwise the existing `agent.system_prompt_version` + `compressor.type` path runs as before.

---

## Auto-Registering Trained Checkpoints

Set `project.register_as=<new_aug_id>` at the end of any training run to automatically:
1. Copy the final compressor weights to `outputs/augmentations/<new_aug_id>/compressor.pt`
2. Register the entry in `augmentations.json` with type derived from the init type
3. Record `parent_init_id` (the init snapshot this run started from) and `training_run_id`

```bash
python scripts/run_training.py \
    agent.prompt_id=v2 \
    project.augmentation_id=ssd-init-001 \
    training=ppo_colab \
    project.register_as=ssd-trained-001
```

---

## Using IDs in the Colab Notebook

In `notebooks/05_colab_rl_training.ipynb` Section 1:

```python
PROMPT_ID        = "v2"           # prompt registry ID
AUGMENTATION_ID  = "tgad-init-001"  # augmentation registry ID
TRAINING_CONFIG  = "ppo_mcts"     # ppo_colab | ppo_mcts | ppo_debug
REGISTER_AS      = None           # set to auto-register final checkpoint
```

A registry preview cell immediately below shows the full metadata for both IDs before you launch any training.

For plain episode generation (no training):
```python
!python {REPO_DIR}/scripts/run_episode.py \
    agent.prompt_id={PROMPT_ID} \
    project.augmentation_id={AUGMENTATION_ID}
```

---

## Listing the Registry

```bash
# All entries
python scripts/list_registry.py

# Filter to specific IDs
python scripts/list_registry.py --prompt-id sweep_D --aug-id ssd-init-001

# Prompts only / augmentations only
python scripts/list_registry.py --prompts-only
python scripts/list_registry.py --augs-only
```

---

## Example Combinations

| Use case | `prompt_id` | `augmentation_id` | `TRAINING_CONFIG` |
|---|---|---|---|
| Raw baseline episode | `v2` | `raw-default` | — |
| LLM summary baseline | `v2` | `llm-summary-default` | — |
| MCTS-only episode | `v2` | `mcts-default` | — |
| Train SSD from init | `v2` | `ssd-init-001` | `ppo_colab` |
| Train TGAD from init | `v5` | `tgad-init-001` | `ppo_mcts` |
| Run trained SSD | `sweep_D` | `ssd-trained-001` | — |
| Run trained TGAD | `v2` | `tgad-trained-001` | — |

---

## Code Reference

| Symbol | File |
|---|---|
| `PromptRegistry`, `PromptEntry` | `src/…/core/registry.py` |
| `AugmentationRegistry`, `AugmentationEntry`, `AugmentationType` | `src/…/core/registry.py` |
| `RunSpec`, `ResolvedRunSpec` | `src/…/core/run_spec.py` |
| `resolve_run_spec()` | `src/…/core/run_spec.py` |
| `apply_run_spec_to_cfg()` | `src/…/core/run_spec.py` |
| `build_compressor_from_entry()` | `src/…/core/run_spec.py` |
| `snapshot_init.py` | `scripts/snapshot_init.py` |
| `list_registry.py` | `scripts/list_registry.py` |
