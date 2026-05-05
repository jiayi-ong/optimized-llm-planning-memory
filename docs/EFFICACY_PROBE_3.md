# Efficacy Probe 3 — TGAD PPO Bug 8 Investigation

**Date:** 2026-05-05  
**Branch:** `feat/baseline-prompt-sweep-v1`  
**Goal:** Identify root cause of `approx_kl=7.019` at first TGAD PPO update (observed in TGAD 640-step probe), confirm fix, and establish a general audit rule for SB3 PPO compressor stability.

---

## 1. Context

Efficacy Probe 2 confirmed that the PPO pipeline is end-to-end functional for the SSD (StructuredSelectiveDistiller) compressor after 7 bug fixes. The TGAD 640-step probe (`MCTSGraphAttentionDistiller`) ran concurrently but produced `approx_kl=7.019` and `policy_gradient_loss=3176` at its first PPO update — matching the symptom pattern of Bug 7 (LoRA dropout eval/train mismatch), despite `lora.dropout` already being set to 0.0.

This probe answers: **what is causing KL explosion in TGAD after Bug 7 is fixed?**

---

## 2. Investigation Strategy

Sequential static code analysis of the TGAD gradient path:

1. Start at the symptom (KL explosion at epoch 2) and eliminate candidate causes one at a time
2. Verify that `policy.py` log_prob paths are fully differentiable (rule out accidental `.detach()`)
3. Inspect every submodule with stochastic behavior inside `MCTSGraphAttentionDistiller`
4. Identify which module has a different output in eval mode vs train mode when N=1 (the `get_log_probs()` path)
5. Confirm the fix matches the exact same pattern as Bug 7

### Scope

- Files inspected: `policy.py`, `mcts_gat_distiller.py`, `tree_gat.py`, `trainer.py`
- No new training runs were performed during the investigation — diagnosis was done from TensorBoard data and code reading

---

## 3. Input Data: TGAD 640-step First PPO Update

TensorBoard event file: `outputs/logs/run_20260505_011329/PPO_1/events.out.tfevents.1777958034.jyo_work_device.39228.0`

| TensorBoard Key | Step 64 | Step 65 |
|----------------|---------|---------|
| `rollout/ep_rew_mean` | **0.465357** | — |
| `train/advantages_mean` | **0.351010** | — |
| `train/advantages_std` | **0.325** | — |
| `train/approx_kl` | 0.000000 | **7.019380** |
| `train/clip_fraction` | 0.000000 | **0.984375** |
| `train/policy_gradient_loss` | — | **3176.638916** |

Run config: `compressor=mcts_gat`, `n_steps=64`, `n_epochs=2`, `num_timesteps=640`, `batch_size=16`

---

## 4. Investigation Steps

### Step 1 — Verify policy.py gradient path

**Question:** Does `_generate_action()` have a `.detach()` that blocks gradients in epoch 1?

**Finding:** The `.detach()` at `policy.py:307` is only on the **exception fallback path** (line 306-308, inside `except Exception`). The normal path at line 293 computes `log_prob = token_log_probs.mean()` with no detach. The `evaluate_actions()` method (the PPO update path) also uses `.mean()` with no detach. **Not the root cause.**

### Step 2 — Interpret the step=64 / step=65 TensorBoard discrepancy

**Question:** Is `approx_kl=0.0` at step=64 a real zero-gradient epoch, or a logging artifact?

**Finding:** It is a logging artifact. The event chain is:

1. After 64 timesteps, SB3 calls `callback.on_rollout_end()` for all callbacks
2. `PPOUpdateMetricsCallback._on_rollout_end()` fires **BEFORE** `train()` (documented in trainer.py line 252 comment). It reads from `model._captured_log_values`, which was last updated by the most recent intermediate `logger.dump()` during the rollout — triggered by `EpisodeLogCallback._on_step()` at episode ends. Those intermediate dumps contain only episode metrics, no `train/` keys.
3. The callback falls back to defaults: `approx_kl=0.0`, `clip_fraction=0.0`
4. It calls `self.logger.record("train/approx_kl", 0.0)` — recording the stale default
5. **Then** `train()` runs, overwrites `train/approx_kl=7.019` into `name_to_value`
6. `train()` calls `logger.dump(64)` → `LoggerCapturePatch` intercepts, snapshots, calls `_orig_dump(64)` → TensorBoard step=64 receives whatever is in `name_to_value` at that moment

**Additional finding on `LoggerCapturePatch._capturing_dump()`:** Every call to `dump()` first calls `_captured.clear()`. Because `EpisodeLogCallback` calls `dump()` on every episode end during the rollout, `_captured` is cleared and repopulated with episode-only metrics multiple times before `train()` runs. This is why `ppo_metrics.jsonl` is empty: `_on_rollout_end()` always reads a `_captured` that contains no `train/` keys.

**Conclusion:** Step=64 `approx_kl=0.0` is the stale default written by `PPOUpdateMetricsCallback` before `train()` runs. Step=65 `approx_kl=7.019` is from SB3's `train()` calling `dump()`, which gets picked up and re-logged by the next `EpisodeLogCallback.dump()` at timestep 65. The **real first PPO update is `approx_kl=7.019`**.

### Step 3 — Check GumbelTopK behavior with N=1

**Question:** Does `GumbelTopK` in train mode produce different outputs than eval mode when `get_log_probs()` uses N=1 node (the fallback path)?

**Finding:** With N=1 node, `F.softmax([single_value], dim=0)` always returns `[1.0]` regardless of Gumbel noise or temperature — softmax of a single-element tensor is identically 1. Therefore `tree_context = 1.0 * node_embs[0] = node_embs[0]` in both modes. **GumbelTopK is not the root cause with N=1.**

### Step 4 — Inspect PathSetEncoder dropout

**Question:** Does `PathSetEncoder` have any stochastic behavior that differs between eval and train modes?

**Finding (tree_gat.py, lines 56-124):** `_MultiHeadAttentionLayer` has:
- `self._dropout = nn.Dropout(dropout)` applied to attention weights after softmax (line 111)
- `self._ff = nn.Sequential(..., nn.Dropout(dropout), ...)` in the feedforward sub-layer (line 78)

`PathSetEncoder` (`tree_gat.py:127`) instantiates two `_MultiHeadAttentionLayer` objects both with `dropout=0.1` (the value passed from `MCTSGraphAttentionDistiller.__init__()`).

**Finding (mcts_gat_distiller.py, line 272-274):**
```python
self._path_encoder = PathSetEncoder(
    dim=hidden_dim, num_heads=4, dropout=0.1
)
```

The value `0.1` is **hardcoded** in the constructor — not read from `mcts_gat.yaml`. The Bug 7 YAML fix (`lora.dropout: 0.0`) only affected the LoRA adapter; `PathSetEncoder` was never in the config.

**Mechanism with N=1:**
- Attention weight matrix is `[H, N, N] = [4, 1, 1]` — one scalar weight per head
- In eval mode: `dropout` is inactive → all 4 weights ≈ 1.0 after softmax
- In train mode: `nn.Dropout(0.1)` applied to `[4, 1, 1]` tensor → each of the 4 weights independently dropped with p=0.1, survivors scaled by `1/(1-0.1) = 1.111`
- Expected: ~3.6/4 heads survive, ~0.4 zeroed; the output is stochastic and scaled differently from eval mode

This is **exactly the Bug 7 mechanism** reproduced in a custom module: rollout (eval mode, deterministic) produces different log_probs than PPO update (train mode, stochastic) → old ≠ new → KL diverges.

---

## 5. Root Cause

**Bug 8: `PathSetEncoder(dropout=0.1)` hardcoded in `MCTSGraphAttentionDistiller.__init__()`.**

The fix for Bug 7 set `lora.dropout: 0.0` in `mcts_gat.yaml`, which resolved the eval/train mismatch in the T5 LoRA adapter. However, `PathSetEncoder` was instantiated with a hardcoded `dropout=0.1` not reachable from the config. SB3's `policy.set_training_mode()` propagates through the full `nn.Module` hierarchy (T5 model → LoRA adapters → `_path_encoder` → `_layer1` → `_dropout`), activating dropout on attention weights and feedforward layers in the path encoder during the PPO update phase.

This is a **generalization failure**: Bug 7 fixed one stochastic module (LoRA) but left another (`PathSetEncoder`) untouched because it was not in the config surface.

---

## 6. Fix Applied

Three files changed (commit `7204fa5`):

### 6.1 `src/.../compressor/mcts_gat_distiller.py`

Added `path_encoder_dropout: float = 0.0` parameter to `MCTSGraphAttentionDistiller.__init__()`:

```python
# Before
def __init__(
    self,
    ...,
    top_k_paths: int = 3,
) -> None:
    ...
    self._path_encoder = PathSetEncoder(
        dim=hidden_dim, num_heads=4, dropout=0.1
    ).to(self._device)

# After
def __init__(
    self,
    ...,
    top_k_paths: int = 3,
    path_encoder_dropout: float = 0.0,
) -> None:
    ...
    # dropout=0.0 for PPO stability: SB3 toggles training mode between rollout
    # (eval, no dropout) and update (train, dropout active). Any dropout>0 causes
    # a stochastic mismatch between old/new log_probs → KL explosion (same as Bug 7).
    self._path_encoder = PathSetEncoder(
        dim=hidden_dim, num_heads=4, dropout=path_encoder_dropout
    ).to(self._device)
```

### 6.2 `configs/compressor/mcts_gat.yaml`

Added:
```yaml
path_encoder_dropout: 0.0  # 0.0 for PPO stability: PathSetEncoder eval/train mismatch
                           # with dropout>0 causes approx_kl explosion (Bug 8, same root as Bug 7)
```

### 6.3 `scripts/run_training.py`

Added pass-through for the new parameter:
```python
compressor = MCTSGraphAttentionDistiller(
    ...
    top_k_paths=OmegaConf.select(cfg, "compressor.top_k_paths", default=3),
    path_encoder_dropout=OmegaConf.select(cfg, "compressor.path_encoder_dropout", default=0.0),
)
```

---

## 7. Raw Code Evidence

| Location | Finding |
|----------|---------|
| `policy.py:307` | `.detach()` only on exception fallback path — not in normal flow |
| `policy.py:185, 293` | Both use `.mean()` — Bug 3 fix confirmed intact |
| `mcts_gat_distiller.py:272-274` | `PathSetEncoder(dropout=0.1)` hardcoded, not from config |
| `tree_gat.py:73` | `self._dropout = nn.Dropout(dropout)` on attention weights |
| `tree_gat.py:78` | `nn.Dropout(dropout)` in feedforward sub-layer |
| `mcts_gat_distiller.py:177-207` | `GumbelTopK.forward()`: with N=1, softmax is always `[1.0]` — deterministic in both modes |
| `trainer.py:252` | Comment confirms `_on_rollout_end()` fires BEFORE `train()` |
| `trainer.py:263-269` | `LoggerCapturePatch._capturing_dump()` calls `_captured.clear()` every time — cleared by episode-end dumps, explaining empty `ppo_metrics.jsonl` |

---

## 8. Aggregate Summary

### Bug Comparison: Bug 7 vs Bug 8

| Attribute | Bug 7 (LoRA dropout) | Bug 8 (PathSetEncoder dropout) |
|-----------|---------------------|--------------------------------|
| Module | T5 LoRA adapter | Custom `PathSetEncoder` + `_MultiHeadAttentionLayer` |
| Dropout value | 0.05 | 0.1 |
| Config path | `mcts_gat.yaml lora.dropout` | Hardcoded in `__init__` (not in YAML before fix) |
| Compressors affected | Both TGAD and SSD | TGAD only (SSD has no PathSetEncoder) |
| `approx_kl` before fix | 18.433 | 7.019 |
| `clip_fraction` before fix | 1.000 | 0.984 |
| `policy_gradient_loss` before fix | 157 | 3176 |
| Mechanism | Rollout (eval) vs update (train) use different dropout masks | Same |
| Fix | `dropout: 0.0` in YAML | `path_encoder_dropout: float = 0.0` param added to constructor |

### Why SSD was not affected

`StructuredSelectiveDistiller` does not use `PathSetEncoder` or any equivalent custom attention module with dropout. Its architecture is: frozen T5 encoder → cross-attention (hardcoded `nn.MultiheadAttention`) → LoRA-patched T5 decoder. With `lora.dropout: 0.0` (Bug 7 fix), SSD has no remaining stochastic layers in the `nn.Module` tree. First update `approx_kl=0.372` confirmed stable.

### General Principle Established

Any `nn.Module` submodule within the compressor whose behavior differs between `model.eval()` and `model.train()` will produce a systematic old/new log_prob mismatch for SB3 PPO. This includes:
- `nn.Dropout` (found in LoRA, PathSetEncoder attention weights, PathSetEncoder feedforward)
- `nn.BatchNorm` (if present — not applicable here)
- Stochastic sampling layers (GumbelTopK in train mode — NOT an issue with N=1, but would be with N>1)

**The fix is always the same: set all internal dropout to 0.0 when using SB3 PPO.**

---

## 9. Steps to Replicate and Verify the Fix

### 9.1 Prerequisites

```bash
# Install dependencies
pip install -e ".[dev]"

# Set API key
export OPENAI_API_KEY=<key>

# Confirm world exists
python scripts/generate_world.py +simulator.seed=42
```

### 9.2 TGAD 640-step Re-run (with Bug 8 fix)

```bash
python scripts/run_training.py training=ppo_probe compressor=mcts_gat agent=react_probe \
  +simulator.world_id=world_42_20260504_075144
```

`mcts_gat.yaml` now has `path_encoder_dropout: 0.0` — no extra overrides needed.

**Expected first PPO update (timestep 64):**
- `approx_kl < 0.5` (matching SSD's `0.372`)
- `clip_fraction < 0.7`
- `policy_gradient_loss < 1.0`

**How to check (TensorBoard):**
```bash
tensorboard --logdir outputs/logs/
```
Navigate to `train/approx_kl` scalar. The real first update value is logged at the step AFTER `rollout/ep_rew_mean` first appears (due to the `PPOUpdateMetricsCallback` timing artifact — step N shows stale 0.0, step N+1 shows real value).

**How to check (stdout log):**  
Look for `"policy_gradient_loss"` in the log file at `outputs/tgad_640_probe.log` (or the new run's log). The `train/` block may not appear in stdout if `EpisodeLogCallback.dump()` races ahead — rely on TensorBoard for authoritative values.

### 9.3 Quick Smoke Test (16 timesteps, ~20 min CPU)

```bash
python scripts/run_training.py training=ppo_probe compressor=mcts_gat agent=react_probe \
  +simulator.world_id=world_42_20260504_075144 \
  training.num_timesteps=16 training.ppo.n_steps=8 training.ppo.n_epochs=2 \
  training.ppo.batch_size=4 training.env.steps_per_compression=3 training.env.max_agent_steps=9
```

This is the same command as the Efficacy Probe 2 TGAD smoke test, but now with Bug 8 fixed. Expected to replicate SSD's stability profile: `approx_kl < 0.5`, `training.complete` within 20 minutes.

---

## 10. Conclusions

1. **Bug 8 is the TGAD-specific extension of Bug 7.** The LoRA dropout YAML fix was necessary but not sufficient — `PathSetEncoder` had its own `dropout=0.1` unreachable from config, producing the same eval/train log_prob mismatch in TGAD's path attention layers.

2. **The `approx_kl=0.0` reading at TensorBoard step 64 was a logging artifact**, not a real zero-gradient epoch. `PPOUpdateMetricsCallback._on_rollout_end()` fires before `train()` and records stale default values. The real first PPO update KL was 7.019 (from `train()`'s own `logger.dump()`). This explains why `ppo_metrics.jsonl` is always empty: by the time `_on_rollout_end()` fires, `LoggerCapturePatch._captured` has been wiped by intermediate episode-end `dump()` calls.

3. **GumbelTopK is not a stability risk in the `get_log_probs()` path.** With N=1 node (the fallback path used by `evaluate_actions()`), softmax is always `[1.0]` regardless of Gumbel noise — both modes produce identical outputs. This is a non-issue.

4. **The general audit rule for SB3 PPO compressors:** Enumerate every `nn.Module` in the compressor's submodule tree with `model.modules()`. For each `nn.Dropout` (or equivalent), verify `p=0.0` or that the module is not reachable from the `nn.Module` tree toggled by `set_training_mode()`. Stochastic layers outside the module tree (e.g., a separate torch.no_grad() generation call) are safe.

5. **ep_rew_mean=0.465 and advantages_mean=0.351 are healthy**, confirming the reward signal is live and non-trivial even before any stable gradient step. The instability is entirely in the gradient computation path, not the environment or reward function.

---

## 11. Recommendations

### Immediate

1. **Kill the current TGAD 640-step probe** (PID 39228 or Ctrl-C) — it ran without Bug 8 fix and its further updates will all be unstable.

2. **Re-run the TGAD 640-step probe** with the fixed config:
   ```bash
   python scripts/run_training.py training=ppo_probe compressor=mcts_gat agent=react_probe \
     +simulator.world_id=world_42_20260504_075144
   ```
   First update should show `approx_kl < 0.5`.

3. **Fix `ppo_metrics.jsonl` logging** (optional, non-blocking): `PPOUpdateMetricsCallback._on_rollout_end()` reads stale captured values because episode-end `dump()` calls wipe `_captured` before `train()` runs. The fix is to move the callback's JSONL write to `_on_training_start` + a `train()` hook, or to read directly from the rollout buffer / optimizer after `train()` completes. For now, TensorBoard is the authoritative source for train/ metrics.

### For Colab GPU Training

The complete set of required settings for stable TGAD PPO (all 8 bugs addressed):

| Config Key | Required Value | Bug Fixed |
|------------|---------------|-----------|
| `training.max_action_tokens` | `512` | Bug 1 |
| `training.normalize_advantage` | `true` | Bug 2 |
| `training.n_epochs` | `2` | Bug 7 (KL compounding) |
| `training.learning_rate` | `3e-6` | Bug 7 (LoRA stability) |
| `training.target_kl` | `0.5` | Guard |
| `compressor.lora.dropout` | `0.0` | Bug 7 |
| `compressor.path_encoder_dropout` | `0.0` | **Bug 8** |

### Debugging Checklist (for future KL explosion)

When `approx_kl > 1.0` at any PPO update:

1. Run `model.modules()` and print all `nn.Dropout` instances — check `p > 0`
2. Is `lora.dropout = 0.0`? (Bug 7)
3. Is `path_encoder_dropout: 0.0` in `mcts_gat.yaml`? (Bug 8)
4. Are there any `BatchNorm` layers in the compressor submodule tree?
5. Does `GumbelTopK` use stochastic noise when N > 1? (potential Bug 9 if TGAD is used with multi-path rollouts in the future)
6. Is `n_epochs > 2`? Reduce if KL compounds across epochs.
7. Is `learning_rate > 3e-6` for T5 LoRA? Reduce by 3–5× if yes.
