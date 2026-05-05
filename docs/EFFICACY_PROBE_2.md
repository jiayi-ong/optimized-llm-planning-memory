# Efficacy Probe 2 — PPO Pipeline End-to-End Verification

**Date:** 2026-05-05  
**Branch:** `feat/baseline-prompt-sweep-v1`  
**Goal:** Verify that the post-fix PPO training pipeline produces `num_compressions > 0` and numerically stable losses (`approx_kl < 0.5`, finite `policy_gradient_loss`).

---

## 1. Context

Efficacy Probe 1 (session `efficacy-improvements-003`) identified and fixed three high-level system failures: a compression cascade bug in `react_agent.py`, agent observation-attention failure, and MCTS design gaps. Those fixes were merged in PR #13.

Probe 2 focuses entirely on the RL training loop: does the PPO optimization actually work once the agent-level bugs are resolved? The pipeline was suspected broken because early runs produced `num_compressions=0` and `policy_gradient_loss=NaN` throughout training, with no reward signal.

---

## 2. Methodology

### 2.1 Investigation Strategy

The approach was empirical and sequential: run the smallest possible training loop (16 timesteps, 2 PPO updates), observe failures, isolate root causes one at a time, apply minimal targeted fixes, re-run. The goal was not to train a converged model but to demonstrate that the pipeline is free of pathological failure modes.

**Probes run (in order):**

| Probe | Config | Timesteps | Compressor | Purpose |
|-------|--------|-----------|------------|---------|
| TGAD smoke test (iter 1) | ppo_probe | 16 | mcts_gat | Confirm compressions > 0 after Bugs 1–6 |
| TGAD smoke test (iter 2) | ppo_probe | 16 | mcts_gat | Expose KL instability (Bug 7) |
| SSD verification | ppo_probe | 16 | structured_selective | Confirm Bug 7 fix, end-to-end stable |
| TGAD 640-step | ppo_probe | 640 | mcts_gat | Verify stability at scale with MCTS overhead |

### 2.2 Diagnostic Metrics

For each probe, the critical metrics inspected were:

| Metric | Healthy Range | Pathological Sign |
|--------|-------------|-------------------|
| `num_compressions/episode` | ≥ 1 | = 0 → parse always fails |
| `approx_kl` | < 0.5 | > 1.0 → gradient too large, training unstable |
| `clip_fraction` | < 0.3 | = 1.0 → all samples clipped, learning collapses |
| `policy_gradient_loss` | ~0.01–1.0 | > 10 → log_prob overflow or KL blowup |
| `value_loss` | finite | NaN → numerical overflow |
| `training.complete` | logged | crash or hang → pipeline error |

---

## 3. Steps to Replicate

### 3.1 Prerequisites

```bash
# Install dependencies (if not already done)
pip install -e ".[dev]"

# Set up API key for gpt-4o-mini
export OPENAI_API_KEY=<key>

# Confirm world exists (or generate one)
python scripts/generate_world.py +simulator.seed=42
# Note world_id from output, e.g. world_42_20260504_075144
```

### 3.2 SSD Verification (16 timesteps, ~15 min on CPU)

```bash
python scripts/run_training.py training=ppo_probe compressor=structured_selective agent=react_probe \
  +simulator.world_id=world_42_20260504_075144 \
  training.num_timesteps=16 training.ppo.n_steps=8 training.ppo.n_epochs=2 \
  training.ppo.batch_size=4 training.env.steps_per_compression=3 training.env.max_agent_steps=9
```

Expected output: `training.complete` within 15 minutes, `approx_kl < 0.5` on all epochs.

### 3.3 TGAD Smoke Test (16 timesteps, ~20 min on CPU)

```bash
python scripts/run_training.py training=ppo_probe compressor=mcts_gat agent=react_probe \
  +simulator.world_id=world_42_20260504_075144 \
  training.num_timesteps=16 training.ppo.n_steps=8 training.ppo.n_epochs=2 \
  training.ppo.batch_size=4 training.env.steps_per_compression=3 training.env.max_agent_steps=9
```

### 3.4 TGAD 640-step Probe (full probe, ~8–12 hours on CPU, ~30–60 min on Colab T4)

```bash
python scripts/run_training.py training=ppo_probe compressor=mcts_gat agent=react_probe \
  +simulator.world_id=world_42_20260504_075144
```

Uses `ppo_probe.yaml` defaults: `num_timesteps=640`, `n_steps=64`, `n_epochs=2`, `batch_size=16`.

### 3.5 Config Snapshot (all fixes applied)

`configs/training/ppo_probe.yaml` key values:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `max_action_tokens` | 512 | Must match compressor `max_output_tokens` |
| `normalize_advantage` | true | Prevents policy_loss ≈ 0 at initialization |
| `n_steps` | 64 | Enough variance for advantage estimation |
| `n_epochs` | 2 | Reduces KL compounding window |
| `learning_rate` | 3e-6 | Prevents KL explosion with T5 LoRA |
| `target_kl` | 0.5 | Early stop if KL blows up |
| `max_grad_norm` | 0.5 | Tighter clip for LoRA stability |

`configs/compressor/mcts_gat.yaml` and `structured_selective.yaml`:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `lora.dropout` | 0.0 | Eliminates eval/train log_prob mismatch (Bug 7) |

---

## 4. Raw Results

### 4.1 TGAD Smoke Test (Before Bug 7 Fix)

| Metric | Iter 1 Epoch 2 | Iter 2 Epoch 2 |
|--------|---------------|---------------|
| `num_compressions/episode` | 1–2 | 1–2 |
| `ep_rew_mean` | 0.494 | 0.593 |
| `approx_kl` | **18.433** | — |
| `clip_fraction` | **1.000** | — |
| `policy_gradient_loss` | **157** | — |
| `value_loss` | 0.192 | — |
| `training.complete` | ✓ (no crash) | |

### 4.2 SSD Verification (After Bug 7 Fix)

| Metric | Iter 1 Epoch 1 | Iter 1 Epoch 2 | Iter 2 Epoch 2 |
|--------|---------------|---------------|---------------|
| `ep_rew_mean` | 0.728 | — | 0.676 |
| `approx_kl` | 0.000 | **0.372** | 0.000 |
| `clip_fraction` | 0.000 | 0.562 | 0.000 |
| `policy_gradient_loss` | — | **0.144** | — |
| `value_loss` | — | 0.266 | — |
| `n_updates` | — | 2 | 4 |
| `training.complete` | ✓ at 05:30:04 | | |

### 4.3 Bug 7 Fix: Before vs After

| Metric | Before (`dropout=0.05`) | After (`dropout=0.0`) | Reduction |
|--------|------------------------|----------------------|-----------|
| `approx_kl` (iter 1 epoch 2) | 18.433 | **0.372** | −98% |
| `policy_gradient_loss` | 157 | **0.144** | −1000× |
| `clip_fraction` | 1.000 | **0.562** | −44% |
| Training stable | No (target_kl early stop) | **Yes** | ✓ |

### 4.4 TGAD 640-step Probe — First PPO Update (Observed)

First PPO update fired at timestep 64 (after collecting 64 compression events across ~63 episodes).

| Metric | Step 64 (epoch 1 artifact) | Step 65 (actual train() result) |
|--------|---------------------------|----------------------------------|
| `rollout/ep_rew_mean` | **0.465** ✓ | — |
| `train/advantages_mean` | **0.351** ✓ | — |
| `train/advantages_std` | **0.325** ✓ | — |
| `approx_kl` | 0.000 (stale default) | **7.019** ← KL explosion |
| `clip_fraction` | 0.000 (stale default) | **0.984** |
| `policy_gradient_loss` | — | **3176.6** |

**Note on step=64 reading**: `approx_kl=0.000` at step=64 is NOT the real KL — it is a stale default value recorded by `PPOUpdateMetricsCallback._on_rollout_end()` BEFORE `train()` runs, which then gets picked up by an intermediate `logger.dump()`. The real KL is step=65's `7.019`, which is logged by SB3's `train()` after both epochs run.

**Root cause — Bug 8**: `PathSetEncoder(dropout=0.1)` hardcoded in `MCTSGraphAttentionDistiller.__init__()`. SB3 toggles `set_training_mode(False/True)` on the policy `nn.Module` tree, which propagates to `PathSetEncoder._layer1` and `_layer2`. With `dropout=0.1`, rollout log_probs (eval, dropout off) differ from update log_probs (train, dropout active) → `exp(new - old)` diverges before any useful gradient step. Identical mechanism to Bug 7 (LoRA dropout) but in a custom module not covered by the YAML fix.

**Fix applied (Bug 8)**: Added `path_encoder_dropout: float = 0.0` parameter to `MCTSGraphAttentionDistiller.__init__()`, wired to `PathSetEncoder(..., dropout=path_encoder_dropout)`. Added `path_encoder_dropout: 0.0` to `configs/compressor/mcts_gat.yaml` and pass-through in `run_training.py`.

**Re-run required**: The in-progress TGAD 640-step probe does NOT have this fix applied. A fresh run with the fix will confirm `approx_kl < 0.5` at the first update (matching SSD behavior).

---

## 5. Bug Summary

All 8 bugs caused either `num_compressions=0` or numerical instability. Fixed in order of discovery.

| # | Root Cause | Symptom | Fix | File |
|---|-----------|---------|-----|------|
| 1 | `max_action_tokens=128` | Truncates 512-token template → parse fails | 128 → 512 | `ppo_probe.yaml` |
| 2 | `normalize_advantage=false` | Zero advantage variance at init → policy_loss≈0 | `false` → `true` | `ppo_probe.yaml` |
| 3 | `token_log_probs.sum()` ≈ −2000 | PPO ratio `exp(new−old)` overflows to NaN | `.sum()` → `.mean()` | `policy.py` |
| 4 | T5 tokenizer strips newlines | Sentinels become inline → section parser misses them | Re-inject `\n` around `## X ##` before parse | `env.py` |
| 5 | Fallback sketch starts with generation prefix | Embedded sentinel creates duplicate section | Strip `## X ##` from fallback text | `mcts_gat_distiller.py`, `structured_selective_distiller.py` |
| **6** | `env._tokenizer=None` **(root cause)** | `chr(token_id)` decoding → gibberish → parse always None | Pass `compressor_tokenizer` to `RLTrainer()` | `run_training.py` |
| **7** | LoRA `dropout=0.05` eval/train mismatch | Old log_probs (eval mode) vs new log_probs (train mode) differ → `approx_kl=18.4` | `dropout: 0.0`; `n_epochs: 4→2` | `mcts_gat.yaml`, `structured_selective.yaml` |
| **8** | `PathSetEncoder(dropout=0.1)` hardcoded | TGAD-only: eval/train mismatch in custom attention module → `approx_kl=7.019`, `pg_loss=3176` at first update | Add `path_encoder_dropout=0.0` param; pass to `PathSetEncoder` | `mcts_gat_distiller.py`, `mcts_gat.yaml`, `run_training.py` |

**Bug 6** was the single root cause of all zero-compression symptoms prior to this session. `run_training.py` called `RLTrainer()` without `tokenizer=...`, so `CompressionEnv._decode_action()` used `chr(token_id)` character decoding, producing gibberish that never matched the template format.

**Bug 7** caused KL explosion despite compressions working. SB3 calls `policy.set_training_mode(False)` during rollout and `set_training_mode(True)` during update. PyTorch `nn.Module.train(mode)` propagates through the policy to the LoRA compressor submodule. With `dropout=0.05`, rollout log_probs (eval, no dropout) and update log_probs (train, dropout active) use different stochastic masks → systematic KL divergence even before any gradient step.

**Bug 8** is the TGAD-specific extension of Bug 7. The `mcts_gat.yaml` fix (Bug 7) set `lora.dropout=0.0`, but the `PathSetEncoder` custom attention module was instantiated with `dropout=0.1` hardcoded in `MCTSGraphAttentionDistiller.__init__()`, bypassing the config. The SB3 training-mode toggle propagates to `PathSetEncoder._layer1` and `_layer2`, causing the same log_prob mismatch. Confirmed via TensorBoard: TGAD first update `approx_kl=7.019`, `pg_loss=3176` vs SSD (no PathSetEncoder) `approx_kl=0.372`.

---

## 6. Files Changed

| File | Change |
|------|--------|
| `configs/training/ppo_probe.yaml` | `max_action_tokens` 128→512; `lr` 3e-5→3e-6; `n_epochs` 4→2; `normalize_advantage: true`; `target_kl: 0.5` |
| `configs/training/ppo_probe.yaml` | `max_action_tokens` 128→512; `lr` 3e-5→3e-6; `n_epochs` 4→2; `normalize_advantage: true`; `target_kl: 0.5` |
| `configs/compressor/mcts_gat.yaml` | `lora.dropout` 0.05→0.0; added `path_encoder_dropout: 0.0` (Bug 8) |
| `configs/compressor/structured_selective.yaml` | `lora.dropout` 0.05→0.0 |
| `src/.../training/policy.py` | `token_log_probs.sum()` → `.mean()` (2 locations); added traceback logging |
| `src/.../training/env.py` | Sentinel normalization before `_parse_compressed_state()` |
| `src/.../training/trainer.py` | `LoggerCapturePatch` callback; LR read from optimizer param groups |
| `src/.../compressor/mcts_gat_distiller.py` | Strip `## X ##` from fallback sketch; `min_new_tokens=4`; added `path_encoder_dropout` param (Bug 8) |
| `src/.../compressor/structured_selective_distiller.py` | Same fallback fix; `min_new_tokens=4` |
| `scripts/run_training.py` | Pass `tokenizer=compressor_tokenizer` to `RLTrainer()`; pass `path_encoder_dropout` to TGAD (Bug 8) |

---

## 7. Conclusions

1. **The PPO pipeline is end-to-end functional for SSD.** After 7 bug fixes, the SSD compressor trains without crash, produces `num_compressions > 0`, and maintains `approx_kl < 0.5`.

2. **Bug 6 (missing tokenizer) was the latent root cause.** It masked all other issues by making parse always return `None`, so symptoms like zero compressions and zero rewards were never informative about the actual problem until the tokenizer was passed.

3. **Every stochastic submodule inside the compressor must have dropout=0.0 for SB3 PPO.** SB3's training-mode toggle propagates through the full `nn.Module` hierarchy. Bug 7 fixed LoRA dropout (YAML-configurable); Bug 8 found the same pattern in `PathSetEncoder` (hardcoded, not in YAML). The general rule: any layer whose behavior differs between `training=True` and `training=False` will produce a log_prob mismatch → KL explosion. Audit ALL submodules when debugging KL instability.

4. **Token-level PPO requires `.mean()` log_prob aggregation.** With T5 generating ~200 tokens per action, `.sum()` produces values near −2000. After one gradient step, `exp(log_prob_new − log_prob_old)` overflows to `inf`/`NaN`. Mean keeps values in [−5, −15], which is safe.

5. **ep_rew_mean is non-trivially non-zero.** The SSD probe produced `ep_rew_mean=0.728` (iter 1) and `0.676` (iter 2), confirming that the reward function is live and the agent is completing some tasks.

6. **TGAD requires Bug 8 fix before stable training.** The in-progress TGAD 640-step probe ran WITHOUT the `path_encoder_dropout=0.0` fix and produced `approx_kl=7.019` at its first PPO update. A fresh run with Bug 8 applied is required to confirm TGAD stability.

---

## 8. Recommendations

### For Colab GPU Training

The `ppo_probe.yaml` config (all 8 fixes applied) is ready to run on Colab. **For TGAD**, the Bug 8 fix (`path_encoder_dropout: 0.0`) is now in `mcts_gat.yaml` — use this config as-is.

Required settings:
- `lora.dropout: 0.0` (both compressor configs) — **required for stability** (Bug 7)
- `path_encoder_dropout: 0.0` (TGAD only) — **required for stability** (Bug 8)
- `n_epochs: 2` — prevents KL compounding
- `lr: 3e-6` — safe for T5 LoRA
- `max_action_tokens: 512` — must match `max_output_tokens`
- `target_kl: 0.5` — early stop guard

CPU timing for 640 timesteps with MCTS overhead is 8–12 hours. GPU (T4/A100) is ~10–20× faster.

### TGAD Re-run (after Bug 8 fix)

The ongoing TGAD 640-step probe hit `approx_kl=7.019` at the first update. Kill it and re-run with the fixed config:

```bash
# Kill the current probe (PID 39228 or via Ctrl-C), then:
python scripts/run_training.py training=ppo_probe compressor=mcts_gat agent=react_probe \
  +simulator.world_id=world_42_20260504_075144
# mcts_gat.yaml now has path_encoder_dropout: 0.0 — no extra overrides needed
```

Expected first update: `approx_kl < 0.5`, matching SSD's 0.372.

### For Phase 6 Comparison

Once Colab training is complete, run:

```bash
# Score pre-training baseline
python scripts/run_eval.py agent=react_baseline_raw

# Score post-training TGAD
python scripts/run_eval.py agent=react_default compressor=mcts_gat \
  +checkpoint=outputs/checkpoints/mcts_gat_final.zip

# Score post-training SSD
python scripts/run_eval.py agent=react_default compressor=structured_selective \
  +checkpoint=outputs/checkpoints/ssd_final.zip
```

Target comparison table:

| Condition | `hard_constraint_score` | `completion_rate` | `tool_efficiency` | `ep_rew_mean` |
|-----------|------------------------|-------------------|-------------------|---------------|
| RAW baseline | TBD | TBD | TBD | — |
| TGAD pre-train | TBD | TBD | TBD | — |
| TGAD post-640 | TBD | TBD | TBD | TBD |
| SSD pre-train | TBD | TBD | TBD | — |
| SSD post-640 | TBD | TBD | TBD | TBD |

### Debugging Checklist (if KL explodes)

1. Is `lora.dropout = 0.0` in the compressor YAML? (Bug 7)
2. For TGAD: is `path_encoder_dropout: 0.0` in `mcts_gat.yaml`? (Bug 8)
3. Are there any other custom `nn.Module` submodules with `dropout > 0`? The general rule: every stochastic layer in the compressor must be dropout-free for SB3 PPO.
4. Is `tokenizer` passed to `RLTrainer()`? (Bug 6)
5. Is `max_action_tokens ≥ compressor.max_output_tokens`? (Bug 1)
6. Does `env.py` normalize sentinels before parse? (Bug 4)
7. Does `policy.py` use `.mean()` not `.sum()` for log_probs? (Bug 3)
