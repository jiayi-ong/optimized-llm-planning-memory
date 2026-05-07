# RL Training Configuration Rationale

This document explains the PPO hyperparameter choices for the low-data training regime
(1–3k total timesteps) that results from the LLM API cost and latency constraints of
this project. It covers the science behind each parameter, the three questions that
motivated the changes, and the val-monitoring design.

---

## Background: Why This Is "Low-Data RL"

Standard PPO benchmarks use millions of timesteps on fast simulation environments
(e.g., 1M steps at 10,000 env.steps/s ≈ 100 s wall-clock). Here:

- 1 RL timestep = 1 compression event = `steps_per_compression=2` LLM API calls
- Each LLM call ≈ 12 s on Colab T4 with GPT-4o-mini
- With `n_envs=2` and SubprocVecEnv, both envs run in parallel:
  - 1 wall-clock step ≈ 2 × 12 s = 24 s
  - T=1500 ≈ 1500 × 24 s ≈ 10 h on T4

This means:

| Budget | PPO updates | Total grad steps | Wall-clock (T4) |
|--------|-------------|-----------------|-----------------|
| T=600  | ~18         | ~288            | ~4 h            |
| T=1000 | ~31         | ~496            | ~6.7 h          |
| T=1500 | ~46         | ~736            | ~10 h           |
| T=2000 | ~62         | ~992            | ~13 h (risk)    |
| T=3000 | ~93         | ~1488           | ~20 h (Pro only)|

*Update counts assume n_steps=16, n_envs=2 → rollout=32 transitions.*
*Grad steps = PPO updates × n_epochs(4) × mini-batches(4) = updates × 16.*

---

## Q1: Val/Test Split Utilization

### Finding

The generator produces three splits (`train/`, `val/`, `test/`) but only `train/` is
consumed during RL training (`run_training.py:54–66`) and `test/` during evaluation
(`run_evaluation.py`). The `val/` directory was generated but never read by any code.

### Fix

`ValEvalCallback` (added to `training/trainer.py`) runs `n_val_episodes=4` complete
episodes on the val split every `val_eval_freq=100` timesteps and logs:

- `val/mean_reward`
- `val/hard_constraint_score`
- `val/soft_constraint_score`

These appear as a separate TensorBoard series (orange, vs. blue for train) and signal
generalization failure (train reward up, val reward flat → overfitting to training requests).

**API overhead**: 15 val events × 4 episodes × ~15 LLM calls ≈ 900 extra calls (~15%
over T=1500). Worthwhile for the diagnostic signal.

**Implementation note**: `ValEvalCallback` always uses `DummyVecEnv` (not `SubprocVecEnv`)
so it evaluates with the live policy weights in the main process — the same weights
that PPO just updated. SubprocVecEnv forks worker processes with stale weights.

### When val split is absent

If `data/user_requests/val/` is empty, `run_training.py` logs
`no val requests — val eval disabled` and skips the callback silently. No error.

---

## Q2: Complexity Tier Configuration

### The Five Dimensions (`src/core/complexity.py`)

| Dimension | Weight | Score=1.0 when | Training impact |
|-----------|--------|----------------|-----------------|
| Constraint density | 25% | ≥5 constraints/day | Sparser reward; more simultaneous objectives |
| Geographic complexity | 25% | ≥3 destination cities | Longer paths; more tool calls per episode |
| Budget tightness | 20% | <$50/traveler/day | Narrower feasible space; more retries |
| Group complexity | 15% | children + access. + dietary | Cross-cutting implicit constraints |
| Preference specificity | 15% | ≥6 soft constraints | More failure dimensions |

Tier thresholds: `low` (<0.35), `medium` (0.35–0.65), `high` (≥0.65).

Empirical distribution at `project.seed=42`, n=90: **66% low, 34% medium, 0% high**.
The generator does not naturally produce high-complexity requests at this seed.

### Recommendations for 1–3k Step Training

**Use low+medium only during training.**

High-complexity episodes have sparser rewards (more simultaneous constraints fail before
the first partial success). With only 67–200 total episodes, the compressor sees too few
non-zero-reward high-complexity episodes to learn anything from them. They just add noise.

**Geographic complexity is the dominant driver (25% weight).** A 2-city request alone
scores geo=0.5, pushing composite toward medium. For the cleanest training signal at
T=1000, use single-city requests (set `geographic_complexity="single_city"` in the
generator, or filter by `metadata.complexity_breakdown.geographic_complexity`).

**Episode length control**: Setting `env.max_agent_steps=20` (reduced from 30) cuts
wasted API calls on truncated high-step episodes. At complexity<0.65, optimal solutions
are found within 15–20 agent steps, so the ceiling is appropriate.

**Test set should include high complexity** for evaluation validity. The test distribution
should represent the deployment difficulty, not the training distribution.

---

## Q3: PPO Configuration for 1–3k Total Timesteps

### Parameter-by-Parameter Justification

#### `n_envs=2` — keep

SubprocVecEnv runs both envs in parallel at no extra wall-clock cost. Going to 3 adds
50% more API calls with diminishing returns (smaller benefit from additional diversity).

#### `n_steps=16` — keep for all budgets

With rollout = 16 × 2 = 32 transitions:

- At T=1500: 46 PPO updates × 16 grad steps = 736 total gradient steps
- Increasing to n_steps=32 (rollout=64) halves the update count at the same T — you
  get better advantage estimates per update but fewer updates overall. The advantage
  quality gain doesn't outweigh the update frequency loss at this budget.
- n_steps=16 maximizes gradient steps per API dollar.

#### `batch_size=8` — keep

With rollout=32 and batch=8: 4 mini-batches per epoch, 4 independent gradient estimates
per PPO update cycle. The minimum batch for meaningful gradient averaging.

Increasing to 16 would give 2 mini-batches per epoch — less diversity per update.

#### `n_epochs=4` — keep

4 passes over the rollout buffer is the standard PPO value. More epochs (6+) risk KL
explosion when the rollout is small (32 samples): repeated gradient steps on the same
data without clipping protection can cause excessive policy changes. The prior
`approx_kl=7.019` incident validates being conservative here.

#### `clip_epsilon=0.2` — keep

Standard PPO clipping threshold. The prior instability was caused by
`path_encoder_dropout=0.1` in TGAD's PathSetEncoder, not by clip_epsilon. After
fixing dropout=0.0, clip=0.2 is stable.

Tighter clipping (0.15) would slow learning unnecessarily at this budget.

#### `learning_rate=3.0e-5` — keep

Conservative but appropriate. With ~46 total updates, a bad update (lr too high, large
gradient) is hard to recover from. The ppo_sweep.yaml grid starts here as the
"theory-backed baseline." The prior approx_kl issue was architecture-level, not LR-level.

#### `ent_coef=0.01 → 0.02` — increased

The compressed state action space is enormous (up to 512 tokens from a 32k vocab).
At initialization, the compressor generates near-random compression outputs.

An entropy bonus of 0.02 adds a small pressure to maintain action diversity, preventing
the policy from collapsing to one dominant compression template before rewards provide
directional signal. This is especially important in the first ~200 timesteps when
rewards may be sparse.

Reduce back to 0.01 if TensorBoard shows `episode/clip_fraction < 0.05` consistently
(policy not moving enough — entropy bonus is too weak relative to reward signal).

Raise to 0.05 if TensorBoard shows `episode/hard_constraint_score` flat for >10
consecutive updates with `clip_fraction > 0.25` (policy is changing but not improving —
exploration is helping, push more).

#### `max_grad_norm=1.0` — keep

The ppo_sweep.yaml explicitly marks this as "never relax — protects from instability."
The prior KL explosion was policy-level (ratio >> 1 per step), not gradient-level.
Gradient clipping is a separate protection layer that should always be active.

#### `gamma=0.99` — keep

High discount correctly values multi-step returns in this planning task. The compressor's
action (compressed state) affects rewards 10+ steps later (through the quality of the
agent's planning context). Lower gamma (0.95) would underweight the terminal bonus (5.0)
which is the strongest reward signal.

#### `gae_lambda=0.95` — keep

Standard for sparse-reward tasks. Balances variance (λ→0 = TD bootstrap, low variance,
high bias) and bias (λ→1 = full Monte Carlo, unbiased but high variance). 0.95 is the
PPO paper default, well-validated across many environments.

#### `lr_schedule=constant` — keep for 1–3k budgets

Linear decay halves the LR by the midpoint of training. At T=1500 midpoint (~750 steps),
the policy is often still in the early phase where it's just beginning to learn from
non-zero rewards. Decaying LR before meaningful improvement is premature.

Switch to `linear` only for T>10k where late-training fine-tuning benefits from a
smaller learning rate.

#### `normalize_advantage=false` — keep

With `batch_size=8`, normalizing advantage within a mini-batch of 8 samples gives
unreliable statistics (mean and variance of 8 values are noisy). At policy initialization
when rewards are near-zero, advantages are near-zero, and normalizing them amplifies
noise → policy_loss ≈ 0 with no gradient signal. Keep disabled.

---

## Summary Table: ppo_colab.yaml Changes

| Parameter | Before (600 steps) | After (1500 steps) | Reason |
|-----------|-------------------|--------------------|--------|
| `num_timesteps` | 600 | **1500** | 3× more grad steps (~736 vs 288); fits in Colab Pro session |
| `ent_coef` | 0.01 | **0.02** | Prevent premature action-space collapse |
| `checkpoint_every_n_steps` | 50 | **100** | 15 checkpoints over run (proportionally same density) |
| `env.max_agent_steps` | 30 | **20** | Appropriate for low/medium complexity; saves ~30% API calls |
| `val_eval_freq` | (new) | **100** | Val reward monitoring every 100 timesteps |
| `n_val_episodes` | (new) | **4** | 4 val episodes per event (~15% API overhead) |
| All other PPO params | unchanged | unchanged | Proven stable; no reason to change |

---

## Val Monitoring Interpretation Guide

| TensorBoard signal | Interpretation | Action |
|-------------------|----------------|--------|
| `val/mean_reward` trending up with `episode/total_reward` | Generalizing to val set | Continue training |
| `episode/total_reward` up, `val/mean_reward` flat/down | Overfitting to train requests | Consider early stopping at last good checkpoint |
| Both flat near 0 after 20+ updates | Policy stuck; reward signal too sparse | Check that reward is non-zero in episode logs; try `ent_coef=0.05` |
| `val/hard_constraint_score` ≈ `episode/hard_constraint_score` | Good transfer | Normal |
| Large gap (>0.3) between train and val hard_score | Hard overfit | Increase training set size or reduce n_train episodes per world |

---

## Related Files

| File | Role |
|------|------|
| `configs/training/ppo_colab.yaml` | Colab-optimized PPO config (the one you edit) |
| `configs/training/ppo_sweep.yaml` | Hyperparameter sweep grid reference |
| `src/training/trainer.py` | `ValEvalCallback`, `RLTrainer` |
| `src/core/config.py` | `TrainingConfig`, `PPOHyperparams`, `EnvConfig` |
| `src/core/complexity.py` | `RequestComplexityScorer` |
| `scripts/run_training.py` | Training entry point (loads train + val splits) |
| `notebooks/05_colab_rl_training.ipynb` | Colab training notebook |
