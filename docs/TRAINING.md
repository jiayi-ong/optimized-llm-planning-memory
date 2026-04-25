# Training — PPO Reinforcement Learning Pipeline

The RL training loop trains a compressor to produce `CompressedState` representations that maximize episode reward. The reward is shaped by constraint satisfaction, tool efficiency, and logical consistency — the same metrics used in evaluation.

---

## Files

| File | Role |
|---|---|
| `training/env.py` | `CompressionEnv` — Gymnasium environment wrapping the episode loop |
| `training/policy.py` | `CompressorPolicy` — SB3-compatible PPO policy |
| `training/reward.py` | `RewardFunction` — multi-component shaped reward |
| `training/trainer.py` | `RLTrainer` — orchestrates training, checkpointing, logging |
| `training/episode_buffer.py` | `EpisodeBuffer` — collects and batches completed episodes |
| `training/run_logger.py` | `RunLogger` — TensorBoard and structured JSONL logging |

---

## How the RL Loop Works

```
reset()
  → create fresh simulator (seeded world)
  → run first compress_every_n_steps ReAct steps (no compressed context)
  → return trajectory token IDs as observation

step(action)
  → decode action token IDs → CompressedState text
  → parse CompressedState (fallback to empty state if parse fails)
  → inject CompressedState into agent context
  → run next compress_every_n_steps ReAct steps
  → compute reward via RewardFunction
  → return updated trajectory tokens as next observation
  → if agent signals DONE or max_steps reached: terminated=True
```

The agent runs inside the environment. The **compressor** is the policy being trained. Each PPO action is the full text of a `CompressedState` (tokenized into the action vector).

---

## CompressionEnv

```python
class CompressionEnv(gymnasium.Env):
    # observation_space: Box(max_obs_tokens,)   int32 — trajectory token IDs
    # action_space:      Box(max_action_tokens,) int32 — compressed state token IDs
```

Key config parameters from `configs/training/*.yaml`:

| Parameter | Default | Effect |
|---|---|---|
| `max_obs_tokens` | 2048 | Observation tensor length (padded/truncated) |
| `max_action_tokens` | 512 | Action tensor length |
| `n_envs` | 4 | Parallel environments (vectorized via SB3 `make_vec_env`) |
| `seed_range` | `[0, 9999]` | Each env draws a unique simulator seed |

Each parallel environment uses a different random seed for the travel world, giving the policy diverse training episodes.

---

## Reward Function

`RewardFunction.compute()` returns a `RewardComponents` object:

```python
@dataclass
class RewardComponents:
    hard_constraint_score:    float  # [0, 1] — fraction of hard constraints satisfied
    soft_constraint_score:    float  # [0, 1] — weighted soft constraint satisfaction
    tool_efficiency_score:    float  # [0, 1] — 1 - redundant_calls / total_calls
    tool_failure_penalty:     float  # ≤ 0    — penalty proportional to failure rate
    logical_consistency_score: float # [0, 1] — date ordering + no double-bookings
    total_reward:             float  # weighted sum, optionally clipped to [-1, 1]
```

All constraint scoring is delegated to `ConstraintSatisfactionEngine` from `core/constraints.py` — the **same class** used by `DeterministicEvaluator`. Training reward ≡ evaluation metric.

### Reward Shaping

Reward weights are set in `configs/reward/default.yaml`:

```yaml
reward:
  weights:
    hard_constraint:    1.0
    soft_constraint:    0.3
    tool_efficiency:    0.2
    logical_consistency: 0.1
  step_penalty:         -0.01   # per-step cost to encourage efficiency
  terminal_bonus:       0.5     # added when episode ends with hard_score = 1.0
  clip_reward:          true    # clip total to [-1, 1]
```

To add a new reward component:

1. Add a field to `RewardComponents` in `core/models.py`.
2. Compute it in `RewardFunction.compute()` in `training/reward.py`.
3. Add the corresponding weight to `RewardConfig` in `core/config.py`.
4. Update `configs/reward/default.yaml`.
5. Check that `DeterministicEvaluator` also exposes the equivalent metric (see [docs/EVALUATION.md](EVALUATION.md)) so training and evaluation remain aligned.

---

## PPO Policy

`CompressorPolicy` is a custom SB3-compatible policy that wraps the compressor:

```
Observation tokens
    │
    ▼
Token Embedding (nn.Embedding)
    │
    ▼
Value MLP (3-layer) → V(s)  (critic)
    │
    ▼
Compressor.compress(trajectory) → action text
    │
    ▼
Tokenize action text → action tensor
    │
    ▼
Compressor.get_log_probs(trajectory, action) → log π(a|s)
```

The value network uses token embeddings (not raw token IDs) to avoid feeding categorical integers directly to a linear layer.

---

## RLTrainer

`RLTrainer` is the main training orchestrator:

```python
trainer = RLTrainer(config=project_config)
trainer.train()
```

Internally:
1. Loads user requests from `data/user_requests/train/`.
2. Creates `CompressionEnv` wrapped in `make_vec_env(n_envs=config.training.n_envs)`.
3. Initializes SB3 `PPO` with `CompressorPolicy`.
4. Trains for `config.training.num_timesteps` total steps.
5. Saves a checkpoint every `checkpoint_every_n_steps`:
   ```
   outputs/checkpoints/ppo_compressor_{step}_steps.zip
   outputs/checkpoints/ppo_compressor_{step}_steps/
       compressor/
       reward_predictor/
   ```
6. Saves a final checkpoint to `outputs/checkpoints/final/`.

### Resuming from a checkpoint

```bash
python scripts/run_training.py training=ppo_colab \
    training.resume_from=outputs/checkpoints/ppo_compressor_10000_steps.zip
```

The `resume_from` path is to the SB3 `.zip` file. The compressor weights are loaded from the matching subdirectory automatically.

---

## Checkpoint Layout

```
outputs/checkpoints/
├── ppo_compressor_5000_steps.zip        # SB3 full policy (optimizer + weights)
├── ppo_compressor_5000_steps/
│   ├── compressor/
│   │   └── {compressor_type}.pt         # compressor weights
│   └── reward_predictor/
│       └── reward_predictor.pt          # linear reward predictor state_dict
├── ppo_compressor_10000_steps.zip
├── ppo_compressor_10000_steps/
│   └── ...
└── final/
    ├── ppo_model.zip
    ├── compressor/
    └── reward_predictor/
```

---

## Hyperparameter Reference

Starting values are theory-backed for text-generation RL tasks:

| Parameter | Colab default | Full default | Notes |
|---|---|---|---|
| `num_timesteps` | 50,000 | 500,000 | Total env steps |
| `n_envs` | 2 | 4 | Parallel simulator instances |
| `n_steps` | 128 | 256 | Steps per env before each PPO update |
| `batch_size` | 32 | 64 | PPO minibatch size |
| `n_epochs` | 4 | 10 | PPO gradient epochs per rollout |
| `learning_rate` | 3e-4 | 3e-4 | Adam LR (anneal with `lr_schedule`) |
| `gamma` | 0.99 | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | 0.95 | GAE λ |
| `clip_range` | 0.2 | 0.2 | PPO ε-clip |
| `ent_coef` | 0.01 | 0.01 | Entropy regularization |
| `vf_coef` | 0.5 | 0.5 | Value function loss weight |

### Hyperparameter tuning strategy

The starting point is the standard PPO recommendation for sparse-reward tasks. Sweep only these three:

| Parameter | Range to sweep | What it controls |
|---|---|---|
| `learning_rate` | `[1e-4, 3e-4, 1e-3]` | Training stability |
| `clip_range` | `[0.1, 0.2, 0.3]` | Policy update aggressiveness |
| `ent_coef` | `[0.0, 0.01, 0.05]` | Exploration vs exploitation |

Use the Colab config (50k steps, 2 envs) for all sweeps. Only promote to the full config once a good combination is found.

---

## TensorBoard Metrics

```bash
# View live during training
%tensorboard --logdir outputs/logs   # in Colab
tensorboard --logdir outputs/logs    # local
```

| Tag | Healthy trend | Red flag |
|---|---|---|
| `episode/total_reward` | Rising | Flat after 10k steps |
| `episode/hard_constraint_score` | Rising toward 1.0 | Stuck at 0 |
| `episode/tool_efficiency_score` | Rising | Stuck at 0 (check API key) |
| `train/policy_gradient_loss` | Decreasing toward 0 | Exploding |
| `train/entropy_loss` | Slowly decreasing | Jumps to 0 (collapsed policy) |
| `train/value_loss` | Decreasing | Never decreasing (value net not learning) |
| `train/clip_fraction` | 0.1–0.2 | >0.3 (LR too high) |
| `train/approx_kl` | ~0.01 | >0.05 (large policy update, consider lower LR) |

---

## Google Colab

### One-time setup

```python
# 1. Set Runtime → GPU (T4)
# 2. Add secrets: ANTHROPIC_API_KEY and/or OPENAI_API_KEY

# 3. Clone repos
!git clone <simulator_url> /content/my-travel-world
!git clone <this_repo_url> /content/optimized-llm-planning-memory

# 4. Install
%cd /content/optimized-llm-planning-memory
!pip install -e ".[dev]" -q
!pip install -e "/content/my-travel-world" -q

# 5. Mount Drive so outputs survive session resets
from google.colab import drive
drive.mount('/content/drive')
!ln -sf /content/drive/MyDrive/optllm-outputs /content/optimized-llm-planning-memory/outputs

# 6. Verify API key is accessible
import os
from google.colab import userdata
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")

# 7. Smoke test (one 5-step episode)
!python scripts/run_episode.py agent.max_steps=5
```

### Training

```python
# Start TensorBoard BEFORE training
%load_ext tensorboard
%tensorboard --logdir outputs/logs

# Train (Colab-friendly config: 50k steps, 2 envs)
!python scripts/run_training.py compressor=identity training=ppo_colab
```

### Resume after session reset

```python
import glob, os
zips = sorted(glob.glob("outputs/checkpoints/*.zip"))
latest = zips[-1] if zips else None
print(f"Resuming from: {latest}")

!python scripts/run_training.py compressor=identity training=ppo_colab \
    training.resume_from={latest}
```

---

## Reward Predictor Diagnostics

`RewardPredictorComponent` fits a `torch.nn.Linear(5, 1)` model every 50 completed episodes, predicting episode reward from five scalar features (hard_constraint_score, soft_constraint_score, tool_efficiency, steps_taken, budget_adherence).

Its weights are saved at each checkpoint to `reward_predictor/reward_predictor.pt`.

To inspect which features correlate most with reward:
```python
import torch
ckpt = torch.load("outputs/checkpoints/final/reward_predictor/reward_predictor.pt")
weights = ckpt["weight"][0]
features = ["hard_constraint", "soft_constraint", "tool_efficiency", "steps", "budget"]
for f, w in zip(features, weights):
    print(f"{f}: {w:.3f}")
```

High absolute weight on `hard_constraint` is expected and validates that the compressor is learning something meaningful.

---

## EpisodeBuffer

`EpisodeBuffer` collects completed `EpisodeLog` objects and batches them for PPO updates. It is separate from SB3's internal rollout buffer.

When to use `EpisodeBuffer` vs. SB3's internal buffer:
- **SB3's buffer** is used automatically when calling `ppo.learn()`. It handles advantage computation (GAE) internally. Use this for standard training.
- **`EpisodeBuffer`** is available for custom training loops that need to inspect full episode logs (e.g., offline evaluation, reward reshaping after episode completion). It does **not** compute GAE — you must implement that separately if bypassing SB3.
