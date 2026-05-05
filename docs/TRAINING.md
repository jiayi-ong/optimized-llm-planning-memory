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
| `training/run_logger.py` | `RLRunLogger` — JSONL training diagnostics (per-episode + per-update) |
| `training/run_manifest.py` | `TrainingRunManifest` — full resolved config captured at run start |
| `simulator/world_pool.py` | `WorldPool` — pre-generated world pool to amortise per-episode cost |
| `utils/colab_utils.py` | Bundle/upload/download helpers for Colab + Google Drive sharing |

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
  → if agent signals DONE, EXIT(<code>), or max_steps reached: terminated=True
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

### Request sampling

`CompressionEnv.reset()` cycles through training requests in round-robin order (`itertools.cycle`) rather than sampling randomly. This eliminates within-rollout serial correlation that would bias GAE advantage estimates when consecutive episodes share the same request.

---

## Simulator World Pool

By default, `RLTrainer` pre-generates `pool_size` travel worlds at startup (configured in `configs/simulator/default.yaml`) rather than generating a fresh world for every `CompressionEnv.reset()` call.

```yaml
# configs/simulator/default.yaml
pool_size: 20              # worlds generated at startup
unique_per_episode: false  # true = fresh world per reset (slow, for debugging)
```

### Why a pool?

| Approach | Startup cost | Per-episode cost | Memory |
|---|---|---|---|
| WorldPool (default) | 20 worlds × ~0.5 s = ~10 s | `O(1)` random dict lookup | ~4 MB (20 × ~200 KB) |
| `unique_per_episode: true` | None | ~0.5 s per episode | Negligible |

With 4 parallel envs × 10k episodes = 40k resets, the unique approach wastes ~5.5 hours of GPU time waiting for world generation. The pool amortises that to a one-time 10-second startup.

### Thread safety

`WorldPool.build()` runs before SB3 forks subprocesses. Each worker process receives a pickled copy of the pool, so no locks are needed during training.

### Tuning pool_size

`pool_size` controls diversity of the training distribution. Too small → the policy overfits to a narrow world. Too large → startup cost grows. 20 is a good default for 50k-step Colab runs.

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

### Optional reward components

Three additional trip-quality signals can be toggled on via `configs/reward/*.yaml` without touching `reward.py`. They are disabled by default to preserve the standard training baseline.

| Component | Config key | Description |
|---|---|---|
| Destination coverage | `optional.destination_coverage` | Fraction of required destination cities that appear in the itinerary. |
| Activity density | `optional.activity_density` | Rewards itineraries that fill each day with 2–4 activities. |
| Budget adherence | `optional.budget_adherence` | Already implicit in `hard_constraint`; disabled by default to avoid double-counting. |

Enable one or more in a custom reward config:

```yaml
# configs/reward/coverage_heavy.yaml
optional:
  destination_coverage:
    enabled: true
    weight: 0.5
  activity_density:
    enabled: true
    weight: 0.3
```

Run with:
```bash
python scripts/run_training.py reward=coverage_heavy compressor=identity
```

Each enabled component is computed by delegating to `DeterministicEvaluator` — the same implementation used in evaluation, preserving the training-evaluation invariant.

**Ablation usage:** run once with `reward=default` and once with `reward=coverage_heavy` on the same seed set. Compare `hard_constraint_ratio` vs `destination_coverage_ratio` in the eval results to see the trade-off.

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

The value network uses token embeddings (not raw token IDs) to avoid feeding categorical integers directly to a linear layer. Padding tokens (ID=0) are masked out before mean-pooling so that long zero-padded observations do not dilute the pooled representation toward zero.

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
| `n_steps` | 64 | 256 | Steps per env before each PPO update |
| `batch_size` | 32 | 64 | PPO minibatch size |
| `n_epochs` | 4 | 10 | PPO gradient epochs per rollout |
| `learning_rate` | 3e-5 | 3e-4 | Adam LR (anneal with `lr_schedule`) |
| `gamma` | 0.99 | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | 0.95 | GAE λ |
| `clip_range` | 0.2 | 0.2 | PPO ε-clip |
| `ent_coef` | 0.01 | 0.01 | Entropy regularization |
| `vf_coef` | 0.5 | 0.5 | Value function loss weight |
| `normalize_advantage` | **false** | true | Set `false` to prevent policy_loss≈0 at init. When `true`, per-mini-batch normalization drives `mean(adv)→0`, which zeroes out `policy_loss` whenever `ratio≈1` (i.e. at initialisation and after small updates). Keep `false` for all Colab runs. |

### Hyperparameter tuning strategy

The starting point is the standard PPO recommendation for sparse-reward tasks. Sweep only these three:

| Parameter | Range to sweep | What it controls |
|---|---|---|
| `learning_rate` | `[1e-4, 3e-4, 1e-3]` | Training stability |
| `clip_range` | `[0.1, 0.2, 0.3]` | Policy update aggressiveness |
| `ent_coef` | `[0.0, 0.01, 0.05]` | Exploration vs exploitation |

Use the Colab config (50k steps, 2 envs) for all sweeps. Only promote to the full config once a good combination is found.

---

## GPU Device Configuration

`RLTrainer` detects the best available device automatically:

```yaml
# configs/training/ppo_default.yaml
device: auto   # "cuda" if GPU present, else "cpu"
```

Valid values: `"auto"`, `"cuda"`, `"cpu"`. Force CPU for debugging with `device: cpu` (see `configs/training/ppo_debug.yaml`).

After SB3's PPO initialises, the compressor's PyTorch parameters are explicitly moved to the selected device:

```python
self._compressor.to(self._device)   # graceful no-op for non-HF compressors
```

SB3 handles its own internal tensors via the `device` argument passed to `PPO(...)`.

---

## TensorBoard vs JSONL Diagnostics

Training produces two complementary log streams:

| Stream | Location | Best for |
|---|---|---|
| TensorBoard events | `outputs/logs/` | Live monitoring in Colab during training (`%tensorboard --logdir outputs/logs`) |
| JSONL logs | `outputs/training/<run_id>/` | Offline analysis: Streamlit Training Dashboard, `notebooks/08_run_comparison.ipynb`, pandas |

The JSONL format — two files per run — does not require TensorBoard to read and works identically in Colab and locally:

```
outputs/training/<run_id>/
├── manifest.json            # full resolved config
├── episode_metrics.jsonl    # one JSON line per episode
└── ppo_metrics.jsonl        # one JSON line per PPO update cycle
```

Load them directly in Python:

```python
from optimized_llm_planning_memory.training.run_logger import load_episode_metrics, load_ppo_metrics

episodes = load_episode_metrics(run_id, training_dir="outputs/training")
updates  = load_ppo_metrics(run_id, training_dir="outputs/training")
```

---

## Run Manifests

Every training run writes a `manifest.json` at startup, capturing the full resolved Hydra config alongside run metadata:

```json
{
  "run_id": "20260501_120000",
  "run_name": "alice_identity_v1",
  "git_sha": "abc123",
  "compressor_type": "identity",
  "agent_mode": "compressor",
  "reward_weights": { "hard_constraint": 2.0, "soft_constraint": 1.0, ... },
  "ppo_hyperparams": { "learning_rate": 3e-4, "clip_epsilon": 0.2, ... },
  "n_envs": 2,
  "num_timesteps": 50000,
  "n_train_requests": 40,
  "checkpoint_dir": "outputs/checkpoints",
  "created_at": "2026-05-01T12:00:00+00:00"
}
```

### Reading manifests

```python
from optimized_llm_planning_memory.training.run_manifest import (
    load_manifest, list_manifests, resolve_checkpoint,
)

# All manifests, newest first
for m in list_manifests("outputs/training"):
    print(m.run_id, m.compressor_type, m.run_name)

# Load one specific run
manifest = load_manifest("20260501_120000", training_dir="outputs/training")

# Find checkpoint path automatically
ckpt = resolve_checkpoint("20260501_120000", output_dir="outputs")
# → Path("outputs/checkpoints/final/ppo_model.zip")  or a numbered zip
```

### Using manifests for evaluation

```bash
# Auto-resolves checkpoint + compressor type from manifest
python scripts/run_evaluation.py \
    +run_id=20260501_120000 \
    eval.deterministic_only=true
```

See [docs/EVALUATION.md — Evaluating from a Training Run ID](EVALUATION.md#evaluating-from-a-training-run-id).

### Run comparison

`notebooks/08_run_comparison.ipynb` loads manifests and JSONL from multiple runs and produces:
- A ranked summary table (sorted by final hard constraint score).
- Side-by-side reward curves for all metrics.
- PPO convergence diagnostics (KL divergence, clip fraction, explained variance).
- A CSV export of the summary table.

---

## Artifact Storage Strategy

Full `EpisodeLog` JSON files are large (~40 KB each). At 10k training episodes, saving every one would consume ~400 MB. The default is to save none during training:

```yaml
# configs/training/ppo_default.yaml
episode_save_freq: 0   # 0 = never; 50 = save every 50th episode
```

The lean training log (JSONL) captures all diagnostics needed for post-training analysis in ~5 MB for 10k episodes.

Enable episode saving during debugging or when you need full trajectory replay:

```yaml
# configs/training/ppo_debug.yaml
episode_save_freq: 1   # save every episode (use for short runs only)
```

### Checkpoint cleanup

Intermediate checkpoint zips (`outputs/checkpoints/ppo_*_steps.zip`) accumulate throughout training. Only `outputs/checkpoints/final/` is needed for sharing. Delete the intermediates before bundling if Drive space is a concern:

```python
import glob, os
for z in glob.glob("outputs/checkpoints/ppo_*_steps.zip"):
    os.remove(z)
    print(f"Removed: {z}")
```

### Bundle and share

After training, package everything into a single `.tar.gz` for sharing with teammates or downloading locally:

```python
from optimized_llm_planning_memory.utils.colab_utils import bundle_run, upload_to_drive

bundle_path = bundle_run(run_id, output_dir="outputs", bundle_dir="outputs/bundles")
upload_to_drive(bundle_path, drive_dir="/content/drive/MyDrive/optllm_team_bundles")
```

See [docs/COLAB_GUIDE.md](COLAB_GUIDE.md) for the full team collaboration workflow.

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
| `train/policy_gradient_loss` | Non-zero, decreasing | **Exactly 0.0 every update** — means `normalize_advantage=True` is collapsing advantages; set `normalize_advantage: false` in the training config |
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

# MCTS-augmented run (higher API cost — each compression event calls the LLM for tree search)
# agent=react_mcts and compressor=llm_mcts must always be paired
!python scripts/run_training.py agent=react_mcts compressor=llm_mcts training=ppo_colab
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
