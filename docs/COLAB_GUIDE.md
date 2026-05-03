# Google Colab Collaboration Guide

How team members set up, run, and share training runs across Colab sessions — without stepping on each other's files.

---

## Mental Model

Each developer owns one or more **named training runs**. A run is identified by its `run_id` (a `YYYYMMDD_HHMMSS` timestamp) and captured in a `manifest.json` alongside the training JSONL logs. Runs are:

- **Stored in Colab** during the session (under `/content/optimized-llm-planning-memory/outputs/`).
- **Persisted to Google Drive** via a Drive symlink so they survive session resets.
- **Shared as bundles** — `.tar.gz` archives containing the JSONL logs, manifest, and final checkpoint — which teammates download and unpack locally for Streamlit and notebook analysis.

You do NOT need to share the full `outputs/` directory or push anything to git.

---

## Per-Developer Setup (one time per account)

### 1. Runtime — select GPU

`Runtime → Change runtime type → T4 GPU`

Training with `compressor=identity` on the T4 takes ~15 min for 50k steps.

### 2. Add API keys to Colab Secrets

`Tools → Secrets` → add:

| Secret name | Value |
|---|---|
| `ANTHROPIC_API_KEY` | `sk-ant-...` |
| `OPENAI_API_KEY` | `sk-...` |

This avoids pasting keys into cells. Keys are per-account and never shared with teammates.

### 3. Clone both repos

```python
!git clone https://github.com/<org>/my-travel-world /content/my-travel-world
!git clone https://github.com/<org>/optimized-llm-planning-memory /content/optimized-llm-planning-memory
```

### 4. Install

```python
%cd /content/optimized-llm-planning-memory
!pip install -e ".[dev]" -q
!pip install -e "/content/my-travel-world" -q
```

### 5. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Make Drive the persistent outputs directory
!mkdir -p /content/drive/MyDrive/optllm-outputs
!ln -sf /content/drive/MyDrive/optllm-outputs \
    /content/optimized-llm-planning-memory/outputs
```

After this, everything written to `outputs/` persists across session resets.

### 6. Load API keys from Secrets

```python
import os
from google.colab import userdata

os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
os.environ["OPENAI_API_KEY"]    = userdata.get("OPENAI_API_KEY")
```

### 7. Smoke-test (one episode, no training)

```python
!python scripts/run_episode.py agent.max_steps=5
```

If this completes without error, the full stack is working.

---

## Running Training

### Start TensorBoard before training

TensorBoard must be started before `ppo.learn()` — it reads event files written during training.

```python
%load_ext tensorboard
%tensorboard --logdir outputs/logs
```

### Train (Colab-friendly config)

```python
# Identity compressor — fastest to test; trains the value head only
!python scripts/run_training.py compressor=identity training=ppo_colab

# Transformer compressor — trains actual compressor weights on GPU
!python scripts/run_training.py compressor=transformer training=ppo_colab

# Custom run name (appears in manifest + Streamlit selector)
!python scripts/run_training.py compressor=identity training=ppo_colab \
    training.run_name=alice_identity_v1
```

After training completes, `outputs/training/<run_id>/` contains:
```
manifest.json             # full resolved config + run metadata
episode_metrics.jsonl     # one line per episode
ppo_metrics.jsonl         # one line per PPO update
```

### Resume after session reset

```python
import glob
zips = sorted(glob.glob("outputs/checkpoints/*.zip"))
latest = zips[-1] if zips else None
print(f"Resuming from: {latest}")

!python scripts/run_training.py compressor=identity training=ppo_colab \
    training.resume_from={latest}
```

Because `outputs/` symlinks to Drive, the checkpoint survives the reset.

### Hyperparameter sweep

Use the 3×3×3 grid defined in `configs/training/ppo_sweep.yaml`. Sweep sequentially — each takes ~15 min on T4:

```python
lrs        = ["1e-4", "3e-4", "1e-3"]
clip_epses = ["0.1", "0.2", "0.3"]
ent_coefs  = ["0.0", "0.01", "0.05"]

for lr in lrs:
    for clip in clip_epses:
        for ent in ent_coefs:
            !python scripts/run_training.py \
                compressor=identity training=ppo_colab \
                training.learning_rate={lr} \
                training.clip_epsilon={clip} \
                training.ent_coef={ent} \
                training.run_name=sweep_{lr}_{clip}_{ent}
```

---

## Artifact Storage Strategy

| Artifact | Location | Size estimate | Keep? |
|---|---|---|---|
| `training/<run_id>/episode_metrics.jsonl` | Drive | ~0.5 MB / 10k ep | Always |
| `training/<run_id>/ppo_metrics.jsonl` | Drive | ~0.1 MB / 10k ep | Always |
| `training/<run_id>/manifest.json` | Drive | < 5 KB | Always |
| `checkpoints/final/` | Drive | ~20–200 MB | Always |
| `checkpoints/<step>.zip` | Drive | ~20–200 MB each | Keep last 2 |
| `logs/` (TensorBoard) | Drive | ~10 MB / run | Keep |
| `episodes/*.json` | Drive | ~40 KB each | By default: never saved during training (`episode_save_freq=0`). Enable with `training.episode_save_freq=50` for debugging. |

> **Storage rule of thumb**: a 50k-step run with `episode_save_freq=0` uses ~30 MB total (JSONL + checkpoint). Enabling `episode_save_freq=1` for every episode inflates this to >400 MB. Keep it off during sweeps.

---

## Sharing Runs with Teammates

After training, package the run into a single shareable archive:

```python
from optimized_llm_planning_memory.utils.colab_utils import (
    bundle_run, upload_to_drive, download_bundle, list_drive_runs,
)
from optimized_llm_planning_memory.training.run_manifest import list_manifests

# Find the latest run
manifests = list_manifests(training_dir="outputs/training")
latest = manifests[0]
run_id = latest.run_id
print(f"Bundling run: {run_id}  ({latest.run_name})")

# Create bundle (~5–30 MB depending on checkpoint size)
bundle_path = bundle_run(run_id, output_dir="outputs", bundle_dir="outputs/bundles")

# Upload to a shared Drive folder (adjust the path to your team's shared Drive)
upload_to_drive(bundle_path, drive_dir="/content/drive/MyDrive/optllm_team_bundles")

# Or trigger a browser download to your local machine
download_bundle(run_id, bundle_dir="outputs/bundles")
```

### What's in the bundle

```
<run_id>.tar.gz
├── training/<run_id>/
│   ├── manifest.json
│   ├── episode_metrics.jsonl
│   └── ppo_metrics.jsonl
├── checkpoints/final/
│   ├── ppo_model.zip
│   ├── compressor/
│   └── reward_predictor/
└── logs/                  # TensorBoard events (only if < 50 MB)
```

---

## Using a Teammate's Bundle Locally

After downloading `<run_id>.tar.gz`, unpack it into your local `outputs/` directory:

```bash
# Unpack into outputs/
tar -xzf <run_id>.tar.gz -C outputs/
```

Then the local Streamlit Training Dashboard picks it up automatically:

```bash
streamlit run app/main.py
# Navigate to "Training Dashboard" — the run appears in the run selector
```

And the run comparison notebook works too:

```bash
uv run jupyter lab notebooks/08_run_comparison.ipynb
```

---

## Evaluating a Training Run Locally

Use the `+run_id` flag to auto-resolve the checkpoint from the manifest:

```bash
# Deterministic only (no API key needed)
python scripts/run_evaluation.py \
    +run_id=<run_id> \
    eval.deterministic_only=true

# Full evaluation with LLM judge
python scripts/run_evaluation.py \
    +run_id=<run_id>
```

The script reads `outputs/training/<run_id>/manifest.json` to find the compressor type and checkpoint path. See [EVALUATION.md — Evaluating from a Training Run ID](EVALUATION.md#evaluating-from-a-training-run-id) for details.

---

## One Colab Per Developer

Each team member works in their own Colab notebook (their copy of `notebooks/05_colab_rl_training.ipynb`). This keeps training runs isolated and avoids Drive write conflicts.

**Do NOT** share a Colab notebook URL with live kernel state — always share the notebook file and let teammates open a fresh copy.

**Coordination pattern:**
1. Each developer trains their assigned configuration.
2. Bundles are uploaded to the shared Drive folder.
3. One developer downloads all bundles locally and runs `notebooks/08_run_comparison.ipynb` to produce the comparison table and plots.
4. The comparison output (`outputs/run_comparison.png`, `outputs/run_comparison_summary.csv`) is committed to the repo as part of the ablation study results.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: Training run directory not found` | Check `run_id` — it must match the subdirectory name under `outputs/training/`. Run `list_manifests()` to see available IDs. |
| Drive runs out of space | Delete intermediate checkpoint zips (`outputs/checkpoints/ppo_*_steps.zip`). Only the `final/` checkpoint is needed for sharing. |
| TensorBoard shows no data | You started `%tensorboard` after `ppo.learn()` started. Stop training, run `%tensorboard --logdir outputs/logs`, then restart. |
| Session reset — checkpoint missing | Check that the Drive symlink is correct: `ls -la outputs/` should show it pointing to a Drive path. |
| LLM API errors during training | Training continues without the LLM judge. Check `outputs/logs/run_training_*.log` for `litellm` error lines. |
| `import google.colab` fails locally | `colab_utils.py` detects non-Colab environments and skips Colab-specific calls. `download_bundle()` just prints the local path. |
