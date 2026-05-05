# Optimized LLM Planning with Memory

**Columbia MSDS — STATGR5293: Generative AI with LLMs — Final Project (Spring 2026)**

---

## What This Is

This repository investigates whether **context compression alone** — without fine-tuning the planning LLM — can outperform raw and summarized ReAct trajectories for constraint-satisfying travel planning.

A trainable **compressor** distills a growing ReAct agent trajectory into a compact structured text state. That state is injected into the agent's context window at each compression event. The compressor is trained end-to-end via **Proximal Policy Optimization (PPO)**, with rewards shaped by hard/soft constraint satisfaction, tool efficiency, and logical consistency.

Three experimental conditions are compared:

| Condition | Config | Description |
|---|---|---|
| **RAW** | `agent=react_baseline_raw` | Full trajectory appended verbatim — no compression |
| **LLM Summary** | `agent=react_default compressor=llm_prompt` | Separate LLM call summarizes trajectory at each step |
| **Trained Compressor** | `agent=react_default compressor=identity` or `transformer` | PPO-trained compressor produces a structured 6-section memory state |

---

## Architecture

```
UserRequest
    │
    ▼
ReActAgent  ──────────────────────────────────────────────────────────┐
    think → act → observe loop                                        │
    │                                                                 │
    ├─► ContextBuilder    (RAW | LLM_SUMMARY | COMPRESSOR mode)      │
    │       builds the LLM prompt from:                              │
    │       [SYSTEM] + [USER REQUEST] + [CURRENT ITINERARY STATE]   │
    │       + [MEMORY] + [TOOL SCHEMA] + [FEW-SHOT EXAMPLES]        │
    │                                                                 │
    ├─► ToolRegistry ──► BaseTool ──► SimulatorAdapter               │
    │       middleware       Template Method   wraps travel_world     │
    │       (validation, tracking, events, error feedback)           │
    │                                                                 │
    └─► Compressor  (CompressorBase ABC)                             │
            every N steps or on token threshold                       │
            trajectory → CompressedState (6 required sections)       │
                                                                      │
EpisodeLog ◄──────────────────────────────────────────────────────────┘
    │   frozen data contract: trajectory, compressed_states,
    │   final_itinerary, reward_components, tool_stats, termination_reason
    │
    ├─► RewardFunction            (PPO training signal)
    │       uses ConstraintSatisfactionEngine
    │
    └─► Evaluator
            DeterministicEvaluator   (15 metrics: 8 v1 + 6 v2 + 1 v3, no LLM)
            LLMJudge                 (up to 10 rubric dimensions)
```

### Critical Design Invariants

1. **`ConstraintSatisfactionEngine`** (`core/constraints.py`) is the **single implementation** imported by both `RewardFunction` and `DeterministicEvaluator`. Training signal ≡ evaluation metric — by construction.
2. **`EpisodeLog`** is the universal data contract. Nothing downstream reads `Trajectory` directly.
3. **`CompressedStateTemplate`** enforces a fixed 6-section output schema, preventing the compressor from drifting toward unstructured actions.
4. **`SimulatorProtocol`** uses `typing.Protocol` structural typing so the external simulator never imports from this codebase.

---

## Module Map

| Module | Purpose | Deep-dive |
|---|---|---|
| `core/` | Pydantic v2 data models, constraint engine, config schema, exceptions | — |
| `simulator/` | `SimulatorProtocol` structural interface + `SimulatorAdapter` + `WorldPool` | — |
| `tools/` | `BaseTool` ABC, `ToolRegistry`, `ToolCallTracker`, `EventBus`, 14 concrete tools (incl. `cancel_booking`) | [docs/TOOLS.md](docs/TOOLS.md) |
| `agent/` | `ReActAgent`, `Trajectory`, `ContextBuilder`, `AgentMode`, prompt templates | [docs/AGENT.md](docs/AGENT.md) |
| `compressor/` | `CompressorBase` + `TrainableCompressorBase` ABCs; 6 implementations; `CompressedStateTemplate` | [docs/COMPRESSOR.md](docs/COMPRESSOR.md) |
| `training/` | `CompressionEnv` (Gymnasium), `CompressorPolicy` (SB3), `RewardFunction`, `RLTrainer`, `RLRunLogger`, `TrainingRunManifest` | [docs/TRAINING.md](docs/TRAINING.md) |
| `evaluation/` | `DeterministicEvaluator` (15 metrics: 8 v1 + 6 v2 + 1 v3), `LLMJudge`, `AblationRunner` | [docs/EVALUATION.md](docs/EVALUATION.md) |
| `mcts/` | Optional MCTS search augmentation (tree, node, controller) | — |
| `utils/` | structlog config, visualization, seed control, episode I/O, `itinerary_export`, `colab_utils` | — |

---

## Project Structure

```
optimized-llm-planning-memory/
├── configs/                    # Hydra config hierarchy
│   ├── config.yaml             #   root: defaults list
│   ├── agent/                  #   react_baseline_raw, react_default, react_mcts
│   ├── compressor/             #   identity, llm_prompt, transformer, hybrid, llm_mcts
│   ├── training/               #   ppo_default, ppo_colab, ppo_debug, ppo_sweep
│   ├── reward/                 #   default, coverage_heavy (optional trip-quality signals)
│   ├── eval/                   #   default (scoring_weights, judge_model)
│   ├── simulator/              #   default (seed_range, pool_size, world_params)
│   └── logging/                #   default
├── src/
│   └── optimized_llm_planning_memory/
│       ├── core/               # models, config, constraints, exceptions
│       ├── simulator/          # protocol, adapter, schemas
│       ├── tools/              # 14 tools + registry + tracker + event bus
│       ├── agent/              # react_agent, context_builder, trajectory, prompts
│       ├── compressor/         # 6 implementations + template + lora_utils
│       ├── training/           # env, policy, reward, trainer, buffer, logger
│       ├── evaluation/         # evaluator, deterministic, llm_judge, ablation
│       ├── mcts/               # tree, node, controller (optional)
│       └── utils/              # logging, io, seed, tensorboard, visualization
├── tests/                      # 613+ passing tests
│   ├── unit/                   #   fast isolated unit tests
│   ├── module/                 #   multi-class workflow tests
│   ├── system/                 #   end-to-end smoke tests (LLM mocked)
│   ├── test_core/              #   constraint engine, config YAML loading
│   ├── test_agent/             #   react agent parse + tool dispatch
│   ├── test_compressor/        #   ABC contract, identity, template, dummy
│   ├── test_training/          #   reward ≡ evaluation regression tests
│   └── test_integration/       #   full pipeline (marked integration)
├── scripts/
│   ├── run_episode.py          # debug single episode
│   ├── run_training.py         # PPO training with Hydra config
│   ├── run_evaluation.py       # batch evaluation
│   ├── run_baseline_eval.py    # quick no-LLM baseline
│   ├── run_eval.py             # standalone eval CLI (re-score saved episodes)
│   └── generate_user_requests.py
├── app/                        # Streamlit developer UI (5 pages)
├── notebooks/                  # 7 interactive development notebooks
├── data/
│   ├── user_requests/          #   train/, val/, test/ + templates/
│   ├── few_shot_examples/      #   react_tool_use.json
│   ├── compressor_dev/         #   4 pre-built trajectory contexts for dev
│   └── rubrics/                #   itinerary_rubric_v1.md
└── outputs/                    # runtime outputs (gitignored)
    ├── episodes/               #   saved EpisodeLog JSONs (off during training by default)
    ├── checkpoints/            #   PPO .zip + compressor/ + reward_predictor/
    ├── training/               #   per-run JSONL diagnostics + manifest.json
    │   └── <run_id>/          #     episode_metrics.jsonl, ppo_metrics.jsonl, manifest.json
    ├── bundles/                #   .tar.gz run archives for sharing
    ├── eval_results/           #   manifest.json + results.jsonl per eval run
    └── logs/                   #   TensorBoard event files
```

---

## Local Setup

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (`pip install uv`)
- `travel_world` simulator as a **sibling package**:

```
workspace/
├── optimized-llm-planning-memory/   ← this repo
└── my-travel-world/                 ← simulator repo
```

### Install

```bash
cd optimized-llm-planning-memory

# Create venv and install all deps (uv.lock guarantees identical versions)
uv sync --extra dev --extra notebook

# Copy env template and fill in at least one LLM API key
cp .env.example .env
```

Minimum `.env` for live LLM calls:
```bash
ANTHROPIC_API_KEY=sk-ant-...    # for claude-* models
OPENAI_API_KEY=sk-...           # for gpt-* models (LLM judge default)
```

### Optional extras

| Extra | When to add |
|---|---|
| `dev` | Always — linting, type-checking, tests |
| `notebook` | Local notebook development |
| `wandb` | Weights & Biases experiment tracking |

---

## Running the Code

### Single episode (debug and inspect)

```bash
# Compressor mode (default)
python scripts/run_episode.py

# RAW baseline — no compression
python scripts/run_episode.py agent=react_baseline_raw

# LLM-summary baseline
python scripts/run_episode.py agent=react_default compressor=llm_prompt

# MCTS-augmented compression (agent and compressor must be paired)
python scripts/run_episode.py agent=react_mcts compressor=llm_mcts

# With a trained checkpoint
python scripts/run_episode.py compressor=transformer \
    training.resume_from=outputs/checkpoints/final/ppo_model.zip
```

> **MCTS config pairing:** `agent=react_mcts` sets `mode: mcts_compressor` and provides the `agent.mcts` config sub-tree that the MCTSController requires. `compressor=llm_mcts` selects `LLMMCTSCompressor`, which has the `compress_with_tree()` method that consumes the MCTS output. These two must always be used together. Mixing them with other agent/compressor configs logs a startup warning but does not crash — MCTS search simply will not run.

### Training

```bash
# Full run — identity compressor (GPU recommended; auto-detects CUDA)
python scripts/run_training.py compressor=identity

# Colab-friendly (50k steps, 2 parallel envs, T4)
python scripts/run_training.py compressor=identity training=ppo_colab

# Named run — appears in manifest and Streamlit selector
python scripts/run_training.py compressor=identity training=ppo_colab \
    training.run_name=alice_identity_v1

# Custom reward profile (adds destination coverage + activity density signals)
python scripts/run_training.py reward=coverage_heavy compressor=identity

# Resume from checkpoint
python scripts/run_training.py compressor=identity training=ppo_colab \
    training.resume_from=outputs/checkpoints/ppo_compressor_10000_steps.zip
```

Each run writes `outputs/training/<run_id>/manifest.json` (full resolved config), `episode_metrics.jsonl`, and `ppo_metrics.jsonl`. Checkpoints land in `outputs/checkpoints/` every `training.checkpoint_every_n_steps` steps.

#### Artifact bundling

After training, bundle the run for sharing with teammates:

```python
from optimized_llm_planning_memory.utils.colab_utils import bundle_run
bundle_path = bundle_run("<run_id>", output_dir="outputs")
# → outputs/bundles/<run_id>.tar.gz  (~5–30 MB)
```

See [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md) for the full team collaboration workflow.

### Evaluation

#### Generate world-aligned requests

Create requests anchored to a specific saved world so city names and dates match the world's actual data:

```bash
python scripts/generate_user_requests.py \
    --world_dir worlds/world_42_20260502_084804 \
    --n_train 40 --n_val 10 --n_test 10 --seed 42
```

Every generated `UserRequest` has `world_id` set and uses real city names from `geo_layer.json`. See [docs/EVALUATION.md — World-Aligned Request Generation](docs/EVALUATION.md#world-aligned-request-generation).

#### Re-score saved episodes (standalone CLI)

Apply the current metrics to any previously saved `EpisodeLog` files without re-running the agent — useful when bumping `METRIC_VERSION` or fixing a constraint engine bug.

Three mutually-exclusive selection flags (`--all`, `--episode_ids`, `--request_ids`):

```bash
# Score ALL episodes, both layers (requires API key in .env)
python scripts/run_eval.py --all

# Score ALL episodes, deterministic only — no API key needed
python scripts/run_eval.py --all --deterministic_only

# Score specific episodes by UUID, write logs to file
python scripts/run_eval.py --episode_ids 08dff70b-548a-... --log_file outputs/logs/run_eval.log

# Score all episodes for specific requests, deterministic only
python scripts/run_eval.py --request_ids req-abc req-def --deterministic_only

# Combine any selection flag with --agent_mode for an extra filter
python scripts/run_eval.py --all --agent_mode raw --note "baseline re-eval after v2 metrics"
```

API keys are loaded automatically from `.env`. Use `--log_file` to persist structured logs alongside the eval output.

Output: `outputs/eval_results/{YYYYMMDD_HHMMSS}_{run_id}/manifest.json` + `results.jsonl`.

#### Full pipeline eval (Hydra)

```bash
# RAW baseline (no API key needed for deterministic-only)
python scripts/run_evaluation.py agent=react_baseline_raw eval.deterministic_only=true

# Trained compressor — auto-resolve checkpoint from training run manifest
python scripts/run_evaluation.py +run_id=<run_id> eval.deterministic_only=true

# Trained compressor — explicit checkpoint
python scripts/run_evaluation.py compressor=identity eval.deterministic_only=true \
    training.resume_from=outputs/checkpoints/final/ppo_model.zip
```

The `+run_id` flag reads `outputs/training/<run_id>/manifest.json`, infers the compressor type, and resolves the checkpoint path automatically.

#### Comparing runs with eval_key

Every `EvalResult` carries an `eval_key` = `{request_id}::{world_seed}::{agent_mode}::{metric_version}`. Two results with the same key are directly comparable. The eval viewer (`app/pages/6_eval_viewer.py`) uses this key for deduplication and cross-mode comparison. See [docs/EVALUATION.md — Eval Key](docs/EVALUATION.md#eval-key-and-uniqueness-contract).

### Interactive notebooks

```bash
uv run python -m ipykernel install --user --name optllm --display-name "OptLLM"
uv run jupyter lab notebooks/
```

| Notebook | Purpose |
|---|---|
| `05_colab_rl_training.ipynb` | Full PPO training in Google Colab (GPU) — includes Section 12: bundle & share |
| `06_evaluation.ipynb` | Interactive evaluation dashboard — compare conditions |
| `07_compressor_dev.ipynb` | Step-by-step guide for building a new compressor |
| `08_run_comparison.ipynb` | Load multiple run bundles and compare side-by-side — reward curves, summary table, PPO diagnostics |

### Developer UI (Streamlit)

```bash
streamlit run app/main.py
```

Five pages: Episode Browser · Trajectory Viewer · Compression Viewer · MCTS Viewer · Training Dashboard.

---

## Test Suite

```bash
# Fast tests (excludes slow model-download and integration tests)
uv run pytest tests/ -q -m "not slow and not integration"

# Unit tests only
uv run pytest tests/ -q -m unit

# Include TransformerCompressor smoke test (downloads flan-t5-small ~300 MB)
uv run pytest tests/ -q -m slow

# Integration tests (requires live API key)
uv run pytest tests/test_integration/ -m integration
```

Markers: `unit`, `slow`, `integration`, `module_test`, `system_test`.

---

## Google Colab Training

See [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md) for the full team collaboration guide and [docs/TRAINING.md](docs/TRAINING.md#google-colab) for training details.

Quick steps:
1. Runtime → **GPU (T4)**
2. Add `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` to Colab Secrets (`Tools → Secrets`)
3. Clone both repos, `pip install -e ".[dev]"`
4. Mount Drive and symlink `outputs/` → Drive folder so artifacts persist across resets
5. `%tensorboard --logdir outputs/logs` **before** training
6. `python scripts/run_training.py training=ppo_colab training.run_name=<your_name>_<config>`
7. After training: `bundle_run(<run_id>)` + `upload_to_drive(...)` to share with teammates

---

## Adding New Components

| What to add | Where to start |
|---|---|
| New compressor | [docs/COMPRESSOR.md → Adding a New Compressor](docs/COMPRESSOR.md#adding-a-new-compressor) |
| New tool | [docs/TOOLS.md → Adding a New Tool](docs/TOOLS.md#adding-a-new-tool) |
| New deterministic metric | [docs/EVALUATION.md → Adding a Deterministic Metric](docs/EVALUATION.md#adding-a-deterministic-metric) |
| New rubric dimension | [docs/EVALUATION.md → Adding a Rubric Dimension](docs/EVALUATION.md#adding-a-rubric-dimension) |
| New agent mode or prompt version | [docs/AGENT.md → Extending the Agent](docs/AGENT.md#extending-the-agent) |
| New reward component (optional/pluggable) | [docs/TRAINING.md → Optional reward components](docs/TRAINING.md#optional-reward-components) |
| New reward component (structural) | [docs/TRAINING.md → Reward Shaping](docs/TRAINING.md#reward-shaping) |
| Colab team collaboration | [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md) |

---

## Dependency Management

```bash
uv add some-package                        # add runtime dep
uv add --optional dev some-dev-tool        # add to an extra
uv lock                                    # regenerate lock after manual edits
uv sync --extra dev --extra notebook       # sync after git pull
uv lock --upgrade-package some-package     # upgrade one package
```

`uv.lock` is committed to git. Run `uv sync` after every pull.

---

## License

MIT
