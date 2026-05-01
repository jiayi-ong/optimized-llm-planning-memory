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
    │       trajectory + compressed_state + tool_schema              │
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
            DeterministicEvaluator   (8 metrics, no LLM)
            LLMJudge                 (6 rubric dimensions)
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
| `simulator/` | `SimulatorProtocol` structural interface + `SimulatorAdapter` thin wrapper | — |
| `tools/` | `BaseTool` ABC, `ToolRegistry`, `ToolCallTracker`, `EventBus`, 13 concrete tools | [docs/TOOLS.md](docs/TOOLS.md) |
| `agent/` | `ReActAgent`, `Trajectory`, `ContextBuilder`, `AgentMode`, prompt templates | [docs/AGENT.md](docs/AGENT.md) |
| `compressor/` | `CompressorBase` + `TrainableCompressorBase` ABCs; 6 implementations; `CompressedStateTemplate` | [docs/COMPRESSOR.md](docs/COMPRESSOR.md) |
| `training/` | `CompressionEnv` (Gymnasium), `CompressorPolicy` (SB3), `RewardFunction`, `RLTrainer` | [docs/TRAINING.md](docs/TRAINING.md) |
| `evaluation/` | `DeterministicEvaluator`, `LLMJudge`, `AblationRunner` | [docs/EVALUATION.md](docs/EVALUATION.md) |
| `mcts/` | Optional MCTS search augmentation (tree, node, controller) | — |
| `utils/` | structlog config, TensorBoard helpers, visualization, seed control, episode I/O | — |

---

## Project Structure

```
optimized-llm-planning-memory/
├── configs/                    # Hydra config hierarchy
│   ├── config.yaml             #   root: defaults list
│   ├── agent/                  #   react_baseline_raw, react_default, react_mcts
│   ├── compressor/             #   identity, llm_prompt, transformer, hybrid, llm_mcts
│   ├── training/               #   ppo_default, ppo_colab, ppo_mcts
│   ├── reward/                 #   default (weights & shaping)
│   ├── eval/                   #   default (scoring_weights, judge_model)
│   ├── simulator/              #   default (seed_range, world_params)
│   └── logging/                #   default
├── src/
│   └── optimized_llm_planning_memory/
│       ├── core/               # models, config, constraints, exceptions
│       ├── simulator/          # protocol, adapter, schemas
│       ├── tools/              # 13 tools + registry + tracker + event bus
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
│   └── generate_user_requests.py
├── app/                        # Streamlit developer UI (5 pages)
├── notebooks/                  # 7 interactive development notebooks
├── data/
│   ├── user_requests/          #   train/, val/, test/ + templates/
│   ├── few_shot_examples/      #   react_tool_use.json
│   ├── compressor_dev/         #   4 pre-built trajectory contexts for dev
│   └── rubrics/                #   itinerary_rubric_v1.md
└── outputs/                    # runtime outputs (gitignored)
    ├── episodes/               #   saved EpisodeLog JSONs
    ├── checkpoints/            #   PPO .zip + compressor/ + reward_predictor/
    ├── eval_results/           #   manifest.json + results.jsonl per run
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

# With a trained checkpoint
python scripts/run_episode.py compressor=transformer \
    training.resume_from=outputs/checkpoints/final/ppo_model.zip
```

### Training

```bash
# Full run — identity compressor (GPU recommended)
python scripts/run_training.py compressor=identity

# Colab-friendly (50k steps, 2 parallel envs)
python scripts/run_training.py compressor=identity training=ppo_colab

# Resume from checkpoint
python scripts/run_training.py compressor=identity training=ppo_colab \
    training.resume_from=outputs/checkpoints/ppo_compressor_10000_steps.zip
```

Checkpoints land in `outputs/checkpoints/` every `training.checkpoint_every_n_steps` steps.

### Evaluation

```bash
# RAW baseline (no API key needed for deterministic-only)
python scripts/run_evaluation.py agent=react_baseline_raw eval.deterministic_only=true

# Trained compressor
python scripts/run_evaluation.py compressor=identity eval.deterministic_only=true \
    training.resume_from=outputs/checkpoints/final/ppo_model.zip
```

Results: `outputs/eval_results/{run_id}/manifest.json` + `results.jsonl`.

### Interactive notebooks

```bash
uv run python -m ipykernel install --user --name optllm --display-name "OptLLM"
uv run jupyter lab notebooks/
```

| Notebook | Purpose |
|---|---|
| `05_colab_rl_training.ipynb` | Full PPO training in Google Colab (GPU) |
| `06_evaluation.ipynb` | Interactive evaluation dashboard — compare conditions |
| `07_compressor_dev.ipynb` | Step-by-step guide for building a new compressor |

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

See [docs/TRAINING.md](docs/TRAINING.md#google-colab) for the full setup guide.

Quick steps:
1. Runtime → **GPU (T4)**
2. Add your LLM API key to Colab Secrets
3. Clone both repos, `pip install -e .[dev]`
4. Mount Drive so `outputs/` persists across session resets
5. `%tensorboard --logdir outputs/logs` before training
6. `python scripts/run_training.py training=ppo_colab`

---

## Adding New Components

| What to add | Where to start |
|---|---|
| New compressor | [docs/COMPRESSOR.md → Adding a New Compressor](docs/COMPRESSOR.md#adding-a-new-compressor) |
| New tool | [docs/TOOLS.md → Adding a New Tool](docs/TOOLS.md#adding-a-new-tool) |
| New deterministic metric | [docs/EVALUATION.md → Adding a Deterministic Metric](docs/EVALUATION.md#adding-a-deterministic-metric) |
| New rubric dimension | [docs/EVALUATION.md → Adding a Rubric Dimension](docs/EVALUATION.md#adding-a-rubric-dimension) |
| New agent mode or prompt version | [docs/AGENT.md → Extending the Agent](docs/AGENT.md#extending-the-agent) |
| New reward component | [docs/TRAINING.md → Reward Shaping](docs/TRAINING.md#reward-shaping) |

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
