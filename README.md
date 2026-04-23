# Optimized LLM Planning with Memory

**Columbia MSDS — STATGR5293: Generative AI with LLMs — Final Project (Spring 2026)**

---

## Overview

This repository trains a **context-compression mechanism** (the *compressor*) that distills a growing ReAct agent trajectory into a compact, structured text representation. The goal is to enable an LLM-based travel planner to satisfy more hard and soft constraints under a **fixed context window budget**.

The compressor is trained with **Proximal Policy Optimization (PPO)** using a custom Gymnasium environment. Three experimental conditions are compared:

| Condition | Description |
|---|---|
| **RAW** | Full trajectory appended verbatim — no compression |
| **LLM Summary** | A separate LLM call summarizes the trajectory at each step |
| **Trained Compressor** | PPO-trained seq2seq model compresses the trajectory into a structured state |

---

## Architecture

```
UserRequest
    │
    ▼
ReActAgent  ──────────────────────────────────────────────────────────┐
  │  think → act → observe loop                                        │
  │                                                                    │
  ├─► ContextBuilder  (RAW | LLM_SUMMARY | COMPRESSOR mode)           │
  │       builds the LLM prompt from trajectory + compressed state    │
  │                                                                    │
  ├─► ToolRegistry  ──►  BaseTool  ──►  SimulatorAdapter              │
  │       middleware layer               wraps the travel simulator    │
  │       (validation, tracking,                                       │
  │        events, error feedback)                                     │
  │                                                                    │
  └─► Compressor  (CompressorBase ABC)                                 │
          LLMCompressor  /  TransformerCompressor  /  DummyCompressor  │
          compress trajectory → CompressedState                        │
                                                                       │
EpisodeLog  ◄──────────────────────────────────────────────────────────┘
    │   (trajectory, compressed_states, final_itinerary,
    │    reward_components, tool_stats)
    │
    ├─► RewardFunction  (PPO training signal)
    │       uses ConstraintSatisfactionEngine (shared with evaluation)
    │
    └─► Evaluator
            DeterministicEvaluator  +  LLMJudge
```

### Module Map

| Module | Purpose |
|---|---|
| `core/` | Shared Pydantic v2 models, constraint engine, config schema, exceptions |
| `simulator/` | `SimulatorProtocol` (structural typing) + `SimulatorAdapter` (thin wrapper) |
| `tools/` | `BaseTool` ABC, `ToolRegistry`, `ToolCallTracker`, `EventBus`, 8 concrete tools |
| `agent/` | `ReActAgent`, `Trajectory`, `ContextBuilder`, `AgentMode` enum, prompt templates |
| `compressor/` | `CompressorBase` + `TrainableCompressorBase` ABCs; **Identity** (baseline, PPO-compatible), LLM, Transformer, Dummy, Hybrid implementations; `CompressedStateTemplate`; `SparkWeightComponent` (PySpark MLlib reward predictor) |
| `training/` | `CompressionEnv` (Gymnasium), `CompressorPolicy` (SB3), `RewardFunction`, `RLTrainer`, `EpisodeBuffer` |
| `evaluation/` | `Evaluator`, `DeterministicEvaluator`, `LLMJudge`, `AblationRunner` |
| `utils/` | Logging, TensorBoard helpers, visualization, seed control, episode I/O |

---

## Key Design Decisions

- **`SimulatorProtocol`** uses `typing.Protocol` (structural subtyping) so the external simulator library never imports from this codebase, and tests can pass a `MockSimulator` without inheritance.
- **`ConstraintSatisfactionEngine`** (`core/constraints.py`) is the single implementation of constraint scoring — both `RewardFunction` (training) and `DeterministicEvaluator` (evaluation) import it, guaranteeing the training signal and evaluation metric are identical.
- **`EpisodeLog`** is the universal data contract between planning, RL training, and evaluation. Nothing downstream reads the `Trajectory` directly.
- **`CompressedStateTemplate`** enforces a fixed section schema on the compressor's action space — the model must fill all six sections, preventing action-space drift.
- **`BaseTool.call()`** is a Template Method — subclasses implement `_execute()` only; validation, tracking, and event emission are handled once in the base class.

---

## Project Structure

```
optimized-llm-planning-memory/
├── pyproject.toml
├── Makefile
├── .env.example
├── configs/                   # Hydra config hierarchy
│   ├── config.yaml
│   ├── agent/
│   ├── compressor/
│   ├── simulator/
│   ├── training/
│   ├── reward/
│   └── eval/
├── src/
│   └── optimized_llm_planning_memory/
│       ├── core/
│       ├── simulator/
│       ├── tools/
│       ├── agent/
│       ├── compressor/
│       ├── training/
│       ├── evaluation/
│       └── utils/
├── tests/
│   ├── test_core/
│   ├── test_compressor/
│   ├── test_tools/
│   ├── test_agent/
│   ├── test_training/
│   ├── test_evaluation/
│   └── test_integration/
├── scripts/
│   ├── run_episode.py
│   ├── run_training.py
│   ├── run_evaluation.py
│   └── generate_user_requests.py
├── data/
│   ├── user_requests/
│   ├── rubrics/
│   └── few_shot_examples/
└── notebooks/
```

---

## Installation

```bash
# Clone and create a virtual environment
git clone <repo-url>
cd optimized-llm-planning-memory
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e ".[dev]"

# Copy and fill in API keys
cp .env.example .env
```

### Requirements

- Python 3.11+
- PyTorch 2.3+
- `pydantic-ai`, `litellm`, `instructor` for LLM orchestration
- `gymnasium`, `stable-baselines3` for RL training
- `hydra-core` for config management

---

## Compressor Development

### Interactive notebook

> **Notebook:** [`notebooks/07_compressor_dev.ipynb`](notebooks/07_compressor_dev.ipynb)

Step-by-step developer guide for building and testing a new compressor:

| Section | What it does |
|---|---|
| **1 Setup** | Path wiring, project imports, dev-context loader utility |
| **2 Minimal non-trainable compressor** | Build `KeywordSummaryCompressor` (rule-based, no PyTorch) |
| **3 Trainable upgrade** | Upgrade to `TrainableKeywordSummaryCompressor` with char embedding |
| **4 Verify trainable contract** | Interactive gradient-flow, optimizer-step, and checkpoint roundtrip checks |
| **5 Run test suite** | `subprocess` cells running `pytest` from within the notebook |
| **6 Manual context inspection** | Compress all four dev contexts; token count comparison table; side-by-side vs. `IdentityCompressor` |
| **7 Key considerations** | Gradient connectivity, template enforcement, log-prob sign, checkpoint error contract, token efficiency |
| **8 Register your compressor** | Hydra config + `__init__.py` export + move to module file |

### Comprehensive test suite

`tests/test_compressor/test_comprehensive.py` covers:

- **ABC contract enforcement** — verifies that `CompressorBase` and `TrainableCompressorBase` refuse instantiation when abstract methods are missing
- **`IdentityCompressor`** — 24 tests: initialisation, `compress()` output, log-probs shape and values, gradient flow to `_dummy_param`, checkpoint roundtrip, metadata
- **Cross-compressor protocol compliance** — parametrized over `IdentityCompressor` + `DummyCompressor`; any new trainable compressor added to the fixture runs all compliance tests for free
- **`CompressedStateTemplate` edge-cases** — multiline content, long itinerary sketch, Unicode, empty lists
- **Checkpoint error paths** — missing file, corrupted file; must raise `CompressorCheckpointError` (not a generic exception)
- **`DummyCompressor` gradient tests** — end-to-end autograd through the decoder; optimizer step changes params; gradient accumulation
- **`TransformerCompressor` smoke test** — marked `@pytest.mark.slow`; downloads flan-t5-small (~300 MB); excluded from the default CI run

```bash
# Run fast compressor tests only
pytest tests/test_compressor/ -v -m "not slow"

# Include TransformerCompressor smoke test (needs internet + HuggingFace cache)
pytest tests/test_compressor/ -v -m slow
```

### Dev context files

Pre-made realistic trajectories for manual compressor inspection:

| File | Steps | Scenario |
|---|---|---|
| `data/compressor_dev/contexts/01_early_episode.json` | 2 | City discovered, hotels listed — no bookings yet |
| `data/compressor_dev/contexts/02_mid_episode.json` | 6 | Hotel booked; one failed tool call (typo), corrected; restaurants found |
| `data/compressor_dev/contexts/03_late_episode.json` | 8 | Hotel + 2 museums + 1 restaurant + event booked; day-4 evening open |
| `data/compressor_dev/contexts/04_stress_context.json` | 12 | Duplicate hotel search, rejected duplicate booking, budget miscalculation in thought |

---

## Running Tests

```bash
# Full test suite (unit + integration, no live LLM needed)
pytest tests/ -q

# Skip slow compressor model tests
pytest tests/ -q --ignore=tests/test_compressor/test_dummy_compressor.py

# Integration tests (requires API key in .env)
pytest tests/test_integration/ -m integration -q
```

---

## Running an Episode

```bash
# Single episode in RAW mode (no compression)
python scripts/run_episode.py agent.mode=raw

# Single episode with LLM summarizer baseline
python scripts/run_episode.py agent.mode=llm_summary compressor=llm_prompt

# Single episode with trained compressor
python scripts/run_episode.py agent.mode=compressor compressor=transformer \
    training.resume_from=outputs/checkpoints/step_500000.zip
```

---

## Training

```bash
# Full PPO training run — identity compressor (GPU recommended)
python scripts/run_training.py compressor=identity

# Identity compressor, Colab-friendly config (2 envs, 50 k steps)
python scripts/run_training.py compressor=identity training=ppo_colab

# Transformer compressor (requires GPU + HuggingFace model)
python scripts/run_training.py compressor=transformer training=ppo_colab
```

Checkpoints are saved to `outputs/checkpoints/` every `training.checkpoint_every_n_steps` steps. The compressor weights and PySpark model are saved alongside each SB3 checkpoint.

---

## Google Colab Training

> **Notebook:** [`notebooks/05_colab_rl_training.ipynb`](notebooks/05_colab_rl_training.ipynb)
>
> Open it directly in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_ORG/optimized-llm-planning-memory/blob/main/notebooks/05_colab_rl_training.ipynb)

### Step-by-step

| Step | What to do |
|---|---|
| **1. Runtime** | Set Runtime → Change runtime type → **GPU (T4)**. Training 50 k steps takes ~20–40 min. |
| **2. Secrets** | Click the 🔑 icon → add `ANTHROPIC_API_KEY` (and `OPENAI_API_KEY` if using OpenAI). Never paste keys into cells. |
| **3. Clone & install** | Run the setup cells. They clone the repo, `pip install -e .[dev]`, and install `travel_world` and `pyspark`. |
| **4. Mount Drive** | Run the Drive mount cell. A symlink is created so `outputs/` writes directly to your Drive — checkpoints survive session resets. |
| **5. Generate requests** | Run the generate cell. It calls `scripts/generate_user_requests.py`, which discovers city IDs from the travel simulator (no LLM needed). |
| **6. Smoke test** | Run one episode (`agent.max_steps=5`) to confirm the stack is wired before committing to a full run. |
| **7. TensorBoard** | Run `%tensorboard --logdir outputs/logs` **before** training so it updates live during the run. |
| **8. Train** | Run the training cell. Progress prints to stdout; TensorBoard updates every rollout. |
| **9. Resume** | If the session resets, re-run steps 1–6, then use the "Resuming from checkpoint" cell. It finds the latest `.zip` automatically. |

### Key TensorBoard metrics to watch

| Tag | Healthy trend |
|---|---|
| `episode/total_reward` | Rising over time |
| `episode/hard_constraint_score` | Rising, approaching 1.0 |
| `episode/tool_efficiency_score` | Rising (fewer wasted tool calls) |
| `train/policy_gradient_loss` | Decreasing toward 0 |
| `train/entropy_loss` | Slowly decreasing (exploration reducing) |
| `train/value_loss` | Decreasing (value network improving) |

**Red flags:** `total_reward` flat after 10 k steps → check `agent.max_steps` is large enough for the LLM to complete a task. `tool_efficiency_score` stuck at 0 → confirm the API key is valid.

### PySpark diagnostics

The `SparkWeightComponent` fits a `LinearRegression` every 50 completed episodes, predicting episode reward from five scalar features. Its weights are saved to `outputs/checkpoints/<step>/spark/spark_weights.json`. Use notebook cell 9 ("PySpark component diagnostics") to inspect which features correlate most with reward — a sanity check that the compressor is learning something meaningful.

### Checkpoint layout

```
outputs/checkpoints/
├── ppo_compressor_5000_steps.zip    # SB3 full policy (optimizer + weights)
├── ppo_compressor_5000_steps/
│   ├── compressor/
│   │   └── identity_compressor.pt   # scalar _dummy_param
│   └── spark/
│       └── spark_weights.json       # PySpark MLlib coefficients
├── ...
└── final/
    ├── ppo_model.zip
    ├── compressor/
    └── spark/
```

To load a checkpoint for evaluation or continued training:

```bash
# Continue training from step 10 000
python scripts/run_training.py \
    compressor=identity training=ppo_colab \
    training.resume_from=outputs/checkpoints/ppo_compressor_10000_steps.zip

# Evaluate with a specific checkpoint
python scripts/run_evaluation.py \
    compressor=identity eval.deterministic_only=true \
    training.resume_from=outputs/checkpoints/final/ppo_model.zip
```

---

## Evaluation

### Architecture

Evaluation runs in two independent layers that can be used together or separately:

| Layer | Class | When to use |
|---|---|---|
| **Deterministic** | `DeterministicEvaluator` | Always — fast, free, reproducible |
| **LLM Judge** | `LLMJudge` | When you need qualitative scoring (requires API key) |

Both layers are orchestrated by `Evaluator`, which writes a `EvalResult` per episode and saves everything under a versioned `EvalRunManifest`.

### Deterministic Metrics (v1)

All eight metrics are computed from the `EpisodeLog` alone — no LLM call needed:

| Metric | Description |
|---|---|
| `hard_constraint_ratio` | Fraction of hard constraints satisfied [0, 1]. Primary success criterion. |
| `soft_constraint_score` | Weighted soft constraint score [0, 1]. |
| `budget_adherence` | 1.0 if within budget; penalised proportionally if over. |
| `logical_consistency` | Date ordering + no duplicate hotel bookings [0, 1]. |
| `tool_efficiency` | 1 − redundant call ratio [0, 1]. Higher is better. |
| `tool_failure_rate` | Fraction of tool calls that failed. Lower is better. |
| `avg_tool_latency_ms` | Mean latency per tool call. |
| `steps_per_episode` | Total ReAct steps taken. |

> **Metric versioning.** `METRIC_VERSION = "v1"` is stamped on every `EvalResult`. When the metric schema evolves, old results retain their original tag, making cross-version comparisons safe to detect and flag.

### LLM Judge Rubric Dimensions

When `USE_LLM_JUDGE = True`, `LLMJudge` scores the final itinerary on six rubric dimensions via `instructor`-structured output. Each dimension returns a score in [0, 1] plus a free-text reasoning string stored in `EvalResult.rubric_breakdown`:

`constraint_satisfaction` · `itinerary_feasibility` · `preference_alignment` · `diversity_balance` · `budget_efficiency` · `overall_quality`

### Overall Score

```
overall = (2 × hard_constraint_ratio + soft_constraint_score
           + tool_efficiency + budget_adherence + logical_consistency
           + all LLM judge scores) / total_components
```

Hard constraint ratio is double-weighted because it is the primary success criterion.

### Persistent Eval Runs

Each evaluation batch is saved as a self-contained run directory:

```
outputs/eval_results/
    {run_id}/
        manifest.json   ← EvalRunManifest: compressor, agent mode, metric version,
                                           checkpoint path, world seeds, timestamps
        results.jsonl   ← one EvalResult JSON per line
```

Load any previous run in Python:

```python
from optimized_llm_planning_memory.utils.episode_io import load_eval_run, list_eval_runs

# List all runs, newest first
manifests = list_eval_runs("outputs/eval_results/")

# Load a specific run
manifest, results = load_eval_run(run_id="abc12345", base_directory="outputs/eval_results/")
```

### Interactive Evaluation Notebook

> **Notebook:** [`notebooks/06_evaluation.ipynb`](notebooks/06_evaluation.ipynb)

The primary developer-facing evaluation tool. Sections:

| Section | What it does |
|---|---|
| **1 Setup** | Imports, path config, metric version display |
| **2 Request Picker** | Load requests from `data/user_requests/`; select by index list or random sample |
| **3 Model & Compressor Selector** | Set compressor type, agent mode, checkpoint path, LLM model |
| **4 Episode Source** | Load pre-existing episodes from disk, or run new ones live |
| **5 Metric Selector** | Toggle deterministic metrics and/or LLM judge; pick rubric dimensions |
| **6 Run Evaluation** | Score all episodes with a tqdm progress bar; save manifest |
| **7 Aggregated Results** | Mean ± std table; bar chart of key metrics |
| **8 Drill-Down** | Full trajectory, per-metric ASCII bars, rubric reasoning for one episode |
| **9 Runs Comparison** | Table comparing up to 5 recent runs; regression detection |
| **10 Next Steps** | Documented improvement opportunities |

No API key is needed for deterministic-only mode (`JUDGE_MODEL_ID = None`).

### CLI Scripts

```bash
# Evaluate all three conditions and compare
python scripts/run_evaluation.py agent.mode=raw
python scripts/run_evaluation.py agent.mode=llm_summary  compressor=llm_prompt
python scripts/run_evaluation.py agent.mode=compressor   compressor=transformer \
    training.resume_from=outputs/checkpoints/step_500000.zip

# Quick baseline eval (no LLM needed — scripted agent + deterministic metrics)
python scripts/run_baseline_eval.py
```

Results are written to `outputs/eval_results/` (notebook runs) or `outputs/baseline_eval/` (scripted baseline).

### Ablation Studies

`AblationRunner` in `evaluation/ablation.py` sweeps a Cartesian product of config axes and aggregates metrics across all combinations. See Section 10.7 of the evaluation notebook for a wiring example.

---

## License

MIT
