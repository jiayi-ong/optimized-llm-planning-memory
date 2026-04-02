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
| `compressor/` | `CompressorBase` + `TrainableCompressorBase` ABCs; LLM, Transformer, Dummy, Hybrid implementations; `CompressedStateTemplate` |
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
# Full PPO training run (GPU recommended)
python scripts/run_training.py

# Colab-friendly config (smaller batch, 2 envs)
python scripts/run_training.py --config-name=ppo_colab
```

Checkpoints are saved to `outputs/checkpoints/` every `training.checkpoint_every_n_steps` steps.

---

## Evaluation

```bash
# Evaluate all three conditions and compare
python scripts/run_evaluation.py agent.mode=raw
python scripts/run_evaluation.py agent.mode=llm_summary  compressor=llm_prompt
python scripts/run_evaluation.py agent.mode=compressor   compressor=transformer \
    training.resume_from=outputs/checkpoints/step_500000.zip
```

Results are written to `outputs/eval_results/`.

---

## License

MIT
