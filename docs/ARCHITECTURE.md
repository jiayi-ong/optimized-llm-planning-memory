# Architecture

This document covers the full data flow, design patterns, and invariants that hold the project together. Read this before making cross-cutting changes.

---

## Data Flow

```
1. SETUP
   UserRequest (JSON) ──► ReActAgent + ToolRegistry + Compressor

2. EPISODE LOOP (ReActAgent.run_episode)
   for step in range(max_steps):
       a. ContextBuilder.build(trajectory, user_request, mode, compressed_state)
              → builds prompt string
       b. LLM call (litellm) → "Thought: ... Action: tool_name({...})"
       c. parse response → (thought, tool_name, arguments)  OR  DONE
       d. ToolRegistry.call(tool_name, arguments)
              → BaseTool.call() [validate → _execute → track → emit]
              → ToolResult (success or failure with feedback)
       e. Trajectory.append(step)
       f. if _should_compress():
              compressed_state = Compressor.compress(trajectory.to_model())
   return EpisodeLog

3. TRAINING (CompressionEnv.step)
   action (CompressedState text) → inject into agent context → run N steps
   → reward = RewardFunction.compute(episode_log, user_request)
   → PPO update (SB3): gradient through Compressor.get_log_probs()

4. EVALUATION (Evaluator.evaluate_dataset)
   EpisodeLog → DeterministicEvaluator.score() → 8 metrics
             → LLMJudge.score()            → 6 rubric scores
   → EvalResult → saved to outputs/eval_results/{run_id}/results.jsonl
```

---

## Core Data Contracts

### `EpisodeLog` — the universal hub

Everything downstream consumes `EpisodeLog`. No component reads `Trajectory` directly after the episode ends.

```python
@dataclass(frozen=True)
class EpisodeLog:
    episode_id:          str
    request_id:          str
    agent_mode:          str                    # "raw" | "llm_summary" | "compressor"
    trajectory:          TrajectoryModel        # all ReActStep objects
    compressed_states:   tuple[CompressedState, ...]  # one per compression event
    final_itinerary:     Itinerary | None       # None if agent failed to produce one
    reward_components:   RewardComponents       # per-component reward scores
    tool_stats:          tuple[ToolCallStats, ...]
    total_steps:         int
    success:             bool
    termination_reason:  str | None             # DONE_ITINERARY | EXIT_<CODE> | MAX_STEPS | PARSE_FAILURE | ERROR_<TYPE>
    config_hash:         str                    # hash of ProjectConfig at run time
    created_at:          str                    # ISO 8601
```

`EpisodeLog` is **frozen** (`frozen=True`). Once created it is never mutated. This makes it safe to cache, serialize, and distribute across parallel workers.

`termination_reason` values:
- `DONE_ITINERARY` — agent produced `Action: DONE` with a full `Itinerary:` block
- `EXIT_CITY_NOT_FOUND` / `EXIT_BUDGET_EXCEEDED` / `EXIT_DATE_INVALID` / `EXIT_NO_AVAILABILITY` / `EXIT_REPEATED_DEAD_END` — agent detected a lethal scenario and exited gracefully
- `MAX_STEPS` — episode reached `agent.max_steps` without a terminal signal
- `PARSE_FAILURE` — parser could not extract a valid action after `max_retries_per_action` retries
- `ERROR_<TYPE>` — unexpected exception (e.g. `ERROR_VALUEERROR`)

### `CompressedState` — the compressor's output

```python
@dataclass
class CompressedState:
    compressed_id:   str
    request_id:      str
    step_index:      int            # which ReAct step this was produced after
    summary:         str            # free-text summary (populated by LLM compressors)
    # Template sections — all required:
    hard_constraint_ledger:      str
    soft_constraints_summary:    str
    decisions_made:              str
    open_questions:              str
    key_discoveries:             str
    current_itinerary_sketch:    str
    # Optional structured fields:
    key_decisions:               list[str]
    active_constraints:          list[Constraint]
    explored_options:            list[str]
    next_actions:                list[str]
```

### `Itinerary` — the planning output

```python
class Itinerary(BaseModel):
    itinerary_id:    str
    request_id:      str
    days:            list[ItineraryDay]
    total_cost_usd:  float   # AUTO-COMPUTED from day costs (model_validator)
```

`total_cost_usd` is automatically recomputed from `sum(day.total_cost_usd for day in days)` via a Pydantic `@model_validator`. Never set it manually — it will be overwritten. If you need an itinerary with a specific cost, add `ActivityBooking` objects with the right `cost_usd` values.

---

## Design Patterns

### Template Method — `BaseTool.call()`

The tool lifecycle (validate → execute → track → emit) is fixed in `BaseTool.call()`. Subclasses override only `_execute()`. This guarantees that adding a new tool never accidentally skips tracking or error handling.

```
BaseTool.call()           ← orchestrates the lifecycle
    │
    ├─ validate(raw_args)  ← Pydantic, never overridden
    ├─ _execute(valid)     ← subclass responsibility
    ├─ tracker.record()    ← never overridden
    └─ bus.emit()          ← never overridden
```

### Strategy — `ContextBuilder`

Three memory strategies are interchangeable at the `ContextBuilder` boundary. The agent never knows which strategy is active — it just calls `context_builder.build(...)` and gets back a prompt string.

```
ContextBuilder.build(mode=RAW)            → full trajectory in prompt
ContextBuilder.build(mode=LLM_SUMMARY)    → LLM summary in prompt
ContextBuilder.build(mode=COMPRESSOR)     → CompressedState in prompt
```

### ABC — `CompressorBase`, `TrainableCompressorBase`, `BaseTool`

ABCs are used for **internal** class hierarchies where shared concrete behavior exists (template method in `BaseTool`, default `get_log_probs` in `CompressorBase`). Python enforces abstract method implementation at class definition time, not at runtime.

### Protocol — `SimulatorProtocol`

`SimulatorProtocol` is a `typing.Protocol` for **external** code (the `travel_world` library). The simulator never imports from this codebase. Tests can pass a `MockSimulator` without inheritance. This is the correct boundary between codebases.

```python
class SimulatorProtocol(Protocol):
    def search_flights(...) -> list[dict]: ...
    def book_hotel(...) -> dict: ...
    ...

# MockSimulator satisfies the protocol structurally — no inheritance needed
class MockSimulator:
    def search_flights(...): return [...]
    def book_hotel(...): return {...}
```

### Registry — `ToolRegistry`

Decouples `ReActAgent` from knowing which tools exist. The agent asks for a tool by name string; the registry handles lookup and raises `ToolNotFoundError` for unknown names. The registry also generates the tool schema section of the system prompt automatically.

### Composition — `RewardFunction` holds `ConstraintSatisfactionEngine`

`RewardFunction` does not inherit from the engine; it holds an instance. This makes the engine mockable in tests and allows different reward configurations to use the same engine instance.

---

## Critical Invariants

These invariants are enforced by tests and must be preserved in all changes.

### 1. Training reward ≡ evaluation metric

`training/reward.py` and `evaluation/deterministic.py` both import `ConstraintSatisfactionEngine` from `core/constraints.py`. They must always call the same methods on the same instance.

**Test:** `tests/test_training/test_reward_eval_consistency.py` — runs 5 parametrized scenarios through both paths and asserts `abs(reward_score - eval_score) < 1e-9`.

Violation consequence: the compressor is trained on a proxy metric that evaluation doesn't measure.

### 2. Soft constraint default is 1.0

When `user_request.soft_constraints` is empty, both `RewardFunction.compute()` and `DeterministicEvaluator.score()` must return `soft_constraint_score = 1.0` (no constraints = no violations).

**Test:** `test_no_soft_constraints_both_default_to_one` in `test_reward_eval_consistency.py`.

### 3. `Itinerary.total_cost_usd` is always the sum of day costs

The `@model_validator` recomputes this on every construction. Tests that need a specific total cost must create `ActivityBooking` / `AccommodationBooking` objects with the right `cost_usd` values — passing `total_cost_usd` directly is silently overwritten.

**Test:** Any budget constraint test that creates itineraries with specific costs.

### 4. `CompressedStateTemplate` validates all 6 sections

A compressor that omits any section raises `CompressedStateRenderError` at compression time, not silently later.

**Test:** `tests/test_compressor/test_template.py` — missing section tests.

### 5. `CompressorBase.get_log_probs()` raises for non-trainable compressors

Non-trainable compressors raise `LogProbsNotSupportedError` when `get_log_probs()` is called. The training loop checks `compressor.is_trainable()` before calling this.

**Test:** `tests/test_compressor/test_interface_contract.py::test_default_get_log_probs_raises`.

---

## Configuration System

All configuration is defined as Pydantic models in `core/config.py` and loaded via Hydra. The root compose file is `configs/config.yaml`.

### Structure

```
configs/
├── config.yaml          ← defaults list (one key per sub-config group)
├── agent/               ← AgentConfig
├── compressor/          ← CompressorConfig
├── training/            ← TrainingConfig
├── reward/              ← RewardConfig
├── eval/                ← EvalConfig
├── simulator/           ← SimulatorConfig
└── logging/             ← LoggingConfig
```

### Loading in scripts

```python
import hydra
from omegaconf import DictConfig, OmegaConf
from optimized_llm_planning_memory.core.config import ProjectConfig

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    config = ProjectConfig.model_validate(
        OmegaConf.to_container(cfg, resolve=True)
    )
```

### Overriding from the command line

Hydra allows dotted key overrides:
```bash
python scripts/run_training.py \
    agent.max_steps=50 \
    training.learning_rate=1e-3 \
    compressor=transformer
```

The `compressor=transformer` syntax swaps the entire `configs/compressor/` group.

---

## Output Locations

| Output | Directory | Format |
|---|---|---|
| Episode logs | `outputs/episodes/` | JSON (`EpisodeLog`) |
| Training checkpoints | `outputs/checkpoints/` | `.zip` + subdirectory |
| Evaluation results | `outputs/eval_results/{run_id}/` | `manifest.json` + `results.jsonl` |
| TensorBoard logs | `outputs/logs/` | TensorBoard event files |
| Baseline eval | `outputs/baseline_eval/` | JSON |

All `outputs/` is gitignored. Use Google Drive mounting in Colab to persist outputs across sessions.

---

## Key Files for Each Kind of Change

| Change | Files to edit |
|---|---|
| New Pydantic model field | `core/models.py` + any serialization/deserialization code |
| New config parameter | `core/config.py` + matching `configs/**/*.yaml` |
| New constraint category | `core/models.py` (enum) + `core/constraints.py` (handler) + test |
| New tool | `tools/my_tool.py` + `tools/registry.py` + `simulator/protocol.py` |
| New compressor | `compressor/my_compressor.py` + `compressor/__init__.py` + `configs/compressor/` |
| New agent mode | `agent/modes.py` + `agent/context_builder.py` + `configs/agent/` |
| New reward component | `training/reward.py` + `core/models.py` (RewardComponents) + `configs/reward/` |
| New deterministic metric | `evaluation/deterministic.py` + version bump + `configs/eval/` |
| New rubric dimension | `data/rubrics/*.md` + `evaluation/rubrics.py` + `evaluation/llm_judge.py` |

---

## Reproducibility

- **Global seed:** `config.project.seed` (default 42). Set via `utils/seed.py::set_seed()` which seeds Python, NumPy, PyTorch, and the simulator.
- **Per-episode seed:** drawn from `SimulatorConfig.seed_range` (`[0, 9999]`). Each parallel env gets a distinct seed.
- **Config hash:** `EpisodeLog.config_hash` stores a hash of `ProjectConfig` at run time, allowing you to verify that compared episodes used the same configuration.
- **`uv.lock`:** pinned transitive dependencies — every developer and Colab session gets identical package versions.
