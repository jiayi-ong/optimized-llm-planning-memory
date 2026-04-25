# Evaluation

Evaluation runs in two independent layers. Layer 1 (deterministic) is fast, free, and always run. Layer 2 (LLM judge) adds qualitative scoring and requires an API key.

---

## Files

| File | Role |
|---|---|
| `evaluation/evaluator.py` | `Evaluator` — orchestrates both layers, writes results |
| `evaluation/deterministic.py` | `DeterministicEvaluator` — 8 rule-based metrics |
| `evaluation/llm_judge.py` | `LLMJudge` — 6 rubric dimensions via instructor |
| `evaluation/rubrics.py` | `RubricDimension` — structured output schema for LLM judge |
| `evaluation/manifest.py` | `EvalRunManifest` — run metadata and result index |
| `evaluation/ablation.py` | `AblationRunner` — Cartesian product sweeps |
| `data/rubrics/itinerary_rubric_v1.md` | Human-readable rubric shown to the LLM judge |

---

## Layer 1 — Deterministic Metrics

`DeterministicEvaluator.score(episode_log, user_request)` returns a dict of 8 metrics. No LLM calls. Stamped with `METRIC_VERSION = "v1"` on every `EvalResult`.

### Metric reference

| Metric key | Range | Description |
|---|---|---|
| `hard_constraint_ratio` | [0, 1] | Fraction of hard constraints satisfied. **Primary success criterion** — double-weighted in the overall score. |
| `soft_constraint_score` | [0, 1] | Weighted average satisfaction across soft constraints. 1.0 when no soft constraints exist. |
| `budget_adherence` | [0, 1] | 1.0 if total itinerary cost ≤ budget; decreases proportionally if over. |
| `logical_consistency` | [0, 1] | 1.0 if: days are sorted chronologically AND no hotel is booked twice. |
| `tool_efficiency` | [0, 1] | `1 - redundant_calls / total_calls`. Higher is better. |
| `tool_failure_rate` | [0, 1] | `failed_calls / total_calls`. Lower is better. |
| `avg_tool_latency_ms` | ≥ 0 | Mean latency per tool call in milliseconds. |
| `steps_per_episode` | ≥ 1 | Total ReAct steps taken. |

### Critical invariant

`DeterministicEvaluator` uses `ConstraintSatisfactionEngine` from `core/constraints.py`. `RewardFunction` uses the **same class**. This guarantees that training reward ≡ evaluation metric for every constraint-based score. Breaking this link (e.g., by adding local logic to either module instead of the engine) invalidates the experiment.

---

## Layer 2 — LLM Judge

`LLMJudge.score(episode_log, user_request)` calls an LLM (default: `openai/gpt-4o-mini`, configured in `configs/eval/default.yaml`) and uses `instructor` to extract structured scores.

The judge reads `data/rubrics/itinerary_rubric_v1.md` and evaluates the final itinerary on six dimensions:

| Dimension key | What it measures |
|---|---|
| `constraint_adherence` | Were all stated hard and soft constraints respected? |
| `itinerary_feasibility` | Can a traveler actually execute this plan (timing, logistics)? |
| `preference_alignment` | Does the itinerary reflect the traveler's stated preferences and style? |
| `diversity_balance` | Good mix of activity types, neighborhoods, price points? |
| `budget_efficiency` | Was the budget used wisely — value for money? |
| `overall_quality` | Holistic judgment of the itinerary as a travel plan. |

Each dimension returns a score in [0, 1] and a `reasoning` string, stored in `EvalResult.rubric_breakdown`.

The judge model is fixed across all conditions for fairness. Do not change `judge_model_id` mid-experiment.

---

## Overall Score

```
overall = (
    2 × hard_constraint_ratio
  + soft_constraint_score
  + tool_efficiency
  + budget_adherence
  + logical_consistency
  + sum(llm_judge_scores)   # if judge ran
) / total_components
```

Scoring weights are configured in `configs/eval/default.yaml`:

```yaml
eval:
  scoring_weights:
    hard_constraint_ratio: 2    # doubled — primary success criterion
    soft_constraint_score: 1
    tool_efficiency: 1
    budget_adherence: 1
    logical_consistency: 1
    llm_judge_dimensions: 1     # each rubric dimension equally weighted
```

---

## Running Evaluation

```bash
# All three conditions, deterministic only (no API key needed)
python scripts/run_evaluation.py agent=react_baseline_raw      eval.deterministic_only=true
python scripts/run_evaluation.py compressor=llm_prompt         eval.deterministic_only=true
python scripts/run_evaluation.py compressor=identity           eval.deterministic_only=true \
    training.resume_from=outputs/checkpoints/final/ppo_model.zip

# With LLM judge (requires API key)
python scripts/run_evaluation.py compressor=identity
```

Results land in `outputs/eval_results/{run_id}/`:
- `manifest.json` — `EvalRunManifest`: compressor type, agent mode, metric version, checkpoint path, world seeds, timestamps
- `results.jsonl` — one `EvalResult` JSON per line

### Load previous results in Python

```python
from optimized_llm_planning_memory.utils.episode_io import load_eval_run, list_eval_runs

manifests = list_eval_runs("outputs/eval_results/")       # newest first
manifest, results = load_eval_run("abc12345", "outputs/eval_results/")
```

---

## Interactive Evaluation Notebook

`notebooks/06_evaluation.ipynb` is the primary developer tool for evaluation. Sections:

| Section | What to do |
|---|---|
| 1 Setup | Imports, `JUDGE_MODEL_ID = None` to skip judge |
| 2 Request Picker | Load requests from `data/user_requests/test/` |
| 3 Model Selector | Set compressor type, checkpoint path, agent mode |
| 4 Episode Source | Load saved episodes or run new ones live |
| 5 Run Evaluation | Score with progress bar; auto-saves manifest |
| 6 Results Table | Mean ± std per metric; bar chart |
| 7 Drill-Down | Full trajectory + metric bars for one episode |
| 8 Runs Comparison | Side-by-side table of up to 5 recent runs |

---

## Adding a Deterministic Metric

### Step 1 — Implement the scoring logic

Open `evaluation/deterministic.py`. Add a private method following the existing pattern:

```python
def _score_my_metric(
    self,
    episode_log: EpisodeLog,
    user_request: UserRequest,
) -> float:
    """
    Compute my_metric. [0, 1] — higher is better.

    Rule: <describe what you measure and how the score is computed>
    """
    if episode_log.final_itinerary is None:
        return 0.0
    # ... your scoring logic using episode_log and user_request ...
    return score  # float in [0, 1]
```

If the metric requires constraint evaluation, call `self._engine.evaluate(itinerary, constraints)` — do not re-implement constraint logic locally.

### Step 2 — Add it to `score()`

```python
def score(
    self,
    episode_log: EpisodeLog,
    user_request: UserRequest,
) -> dict[str, float]:
    ...
    return {
        "hard_constraint_ratio": ...,
        "soft_constraint_score": ...,
        # ... existing metrics ...
        "my_metric": self._score_my_metric(episode_log, user_request),  # add here
    }
```

### Step 3 — Bump the metric version

```python
METRIC_VERSION = "v2"   # was "v1"

METRIC_CHANGELOG: dict[str, str] = {
    "v1": "...",
    "v2": "Added my_metric: <description>.",
}
```

Bumping the version stamps new `EvalResult` objects with `"v2"`, allowing old and new results to coexist without confusion.

### Step 4 — Add a scoring weight

```yaml
# configs/eval/default.yaml
eval:
  scoring_weights:
    ...
    my_metric: 1       # document the rationale in a comment
```

### Step 5 — Write a test

```python
# tests/test_evaluation/test_deterministic.py
def test_my_metric_perfect_case(sample_episode_log, sample_user_request):
    evaluator = DeterministicEvaluator()
    scores = evaluator.score(sample_episode_log, sample_user_request)
    assert "my_metric" in scores
    assert 0.0 <= scores["my_metric"] <= 1.0

def test_my_metric_zero_on_failure(failing_episode_log, sample_user_request):
    scores = DeterministicEvaluator().score(failing_episode_log, sample_user_request)
    assert scores["my_metric"] == 0.0
```

---

## Adding a Rubric Dimension

### Step 1 — Add the dimension to the rubric file

Open `data/rubrics/itinerary_rubric_v1.md` and add a new `##` section:

```markdown
## my_dimension (0–1 scale)

**What to assess:** <clear description for the judge LLM>

**Score 1.0:** <what perfect looks like>
**Score 0.5:** <what adequate looks like>
**Score 0.0:** <what failure looks like>
```

The LLM judge reads this file verbatim. Be specific and give concrete examples at each anchor point.

### Step 2 — Add the dimension key to `RubricDimension`

Open `evaluation/rubrics.py`:

```python
class RubricDimension(BaseModel):
    constraint_adherence: float = Field(ge=0.0, le=1.0)
    itinerary_feasibility: float = Field(ge=0.0, le=1.0)
    preference_alignment: float = Field(ge=0.0, le=1.0)
    diversity_balance: float = Field(ge=0.0, le=1.0)
    budget_efficiency: float = Field(ge=0.0, le=1.0)
    overall_quality: float = Field(ge=0.0, le=1.0)
    my_dimension: float = Field(ge=0.0, le=1.0)  # add here

    # Also add a reasoning field:
    my_dimension_reasoning: str = Field(default="")
```

### Step 3 — Update the LLM judge prompt

Open `evaluation/llm_judge.py`. The judge prompt references the rubric dimensions. Add `my_dimension` to the dimension list in the prompt template so the LLM knows to score it.

### Step 4 — Add a scoring weight

```yaml
# configs/eval/default.yaml
eval:
  judge_dimensions:
    - constraint_adherence
    - itinerary_feasibility
    - preference_alignment
    - diversity_balance
    - budget_efficiency
    - overall_quality
    - my_dimension       # add here
```

---

## Constraint Satisfaction Engine

`core/constraints.py` is the single scoring implementation for both training and evaluation. When adding a new `ConstraintCategory`, update **both** the engine **and** add a test in `tests/test_core/test_constraint_engine_unit.py`.

### Adding a new constraint category

1. Add the enum value to `ConstraintCategory` in `core/models.py`:
   ```python
   class ConstraintCategory(str, Enum):
       BUDGET     = "budget"
       DATE       = "date"
       DURATION   = "duration"
       CITY       = "city"
       CUISINE    = "cuisine"    # example new category
       ...
   ```

2. Add an `_evaluate_cuisine()` method in `ConstraintSatisfactionEngine`:
   ```python
   def _evaluate_cuisine(
       self, itinerary: Itinerary, constraint: Constraint
   ) -> ConstraintSatisfactionResult:
       # constraint.value is the required cuisine string
       # check if any restaurant booking matches
       ...
   ```

3. Wire it into the `_evaluate_single()` dispatcher:
   ```python
   elif constraint.category == ConstraintCategory.CUISINE:
       return self._evaluate_cuisine(itinerary, constraint)
   ```

4. Add unit tests in `tests/test_core/test_constraint_engine_unit.py`.

---

## Ablation Studies

`AblationRunner` in `evaluation/ablation.py` sweeps a Cartesian product of config axes:

```python
from optimized_llm_planning_memory.evaluation.ablation import AblationRunner

runner = AblationRunner(
    axes={
        "agent.mode":            ["raw", "llm_summary", "compressor"],
        "compressor.type":       ["identity", "transformer"],
        "training.num_timesteps": [50_000, 200_000],
    },
    base_config=base_config,
    n_episodes_per_cell=10,
)
results = runner.run()     # dict[condition_key → list[EvalResult]]
runner.report(results)     # prints mean ± std table
```

See `notebooks/06_evaluation.ipynb` Section 8 for an interactive wiring example.

---

## Metric Versioning

Every `EvalResult` is stamped with `METRIC_VERSION`. When the metric schema changes:
- Old results retain their original version tag.
- The `runs_comparison` notebook section detects version mismatches and flags them rather than silently averaging incompatible numbers.
- Never remove a version from `METRIC_CHANGELOG` — it's the audit trail.
