# Evaluation

Evaluation runs in two independent layers. Layer 1 (deterministic) is fast, free, and always run. Layer 2 (LLM judge) adds qualitative scoring and requires an API key.

---

## Files

| File | Role |
|---|---|
| `evaluation/evaluator.py` | `Evaluator` — orchestrates both layers, writes results |
| `evaluation/deterministic.py` | `DeterministicEvaluator` — 14 rule-based metrics (8 v1 + 6 v2) |
| `evaluation/llm_judge.py` | `LLMJudge` — up to 10 rubric dimensions via instructor |
| `evaluation/rubrics.py` | Rubric text constants; `DEFAULT_RUBRIC_DIMENSIONS` (6 v1) and `RUBRIC_DIMENSIONS_V2` (all 10) |
| `evaluation/manifest.py` | `EvalRunManifest` — run metadata and result index |
| `evaluation/ablation.py` | `AblationRunner` — Cartesian product sweeps |
| `scripts/run_eval.py` | **Standalone eval CLI** — re-score any saved episodes without re-running the agent |
| `data/rubrics/itinerary_rubric_v1.md` | Human-readable rubric shown to the LLM judge |
| `utils/itinerary_export.py` | `to_itinerary_manifest()` — converts agent-internal `Itinerary` → `ItineraryManifest` for compatibility with the `my-travel-world` evaluation UI |
| `utils/episode_io.py` | `save_eval_run`, `load_eval_run`, `list_eval_runs`, `list_episodes_by_request` |

---

## Layer 1 — Deterministic Metrics

`DeterministicEvaluator.score(episode_log, user_request)` returns a dict of metrics. No LLM calls. Stamped with `METRIC_VERSION` on every `EvalResult`.

### v1 Metrics (original 8)

| Metric key | Range | Description |
|---|---|---|
| `hard_constraint_ratio` | [0, 1] | Fraction of hard constraints satisfied. **Primary success criterion** — double-weighted in the overall score. |
| `soft_constraint_score` | [0, 1] | Weighted average satisfaction across soft constraints. 1.0 when no soft constraints exist. |
| `budget_adherence` | [0, 1] | 1.0 if total itinerary cost ≤ budget; decreases proportionally if over. |
| `logical_consistency` | [0, 1] | Four sub-checks: dates sorted chronologically, no duplicate hotel bookings, no activity time overlaps within a day, flight arrival ≤ hotel check-in. Score = 1 − issues/checks. |
| `tool_efficiency` | [0, 1] | `1 - redundant_calls / total_calls`. Higher is better. |
| `tool_failure_rate` | [0, 1] | `failed_calls / total_calls`. Lower is better. |
| `avg_tool_latency_ms` | ≥ 0 | Mean latency per tool call in milliseconds. |
| `steps_per_episode` | ≥ 1 | Total ReAct steps taken. |

### v2 Metrics (added in v2, 6 new)

| Metric key | Range | Formula / Description |
|---|---|---|
| `destination_coverage_ratio` | [0, 1] | `covered / total` required destination cities. A city is covered if it appears in ≥1 `ItineraryDay.city` field (fuzzy case-insensitive match against `metadata.dest_names`). |
| `accommodation_coverage_ratio` | [0, 1] | `booked_nights / (total_days − 1)`. Fraction of non-departure days with a hotel booking. |
| `activity_density_score` | [0, 1] | Mean of per-day density scores. Per-day formula: 0 acts→0.0, 1→0.5, 2–4→1.0, 5+→max(0, 1−0.2×(n−4)). |
| `rest_day_ratio` | [0, 1] | For trips >5 days: `min(1.0, actual_rest_days / target)` where target = 1 rest day per 4 travel days (a day with ≤1 activity counts as a rest day). Returns 1.0 for trips ≤5 days. |
| `schedule_overlap_score` | [0, 1] | `1 − overlapping_pairs / total_checked_pairs` across all days. Granular complement to `logical_consistency` sub-check 3. |
| `intra_city_feasibility` | [0, 1] | `1 − violated_gaps / total_gaps`. A gap is violated when consecutive same-city activities have < 15 minutes between them. |

### Critical invariant

`DeterministicEvaluator` uses `ConstraintSatisfactionEngine` from `core/constraints.py`. `RewardFunction` uses the **same class**. This guarantees that training reward ≡ evaluation metric for every constraint-based score. Breaking this link (e.g., by adding local logic to either module instead of the engine) invalidates the experiment.

---

## Layer 2 — LLM Judge

`LLMJudge.score(episode_log, user_request)` calls an LLM (default: `openai/gpt-4o-mini`) and uses `instructor` to extract structured scores.

Each dimension returns a score in [0, 1] and a `reasoning` string, stored in `EvalResult.rubric_breakdown`.

### v1 Dimensions (default 6, `DEFAULT_RUBRIC_DIMENSIONS`)

| Dimension key | What it measures |
|---|---|
| `constraint_adherence` | Were all hard constraints explicitly met? |
| `logical_coherence` | Are dates, transitions, and bookings internally consistent? |
| `activity_diversity` | Is there a good mix of activity types tailored to stated preferences? |
| `budget_efficiency` | Was the budget used wisely — value for money, sensible allocation? |
| `feasibility` | Can the itinerary realistically be executed? (booking refs, timing, logistics) |
| `creativity` | Does the plan go beyond generic tourist suggestions? |

### v2 Dimensions (all 10, `RUBRIC_DIMENSIONS_V2`)

Includes the 6 v1 dimensions plus 4 additional qualitative dimensions.

| Dimension key | What it measures |
|---|---|
| `pacing_and_rest` | Energy management: rest days after demanding stretches; healthy rhythm across the whole trip |
| `practical_logistics` | Realistic check-in times, airport transfer windows, activity buffers; no impossibly tight connections |
| `local_experience_quality` | Authentic local experiences vs. generic tourist checklist; place-specific cultural character |
| `preference_embodiment` | Does the trip's overall character holistically reflect the traveler's stated preferences? |

To use all 10 dimensions:
```python
from optimized_llm_planning_memory.evaluation.rubrics import RUBRIC_DIMENSIONS_V2
judge = LLMJudge(model_id="openai/gpt-4o-mini", rubric_dimensions=RUBRIC_DIMENSIONS_V2)
```

The judge model is fixed across all conditions for fairness. Do not change `judge_model_id` mid-experiment.

---

## Eval Key and Uniqueness Contract

Every `EvalResult` carries a `eval_key` computed field:

```
eval_key = f"{request_id}::{world_seed}::{agent_mode}::{metric_version}"
```

This key uniquely identifies a `(request × world × agent × metrics)` combination. Two results with the same `eval_key` are directly comparable.

**Why `world_seed` and not `world_id`?** Two worlds generated with the same seed have identical hotels, events, and prices — they are logically the same world regardless of when they were generated. The seed is the stable identifier.

**Migration note:** Requests generated before `world_id` was added to `UserRequest` will have `world_id=None` and `world_seed=None` in their eval results. Their `eval_key` uses `"none"` as the seed component. These results remain valid for single-run analysis but should not be mixed with seeded results in cross-run comparisons.

### Using eval_key for deduplication

```python
# Keep only the most recent result per unique (request, world, agent, metrics) config
df = df.sort_values("created_at", ascending=False)
df = df.drop_duplicates(subset="eval_key", keep="first")
```

The eval viewer (`app/pages/6_eval_viewer.py`) does this automatically when "Latest per config only" is enabled (default on).

---

## World-Aligned Request Generation

Generate requests anchored to a specific saved world so that city names match the actual world data and trip dates are consistent with the world's `sim_date`:

```bash
# Generate 40 train + 10 val + 10 test requests from a saved world
python scripts/generate_user_requests.py \
    --world_dir worlds/world_42_20260502_084804 \
    --n_train 40 --n_val 10 --n_test 10 --seed 42
```

The generator reads `worlds/world_42_*/geo_layer.json` for real city names and `meta.json` for `sim_date`. Trip start dates are anchored 7–60 days after `sim_date` so trips are in the near future relative to the world.

Every generated `UserRequest` has:
- `world_id` set to the world folder ID (e.g., `"world_42_20260502_084804"`)
- `destination_cities` using actual city names from `geo_layer.json` (not placeholders like `city_1`, `city_2`)
- `metadata["world_id"]` for raw JSON introspection

The original Hydra-based generator (`python scripts/generate_user_requests.py` without `--world_dir`) still works for backward compatibility.

---

## Overall Score

```
overall = (
    2 × hard_constraint_ratio
  + soft_constraint_score
  + tool_efficiency
  + budget_adherence
  + logical_consistency
  + sum(llm_judge_scores)   # if judge ran; each dimension equally weighted
) / total_components
```

Scoring weights are configured in `configs/eval/default.yaml`.

---

## Running Evaluation

### Standalone eval CLI (`scripts/run_eval.py`)

Re-score any saved `EpisodeLog` files without re-running the agent. Useful for applying updated metrics to existing episodes or re-evaluating after a constraint engine fix.

**Episode selection flags (mutually exclusive — pick exactly one):**

| Flag | Selects |
|---|---|
| `--all` | Every `ep_*.json` in `episodes_dir` |
| `--episode_ids ID [ID ...]` | Specific episodes by UUID |
| `--request_ids ID [ID ...]` | All episodes whose `request_id` matches |
| *(none)* | Same as `--all` — backward compatible |

**Other flags:**

| Flag | Default | Description |
|---|---|---|
| `--deterministic_only` | off | Skip LLM judge — no API key needed |
| `--judge_model MODEL` | `openai/gpt-4o-mini` | litellm model string for the LLM judge |
| `--agent_mode MODE` | *(all modes)* | Additional filter; can combine with any selection flag |
| `--note TEXT` | — | Free-text note stored in the run manifest |
| `--log_file PATH` | — | Append structured logs to a file in addition to stdout |

```bash
# Score ALL episodes, both layers (requires API key in .env)
python scripts/run_eval.py --all

# Score ALL episodes, deterministic only — no API key needed
python scripts/run_eval.py --all --deterministic_only

# Score specific episodes by UUID, both layers, write logs to file
python scripts/run_eval.py --episode_ids 08dff70b-548a-... a1b2c3d4-... \
    --log_file outputs/logs/run_eval.log

# Score all episodes for specific requests, deterministic only
python scripts/run_eval.py --request_ids req-abc req-def --deterministic_only

# All episodes, custom judge model, filter by agent mode, annotate the run
python scripts/run_eval.py --all --judge_model openai/gpt-4o \
    --agent_mode raw --note "baseline re-eval after v2 metrics"
```

API keys are loaded from `.env` at the repo root (same pattern as all other scripts). `--log_file` opens in append mode; parent directories are created automatically.

Output directory: `outputs/eval_results/{YYYYMMDD_HHMMSS}_{run_id}/`

### Hydra-based eval (full pipeline)

```bash
# All three conditions, deterministic only (no API key needed)
python scripts/run_evaluation.py agent=react_baseline_raw      eval.deterministic_only=true
python scripts/run_evaluation.py compressor=llm_prompt         eval.deterministic_only=true
python scripts/run_evaluation.py compressor=identity           eval.deterministic_only=true \
    training.resume_from=outputs/checkpoints/final/ppo_model.zip

# With LLM judge (requires API key)
python scripts/run_evaluation.py compressor=identity
```

### Load results in Python

```python
from optimized_llm_planning_memory.utils.episode_io import load_eval_run, list_eval_runs

manifests = list_eval_runs("outputs/eval_results/")      # newest first
manifest, results = load_eval_run("20260502_143022_3fa8c10b", "outputs/eval_results/")

# Filter to a specific seed
seeded = [r for r in results if r.world_seed == 42]
```

---

## Output Directory Layout

```
outputs/eval_results/
└── {YYYYMMDD_HHMMSS}_{run_id_short}/   # e.g. 20260502_143022_3fa8c10b
    ├── manifest.json                    # EvalRunManifest: compressor, agent_mode, seeds, metric_version
    └── results.jsonl                    # one EvalResult JSON per line
```

The timestamp prefix ensures `ls` returns runs in chronological order. The 8-char UUID fragment (`run_id_short`) disambiguates runs created within the same second.

Each `EvalResult` line in `results.jsonl` includes `eval_key`, `world_seed`, `metric_version`, `deterministic_scores`, and optionally `llm_judge_scores` and `rubric_breakdown`.

---

## End-to-End Evaluation Pipeline

```
1. Generate world
   python scripts/generate_world.py --seed 42
   → worlds/world_42_{timestamp}/

2. Generate world-aligned requests
   python scripts/generate_user_requests.py \
       --world_dir worlds/world_42_{timestamp} --n_train 40 --n_val 10 --n_test 10
   → data/user_requests/{train,val,test}/request_*.json  (world_id set)

3. Run agent episodes
   python scripts/run_episode.py agent=react_baseline_raw
   → outputs/episodes/ep_{uuid}.json

4. Score saved episodes
   python scripts/run_eval.py --deterministic_only
   → outputs/eval_results/{timestamp}_{run_id}/

5. View results
   streamlit run app/main.py  →  navigate to Eval Viewer
   # or use notebooks/06_evaluation.ipynb Section 9 for cross-run comparison
```

### Converting to `ItineraryManifest` for the travel world UI

`EpisodeLog.final_itinerary` uses the agent-internal `Itinerary` model. To display the result in the `my-travel-world` evaluation UI (which expects an `ItineraryManifest`), use the conversion utility:

```python
from optimized_llm_planning_memory.utils.itinerary_export import to_itinerary_manifest

manifest = to_itinerary_manifest(
    itinerary=episode_log.final_itinerary,
    request=user_request,
    world_id="world-001",
    episode_id=episode_log.episode_id,
    simulator=sim,   # optional; enables location_id/coordinates lookup
)
```

**Fidelity notes:**
- `location_id` and `coordinates` on each `ItineraryItem` require the optional `simulator` argument. Without it they default to `""` and `{"lat": 0.0, "lon": 0.0}`.
- `transit_segments` is always empty — the agent does not track explicit transit legs between bookings.
- `trip_date_range` is derived from the earliest and latest dates across all days in the itinerary.

---

## Interactive Evaluation Notebook

`notebooks/06_evaluation.ipynb` is the primary developer tool for evaluation.

| Section | What to do |
|---|---|
| 1 Setup | Imports, metric version display (`METRIC_VERSION`) |
| 2 Request Picker | Load and filter requests; shows `world_id` column |
| 3 Model Selector | Set compressor type, checkpoint path, agent mode |
| 4 Episode Source | Load saved episodes or run new ones live |
| 5 Metric Selector | Deterministic, LLM judge, rubric dimensions |
| 6 Run Evaluation | Score + save versioned manifest |
| 7 Aggregated Results | Summary table, bar chart |
| 7b Metric Heatmap | `request_id × metric` heatmap for the current run |
| 8 Drill-Down | Full trajectory + metric bars + rubric reasoning for one episode |
| 9 Runs Comparison | Side-by-side table of recent runs; regression detection |
| 9b Statistical Tests | Paired Wilcoxon signed-rank test between any two runs |
| 10 Next Steps | Documented improvement opportunities |

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
def score(self, episode_log, user_request) -> dict[str, float]:
    ...
    return {
        # ... existing metrics ...
        "my_metric": self._score_my_metric(episode_log, user_request),
    }
```

### Step 3 — Bump the metric version

```python
METRIC_VERSION = "v3"   # increment from current

METRIC_CHANGELOG: dict[str, str] = {
    "v1": "...",
    "v2": "...",
    "v3": "Added my_metric: <description>.",
}
```

Bumping the version stamps new `EvalResult` objects with `"v3"`, allowing old and new results to coexist without confusion in the eval viewer.

### Step 4 — Add a scoring weight (Hydra config)

```yaml
# configs/eval/default.yaml
eval:
  scoring_weights:
    ...
    my_metric: 1
```

### Step 5 — Write a test

```python
# tests/test_evaluation/test_deterministic.py
def test_my_metric_perfect_case(sample_episode_log, sample_user_request):
    evaluator = DeterministicEvaluator()
    scores = evaluator.score(sample_episode_log, sample_user_request)
    assert "my_metric" in scores
    assert 0.0 <= scores["my_metric"] <= 1.0
```

---

## Adding a Rubric Dimension

### Step 1 — Add the dimension constant to `rubrics.py`

```python
MY_DIMENSION = """\
Dimension: My Dimension

<clear description for the judge LLM>

Score 1.0 — <what perfect looks like>
Score 0.75 — <good, minor gaps>
Score 0.5  — <adequate>
Score 0.25 — <poor>
Score 0.0  — <failure>
"""
```

### Step 2 — Register it

```python
RUBRIC_DIMENSIONS: dict[str, str] = {
    # ... existing ...
    "my_dimension": MY_DIMENSION,
}
```

### Step 3 — Use it

```python
judge = LLMJudge(model_id="...", rubric_dimensions=["constraint_adherence", ..., "my_dimension"])
```

The `DEFAULT_RUBRIC_DIMENSIONS` list (6 v1 dims) is unchanged so existing callers are unaffected.

---

## Constraint Satisfaction Engine

`core/constraints.py` is the single scoring implementation for both training and evaluation. When adding a new `ConstraintCategory`, update **both** the engine **and** add a test in `tests/test_core/test_constraint_engine_unit.py`.

### Adding a new constraint category

1. Add the enum value to `ConstraintCategory` in `core/models.py`
2. Add an `_evaluate_<category>()` method in `ConstraintSatisfactionEngine`
3. Wire it into the `_evaluate_single()` dispatcher
4. Add unit tests

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

See `notebooks/06_evaluation.ipynb` Section 10.5 for an interactive wiring example.

---

## Metric Versioning

Every `EvalResult` is stamped with `METRIC_VERSION`. History:

| Version | What changed |
|---|---|
| `v1` | Initial 8 metrics: `hard_constraint_ratio`, `soft_constraint_score`, `tool_efficiency`, `tool_failure_rate`, `avg_tool_latency_ms`, `steps_per_episode`, `budget_adherence`, `logical_consistency`. Constraint evaluation via `ConstraintSatisfactionEngine`. |
| `v2` | Added 6 metrics: `destination_coverage_ratio`, `accommodation_coverage_ratio`, `activity_density_score`, `rest_day_ratio`, `schedule_overlap_score`, `intra_city_feasibility`. Also fixed multi-hotel star-rating bug in `_evaluate_accommodation()`. |

When the metric schema changes:
- Old results retain their original version tag.
- The eval viewer detects version mismatches and shows a warning banner rather than silently averaging incompatible numbers.
- Never remove a version from `METRIC_CHANGELOG` — it is the audit trail.
