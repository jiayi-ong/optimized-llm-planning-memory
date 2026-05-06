# Evaluation UI — Running Locally and Experiment Manual

The evaluation UI is a locally-hosted Streamlit app. It combines a control panel for launching experiments, a historical log of all evaluation results, and ablation comparison views. This document covers everything from environment setup to running a full experiment cycle.

---

## 1. Prerequisites

Python 3.10+ and the project's dependencies must be installed. From the repo root:

```bash
pip install -e ".[dev]"          # installs src package + dev extras
pip install streamlit plotly     # UI-specific dependencies (not in core extras)
```

Confirm installation:

```bash
python -c "import streamlit, plotly; print('UI deps OK')"
```

API keys are loaded from a `.env` file at the repo root. Copy the template and fill in your key:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (required only for LLM judge evals)
```

---

## 2. Starting the UI

From the repo root:

```bash
streamlit run app/main.py
```

Streamlit opens a browser tab at `http://localhost:8501`. The sidebar lists all 11 pages. If port 8501 is in use, specify an alternative:

```bash
streamlit run app/main.py --server.port 8502
```

To allow access from other machines on the same network (e.g., from your laptop while Streamlit runs on a workstation):

```bash
streamlit run app/main.py --server.address 0.0.0.0
```

---

## 3. Page Overview

| Page | What it does |
|------|-------------|
| **1 Eval Dashboard** | Landing page. Table of all eval runs with aggregate stats. Sidebar control panel to configure and start new runs or re-runs. Drill into any run to see mean ± std per metric. |
| **2 Eval Episode Log** | Full history of evaluated episodes. Filter by run, agent mode, augmentation ID, archetype, complexity tier, overall score, and date. Scrollable table. |
| **3 Eval Episode Detail** | Four-tab drill-down for a single episode: user request card, day-by-day itinerary, paginated trajectory viewer, full eval scores and rubric reasoning. |
| **4 Ablation Comparison** | Side-by-side metric table, bar chart, radar chart, and complexity-stratification chart for 2–4 selected runs or augmentation IDs. |
| **5 Request Browser** | Browse all user requests. Filters by archetype, complexity tier, group type, geographic spread, and budget. Distribution charts and per-request detail with constraint breakdown. |
| **6 Episode Browser** | Raw episode log (no eval scores). Browse and filter `EpisodeLog` objects by agent mode, date, success/failure. |
| **7 Trajectory Viewer** | Step-by-step ReAct trajectory with live-polling mode for in-progress episodes. |
| **8 Compression Viewer** | Side-by-side view of raw trajectory input and compressed state output at each compression event. |
| **9 MCTS Viewer** | MCTS search statistics, node values, and top candidate plans (MCTS-mode episodes only). |
| **10 Training Dashboard** | PPO loss curves, KL divergence, clip fraction, and episode reward trends from training runs. |
| **11 Itinerary Viewer** | Day-by-day formatted itinerary for any saved episode. |

---

## 4. Running an Experiment — Step by Step

A complete experiment cycle has five stages. The eval UI covers stages 4 and 5; stages 1–3 are CLI-only.

```
1. Generate world(s)
2. Generate user requests
3. Run agent episodes
4. Evaluate episodes         ← UI control panel or CLI
5. Analyse results           ← UI pages 1, 2, 3, 4
```

### Stage 1 — Generate the travel world

```bash
python scripts/generate_world.py --seed 42
# → worlds/world_42_{timestamp}/
```

Use the same seed for all conditions in an experiment so the world is held constant. For multi-seed experiments, generate one world per seed:

```bash
for seed in 42 123 456; do
    python scripts/generate_world.py --seed $seed
done
```

### Stage 2 — Generate user requests

Requests must be anchored to the world so city names match real simulator IDs:

```bash
python scripts/generate_user_requests.py \
    --world_dir worlds/world_42_{timestamp} \
    --n_train 40 --n_val 10 --n_test 10 \
    --seed 42
# → data/user_requests/{train,val,test}/request_*.json
```

Each request now includes a `metadata.complexity_breakdown` dict with five difficulty dimensions and a `complexity_tier` (`"low"` / `"medium"` / `"high"`). The **Request Browser** (page 5) lets you inspect the distribution before running any episodes.

**Complexity tier definitions:**

| Tier | Composite score | Typical profile |
|------|----------------|----------------|
| `low` | < 0.35 | 1 destination, short trip, loose budget, no special needs |
| `medium` | 0.35–0.65 | 2 destinations, moderate constraints, standard group |
| `high` | ≥ 0.65 | 3+ destinations, tight budget, children or accessibility needs, many soft constraints |

To generate a targeted set of `high`-complexity requests only:

```bash
python scripts/generate_user_requests.py \
    --world_dir worlds/world_42_{timestamp} \
    --n_test 20 --seed 42 --complexity_tier high
```

### Stage 3 — Run agent episodes

Three conditions to generate episodes for comparison:

```bash
# Condition 1: raw (no compression)
python scripts/run_episode.py agent=react_baseline_raw

# Condition 2: LLM summary baseline
python scripts/run_episode.py agent=react_default compressor=llm_prompt

# Condition 3: trained compressor (replace path with your checkpoint)
python scripts/run_episode.py agent=react_default compressor=transformer \
    training.resume_from=outputs/checkpoints/final/ppo_model.zip
```

All episodes land in `outputs/episodes/ep_{uuid}.json`. The **Episode Browser** (page 6) and **Trajectory Viewer** (page 7) let you inspect individual episodes immediately, including live-polling during a running episode.

---

## 5. Evaluating Episodes

### 5a. From the UI (Control Panel)

Open the **Eval Dashboard** (page 1) and expand **"▶ New Eval Run"** in the sidebar.

| Field | What to set | Notes |
|-------|------------|-------|
| Augmentation ID | Registry ID of the compressor, e.g. `tgad-trained-001` | Optional; leave blank for raw/llm_summary baselines |
| Prompt ID | Registry ID of the system prompt, e.g. `sweep_D` | Optional |
| Eval mode | `deterministic` / `full` / `llm_judge` | Start with `deterministic` — it's free and fast |
| Judge model | `openai/gpt-4o-mini` | Only active when mode is `full` or `llm_judge` |
| Episode selection | All episodes / filter by agent_mode / manual IDs | Use "filter by agent_mode" to score one condition at a time |
| Parent run | Select a previous run to mark this as a re-run | Enables re-run lineage tracking |
| Notes | Free-text description | Stored in manifest; searchable in page 1 |

Click **▶ Start Eval Run**. A progress bar appears while the job runs. Results appear in the run table as soon as the job completes.

### 5b. From the CLI

```bash
# Fast deterministic only — no API key needed
python scripts/run_eval.py --all --deterministic_only \
    --augmentation_id tgad-trained-001 --note "v3 metrics, world 42"

# Full eval (deterministic + LLM judge)
python scripts/run_eval.py --all \
    --augmentation_id tgad-trained-001 \
    --judge_model openai/gpt-4o-mini

# Score a specific set of episodes
python scripts/run_eval.py \
    --episode_ids 08dff70b-548a-... a1b2c3d4-... \
    --deterministic_only

# Re-run only the LLM judge on a previous run's episodes
python scripts/run_eval.py \
    --episode_ids $(jq -r '.episode_ids[]' outputs/eval_results/20260506_120000_abc12345/manifest.json) \
    --eval_mode llm_judge \
    --parent_run_id 20260506_120000_abc12345
```

**`--eval_mode` values:**

| Value | Runs | Notes |
|-------|------|-------|
| `deterministic` | Deterministic layer only | Free, fast, no API key |
| `full` | Both layers | Requires `OPENAI_API_KEY` in `.env` |
| `llm_judge` | LLM judge only | Use to re-score an existing deterministic run |

Output lands in `outputs/eval_results/{YYYYMMDD_HHMMSS}_{run_id_short}/` with three files:

```
outputs/eval_results/{run_id}/
    manifest.json        # run metadata: augmentation_id, episode_ids, status, ...
    results.jsonl        # one EvalResult JSON per line
    aggregate.json       # pre-computed mean/std/min/max per metric per agent_mode
```

---

## 6. Reading Results

### 6a. Run Dashboard (page 1)

The run table shows every eval run, newest first. Columns include `augmentation_id`, `agent_mode`, `n_episodes`, and overall/hard-constraint mean scores read from `aggregate.json` — no full JSONL scan on load.

Select a run to expand its detail view: a table of mean ± std per metric per agent_mode, plus a bar chart of overall score with error bars.

Click **"→ View episodes in this run"** to navigate to page 2 pre-filtered to that run.

### 6b. Episode Log (page 2)

Use the sidebar filters to scope the view:

- **Run ID** — isolate results from a single eval run
- **Augmentation ID** — compare init vs. trained checkpoints by ID
- **Complexity tier** — check if performance degrades on hard requests
- **Latest per config only** — deduplicate by `eval_key` so each `(request × world × agent × metric_version)` tuple appears once, keeping the newest result

Click **"Open Episode Detail"** and copy the episode ID to navigate to page 3.

### 6c. Episode Detail (page 3)

Enter an episode ID in the text box. Four tabs:

| Tab | Content |
|-----|---------|
| **Overview** | Formatted user request, complexity breakdown, episode summary (steps/tokens/success), key metric gauges |
| **Itinerary** | Day-by-day view: transport segments, hotel bookings, activities with timestamps and costs; hard constraint satisfaction overlay |
| **Trajectory** | Paginated step viewer (8 steps/page). Per step: thought (orange), action with arguments (blue), observation with success/failure colouring |
| **Evaluation** | All deterministic scores, all LLM judge scores, full rubric breakdown (expandable per dimension with LLM reasoning) |

### 6d. Ablation Comparison (page 4)

Select 2–4 conditions on the **"Compare by"** axis:

| Axis | Use for |
|------|---------|
| `agent_mode` | raw vs. llm_summary vs. compressor vs. mcts_compressor |
| `augmentation_id` | init checkpoint vs. trained checkpoint |
| `run_id` | any two historical runs |

The page renders four panels automatically:
1. Metric comparison table (mean ± std per condition)
2. Bar chart of overall score with error bars
3. Radar chart across 6 key metrics
4. Complexity-stratification bar chart: overall score grouped by `low`/`medium`/`high` complexity tier

Use the **archetype** and **complexity tier** sidebar filters to isolate performance on specific request types.

### 6e. In Python

```python
from optimized_llm_planning_memory.evaluation.eval_store import EvalStore
from pathlib import Path

store = EvalStore(
    eval_dir=Path("outputs/eval_results"),
    episodes_dir=Path("outputs/episodes"),
    requests_dir=Path("data/user_requests"),
)

# All runs
manifests = store.list_runs()            # newest first

# Per-run aggregate stats (O(1) — reads aggregate.json)
agg = store.load_aggregate(manifests[0].run_id)
print(agg["by_agent_mode"]["raw"]["hard_constraint_ratio"])
# → {"mean": 0.42, "std": 0.15, "min": 0.0, "max": 0.8, "n": 24}

# Flat DataFrame across all runs with optional filters
df = store.flat_results(
    agent_modes=["raw", "compressor"],
    complexity_tiers=["high"],           # requires complexity_breakdown in request metadata
    metric_version="v3",
)
```

---

## 7. Configuring an Experiment

### Identifying components with registry IDs

Every compressor checkpoint and system prompt should be registered before running experiments. This makes every eval result fully traceable to the exact component versions.

```python
from optimized_llm_planning_memory.core.registry import AugmentationRegistry, AugmentationEntry, AugmentationType
from datetime import datetime, timezone

registry = AugmentationRegistry.load()
registry.register(AugmentationEntry(
    aug_id="tgad-trained-001",
    type=AugmentationType.TGAD_TRAINED,
    checkpoint_path="outputs/checkpoints/20260506_120000/final/ppo_model.zip",
    parent_init_id="tgad-init-001",
    training_run_id="20260506_120000",
    config_snapshot={"path_encoder_dropout": 0.0, "n_heads": 4, "n_layers": 2},
    created_at=datetime.now(timezone.utc).isoformat(),
))
registry.save()
```

Once registered, pass the ID to every eval run so it appears in all manifests and results:

```bash
python scripts/run_eval.py --all --augmentation_id tgad-trained-001
```

### Choosing metric version

Set `metric_version` via the eval config. All metrics from earlier versions are included in later ones — the version tag is a stamp on `EvalResult` for traceability, not a filter.

| Version | Primary new addition |
|---------|---------------------|
| `v1` | Original 8 metrics |
| `v2` | 6 itinerary-structure metrics (coverage, density, feasibility) |
| `v3` | `completion_rate` — guards against empty-itinerary score inflation |

Always use the same `metric_version` across all conditions in one experiment. Mixed-version comparisons trigger a warning banner in the UI.

### Reproducibility checklist

Before running a new experiment condition, confirm:

1. **World seed fixed** — all conditions use the same world(s). Check `EvalRunManifest.world_seeds`.
2. **Request set fixed** — all conditions use the same `request_ids`. Compare `manifest.request_ids` across runs.
3. **Judge model fixed** — do not change `judge_model_id` between conditions. The LLM judge must be a constant proxy.
4. **Metric version fixed** — all conditions produce results with the same `metric_version`.
5. **Augmentation ID registered** — trained compressor checkpoint is in the registry with a stable `aug_id`.
6. **Notes filled in** — the `notes` field in the manifest is the only free-text hook. Use it: e.g. `"v3 metrics, world42, tgad epoch 50"`.

---

## 8. Re-running Sub-Components

A common workflow is to run deterministic eval first (free), then add LLM judge scores later (costs API calls). Use `--parent_run_id` to link the two runs:

```bash
# Step 1: fast deterministic pass
python scripts/run_eval.py --all --deterministic_only \
    --augmentation_id tgad-trained-001 \
    --note "det-only pass"
# Produces: outputs/eval_results/20260506_130000_abc12345/

# Step 2: add LLM judge for the same episodes
python scripts/run_eval.py \
    --episode_ids ep1 ep2 ep3 ... \
    --eval_mode llm_judge \
    --parent_run_id 20260506_130000_abc12345 \
    --augmentation_id tgad-trained-001 \
    --note "llm_judge pass, gpt-4o-mini"
```

The second run's `manifest.parent_run_id` records the lineage. In the UI run table, both runs appear separately; use `parent_run_id` in Python to join them if needed.

From the UI control panel (page 1 sidebar): select the original run in the **"Parent run (re-run)"** dropdown, set **Eval mode** to `llm_judge`, manually enter the episode IDs, and click **▶ Start Eval Run**.

---

## 9. Output Directory Reference

```
outputs/
├── episodes/
│   └── ep_{uuid}.json               # EpisodeLog: trajectory, itinerary, rewards
├── eval_results/
│   └── {YYYYMMDD_HHMMSS}_{short}/
│       ├── manifest.json            # EvalRunManifest: all provenance fields
│       ├── results.jsonl            # one EvalResult per line
│       └── aggregate.json           # pre-computed stats (mean/std/min/max)
├── training/
│   └── {YYYYMMDD_HHMMSS}/
│       ├── manifest.json            # TrainingRunManifest
│       ├── ppo_metrics.jsonl        # per-update: loss, KL, clip_fraction
│       └── episode_metrics.jsonl    # per-episode: reward, constraint scores
├── checkpoints/
│   └── {run_id}/final/ppo_model.zip
└── .eval_jobs/
    └── {job_id}.json                # job tracker (UI control panel only)
```

The `.eval_jobs/` directory is created automatically when a job is launched from the UI. It is not needed when running `run_eval.py` directly from the CLI.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Page 1 shows "No evaluation runs found" | `outputs/eval_results/` is empty or path is wrong | Run `python scripts/run_eval.py --all --deterministic_only` from the repo root |
| Complexity tier shows `"?"` in page 2/5 | Requests were generated before complexity scoring was added | Re-generate with `python scripts/generate_user_requests.py` |
| `aggregate.json` missing for old runs | Runs pre-date this feature | Re-save them: load and re-save via `save_eval_run()` or just re-run eval |
| LLM judge scores all `0.0` | Missing `OPENAI_API_KEY` in `.env` | Add `OPENAI_API_KEY=sk-...` to `.env` and restart |
| UI hangs on page load | Large `results.jsonl` being scanned | Use `store.flat_results(run_ids=[...])` to scope to specific runs |
| Job shows "running" indefinitely | Process exited before updating status | Click **Refresh** in the sidebar; the manager will detect the dead PID |
| Mixed metric version warning in page 2 | Different eval runs used different `metric_version` | Filter to a single version using the "Metric version" sidebar filter |
