# Compressor — Context Compression Strategies

The compressor distills a growing ReAct trajectory into a structured `CompressedState`. That state replaces the full trajectory in the agent's context window, freeing space for more planning steps within the LLM's context budget.

The compressor is the **only component being trained** in this project. The planning LLM is never fine-tuned.

---

## Files

| File | Role |
|---|---|
| `compressor/base.py` | `CompressorBase` ABC — universal interface |
| `compressor/trainable_base.py` | `TrainableCompressorBase` ABC — PPO training contract |
| `compressor/mcts_aware.py` | `MCTSAwareCompressor` ABC — intermediate ABC for compressors that consume an MCTS tree |
| `compressor/template.py` | `CompressedStateTemplate` — fixed 6-section schema renderer/parser |
| `compressor/identity_compressor.py` | `IdentityCompressor` — full trajectory + trainable scalar param (PPO baseline) |
| `compressor/dummy_compressor.py` | `DummyCompressor` — simple RNN/seq2seq baseline |
| `compressor/llm_compressor.py` | `LLMCompressor` — LLM-as-compressor (non-trainable, evaluation baseline) |
| `compressor/transformer_compressor.py` | `TransformerCompressor` — fine-tunable seq2seq (flan-t5, BART) |
| `compressor/hybrid_compressor.py` | `HybridCompressor` — structured slots + free-form LLM output |
| `compressor/llm_mcts_compressor.py` | `LLMMCTSCompressor` — LLM compressor with MCTS tree awareness |
| `compressor/structured_selective_distiller.py` | `StructuredSelectiveDistiller` (SSD) — Design 1: section-aware cross-attention distillation (standalone, no MCTS) |
| `compressor/tree_gat.py` | `PathSetEncoder` — 2-layer dense multi-head attention for encoding MCTS path-sets |
| `compressor/mcts_gat_distiller.py` | `MCTSGraphAttentionDistiller` (TGAD) — Design 2: T5+GAT tree-conditioned distillation |
| `compressor/lora_utils.py` | LoRA injection helpers |
| `compressor/reward_predictor.py` | `RewardPredictorComponent` — lightweight linear reward predictor |

---

## The Two Base Classes

### `CompressorBase` (all compressors)

```python
class CompressorBase(ABC):

    @abstractmethod
    def compress(
        self,
        trajectory: TrajectoryModel,
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        """
        trajectory      — frozen TrajectoryModel from Trajectory.to_model()
        previous_state  — last CompressedState, or None on first compression
        """

    def get_log_probs(self, trajectory_text: str, compressed_text: str) -> torch.Tensor:
        # Non-trainable default: raises LogProbsNotSupportedError
        raise LogProbsNotSupportedError(...)

    def get_trainable_parameters(self) -> list:
        return []   # non-trainable default

    def is_trainable(self) -> bool:
        return len(self.get_trainable_parameters()) > 0
```

All compressors implement `compress()`. Non-trainable compressors (LLM, Hybrid) stop here.

### `TrainableCompressorBase` (PPO-compatible compressors)

> **WARNING — `get_log_probs` implementation:** do **not** pass `labels=target_ids` to a seq2seq model's `forward()` inside `get_log_probs`. That computes an internal cross-entropy loss and returns the wrong tensor — the per-token log-probs you need are not in the loss scalar. Instead, run the decoder forward pass with `decoder_input_ids=action_ids`, then gather `log_softmax(logits, dim=-1)` at the actual action token indices. See the *Common Mistakes* table below for the symptom this causes.

```python
class TrainableCompressorBase(CompressorBase):

    @abstractmethod
    def get_log_probs(
        self, trajectory_text: str, compressed_text: str
    ) -> torch.Tensor:
        """
        Per-token log p(compressed_token | trajectory_context).
        Shape: (sequence_length,)
        Used by CompressorPolicy to compute the PPO policy ratio.
        """

    @abstractmethod
    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return params that receive gradient updates during PPO."""

    @abstractmethod
    def save_checkpoint(self, path: str) -> None: ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> None: ...

    # Concrete helpers provided by the base class:
    def apply_lora(self, lora_config: LoRAConfig) -> None: ...
    def freeze_base_layers(self) -> None: ...
```

### `MCTSAwareCompressor` (MCTS-aware compressors)

Intermediate ABC extending `CompressorBase`. Compressors that consume MCTS search trees inherit from this instead of directly from `CompressorBase` or `TrainableCompressorBase`:

```python
class MCTSAwareCompressor(CompressorBase):
    @abstractmethod
    def compress_with_tree(
        self,
        trajectory: TrajectoryModel,
        mcts_tree: MCTSTreeRepresentation,
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        """Tree-aware distillation. Called instead of compress() when MCTS is active."""
```

Both `compress()` **and** `compress_with_tree()` must be implemented. `compress()` serves as the non-MCTS fallback — it runs whenever `agent.mode != mcts_compressor`, when no `MCTSController` is wired in, or when the compressor is used standalone (e.g., in ablation studies).

`compress_with_tree()` is only called when all three conditions hold in `ReActAgent`: mode is `MCTS_COMPRESSOR`, the compressor is an `MCTSAwareCompressor` instance, and `mcts_controller is not None`. This triple-guard prevents runtime errors from mismatched configs.

---

## CompressedStateTemplate — The 6-Section Schema

The 6-section schema is deliberately structured rather than free-form. A free-form summary works for human readers but causes two problems for an agent: (1) the rendering pipeline cannot validate that all required information was produced, and (2) re-injection requires templated anchors so the agent knows *which part of the summary* to update its constraint ledger from. The `## SECTION_NAME ##` sentinel headers satisfy both: `template.validate()` checks all headers exist, and the agent's prompt instructs it to locate the `HARD_CONSTRAINT_LEDGER` section specifically when tracking constraint status.

Every `CompressedState` must contain exactly these sections, in this order:

| Section | What to put here |
|---|---|
| `HARD_CONSTRAINT_LEDGER` | List all hard constraints with status: `satisfied` / `violated` / `unknown`. Include constraint IDs so the reward function can evaluate them. |
| `SOFT_CONSTRAINTS_SUMMARY` | Free-text summary of soft constraint progress (preferences partially or fully met). |
| `DECISIONS_MADE` | Bullet list of all confirmed bookings: flights (with `booking_ref`), hotels, events. Include costs. |
| `OPEN_QUESTIONS` | What the agent still needs to resolve before it can finish planning. |
| `KEY_DISCOVERIES` | Relevant facts learned from tool calls: city IDs, price ranges, availability info. |
| `CURRENT_ITINERARY_SKETCH` | Day-by-day outline of the emerging itinerary (partial is fine). |

The template renders and parses these sections using `## SECTION_NAME ##` sentinel headers:

```python
template = CompressedStateTemplate()

# Render a CompressedState to string (for agent context injection)
text = template.render(state)

# Parse LLM output back to CompressedState (for storing in EpisodeLog)
state = template.parse(llm_output_text)

# Validate (raises CompressedStateRenderError if sections are missing)
template.validate(state)
```

A compressor that omits any required section raises `CompressedStateRenderError`. This is intentional — it prevents the RL policy from exploiting partial outputs.

---

## Compressor Implementations

### `IdentityCompressor`

The **PPO baseline**. Returns the full trajectory as-is (no information loss), plus a single trainable scalar parameter `_dummy_param` that makes it compatible with the PPO training loop. Use this to verify the training infrastructure is wired correctly before training a real compressor.

```yaml
# configs/compressor/identity.yaml
compressor:
  type: identity
```

### `LLMCompressor`

The **LLM-summary baseline**. Makes one LLM API call per compression event to summarize the trajectory into the 6-section template. Not trainable. Use this as the "LLM Summary" evaluation condition.

```yaml
compressor:
  type: llm
  llm_model_id: openai/gpt-4o-mini
```

### `TransformerCompressor`

The **primary trainable compressor**. Wraps a HuggingFace seq2seq model (default: `google/flan-t5-small`). The encoder processes the trajectory, the decoder generates the 6-section CompressedState. Supports LoRA for parameter-efficient fine-tuning.

```yaml
compressor:
  type: transformer
  model_name_or_path: google/flan-t5-small
  use_lora: false    # set true to enable LoRA adapters
  lora:
    r: 8
    alpha: 16
    dropout: 0.05
    target_modules: ["q", "v"]
```

### `DummyCompressor`

A minimal trainable compressor used for gradient-flow tests. Implements a tiny character-level LSTM encoder + linear decoder. Never produces meaningful compressions — its purpose is to verify the training loop plumbing.

### `HybridCompressor`

Combines structured slot-filling (budget remaining, constraint statuses extracted by regex) with a free-form LLM summary for the narrative sections. Not currently trainable (inherits `CompressorBase`).

### `StructuredSelectiveDistiller` (SSD — Design 1)

Section-aware cross-attention distillation. No MCTS required.

```yaml
compressor:
  type: structured_selective
```

**Architecture:** Each trajectory step is encoded independently by a frozen T5-small encoder. Five learned section-query vectors (one per free-form template section) cross-attend over all step embeddings. The attention weights are the routing policy: they select which trajectory steps matter most for each section (`DECISIONS_MADE`, `OPEN_QUESTIONS`, etc.). The five attended context vectors (plus a constraint-ledger embedding) are packed into a 6-vector sequence and fed to a T5 decoder (with LoRA) to generate the full 6-section output.

**Why explicit routing?** The baseline `TransformerCompressor` compresses the full trajectory as a single token stream, forcing the T5 decoder to learn implicit content routing from sparse RL reward signals alone. SSD makes routing explicit — the cross-attention weights are differentiable and directly trainable end-to-end via PPO, and they can be visualised per-step for ablation analysis.

**Memory (Colab T4):** ~600 MB total (T5-small encoder fp16 frozen + LoRA adapters + training activations at batch=32).

### `MCTSGraphAttentionDistiller` (TGAD — Design 2)

T5 + Graph Attention Network tree-conditioned distillation. Requires `agent=react_mcts`.

```yaml
compressor:
  type: mcts_gat
```

**Architecture pipeline:**
1. Materialise paths from `MCTSTreeRepresentation` (best path + up to K alternative paths).
2. Encode each path via a **frozen** T5-small encoder → mean-pool to a fixed-size embedding.
3. Concatenate with structural path features (`q_value`, `is_verified`, `is_best_path`, depth) projected through a small MLP.
4. Run a 2-layer `PathSetEncoder` (dense multi-head attention, implemented in `tree_gat.py`) over the N path node embeddings.
5. Score each path with a linear → sigmoid Importance Scorer. During training: Gumbel-softmax top-K (differentiable); during inference: hard top-K.
6. Attention-weighted pool selected paths → single context vector.
7. T5 decoder + LoRA generates all 6 template sections plus `top_candidates` and `tradeoffs`.

**Key advantage over `LLMMCTSCompressor`:** Fully neural — zero LLM API calls at distillation time (~10 ms on T4 vs. ~1–3 s per LLM call). The `is_verified` feature explicitly distinguishes real tool observations from synthetic MCTS expansion steps; the GAT learns to down-weight speculative branches.

**Recommended training curriculum:**
- Phase 1 (supervised pre-training, ~1 hr): Run `LLMMCTSCompressor` on 200 offline episodes to collect `(MCTSTreeRepresentation, CompressedState)` pairs. Train TGAD via cross-entropy on the token sequences to warm up the GAT + decoder weights.
- Phase 2 (PPO fine-tuning, ~4 hr): PPO with `num_simulations=10` (reduced from eval-time 50) using `ppo_mcts.yaml` config.

See [docs/TRAINING.md — MCTSGATDistiller Training Notes](TRAINING.md#mctsgat-distiller-training-notes) for hyperparameter guidance and known instability patterns.

**Memory (Colab T4):** ~300 MB total (T5-small encoder fp16 frozen + T5-small decoder + LoRA + PathSetEncoder + optimizer state).

#### `PathSetEncoder` (`tree_gat.py`)

The attention backbone shared between TGAD layers. Implements dense multi-head attention (not sparse GAT) because MCTS trees in this project have at most 3–5 paths — for this small cardinality, PyTorch Geometric or DGL add dependency overhead with no practical benefit. Two attention layers capture:
- Layer 1 (bottom-up proxy): all nodes attend to each other freely.
- Layer 2 (top-down proxy): all nodes attend to each other, with the best-path node boosted via an additive bias (root anchor).

No external graph library required — standard `torch.nn.MultiheadAttention` with a node mask.

### `LLMMCTSCompressor`

Extends `LLMCompressor` with MCTS tree information injected into the compression prompt. Used with `agent=react_mcts`. Not trainable.

**Required config pairing:** `LLMMCTSCompressor` only produces MCTS-enriched compressions when run with `agent=react_mcts` (which sets `mode: mcts_compressor` and provides the `agent.mcts` config sub-tree for `MCTSController`). Using it with any other agent config logs a startup warning and falls back to standard LLM compression — the MCTS search never runs.

```bash
# Correct — MCTS search runs, MCTSStats populated in EpisodeLog
python scripts/run_episode.py agent=react_mcts compressor=llm_mcts

# Incorrect — MCTS search does NOT run (warning logged at startup)
python scripts/run_episode.py agent=react_default compressor=llm_mcts
```

**Architectural role — "lookahead-informed compression":** MCTS in this project is not a step-level action selector. It runs at each compression event (every `compress_every_n_steps` steps) and generates a shallow tree of synthetic candidate trajectories using the LLM. Each candidate is scored heuristically (tool success rate, booking depth, constraint coverage). The resulting `MCTSTreeRepresentation` is passed to `compress_with_tree()`, which distills the best-path and alternatives into `CompressedState.top_candidates` and `.tradeoffs`. This enriched compressed state is then injected into the agent's context for the next planning window.

MCTS does not execute real tool calls — all branches are synthetic. The tree is therefore a lookahead over plausible (but unconfirmed) futures, not ground-truth outcomes. With a 5-step compression window, the tree explores shallow futures. Moving compression (and MCTS) to run before every action would increase signal fidelity at higher API cost.

---

## Adding a New Compressor

### Non-trainable compressor

Use this when the compression logic is rule-based or delegates to an external LLM.

```python
# compressor/my_compressor.py
from optimized_llm_planning_memory.compressor.base import CompressorBase
from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate
from optimized_llm_planning_memory.core.models import CompressedState, TrajectoryModel


class MyCompressor(CompressorBase):
    def __init__(self) -> None:
        self._template = CompressedStateTemplate()

    def compress(
        self,
        trajectory: TrajectoryModel,
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        # Build the CompressedState from the trajectory
        # All 6 template sections must be populated.
        state = CompressedState(
            compressed_id=str(uuid.uuid4()),
            request_id=trajectory.request_id,
            step_index=trajectory.total_steps,
            summary="...",
            key_decisions=["Booked flight FL-001"],
            active_constraints=[],
            explored_options=[],
            next_actions=["Book hotel in Paris"],
            hard_constraint_ledger=...,
            soft_constraints_summary="...",
            decisions_made="...",
            open_questions="...",
            key_discoveries="...",
            current_itinerary_sketch="...",
        )
        self._template.validate(state)
        return state
```

### Trainable compressor (PPO-compatible)

```python
# compressor/my_trainable_compressor.py
import torch
import torch.nn as nn
from optimized_llm_planning_memory.compressor.trainable_base import TrainableCompressorBase
from optimized_llm_planning_memory.core.exceptions import CompressorCheckpointError


class MyTrainableCompressor(TrainableCompressorBase):
    def __init__(self) -> None:
        # Initialize your model architecture here
        self._encoder = nn.Linear(256, 128)
        self._decoder = nn.Linear(128, 256)

    def compress(self, trajectory, previous_state=None) -> CompressedState:
        # Run your model, render output via CompressedStateTemplate
        ...

    def get_log_probs(
        self, trajectory_text: str, compressed_text: str
    ) -> torch.Tensor:
        # Compute per-token log p(compressed | trajectory)
        # Shape: (sequence_length,)
        # IMPORTANT:
        #   - Use torch.no_grad() — gradients flow through compress(), not here
        #   - Do NOT use teacher forcing (labels=...) in seq2seq forward pass
        #   - Return log_softmax gathered at actual action token indices
        with torch.no_grad():
            ...
        return token_log_probs  # shape (seq_len,)

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        return list(self._encoder.parameters()) + list(self._decoder.parameters())

    def save_checkpoint(self, path: str) -> None:
        try:
            torch.save({"encoder": self._encoder.state_dict(),
                        "decoder": self._decoder.state_dict()}, path)
        except Exception as e:
            raise CompressorCheckpointError(f"Failed to save: {e}") from e

    def load_checkpoint(self, path: str) -> None:
        try:
            ckpt = torch.load(path, map_location="cpu")
            self._encoder.load_state_dict(ckpt["encoder"])
            self._decoder.load_state_dict(ckpt["decoder"])
        except Exception as e:
            raise CompressorCheckpointError(f"Failed to load: {e}") from e
```

### Register it

**1. Export from `compressor/__init__.py`:**
```python
from .my_trainable_compressor import MyTrainableCompressor
```

**2. Add a config file `configs/compressor/my_compressor.yaml`:**
```yaml
compressor:
  type: my_compressor
  # add any compressor-specific params here
```

**3. Wire the type string in the factory** (wherever compressors are constructed from config, typically `training/trainer.py` or `scripts/run_episode.py`):
```python
if config.compressor.type == "my_compressor":
    compressor = MyTrainableCompressor()
```

**4. Write tests** in `tests/test_compressor/`. At minimum, run the cross-compressor compliance fixture in `test_comprehensive.py` by adding your class to the `trainable_compressor` parametrize list.

---

## LoRA Configuration

LoRA adapters can be applied to any `TrainableCompressorBase` subclass that holds a HuggingFace model:

```python
from optimized_llm_planning_memory.core.config import LoRAConfig

lora_config = LoRAConfig(r=8, alpha=16, dropout=0.05, target_modules=["q", "v"])
compressor.apply_lora(lora_config)

# After applying LoRA, get_trainable_parameters() returns only adapter params
params = compressor.get_trainable_parameters()
```

Enable LoRA via YAML:
```yaml
compressor:
  type: transformer
  use_lora: true
  lora:
    r: 16
    alpha: 32
    dropout: 0.1
    target_modules: ["q", "v", "k"]
```

---

## Compressor Development Workflow

1. Start in `notebooks/07_compressor_dev.ipynb` — it loads pre-built 2/6/8/12-step trajectory contexts from `data/compressor_dev/contexts/` so you can test `compress()` output interactively without running a full episode.
2. Verify the 6 sections are all populated and pass `template.validate()`.
3. For trainable compressors: verify gradient flow with `optimizer.step()`, then run `save_checkpoint` → `load_checkpoint` roundtrip.
4. Run the test suite: `pytest tests/test_compressor/ -v -m "not slow"`.
5. Add a YAML config and test the compressor end-to-end: `python scripts/run_episode.py compressor=my_compressor`.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Calling forward with `labels=target_ids` in `get_log_probs` | Gradients computed at eval time; wrong log-probs (teacher-forced) | Use `decoder_input_ids=action_ids` and gather log-probs manually |
| Returning the wrong log-prob shape | PPO policy ratio is wrong; training is unstable | Shape must be `(sequence_length,)` — sum over vocab, not over tokens |
| `save_checkpoint` raises a generic `Exception` | Test `test_checkpoint_error_paths` fails | Always raise `CompressorCheckpointError` on failure |
| Missing template section | `CompressedStateRenderError` at runtime | Ensure all 6 sections are set before calling `template.validate()` |
| `get_trainable_parameters()` returns `[]` | `is_trainable()` returns `False`, training loop skips PPO updates | Return all parameters that should receive gradients |
