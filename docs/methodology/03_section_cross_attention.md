# Section-Aware Cross-Attention Compressor (LoRA)

> **What this doc covers:** The `StructuredSelectiveDistiller` (`structured_selective_distiller.py`). How five learnable query vectors each specialize to extract one section of the CompressedState from the trajectory. The mathematics of cross-attention and why it is the right tool for this task. Temporal position embeddings. LoRA on both encoder and decoder. How the compressor becomes interpretable — and how PPO trains it.

---

## 1. Intuition First: Five Focused Readers

Imagine you have a long log of a travel planning session — 30 steps, each with a thought, a tool call, and the result. You need to summarize it into six fixed sections: constraint status, soft preference satisfaction, decisions made, open questions, key discoveries, and the current itinerary sketch.

A naive approach: feed the entire log to a model and ask it to fill in all sections at once. The model must simultaneously track budget exhaustion, city coverage, booking confirmations, and culinary preferences while generating the output — too much to juggle at once.

The **Section-Aware Cross-Attention** approach is different. It assigns **five dedicated "readers"**, each responsible for exactly one section:

| Reader | Section it writes | What it looks for in the steps |
|---|---|---|
| Query 0 | `soft_constraints_summary` | Steps mentioning food, hotel stars, ambiance — preference signals |
| Query 1 | `decisions_made` | Steps where `book_*` or `select_*` tools succeeded |
| Query 2 | `open_questions` | Constraints not yet satisfied, failed tool calls |
| Query 3 | `key_discoveries` | `search_*` results with city IDs, price ranges, availability |
| Query 4 | `current_itinerary_sketch` | Booking confirmations in chronological order |

Each reader is a **learnable vector** that cross-attends over all step embeddings, weighting each step by how relevant it is for that section. The five resulting context vectors are packed as the encoder output for a T5 decoder, which generates the actual section text.

---

## 2. Architecture Overview

```
TrajectoryModel (N steps)
        │
        ▼
Step 1: _extract_step_texts()
        │ → ["[Step 0]\nThought: ...\nAction: ...\nObservation: ...", ...]
        ▼
Step 2: _encode_steps()
        │ T5 encoder encodes each step independently → mean-pool → [N, H]
        │ + temporal position embedding → [N, H]   (step index matters!)
        ▼
Step 3: SectionQueryAttention
        │ 5 learnable queries [5, H] × step embeddings [N, H]
        │ → cross-attention → [5, H]   (one context vector per section)
        ▼
Step 4: _build_ledger_text() + _encode_single_text()
        │ → hard constraint ledger embedding [1, H]
        ▼
Step 5: _pack_encoder_output()
        │ → [1, 6, H]  (ledger + 5 section contexts as 6 "virtual tokens")
        ▼
Step 6: T5 Decoder (LoRA on encoder + decoder)
        │ → generated template text (6 sections)
        ▼
Step 7: _parse_or_fallback()
        └── CompressedState
```

---

## 3. Step Encoding with Temporal Embeddings

### 3.1 Per-Step T5 Encoding

Each trajectory step is converted to a text string:

```
[Step 7]
Thought: Per memory, I should book the Eiffel Tower. Let me search for it.
Action: search_attractions({"city_id": "par_001", "category": "landmark"})
Observation: [{"attraction_id": "att_eiffel", "name": "Eiffel Tower", "price": 30}]
```

This string is tokenized and passed through the T5 encoder. The encoder produces a sequence of hidden states, one per token. We **mean-pool** across the token sequence to get a single step embedding:

$$\mathbf{e}_i^{\text{text}} = \frac{1}{|x_i|} \sum_{j=1}^{|x_i|} \text{T5Encoder}(x_i)_j \in \mathbb{R}^H$$

where $|x_i|$ is the number of tokens in step $i$'s text (truncated to `max_step_tokens=128`).

Each step is encoded **independently** — the encoder does not see other steps. This is intentional: the cross-attention mechanism in Step 3 is responsible for relating steps to each other, not the encoder.

### 3.2 Temporal Position Embeddings

Step order matters: "booked hotel on step 5" is less important than "cancelled hotel on step 20" when deciding what's currently active. We add a **learnable temporal embedding** indexed by step position:

$$\mathbf{e}_i = \mathbf{e}_i^{\text{text}} + \text{TemporalEmbed}(\min(i, T_{\max} - 1))$$

where `TemporalEmbed` is an `nn.Embedding` table of size $T_{\max} \times H$ (with $T_{\max} = 48$). The step index $i$ is clamped to $T_{\max} - 1$ for long episodes.

**Intuition:** Without temporal embeddings, step 2 and step 30 look identical to the cross-attention if they have the same text content (e.g., both search for Paris hotels). The temporal embedding injects positional information: "this booking is recent (step 28) vs. early (step 2)." This is analogous to positional encodings in the original Transformer, but learned (not sinusoidal) and applied at the step level (not the token level).

### 3.3 Step Windowing

For very long trajectories, keeping all steps would exceed memory limits. A sliding window is applied:

```python
# Keep first 4 anchor steps + most recent (MAX_STEPS - 4) steps
anchors = step_texts[:4]
recent  = step_texts[-(MAX_STEPS - 4):]  # MAX_STEPS = 48 total
```

The first 4 steps are always kept (they contain the initial world discovery and first bookings — foundational information). The remaining budget goes to the most recent steps (latest decisions, most current state).

---

## 4. Section Query Cross-Attention

This is the mathematical core of the compressor.

### 4.1 Setup

We have:
- **Step embeddings** $E \in \mathbb{R}^{N \times H}$ — one per trajectory step.
- **Section queries** $Q_{\text{sec}} \in \mathbb{R}^{5 \times H}$ — five learnable vectors (one per section), initialized as $\mathcal{N}(0, 0.02^2)$ and learned end-to-end via backpropagation.

The goal: use the section queries to "look up" relevant information from the step embeddings.

### 4.2 Cross-Attention Mathematics

In standard self-attention, queries and keys come from the same sequence. In **cross-attention**, queries come from one set (section queries) and keys/values come from another (step embeddings):

$$K = E W_K \in \mathbb{R}^{N \times H}, \quad V = E W_V \in \mathbb{R}^{N \times H}$$
$$Q = Q_{\text{sec}} W_Q \in \mathbb{R}^{5 \times H}$$

With $h$ attention heads and head dimension $d_h = H/h$:

**Attention scores** (how relevant is step $j$ for section $i$?):

$$S_{ij} = \frac{q_i^\top k_j}{\sqrt{d_h}} \in \mathbb{R}$$

Here $q_i \in \mathbb{R}^{d_h}$ is one head's slice of query $i$, and $k_j \in \mathbb{R}^{d_h}$ is the corresponding key for step $j$.

**Softmax normalization** (across all steps $j = 1, \ldots, N$):

$$\alpha_{ij} = \frac{\exp(S_{ij})}{\sum_{j'=1}^{N} \exp(S_{ij'})}$$

This gives a proper probability distribution: $\sum_j \alpha_{ij} = 1$ for each section $i$. The weight $\alpha_{ij}$ tells us "what fraction of section $i$'s context comes from step $j$?"

**Weighted aggregation**:

$$\mathbf{c}_i = \sum_{j=1}^{N} \alpha_{ij} v_j \in \mathbb{R}^{d_h}$$

Repeating across all $h$ heads and concatenating gives the context vector for section $i \in \mathbb{R}^H$.

**Residual connection** (adds query back to context):

$$\mathbf{c}_i^{\text{final}} = \text{LayerNorm}(\mathbf{c}_i + Q_{\text{sec}, i})$$

This is important: by adding the section query back to the output, each section's context vector retains a "memory" of what kind of information it was looking for, not just what it found.

### 4.3 Multi-Head Version

The code uses `num_heads=4`:
- Head dimension $d_h = H / 4 = 128$.
- 4 parallel attention computations with independent $W_Q^{(m)}, W_K^{(m)}, W_V^{(m)}$ per head $m$.
- Each head can specialize: one head may focus on the action verb ("book", "search", "cancel"), another on the location, another on the price.
- Outputs concatenated: $[\mathbf{c}_i^{(1)} \| \mathbf{c}_i^{(2)} \| \mathbf{c}_i^{(3)} \| \mathbf{c}_i^{(4)}] \in \mathbb{R}^H$.

### 4.4 What Each Query Learns to Attend To

After training via PPO, the queries specialize. We can visualize this by logging the attention weights $\alpha_{ij}$ during evaluation:

| Section query | Expected specialization (learned) |
|---|---|
| Query 1 (`decisions_made`) | Peaks at steps where `book_hotel`, `select_flight`, `book_event` succeeded |
| Query 2 (`open_questions`) | Peaks at steps where hard constraints appear in the request, and NO corresponding booking yet |
| Query 3 (`key_discoveries`) | Peaks at steps where `search_flights`, `search_hotels` returned new results |
| Query 4 (`current_itinerary_sketch`) | Peaks at most recent booking steps (high temporal embedding index) |
| Query 0 (`soft_constraints_summary`) | Peaks at steps mentioning hotel stars, cuisine preferences, activity categories |

The attention weight matrix $\alpha \in \mathbb{R}^{5 \times N}$ is exposed via `forward_with_weights()` for inspection and ablation studies.

---

## 5. Constraint Ledger as a Separate Channel

The hard constraint ledger is **not** generated by the section query attention. It is computed **deterministically** by `ConstraintSatisfactionEngine`:

```python
ledger_text = self._build_ledger_text(previous_state)
# → "SATISFIED: MUST_VISIT_PARIS. UNKNOWN: MUST_VISIT_ROME, BUDGET."
ledger_emb = self._encode_single_text(ledger_text)  # T5 encoder → mean-pool → [1, H]
```

This deterministic channel serves two purposes:
1. **Training ≡ Evaluation invariant:** The same `ConstraintSatisfactionEngine` used in the reward function populates the ledger. The decoder always receives ground-truth constraint status, not a neural approximation.
2. **Separating concerns:** The neural components (section queries, decoder) can focus on the natural-language sections; constraint tracking is delegated to rule-based logic that is always correct.

---

## 6. Packing the Encoder Output

The T5 decoder expects a sequence of encoder hidden states to cross-attend to. We pack 6 "virtual tokens":

```
Positions:  [0]         [1]            [2]              [3]             [4]           [5]
Content:    ledger_emb  section_ctx_0  section_ctx_1    section_ctx_2   section_ctx_3 section_ctx_4
Section:    HARD_LEDGER SOFT_CONSTR    DECISIONS_MADE   OPEN_QUESTIONS  DISCOVERIES   ITINERARY
```

$$E_{\text{decoder}} = W_{\text{proj}} \cdot \text{Concat}(\mathbf{e}_{\text{ledger}}, \mathbf{c}_0, \ldots, \mathbf{c}_4) \in \mathbb{R}^{1 \times 6 \times H}$$

where $W_{\text{proj}} \in \mathbb{R}^{H \times H}$ is a learned projection applied independently to each of the 6 positions.

**What happens at decoding time:** The T5 decoder cross-attends to this 6-token sequence. When generating the `## DECISIONS MADE ##` section, the decoder should learn to focus on position 2 (`section_ctx_1`). When generating `## HARD CONSTRAINT LEDGER ##`, it attends to position 0 (the deterministic ledger). This routing is learned implicitly through training.

---

## 7. LoRA on Both Encoder and Decoder

Unlike the GAT Distiller (which only applies LoRA to the decoder), the Section-Aware Cross-Attention compressor applies LoRA to **both** the T5 encoder and decoder.

### Why Both?

The encoder here plays a more active role: it encodes individual steps, and the quality of step embeddings directly determines how well the section queries can attend to relevant steps. If a "book_hotel" step and a "search_hotels" step produce nearly identical embeddings, the DECISIONS_MADE query cannot distinguish them.

By applying LoRA to the encoder, the system can learn step-type-specific embedding adjustments. Specifically, the encoder learns to emphasize the action verb in each step's embedding, making "book" steps clearly distinguishable from "search" steps.

### LoRA Formula (Recap)

For any attention projection matrix $W \in \mathbb{R}^{d \times k}$:

$$W' = W + \frac{\alpha}{r} B A, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k}$$

with $B_0 = 0$, $A_0 \sim \mathcal{N}(0, \sigma^2)$ at initialization.

The scaling factor $\alpha/r$ controls the effective learning rate of the adapter relative to the base weights. A common setting is $\alpha = 2r$, giving a scale of 2.

### Parameter Counts (Example)

For `flan-t5-small` with $H = 512$ and $r = 8$:

| Module | Full fine-tune params | LoRA params | Reduction |
|---|---|---|---|
| Encoder Q-projection | 262,144 | 8,192 | 32× |
| Encoder V-projection | 262,144 | 8,192 | 32× |
| Decoder Q-projection | 262,144 | 8,192 | 32× |
| Decoder V-projection | 262,144 | 8,192 | 32× |
| **Total (4 modules)** | **1,048,576** | **32,768** | **32×** |

Plus the section query parameters: $5 \times 512 = 2,560$ params — tiny but high-leverage.

---

## 8. RL Training: get_log_probs()

PPO requires computing $\log \pi_\theta(a | s)$ — the log-probability of producing the compressed text $a$ from trajectory text $s$.

The `get_log_probs()` method:

1. **Split trajectory text** back into individual step strings (reversing `to_text()`).
2. **Apply windowing** (same `_apply_step_window()` as in `compress()`).
3. **Encode steps** → `[N, H]` with temporal embeddings.
4. **Section query attention** → `[5, H]`.
5. **Encode ledger** from compressed text → `[1, H]`.
6. **Pack** → `[1, 6, H]`.
7. **Teacher-forcing forward pass**: feed decoder the target compressed tokens shifted right by one, compute logits:

$$\text{logits}_t = \text{T5Decoder}(y_{<t}, E_{\text{decoder}}) \in \mathbb{R}^{V}$$

8. **Log-softmax and gather**:
$$\log \pi_\theta(y_t | y_{<t}) = \log \frac{\exp(\text{logits}_t[y_t])}{\sum_{v} \exp(\text{logits}_t[v])}$$

9. **Return** the vector of per-token log-probs (masking padding tokens).

**Why teacher-forcing?** During training, we know the target sequence (the compressor's own output from the forward pass). Feeding the true tokens as decoder input is faster and more stable than sampling: each position's log-prob is computed in parallel (one forward pass for all $T$ positions), rather than autoregressively.

---

## 9. Interpretability: Attention Weight Logging

A unique advantage of the section query design is **interpretability**. The attention weight matrix $\alpha \in \mathbb{R}^{5 \times N}$ (5 sections × $N$ steps) can be logged during evaluation:

```python
# From structured_selective_distiller.py
context, attn_weights = self._section_attention.forward_with_weights(step_embs)
# attn_weights: [5, N] — which steps each section attended to most

# Log for analysis:
for section_idx, section_name in enumerate(SECTION_NAMES):
    top_step = attn_weights[section_idx].argmax().item()
    top_weight = attn_weights[section_idx].max().item()
    logger.debug(f"{section_name}: top step = {top_step} (weight = {top_weight:.3f})")
```

This supports the project's ablation study goals: we can check whether the DECISIONS_MADE query is actually peaking at booking steps, or whether the model is doing something unexpected. If the attention patterns are interpretable, we have evidence that the compressor has learned the right inductive bias.

---

## 10. Comparison with GAT Distiller

| Dimension | GAT Distiller | Section Cross-Attention |
|---|---|---|
| **Primary input** | MCTS tree (best + alt paths) | Linear trajectory |
| **Key mechanism** | Path-set attention (paths attend to paths) | Section-query cross-attention (queries attend to steps) |
| **LoRA scope** | Decoder only | Encoder + Decoder |
| **MCTS required?** | Yes (for `compress_with_tree()`; has fallback) | No |
| **Forward direction** | Tree → compress → agent | Trajectory → compress → agent |
| **Interpretability** | Via Q-values, structural features | Via per-section attention weights |
| **Inductive bias** | Best-path dominance, tree context | Section-specific step retrieval |
| **MCTS-specific fields** | `top_candidates`, `tradeoffs` | None |

They are complementary: the GAT Distiller leverages lookahead (MCTS tree), while the Section-Attention model leverages learned retrospection (trajectory cross-attention). The research design allows comparing them directly under the same RL training setup.

---

## 11. Big Picture Tie-In

The Section-Aware Cross-Attention compressor addresses the question: *can a neural architecture alone — without explicit lookahead search — learn to compress a planning trajectory into a compact, action-guiding memory state?*

The answer after RL training: the section queries learn to route their attention to the trajectory steps most relevant for each output section. The DECISIONS_MADE query concentrates on booking confirmations; the OPEN_QUESTIONS query attends to unresolved constraint mentions; the KEY_DISCOVERIES query focuses on search results.

This is fundamentally a **learned routing** problem: given a long heterogeneous trajectory, extract the right information for each downstream use. The cross-attention architecture solves this cleanly by parameterizing "what each section is looking for" as a learnable vector, and training the whole system end-to-end via PPO reward signal.

If this compressor matches or exceeds the MCTS-based baseline, it suggests that *structural lookahead is not necessary* — a well-trained compressor can implicitly learn what to include in the memory state through experience alone. This is a key empirical question the ablation study is designed to answer.
