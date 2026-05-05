# Graph-Attention Distiller over MCTS Tree

> **What this doc covers:** The `MCTSGraphAttentionDistiller` (`mcts_gat_distiller.py`, `tree_gat.py`). How it reads the MCTS tree as a set of path embeddings, runs graph-style attention across them, selects the most important via Gumbel-softmax top-K, and generates a structured `CompressedState` via a T5 decoder. How LoRA makes this trainable at low cost. How PPO's reward signal improves it over time.

---

## 1. Intuition First: Reading a Tree of Plans

After MCTS runs 50 simulations, it has identified several promising future paths through the planning space:
- **Best path:** Book Eiffel Tower → Paris→Rome flight → Rome hotel
- **Alt path 1:** Rome flight first → Eiffel Tower → Rome hotel (slightly lower Q-value)
- **Alt path 2:** Eiffel Tower → food market → Rome (different activity mix)

Each path is a trajectory — a sequence of (thought, action, observation) steps. The **GAT Distiller** answers the question: *given these N candidate paths, what single compact memory state should we give the agent to maximize its planning quality in the next steps?*

The answer requires:
1. **Understanding each path** — encode its trajectory text into a dense vector.
2. **Blending path information** — let paths "talk to each other" to identify what they agree on (book Eiffel Tower soon) vs. where they differ (Rome flight timing).
3. **Weighting by importance** — the best path should dominate, but alternatives should contribute context about tradeoffs.
4. **Generating structured output** — a T5 decoder produces the 6-section `CompressedState` from the blended representation.

---

## 2. Architecture Overview

```
MCTSTreeRepresentation
  best_path_trajectory (TrajectoryModel)
  alternative_paths    (list[TrajectoryModel])
  top_candidates       (list[str])
  tradeoffs            (str)
        │
        ▼
Step 1: _materialise_nodes()
        │ → N node dicts (trajectory text, Q-value, is_best, has_bookings, ...)
        ▼
Step 2: _encode_nodes()
        │ → text_embs  [N, H]  (frozen T5 encoder mean-pools each path's tokens)
        │ → struct_feats [N, 5] (Q-value, verified, is_best, path_idx, bookings)
        ▼
Step 3: StructuralFeatureProjector
        │ → node_feats [N, H]  (text + structural features fused)
        ▼
Step 4: PathSetEncoder (2-layer attention)
        │ Layer 1: all-to-all attention (paths attend to each other)
        │ Layer 2: root-anchored attention (best path given extra weight)
        │ → refined_feats [N, H]
        ▼
Step 5: ImportanceScorer + GumbelTopK
        │ importance_logits = Linear(refined_feats) → [N]
        │ GumbelTopK → attention-weighted pool → tree_context [H]
        │                                         best_path_emb [H]
        ▼
Step 6: _pack_encoder_output()
        │ → encoder_out [1, 2, H]  (2 "virtual tokens" for T5 decoder)
        ▼
Step 7: T5 Decoder (LoRA on decoder only)
        │ → generated_text (full 6-section template)
        ▼
Step 8: _parse_or_fallback()
        └── CompressedState
```

---

## 3. Step-by-Step Mathematics

### 3.1 Node Encoding

Each node (path) $i$ is encoded by running the **frozen T5 encoder** over the path's trajectory text:

$$\mathbf{t}_i = \text{MeanPool}(\text{T5Encoder}(\text{tokenize}(\tau_i))) \in \mathbb{R}^H$$

where $H$ is the T5 hidden dimension (e.g., 512 for flan-t5-small). Mean-pooling over the token sequence gives a single vector representing the entire path.

The **structural features** for node $i$ are a 5-dimensional vector:

$$\mathbf{s}_i = \begin{bmatrix} q_i \\ \mathbb{1}[\text{verified}] \\ \mathbb{1}[\text{best path}] \\ i / (N-1) \\ \mathbb{1}[\text{has bookings}] \end{bmatrix} \in \mathbb{R}^5$$

- $q_i$: the MCTS Q-value (mean simulated return) for path $i$.
- $\mathbb{1}[\text{verified}]$: 1 if all steps in the path have real tool observations, 0 if any are synthetic.
- $\mathbb{1}[\text{best path}]$: 1 only for the root's best-child path (index 0).
- $i / (N-1)$: normalized path index — encodes rank order.
- $\mathbb{1}[\text{has bookings}]$: 1 if the path includes confirmed booking actions.

These structural features let the model learn that "high-Q, best-path, has-bookings" nodes deserve different treatment than "low-Q, synthetic, alternative" nodes.

### 3.2 Structural Feature Projection (Fusion)

The `StructuralFeatureProjector` maps the 5-dim structural vector into the same space as the text embedding, then fuses them:

$$\mathbf{p}_i = \text{MLP}(\mathbf{s}_i) \in \mathbb{R}^{64}$$

$$\mathbf{f}_i = \text{LayerNorm}\left(W_{\text{fuse}}\begin{bmatrix}\mathbf{t}_i \\ \mathbf{p}_i\end{bmatrix}\right) \in \mathbb{R}^H$$

where $W_{\text{fuse}} \in \mathbb{R}^{H \times (H+64)}$ is a learned projection. LayerNorm stabilizes training by normalizing the fused activations.

**Intuition:** The structural features inject numeric information (Q-values, verification status) that the text encoder cannot reliably extract from natural language. Fusing them at this stage means the attention mechanism will treat paths with different Q-values differently even if their text embeddings are similar.

### 3.3 PathSetEncoder: Two Layers of Multi-Head Attention

The `PathSetEncoder` takes $N$ fused node embeddings $\{\mathbf{f}_1, \ldots, \mathbf{f}_N\}$ (arranged as matrix $F \in \mathbb{R}^{N \times H}$) and produces contextually refined embeddings by letting paths attend to each other.

#### Standard Multi-Head Attention (Review)

Given $N$ vectors stacked as $X \in \mathbb{R}^{N \times H}$, multi-head attention with $h$ heads proceeds:

1. Project to queries, keys, values using a fused projection:
$$[Q; K; V] = X W_{QKV}, \quad W_{QKV} \in \mathbb{R}^{H \times 3H}$$

2. Split into $h$ heads; for head $j$ with head dimension $d_h = H/h$:
$$Q_j, K_j, V_j \in \mathbb{R}^{N \times d_h}$$

3. Scaled dot-product attention:
$$A_j = \text{softmax}\left(\frac{Q_j K_j^\top}{\sqrt{d_h}} + B\right) \in \mathbb{R}^{N \times N}$$

where $B \in \mathbb{R}^{N \times N}$ is an optional **additive bias** (used here to implement root-anchoring).

4. Apply attention weights to values:
$$O_j = A_j V_j \in \mathbb{R}^{N \times d_h}$$

5. Concatenate heads and project:
$$O = \text{Concat}(O_1, \ldots, O_h) W^O \in \mathbb{R}^{N \times H}$$

6. Residual + LayerNorm:
$$X' = \text{LayerNorm}(X + O)$$

7. Feed-forward sublayer (GELU activation):
$$X'' = \text{LayerNorm}(X' + \text{FF}(X')), \quad \text{FF}(x) = W_2\,\text{GELU}(W_1 x)$$

#### Layer 1: All-to-All (Bottom-Up)

The first layer applies the above with **no additive bias** ($B = 0$, only padding mask). Every path can freely attend to every other path. After this layer, each path embedding contains information from all other paths.

**Intuition:** This is the "committee meeting" step — every candidate path shares its information with every other. Paths that book the Eiffel Tower will influence the representations of paths that don't, signaling the constraint gap.

#### Layer 2: Root-Anchored (Top-Down)

The second layer applies attention with an **additive bias** that boosts column 0 (the best-path node):

$$B_{ij} = \begin{cases} \gamma & j = 0 \quad \text{(best path column)} \\ 0 & \text{otherwise} \end{cases}$$

where $\gamma = \texttt{root\_anchor\_bias}$ is a **learnable scalar** (initialized to 0, so the network learns whether to emphasize the best path or not).

Adding $\gamma$ to the attention logits before softmax increases attention to the best-path node for every row $i$:

$$A_{i0} \propto \exp\left(\frac{q_i \cdot k_0}{\sqrt{d_h}} + \gamma\right)$$

**Intuition:** This is the "ground truth pull" step. After the committee meeting, everyone re-reads the best path more carefully. If $\gamma$ is learned to be large and positive, the model is saying "the best MCTS path is the anchor; alternatives provide context but the best path dominates."

The output is `refined_feats` $\in \mathbb{R}^{N \times H}$.

### 3.4 Importance Scoring and Gumbel Top-K

A linear scorer maps each refined node embedding to a scalar logit:

$$\ell_i = W_{\text{score}} \mathbf{r}_i \in \mathbb{R}, \quad W_{\text{score}} \in \mathbb{R}^{1 \times H}$$

These logits measure "how much does path $i$ contribute to the compression?"

#### Gumbel-Softmax for Differentiable Selection

We want to select the top-$K$ paths by importance and aggregate them, but hard argmax/top-K is not differentiable — gradients cannot flow back to learn $\ell_i$. The **Gumbel-softmax trick** provides a differentiable approximation.

**During training:** Add i.i.d. Gumbel noise and take a softmax at temperature $\tau$:

$$g_i \stackrel{\text{iid}}{\sim} \text{Gumbel}(0, 1) = -\log(-\log(u_i)), \quad u_i \stackrel{\text{iid}}{\sim} \text{Uniform}(0,1)$$

$$w_i = \frac{\exp((\ell_i + g_i)/\tau)}{\sum_j \exp((\ell_j + g_j)/\tau)}$$

The key property: in the limit $\tau \to 0$, the distribution concentrates on the argmax, mimicking hard selection. For $\tau = 1$ (training), the distribution is spread out, allowing gradient flow through all paths.

**During inference:** Hard top-$K$ selection with a very low temperature $\tau = 0.01$ (near-deterministic).

The aggregated **tree context** vector is an attention-weighted sum:

$$\mathbf{c}_{\text{tree}} = \sum_{i=1}^N w_i \mathbf{r}_i \in \mathbb{R}^H$$

**Intuition:** Instead of picking one "winning" path and discarding the rest, Gumbel top-K produces a soft blend. During training with $\tau = 1$, the gradient flows to all paths — the model learns which paths are worth attending to. During inference with $\tau \to 0$, it converges to a hard winner.

### 3.5 Packing the Encoder Output

The T5 decoder expects an "encoder output" — a sequence of hidden states to cross-attend to. We create a minimal 2-token sequence:

$$E = \begin{bmatrix} W_{\text{ctx}} \mathbf{c}_{\text{tree}} \\ W_{\text{ctx}} \mathbf{r}_0 \end{bmatrix} \in \mathbb{R}^{1 \times 2 \times H}$$

- **Position 0**: `tree_context` — the weighted aggregate of all paths.
- **Position 1**: `best_path_emb` = $\mathbf{r}_0$ — the best path's refined embedding.

The decoder's cross-attention can then attend to both positions. In practice, the decoder learns to use position 0 for overall structure and position 1 for the best-path specifics.

### 3.6 T5 Decoder: Generating the CompressedState

The T5 decoder autoregressively generates the 6-section template:

$$P(\mathbf{y} \mid E) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, E)$$

where $\mathbf{y}$ is the sequence of output tokens and $E$ is the packed encoder representation from above. The decoder uses **beam search** with 4 beams, generating up to `max_output_tokens=512` tokens.

**Crucially:** The LoRA adapters on the decoder are the only trainable parameters that touch the T5 model weights. The T5 encoder (used to encode path texts) is completely frozen — this saves memory and avoids catastrophic forgetting of the pretrained text understanding.

---

## 4. LoRA: Training the Decoder Efficiently

### The Problem with Full Fine-Tuning

A T5-small model has ~60M parameters. Fine-tuning all of them with PPO would be:
- **Memory-expensive:** storing gradients and optimizer states for 60M params.
- **Data-hungry:** PPO has high sample complexity; 60M params need many episodes.
- **Risk of forgetting:** the T5 decoder has learned useful language generation; aggressive updates can destroy this.

### The LoRA Solution

**Low-Rank Adaptation (LoRA)** freezes the original weight matrices and adds small trainable *rank-$r$ perturbations*:

$$W' = W + \Delta W = W + B A$$

where:
- $W \in \mathbb{R}^{d \times k}$ is the original (frozen) weight matrix.
- $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are the trainable LoRA matrices.
- $r \ll \min(d, k)$ is the **rank** (typically 4–16).

The number of new parameters is $r(d + k)$ instead of $dk$, a reduction by a factor of $\min(d,k)/r$.

**Example:** For a T5-small attention projection with $d = k = 512$ and $r = 8$:
- Full fine-tuning: $512 \times 512 = 262,144$ params per matrix.
- LoRA: $8 \times (512 + 512) = 8,192$ params — **32× fewer**.

### Initialization

$A$ is initialized from $\mathcal{N}(0, \sigma^2)$ (usually $\sigma = 1/r$), and $B = 0$. This ensures $\Delta W = 0$ at initialization: the model starts from the pretrained weights and gradually learns deviations.

### In This System

LoRA is applied to the **T5 decoder's attention projections** (the query and value matrices by default). The encoder is frozen entirely — it just encodes path texts into dense vectors.

```python
# From mcts_gat_distiller.py _apply_decoder_lora():
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=lora_config.r,          # e.g., 8
    lora_alpha=lora_config.alpha,   # e.g., 16  (scaling factor = alpha/r = 2)
    lora_dropout=lora_config.dropout,
    target_modules=lora_config.target_modules,  # e.g., ["q", "v"]
    bias="none",
)
self._model = get_peft_model(self._model, peft_config)
```

**Scaling:** In practice, the LoRA update is scaled by $\alpha / r$ to keep the effective learning rate stable regardless of $r$:
$$\Delta W_{\text{effective}} = \frac{\alpha}{r} B A$$

---

## 5. RL Training: How PPO Improves the GAT Distiller

### The get_log_probs() Method

PPO needs to compute the log-probability of the compressor's output tokens under the current policy, to calculate the probability ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

For a sequence output, this is the product (sum in log space) of per-token log-probabilities. The GAT Distiller's `get_log_probs()` method:

1. Encodes the input trajectory text through the same pipeline (fallback path, no tree).
2. Runs teacher-forcing: feeds the **target** compressed text as decoder input (one token shifted right), and computes the decoder logits.
3. Extracts the log-probability of each actual target token:

$$\log \pi_\theta(y_t | y_{<t}, E) = \log \text{softmax}(\text{logits}_t)_{y_t}$$

4. Returns the per-token log-probability vector (padded tokens masked to 0).

The `CompressorPolicy` then takes the **mean** across tokens (not sum) to get a scalar log-prob per sample:

$$\log \pi_\theta(a | s) = \frac{1}{T} \sum_{t=1}^{T} \log \pi_\theta(y_t | y_{<t}, E)$$

**Why mean, not sum?** The sum grows linearly with sequence length, making log-probs range from $-50$ (short) to $-5000$ (long). The PPO probability ratio $r_t = e^{\log \pi - \log \pi_{\text{old}}}$ would then be astronomically large or small. Using the mean keeps log-probs in a stable $[-15, -3]$ range regardless of sequence length.

### What Improves After Training

| Component | Before RL training | After RL training |
|---|---|---|
| `root_anchor_bias` | 0 (equal bias to all paths) | Learned value — network finds whether to weight best path heavily |
| LoRA matrices $B, A$ | $B=0, A\sim\mathcal{N}$ | Learned to produce decoder outputs that lead to high reward |
| `importance_scorer` $W_{\text{score}}$ | Random | Learned to score paths by their downstream planning utility |
| `_struct_projector` MLP | Random | Learned to weight Q-value, verification, booking flags appropriately |
| T5 decoder attention | Pretrained T5 | Fine-tuned via LoRA to generate constraint-aware template text |

---

## 6. End-to-End Forward Pass (Code Walkthrough)

```python
# mcts_gat_distiller.py compress_with_tree()

# 1. Materialise MCTS paths as node dicts
nodes = self._materialise_nodes(tree_repr)
# nodes[0] = {"text": "...", "q_value": 0.85, "is_best": True, ...}
# nodes[1] = {"text": "...", "q_value": 0.72, "is_best": False, ...}

# 2. Encode: text → [N, H], struct → [N, 5]
path_embs, struct_feats = self._encode_nodes(nodes)
# path_embs: frozen T5 encoder mean-pools each path's tokens → [N, 512]

# 3. Fuse text + structural features
node_feats = self._struct_projector(path_embs, struct_feats)  # [N, 512]
#   struct_feats[:, 0] = Q-values (e.g., [0.85, 0.72, 0.50])
#   struct_feats[:, 1] = is_verified
#   struct_feats[:, 2] = is_best   (= [1, 0, 0])
#   after fusion: all signals combined into [N, 512]

# 4. Two-layer path-set attention
refined_feats = self._path_encoder(node_feats, n_nodes=len(nodes))  # [N, 512]
#   Layer 1: all paths attend to all others
#   Layer 2: all paths additionally attend to best path (index 0) with extra bias

# 5. Score paths and soft-select top-K
importance_logits = self._importance_scorer(refined_feats).squeeze(-1)  # [N]
tree_context = self._gumbel_topk(importance_logits, refined_feats)       # [512]
best_path_emb = refined_feats[0]                                         # [512]

# 6. Pack 2 virtual tokens for T5 decoder
encoder_out = self._pack_encoder_output(tree_context, best_path_emb)  # [1, 2, 512]

# 7. Generate template via T5 decoder (LoRA active)
output_ids = self._model.generate(
    encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out),
    decoder_input_ids=prefix_ids,
    max_new_tokens=512,
    num_beams=4,
)
generated_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 8. Parse into CompressedState
return self._parse_or_fallback(
    generated_text,
    top_candidates=list(tree_repr.top_candidates[:3]),
    tradeoffs=tree_repr.tradeoffs,
    ...
)
```

---

## 7. Design Tradeoffs and Architectural Choices

| Choice | Rationale |
|---|---|
| Freeze T5 encoder | Path encoding runs N times (once per candidate path); freezing keeps it cheap and preserves pretrained text understanding |
| LoRA on decoder only | The decoder generates novel structured text; it benefits from fine-tuning. The encoder only extracts features. |
| 2 virtual tokens (tree_context + best_path_emb) | Minimal interface to T5 decoder; position 0 = global tree summary, position 1 = best path anchor |
| Gumbel-softmax over hard argmax | Differentiability: gradients flow from the decoder loss back through the importance scorer and path encoder |
| `root_anchor_bias` as learnable scalar | The network decides how much to emphasize the best MCTS path; initialized to 0 (neutral) |
| Structural features (Q-value, is_best, etc.) | Numeric signals the text encoder cannot reliably extract; injected as a side channel |

---

## 8. Big Picture Tie-In

The GAT Distiller addresses the project's central challenge — *what to remember, and what to discard* — through the lens of MCTS lookahead. Its distinguishing contribution over the LLM-evaluated MCTS baseline:

1. **Speed:** Compression is a neural forward pass (~80ms on a T4 GPU), not a sequence of LLM API calls.
2. **Calibration:** The LoRA-tuned decoder has been trained on episode outcomes; it has learned that "open hard constraint" text in the CompressedState correlates with high future reward, and generates such text accordingly.
3. **Tree awareness:** By explicitly reading the MCTS tree structure — best path vs. alternatives, Q-values, verification status — it produces compressions that encode not just "what happened" but "which future paths are worth pursuing."

After sufficient RL training, the trained value head (`CompressorPolicy._compute_value()`) can replace the LLM evaluator inside MCTS itself, collapsing the per-simulation cost from an API call to a millisecond MLP forward pass. This is the AlphaZero-style synergy described in [`04_rl_training.md`](04_rl_training.md).
