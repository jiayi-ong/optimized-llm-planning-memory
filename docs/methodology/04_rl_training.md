# RL Training Pipeline: PPO, Rewards, and the MCTS–RL Synergy

> **What this doc covers:** How the compressor is framed as a Reinforcement Learning policy. The Gymnasium environment (`CompressionEnv`). The reward function and its components. The PPO algorithm and why it is the right choice. How the value head learns to replace the MCTS evaluator. The AlphaZero-style feedback cycle.

---

## 1. The RL Framing

Training the compressor requires answering: *how do we know if a compression was good?*

The answer is causal: a compression is good if the agent plans better in the steps that follow it. "Better" means more hard constraints satisfied, fewer wasted tool calls, and a more logical itinerary.

This is precisely a **Reinforcement Learning** setup:

| RL concept | In this system |
|---|---|
| **Agent** | The compressor (either GAT Distiller or Section Cross-Attention) |
| **Environment** | The travel planning world + ReAct agent + travel simulator |
| **State** $s_t$ | The encoded trajectory text after $t$ ReAct steps |
| **Action** $a_t$ | The CompressedState token sequence produced by the compressor |
| **Reward** $r_t$ | How well the planning agent performed in the *next* $N$ ReAct steps |
| **Policy** $\pi_\theta$ | The compressor parameterized by weights $\theta$ |
| **Episode** | One complete planning task (one UserRequest) |

**The key causal chain:**

$$\underbrace{\pi_\theta(a_t | s_t)}_{\text{compressor}} \rightarrow \underbrace{a_t}_{\text{CompressedState}} \rightarrow \underbrace{\text{agent gets MemoryContext}}_{\text{injected into LLM prompt}} \rightarrow \underbrace{r_t}_{\text{constraint satisfaction in next } N \text{ steps}}$$

The compressor learns that certain compression choices (e.g., prominently flagging unsatisfied hard constraints) lead to higher agent performance and thus higher reward.

---

## 2. The Gymnasium Environment: `CompressionEnv`

### 2.1 Observation and Action Spaces

```python
# From env.py
observation_space = Box(
    low=0, high=vocab_size-1,
    shape=(max_obs_tokens,),    # e.g., 2048
    dtype=np.int32
)
action_space = Box(
    low=0, high=vocab_size-1,
    shape=(max_action_tokens,), # e.g., 512
    dtype=np.int32
)
```

Both are integer token ID sequences. The observation is the trajectory encoded as token IDs (padded with 0s). The action is the compressed state encoded as token IDs (also padded).

**Why token IDs as the action space?** The compressor generates text autoregressively. By representing the action as a token ID sequence, the Gymnasium interface can pass the compressor's output back to the environment, which decodes it, parses it as a `CompressedState`, and injects it into the agent's context.

### 2.2 Episode Structure

```
reset() ─────────────────────────────────────────────────────────────────────────────
   │ 1. Sample a UserRequest (round-robin across training set)
   │ 2. Create fresh SimulatorAdapter with episode seed
   │ 3. Create fresh ReActAgent, Trajectory, ToolRegistry
   │ 4. Run first N=steps_per_compression ReAct steps (RAW mode, no compression)
   └──► Return obs = encode(trajectory_text)

step(action: token_ids) ────────────────────────────────────────────────────────────
   │ 1. Decode action tokens → compressed_state_text
   │ 2. Parse as CompressedState (fallback to previous on failure)
   │ 3. Run next N ReAct steps WITH compressed context injected
   │ 4. Compute reward from RewardFunction
   └──► Return (obs, reward, terminated, truncated, info)
```

One **gymnasium step** = one compression event (every $N$ ReAct steps).
One **gymnasium episode** = one complete planning task.

For a 30-step episode with $N = 5$ steps per compression, there are 5 gymnasium steps per episode (plus the initial raw window = 6 total windows).

### 2.3 Why This Framing Is Correct

The environment correctly frames the compressor as a multi-step decision maker:

- **Each action's consequence is delayed.** The compressor at step $t$ produces a CompressedState that affects the agent's performance in the *next* $N$ steps, not the current ones. The environment captures this: `step()` runs the next $N$ ReAct steps *after* the action, then measures the reward.

- **States are informative.** The observation at step $t$ (the trajectory so far) contains all information the compressor needs to decide what to compress. The MDP is fully observed (from the compressor's perspective).

- **Episodes are diverse.** The training set contains multiple `UserRequest` objects with different destination combinations, budget constraints, and soft preferences. Round-robin sampling ensures the compressor sees diverse tasks and learns generalizable compression strategies.

---

## 3. The Reward Function

### 3.1 Components

The `RewardFunction.compute()` method returns a `RewardComponents` object:

$$r_t = w_{\text{hard}} \cdot r_{\text{hard}} + w_{\text{soft}} \cdot r_{\text{soft}} + w_{\text{eff}} \cdot r_{\text{eff}} + w_{\text{fail}} \cdot r_{\text{fail}} + w_{\text{cons}} \cdot r_{\text{cons}} + c_{\text{step}}$$

and at termination:

$$r_T += w_{\text{term}} \cdot r_{\text{term}} + c_{\text{bonus}} \cdot \mathbb{1}[r_{\text{hard}} = 1]$$

#### Hard Constraint Score $r_{\text{hard}} \in [0, 1]$

$$r_{\text{hard}} = \frac{|\{\text{satisfied hard constraints}\}|}{|\{\text{all hard constraints}\}|}$$

This is the fraction of must-have requirements satisfied by the current itinerary. It uses the `ConstraintSatisfactionEngine` — the **same engine** used in final evaluation, ensuring the training signal is perfectly aligned with the evaluation metric.

#### Soft Constraint Score $r_{\text{soft}} \in [0, 1]$

$$r_{\text{soft}} = \frac{\sum_i w_i \cdot \text{score}(c_i^{\text{soft}})}{\sum_i w_i}$$

A weighted average of satisfaction scores for soft (preference) constraints. Returns 1.0 when there are no soft constraints, preserving the training $\equiv$ evaluation invariant.

#### Tool Efficiency Score $r_{\text{eff}} \in [0, 1]$

$$r_{\text{eff}} = 1 - \frac{\text{redundant calls}}{\text{total calls}}$$

Penalizes repeated identical tool calls (e.g., searching the same flight route twice). Tracked by `ToolCallTracker` via argument hashing.

#### Tool Failure Penalty $r_{\text{fail}} \leq 0$

$$r_{\text{fail}} = -\frac{\text{failed calls}}{\text{total calls}} \in [-1, 0]$$

Penalizes invalid tool calls (wrong argument types, nonexistent IDs). The negative sign ensures this is always a penalty.

#### Logical Consistency Score $r_{\text{cons}} \in [0, 1]$

Checks four structural properties:
1. Dates are in chronological order across days.
2. No hotel is double-booked on overlapping nights.
3. No activities overlap within a single day.
4. Flight arrival date $\leq$ hotel check-in date.

Each check is binary (0 or 1); the score is their mean.

#### Per-Step Penalty $c_{\text{step}} < 0$

A small negative constant (e.g., $-0.01$) applied every gymnasium step. This encourages the agent to satisfy constraints efficiently rather than taking many exploratory steps.

#### Terminal Bonus $c_{\text{bonus}}$

Applied only when the episode terminates AND all hard constraints are satisfied ($r_{\text{hard}} = 1$). A large positive signal (e.g., $+0.5$) provides strong feedback for complete constraint satisfaction.

### 3.2 Reward Normalization

If `normalize=True`, the total reward is clipped to $[-1, 1]$ by dividing by the maximum achievable reward:

$$r_{\text{normalized}} = \frac{r}{\max\_possible}$$

This stabilizes PPO training by preventing very large rewards from dominating gradient updates.

---

## 4. Proximal Policy Optimization (PPO)

### 4.1 The Core Problem: Policy Gradient

Standard policy gradient estimates the gradient of expected return:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \hat{A}_t \right]$$

where $\hat{A}_t$ is the **advantage** — "how much better was this action than the average action in this state?"

**The problem:** If we take a large gradient step, the new policy $\pi_{\theta + \Delta\theta}$ might differ drastically from $\pi_\theta$. The next rollout is then collected under a very different policy, and the gradient estimate (which assumed we're near $\pi_\theta$) is no longer valid. Training can become unstable or diverge.

### 4.2 PPO's Solution: Clipped Probability Ratio

PPO reformulates the objective using the **probability ratio**:

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

If $r_t(\theta) > 1$: the new policy assigns higher probability to this action than the old one.
If $r_t(\theta) < 1$: the new policy assigns lower probability.

The PPO **clipped objective** prevents ratio from going too far from 1:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\epsilon,\ 1+\epsilon) \hat{A}_t \right) \right]$$

**Intuition:** If $\hat{A}_t > 0$ (this action was better than average) and $r_t > 1 + \epsilon$ (we're already increasing its probability too aggressively), the clipped version stops the increase. Conversely, if $\hat{A}_t < 0$ and $r_t < 1 - \epsilon$, clipping prevents further decrease. The result: the policy can only move a bounded distance from the old policy per update, preventing instability.

The parameter $\epsilon$ (called `clip_epsilon` or `clip_range` in the codebase) is typically 0.2.

### 4.3 The Advantage Estimate

The advantage $\hat{A}_t$ answers: "compared to the value function's prediction, how good was the reward actually received?"

PPO uses **Generalized Advantage Estimation (GAE)** for a lower-variance estimate:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the **temporal difference (TD) error**.

Parameters:
- $\gamma \in (0, 1)$ is the **discount factor** — future rewards are worth less than immediate ones (e.g., $\gamma = 0.99$).
- $\lambda \in [0, 1]$ is the **GAE parameter** — controls the bias-variance tradeoff: $\lambda = 0$ gives 1-step TD (low variance, high bias); $\lambda = 1$ gives Monte Carlo returns (unbiased, high variance).

**Intuition:** The TD error $\delta_t$ tells us the one-step surprise: "we expected $V(s_t)$ worth of future reward from this state, but we actually got $r_t + \gamma V(s_{t+1})$." The GAE sums these surprises over a geometric-decay window, giving a better estimate than either 1-step or full-rollout.

### 4.4 Full PPO Objective

PPO optimizes three terms jointly:

$$L(\theta) = \underbrace{L^{\text{CLIP}}(\theta)}_{\text{policy improvement}} - \underbrace{c_1 L^{\text{VF}}(\theta)}_{\text{value function accuracy}} + \underbrace{c_2 H[\pi_\theta]}_{\text{entropy bonus}}$$

- **Value function loss** $L^{\text{VF}}$: Mean squared error between the value head's prediction $V_\theta(s_t)$ and the empirical return $\hat{R}_t = \sum_{l \geq 0} \gamma^l r_{t+l}$. This trains the value head to accurately predict future reward.

- **Entropy bonus** $H[\pi_\theta] = -\mathbb{E}[\log \pi_\theta(a|s)]$: Encourages the policy to remain stochastic (exploratory). Without this, the policy collapses to a deterministic strategy that may be locally optimal but globally poor.

- $c_1, c_2$ are weighting coefficients (`vf_coef` and `ent_coef` in the config).

### 4.5 The PPO Update Cycle

```
Collect rollout (n_steps episodes under π_θ_old):
  For each gymnasium step:
    obs → CompressorPolicy.forward() → action tokens
    action → env.step() → (next_obs, reward, done)
    Store (obs, action, reward, done, log_prob, value_estimate)

For n_epochs passes over the rollout buffer:
  For each minibatch of size batch_size:
    Compute r_t(θ) = exp(log_π_θ(a|s) - log_π_θ_old(a|s))
    Compute Â_t via GAE
    Compute L^CLIP, L^VF, H[π_θ]
    Gradient step: θ ← θ - α ∇_θ (-L(θ))
    Gradient clip: ||∇_θ|| ≤ max_grad_norm

Update θ_old ← θ
```

---

## 5. The Value Head

### 5.1 Architecture

The `CompressorPolicy` in `training/policy.py` has a small **value head** that estimates the expected future reward from any state:

```python
# Token embedding + mean-pooling + MLP
self._token_embed = nn.Embedding(vocab_size, 64)    # vocab_size → 64-dim
self._value_net = nn.Sequential(
    nn.Linear(64, value_hidden_dim),  # e.g., 64 → 256
    nn.ReLU(),
    nn.Linear(value_hidden_dim, 1),   # 256 → scalar
)
```

**Forward pass:**

1. Embed observation token IDs: $E_{\text{obs}} \in \mathbb{R}^{L \times 64}$ (L = max_obs_tokens).
2. Apply padding mask: tokens with ID 0 (padding) are zeroed out before pooling.
3. Mean-pool over non-padding positions: $\bar{e} \in \mathbb{R}^{64}$.
4. Pass through MLP: $V(s) = W_2 \text{ReLU}(W_1 \bar{e}) \in \mathbb{R}$.

**Why not use the full T5 encoder?** The T5 encoder is the compressor model itself, which can be large. The value head needs to be cheap (it runs on every observation during rollout collection). A small embedding + MLP is fast and gives the value head independence from the policy — important for PPO's actor-critic stability.

### 5.2 What the Value Head Learns

Initially, the value head predicts random values. After training:

- States with many confirmed bookings and satisfied constraints → high predicted value.
- States with unsatisfied hard constraints and low budget remaining → low predicted value.
- States where the agent has been looping (many redundant calls) → low predicted value.

This is calibrated to the *actual* reward function (constraint satisfaction scores), not a natural-language heuristic. This is what makes the trained value head superior to the LLM evaluator in MCTS simulations.

---

## 6. MCTS + RL: The Synergy (AlphaZero-Style)

This is the key theoretical contribution of combining MCTS with RL in this system.

### 6.1 Before RL Training

```
MCTS simulation scoring: leaf node → gpt-4o-mini API call → float ∈ [0,1]
Cost: ~1 API call per simulation × 50 simulations = 50 API calls per compression
```

The LLM evaluator is:
- **Expensive**: ~$0.01 per 1K tokens × ~300 tokens = ~$0.003 per simulation
- **Slow**: 50–200ms per call
- **Uncalibrated**: not aligned with the actual reward function

### 6.2 After RL Training

```
MCTS simulation scoring: leaf node → value_net(encode(trajectory)) → float ∈ [0,1]
Cost: ~1 MLP forward pass per simulation × 50 simulations = negligible
```

The trained value head is:
- **Cheap**: microseconds per forward pass
- **Fast**: batched, GPU-accelerated
- **Calibrated**: trained on the same reward signal as the full system

This collapses the MCTS evaluation cost by 3–4 orders of magnitude.

### 6.3 The Positive Feedback Loop

```
                    ┌──────────────────────────────────────────┐
                    │                                          │
                    ▼                                          │
Episode rollout: MCTS.search()                                │
  ↓ Uses trained value_net for simulation scoring             │
  ↓ Uses trained compressor policy for context injection      │
CompressedState produced                                      │
  ↓ Agent uses compressed context for next N steps            │
Episode completes → RewardFunction.compute()                  │
  ↓                                                           │
PPO update:                                                   │
  ↓ Improves compressor policy (better compressions)         │
  ↓ Improves value head (better simulation scoring)           │
                    └──────────────────────────────────────────┘
```

Each iteration:
1. **Better compressions** → agent plans better → higher reward.
2. **Better value head** → MCTS explores more efficiently → better compressed states.
3. **Better compressed states** → agent sees clearer priorities → fewer constraint violations.

This is exactly the AlphaZero cycle: policy improvement via tree search, value improvement via self-play outcomes.

### 6.4 Why This Is Better Than Either Alone

| System | Lookahead? | Cross-episode learning? | Evaluation quality | Cost per episode |
|---|---|---|---|---|
| Greedy agent | No | No | N/A | Low |
| MCTS alone | Yes | No | LLM heuristic (noisy) | High (LLM calls) |
| RL alone (no MCTS) | No | Yes | Reward-calibrated | Medium |
| MCTS + RL | Yes | Yes | Reward-calibrated | Low (value net) |

---

## 7. Training Configuration and Hyperparameters

### 7.1 Key Hyperparameters

| Parameter | Typical value | What it controls |
|---|---|---|
| `n_steps` | 128 | Rollout buffer size (gymnasium steps before each PPO update) |
| `batch_size` | 32 | Minibatch size within each PPO epoch |
| `n_epochs` | 4 | Number of passes over the rollout buffer per update |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE bias-variance tradeoff |
| `clip_epsilon` | 0.2 | PPO probability ratio clip range |
| `ent_coef` | 0.01 | Entropy bonus coefficient |
| `vf_coef` | 0.5 | Value function loss coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |
| `base_lr` | 3e-4 | Base learning rate |
| `lr_schedule` | "cosine" | Learning rate decay |

### 7.2 Learning Rate Schedule

A cosine decay from `base_lr` to `0.1 × base_lr`:

$$\text{lr}(p) = \text{lr}_{\min} + \frac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min})\left(1 + \cos(\pi p)\right)$$

where $p \in [0, 1]$ is the fraction of training elapsed (1 = done). This is computed from `progress_remaining = 1 - p`.

### 7.3 Reward Weights

```python
weights = RewardWeights(
    hard_constraint   = 2.0,   # Highest — this is the primary objective
    soft_constraint   = 0.5,
    tool_efficiency   = 0.3,
    tool_failure_penalty = 0.2,
    logical_consistency  = 0.3,
    terminal_itinerary   = 1.0,
)
step_penalty   = -0.01   # per gymnasium step
terminal_bonus = +0.5    # if all hard constraints satisfied at termination
```

The hard constraint weight of 2.0 ensures the policy optimization focuses primarily on satisfying must-have requirements, with efficiency and soft preferences as secondary objectives.

---

## 8. Callbacks and Monitoring

The training loop uses five callbacks to log training dynamics:

| Callback | What it logs |
|---|---|
| `EpisodeLogCallback` | Per-episode hard/soft/efficiency scores, rolling reward mean |
| `CompressorCheckpointCallback` | Compressor weights at regular intervals |
| `RewardPredictorCallback` | Trains optional reward predictor model |
| `MCTSMetricsCallback` | MCTS search statistics (nodes, depth, root Q-value) |
| `PPOUpdateMetricsCallback` | Policy loss, value loss, entropy, KL, clip fraction |

All metrics are logged to TensorBoard and to JSONL files for custom analysis.

**Key diagnostic metrics:**
- **approx_kl**: KL divergence between old and new policy. Should stay < 0.02; spikes indicate unstable updates.
- **clip_fraction**: Fraction of samples where the ratio was clipped. Should be 5–15%; too high means policy is changing too fast.
- **value_loss**: Should decrease over training. If it plateaus, the value head is not learning.
- **hard_constraint_score** (rolling mean): The primary evaluation metric. Should trend upward over training.

---

## 9. Big Picture Tie-In

The RL training pipeline is what transforms the compressor from a static heuristic into a learning system. The key contributions:

1. **Closed the feedback loop.** The compressor now receives a signal about whether its compressions actually helped the agent plan better. Without RL, a compressor can only be evaluated manually after the fact; with RL, it improves automatically.

2. **Aligned training with evaluation.** The reward function uses `ConstraintSatisfactionEngine` — exactly the same engine used in the final evaluation metrics. There is no train/eval metric mismatch: what PPO optimizes is what we care about.

3. **Enabled the MCTS synergy.** After training, the value head can replace the LLM evaluator in MCTS, making real-time lookahead search cheap enough to run on every compression event without prohibitive API costs.

4. **Made the ablation study possible.** By training the same PPO loop with different compressor architectures (GAT Distiller vs. Section Cross-Attention, with and without MCTS), we can isolate the contribution of each component and answer the project's core research question.
