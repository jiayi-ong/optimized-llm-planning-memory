# Methodology Overview: Compressing Planning Memory with MCTS and Reinforcement Learning

> **Audience:** Undergraduate ML/stats students comfortable with linear algebra, multivariable calculus, and common algorithms (attention, backpropagation, dynamic programming).
>
> **Goal of this series:** Explain *why* we built this system, *how* the pieces fit together, and *what the math is doing* — then tie every technical detail back to the big picture.

---

## 1. The Core Problem: Agents That Forget

A **ReAct agent** (short for *Reasoning + Acting*) solves tasks by alternating between two operations:

1. **Think** — the language model produces a natural-language *thought* describing its current reasoning.
2. **Act** — the model issues a *tool call* (e.g., `search_flights`, `book_hotel`) and receives an *observation* (the tool's response).

Each thought–action–observation triple is a **step**, and the growing sequence of steps is the **trajectory**. For travel planning, a full episode might take 30–60 steps, covering flight search, hotel booking, attraction discovery, route planning, and cancellation/rebooking loops.

The challenge: **the LLM's context window has a finite capacity.** As the trajectory grows, two things go wrong:

| Problem | What happens |
|---|---|
| **Context overflow** | Old steps get truncated and disappear entirely. |
| **Attention dilution** | Even before truncation, the model spreads attention across hundreds of tokens of prior history, losing focus on the most decision-relevant facts. |

The naive fix — feed the model everything — is prohibitively expensive in tokens and in practice makes the model *worse*, not better, because the relevant signal drowns in noise.

**Our thesis:** A *trained compressor* can distill the trajectory into a compact, structured *memory state* that contains exactly what the agent needs to plan well — constraints still unsatisfied, key facts discovered, decisions already locked in — and discard everything else.

---

## 2. The Three-Layer System

```
┌──────────────────────────────────────────────────────────────┐
│                   USER REQUEST                               │
│  "Plan 5 days NYC → Paris → Rome, $5000, must see Eiffel"   │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                 REACT PLANNING AGENT                         │
│  Steps: search flights → book hotel → search attractions...  │
│  Tools: 14 travel simulator APIs                            │
│  State: growing trajectory (thought, action, observation)   │
└──────────────────┬───────────────────────────────────────────┘
                   │ every N steps
                   ▼
┌──────────────────────────────────────────────────────────────┐
│               COMPRESSOR                                     │
│  Input:  raw trajectory + (optionally) MCTS tree            │
│  Output: structured CompressedState (6 fixed sections)      │
└──────────────────┬───────────────────────────────────────────┘
                   │ compressed state injected back into agent
                   ▼
            agent continues planning
            with compact memory
```

The **CompressedState** has six sections:

| Section | What it contains |
|---|---|
| `hard_constraint_ledger` | Which must-have requirements are ✓ satisfied / ✗ violated / ? unknown |
| `soft_constraints_summary` | Preference satisfaction narrative |
| `decisions_made` | Confirmed bookings and irreversible choices |
| `open_questions` | Planning gaps that still need resolution |
| `key_discoveries` | World facts learned (city IDs, price ranges, availability) |
| `current_itinerary_sketch` | Compact prose summary of the plan so far |

---

## 3. The Four Variants

This repo implements four distinct compressor strategies, treated as ablations:

| Variant | Compressor | Lookahead? | Learns from episodes? |
|---|---|---|---|
| **RAW** | None (full trajectory in context) | No | No |
| **LLM Summary** | GPT-4o-mini summarizes the old prefix | No | No |
| **Agent + MCTS** | MCTS tree → LLM evaluator → structured CompressedState | Yes | No |
| **Agent + MCTS + RL (GAT)** | MCTS tree → Graph-Attention Distiller → CompressedState | Yes | Yes |
| **Agent + MCTS + RL (Section-Attn)** | Trajectory → Section-Aware Cross-Attention → CompressedState | No (MCTS optional) | Yes |

The two RL variants are the focus of this series. They differ in *architecture* (how they process trajectory/tree information) but share the same RL training loop (PPO).

---

## 4. The Intuition

### Analogy: Chess Commentator vs. Chess Engine

Think of a chess player who has played 30 moves and needs to plan the next 10.

- **RAW baseline**: The player re-reads the full game notation from move 1 every time they want to think. Slow, noisy, and they'll miss the forest for the trees.
- **LLM Summary**: A commentator narrates what happened. Natural language, but the commentator's summary is uncalibrated — they might emphasize drama over strategic importance.
- **Agent + MCTS**: Before the player acts, an engine simulates 50 possible continuations 10 moves ahead. The best line is shown to the player as a recommended plan. The player still uses the raw LLM to decide — but the plan is informed by lookahead.
- **Agent + MCTS + RL**: Now the engine also has a *learned value function* trained on thousands of prior games. It doesn't need to simulate 50 moves to evaluate a position — a single neural network forward pass gives a calibrated estimate. The engine gets faster and more accurate over time.

### The Travel Planning Equivalent

- **MCTS without RL**: Simulates "if I book Eiffel Tower tickets now vs. later" using GPT-4o-mini as a rough value estimator. Expensive but better than no lookahead.
- **GAT Distiller + RL**: Reads the MCTS tree as a graph — best path + alternatives — runs graph attention to blend their embeddings, then generates the CompressedState. After training, the neural value head replaces the expensive LLM evaluator.
- **Section-Attn + RL**: Skips the MCTS tree entirely. Five learnable query vectors each "ask a specific question" about the trajectory (What decisions are made? What's still open?) by cross-attending to each ReAct step. After training, each query has specialized to extract its respective section.

---

## 5. What "Training" Means Here

The compressor is trained via **Proximal Policy Optimization (PPO)**, a policy-gradient RL algorithm. The setup:

- **State** $s_t$: the encoded trajectory text after $t$ ReAct steps.
- **Action** $a_t$: the compressed state text tokens produced by the compressor.
- **Reward** $r_t$: how well the agent plans in the *next* window of steps, measured by constraint satisfaction, tool efficiency, and logical consistency.
- **Policy** $\pi_\theta(a_t | s_t)$: the compressor itself, parameterized by weights $\theta$.

The key insight: the compressor's output directly determines what the agent *sees* for the next $N$ steps. A better compression → better agent decisions → higher reward → PPO updates the compressor to produce better compressions. This is a **self-improving feedback loop**.

---

## 6. Document Map

| Document | What you'll learn |
|---|---|
| [`01_agent_mcts.md`](01_agent_mcts.md) | How the ReAct agent works; MCTS phases and UCB1; step-by-step agent–MCTS interaction |
| [`02_gat_distiller.md`](02_gat_distiller.md) | Graph-Attention Distiller: PathSetEncoder, StructuralFeatureProjector, GumbelTopK, LoRA on decoder |
| [`03_section_cross_attention.md`](03_section_cross_attention.md) | Section-Aware Cross-Attention: learnable section queries, temporal embeddings, LoRA on encoder+decoder |
| [`04_rl_training.md`](04_rl_training.md) | PPO training loop: CompressionEnv, reward shaping, value head, MCTS + RL synergy |

---

## 7. How This Advances the Project's Goals

The project's primary research question is:

> *Is context compression alone — without fine-tuning the planning LLM — sufficient to outperform baselines on travel itinerary planning?*

The methodology is designed to answer this cleanly:
- **Baselines** (RAW, LLM Summary) establish floor performance.
- **MCTS alone** tests whether structured lookahead helps without any learned compressor.
- **MCTS + RL** tests whether a *trained* compressor that has seen thousands of episodes improves further.
- **Section-Attn + RL** (no MCTS) tests whether the architecture alone, trained from episode rewards, can learn to compress without an explicit search tree.

By comparing these four, we isolate the contribution of (a) lookahead search, (b) neural compression, and (c) RL training — independently and in combination.
