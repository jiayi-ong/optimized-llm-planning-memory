# Agent + MCTS: Lookahead Planning Without Learning

> **What this doc covers:** How the ReAct planning agent works; why MCTS is a natural fit for improving it; the four MCTS phases with their mathematics; a step-by-step concrete trace; and the limits of MCTS without a trained value function.

---

## 1. The ReAct Planning Agent

### 1.1 One Episode = One Planning Task

An **episode** begins when the agent receives a `UserRequest`:

```python
UserRequest(
    raw_text="Plan a 5-day trip NYC → Paris → Rome, $5000 budget. Must see Eiffel Tower.",
    hard_constraints=[
        Constraint(category=CITY,   description="Must visit Paris"),
        Constraint(category=CITY,   description="Must visit Rome"),
        Constraint(category=BUDGET, value=5000.0, unit="USD"),
        Constraint(category=EVENT,  description="Must visit Eiffel Tower"),
    ],
    soft_constraints=[
        Constraint(category=ACCOMMODATION, description="Prefer 4-star hotels"),
        Constraint(category=FOOD,          description="Interested in local food markets"),
    ],
)
```

The agent must build an `Itinerary` — a structured collection of booked flights, hotels, and activities — that satisfies as many constraints as possible within the step budget.

### 1.2 The ReAct Loop

At each step, three things happen:

```
[Context: system prompt + user request + compressed memory + trajectory]
         │
         ▼
    LLM generates
    ┌─────────────────────────────────────────────────────────┐
    │ Thought: I need to confirm the Eiffel Tower visit.      │
    │ Action:  search_attractions({"city_id": "par_001",      │
    │                              "category": "landmark"})   │
    └─────────────────────────────────────────────────────────┘
         │
         ▼
    Tool middleware executes → ToolResult
    ┌─────────────────────────────────────────────────────────┐
    │ Observation: [{"attraction_id": "att_eiffel",           │
    │                "name": "Eiffel Tower", "price": 30}]    │
    └─────────────────────────────────────────────────────────┘
         │
         ▼
    ReActStep recorded in Trajectory
```

This produces one `ReActStep`:
$$\text{step}_t = (\text{thought}_t,\ \text{action}_t,\ \text{observation}_t)$$

The trajectory after $T$ steps is the sequence:
$$\tau_T = (\text{step}_0, \text{step}_1, \ldots, \text{step}_{T-1})$$

### 1.3 How the Agent Updates Its Plan

Certain tool calls directly modify the `Itinerary`:

| Tool | Effect |
|---|---|
| `select_flight` | Adds a `TransportSegment` to the departure day |
| `book_hotel` | Sets `AccommodationBooking` on the check-in day |
| `book_event` | Appends an `ActivityBooking` to the event day |
| `cancel_booking` | Removes a prior booking by reference |

The total cost is recomputed after each booking. The agent can see the current itinerary state in every prompt under `[CURRENT ITINERARY STATE]`.

### 1.4 Termination

The episode ends when:
- The agent emits `Action: DONE` — it believes the itinerary is complete.
- The agent emits `Action: EXIT(reason=X)` — it encounters an unrecoverable situation.
- The step budget `max_steps` is exhausted.

---

## 2. Why the Greedy Agent Falls Short

The LLM generates actions greedily: each step produces the locally best-looking action given the current context. This works well for the first few steps but breaks down in two systematic ways.

### 2.1 Constraint Neglect

The agent may spend its step budget on activities it finds interesting (restaurant searches, attraction browsing) while a **hard constraint** (e.g., Eiffel Tower) goes unbooked. The trajectory contains everything needed to detect this, but the LLM's attention is spread across dozens of steps.

### 2.2 Premature Commitment

The agent books a hotel early — say, a budget option in Paris — that forecloses a better hotel+attraction bundle. To fix this it must `cancel_booking` and rebook, wasting 2–3 steps. A planning agent that had looked one step further would have avoided the suboptimal choice.

Both problems arise because the agent has **no lookahead**: it cannot simulate "what happens if I take this action?" before committing.

---

## 3. Monte Carlo Tree Search (MCTS)

MCTS is a framework for making decisions in large state spaces by building a search tree incrementally, guided by a balance of *exploration* (trying actions we know little about) and *exploitation* (following actions that have looked good so far).

### 3.1 State and Action Definitions

In this system:
- **State** $s$: the trajectory $\tau_t$ up to step $t$, plus the current itinerary.
- **Action** $a$: any tool call the agent can issue (e.g., `search_flights`, `book_hotel`, `DONE`).
- **Transition** $T(s, a)$: executing action $a$ — calling the tool and appending the result — yields the next state $s'$.
- **Value** $V(s)$: expected cumulative reward (constraint satisfaction + efficiency) from state $s$ to episode end.

The agent's goal is to find the sequence of actions from the current state that maximizes expected value.

### 3.2 The Four MCTS Phases

MCTS builds a tree of `MCTSNode` objects. Each node stores:
- `trajectory_snapshot` — the trajectory at that node
- `visit_count` $n_s$ — number of times this node has been visited
- `value_sum` $v_s$ — sum of simulation values that passed through this node
- `children` — list of child nodes

One iteration of MCTS has four phases:

#### Phase 1: Selection — Follow the Tree to a Promising Leaf

Starting from the root (current real trajectory), descend the tree by choosing the child with the highest **UCB1 score** at each level:

$$\text{UCB1}(s) = \underbrace{\frac{v_s}{n_s}}_{\text{exploitation}} + C_p \underbrace{\sqrt{\frac{\ln N}{n_s}}}_{\text{exploration}}$$

where:
- $v_s / n_s$ is the *empirical mean value* of simulations through node $s$ — how good this path has looked.
- $\sqrt{\ln N / n_s}$ grows when $n_s$ is small relative to the parent's visits $N$ — ensuring rarely-visited nodes are eventually explored.
- $C_p = \sqrt{2}$ is the **exploration constant** (from UCB1 theory; set to `1.414` in `mcts/config.py`).

**Intuition:** UCB1 implements "optimism in the face of uncertainty." If we haven't tried a path much, we assume it might be great. Once we've tried it enough, we believe the empirical average. The balance between the two terms shifts as $n_s$ grows.

Selection stops when it reaches a **leaf** — a node with no children (unexplored) or a terminal node.

#### Phase 2: Expansion — Generate Candidate Actions

At the leaf node, the LLM is called at `temperature=0.7` to generate `branching_factor=3` diverse candidate next actions. This uses a prompt that shows the current trajectory, confirmed bookings, and open constraints, asking for `branching_factor` different plausible next actions.

```python
# From mcts/controller.py _sample_candidate_actions()
# Prompt shows: user request + completed bookings + recent steps
# LLM generates 3 action strings like:
# "Action: search_attractions({\"city_id\": \"par_001\", ...})"
# "Action: search_flights({\"origin_city_id\": \"par_001\", ...})"
# "Action: book_event({\"event_id\": \"evt_eiffel\"})"
```

For each candidate action, a **synthetic trajectory** is created: the candidate is appended as a new `ReActStep` *without actually executing the tool*. This creates 3 child nodes, each representing a hypothetical future.

> **Why not execute the tool?** Real tool execution would require API calls for each candidate at each simulation. With 50 simulations × 3 branches × $\leq 5$ rollout steps, that's up to 750 tool calls per compression event — far too expensive. Synthetic steps trade accuracy for speed.

#### Phase 3: Simulation — Estimate the Leaf's Value

For each newly created child node, the **evaluator LLM** (`evaluator_model_id`, e.g., GPT-4o-mini) reads the synthetic trajectory and returns a value estimate in $[0, 1]$:

$$\hat{V}(s) = \text{EvaluatorLLM}(\tau_{\text{synthetic}}) \in [0, 1]$$

The evaluator is prompted with the user request + the trajectory and asked: "How well is this trajectory positioned to satisfy all hard and soft constraints? Score 0–1."

This is the **simulation** or **rollout** phase. In classical MCTS (e.g., AlphaGo's early version), you'd play out the game randomly to the end; here we substitute an LLM heuristic for the random rollout.

#### Phase 4: Backpropagation — Update Ancestor Values

The simulation value $\hat{V}$ propagates back up the path from the evaluated node to the root:

```
for node in path_from_leaf_to_root:
    node.visit_count += 1
    node.value_sum  += V̂
```

This updates the exploitation term $v_s / n_s$ for every ancestor: if the child was promising, the parent's mean value rises, making it more likely to be selected in future iterations.

### 3.3 After All Iterations: Extract the Tree Representation

After `num_simulations=50` iterations, the tree encodes which future action sequences look most promising. The `MCTSController` extracts a `MCTSTreeRepresentation`:

```python
MCTSTreeRepresentation(
    best_path_trajectory=...,      # trajectory along the highest-Q path
    alternative_paths=[...],       # up to 3 next-best paths
    top_candidates=[               # human-readable descriptions
        "Book Eiffel Tower tickets (hard constraint)",
        "Search Paris→Rome flight for June 3",
    ],
    tradeoffs="Direct flight $180 vs. train $80, 3hr — train cheaper",
    stats=MCTSStats(nodes_explored=37, num_simulations=50, root_value=0.72),
)
```

---

## 4. How MCTS Output Flows Into the Agent

```
MCTS.search(trajectory, request)
      │
      ▼
MCTSTreeRepresentation
      │
      ▼
Compressor.compress_with_tree(tree_repr)
      │
      ▼
CompressedState {
    top_candidates: ["Book Eiffel Tower...", "Plan Rome flight..."],
    tradeoffs: "...",
    open_questions: ["Eiffel Tower NOT BOOKED — hard constraint!"],
    decisions_made: ["NYC→Paris flight FLT-001 $450"],
    ...
}
      │
      ▼
Injected into agent's context as [MEMORY] for next N steps
```

The `top_candidates` and `open_questions` fields directly steer the agent: the LLM, seeing "Eiffel Tower NOT BOOKED — hard constraint!" in its memory, will prioritize booking it in the next step.

---

## 5. Step-by-Step Trace: Two MCTS Compression Events

### Episode: "5-day NYC → Paris → Rome, $5000, must see Eiffel Tower"

#### Steps 1–5 (Raw window, no compression yet)

```
Step 1 — Action: get_available_routes({})
         Obs:    [nyc_001, par_001, rom_001, ...]

Step 2 — Action: search_flights({origin:"nyc_001", dest:"par_001", date:"2025-06-01"})
         Obs:    [Flight A $450, Flight B $380, ...]

Step 3 — Action: select_flight({edge_id:"e_A"})
         Obs:    {booking_id:"FLT-001", cost:$450}

Step 4 — Action: search_hotels({city_id:"par_001", check_in:"2025-06-01",
                                 check_out:"2025-06-03", max_price:200, min_stars:4})
         Obs:    [Hotel Marais $175/night ★★★★, Hotel Louvre $220/night ★★★★]

Step 5 — Action: book_hotel({hotel_id:"h_marais", check_in:"2025-06-01",
                              check_out:"2025-06-03"})
         Obs:    {booking_id:"HT-002", cost:$350}

Itinerary so far: Flight NYC→Paris + Hotel Paris 2 nights = $800
```

#### First MCTS Trigger (after step 5)

`_should_compress()` fires. MCTS runs 50 iterations on the trajectory.

**Iteration 1:**
- *Select:* Root (visit=0) → select root
- *Expand:* LLM generates 3 candidates:
  - **A** `search_attractions(par_001, "landmark")` — finds Eiffel Tower path
  - **B** `search_flights(par_001 → rom_001, 2025-06-03)` — Rome planning path
  - **C** `search_restaurants(par_001, "french")` — soft preference path
- *Simulate:* Evaluator scores A=0.65, B=0.45, C=0.50
  - (A scores highest: hard constraint Eiffel Tower is closer to being satisfied)
- *Backprop:* Root.value_sum += 0.65, Root.visit_count = 1

**Iterations 2–15:**
- UCB1 selects **A** most often (highest Q + exploration):
  $$\text{UCB1}(A) = 0.65 + 1.414 \cdot \sqrt{\frac{\ln 5}{2}} \approx 0.65 + 1.01 = 1.66$$
- **A** expands into:
  - **A1** `book_event(evt_eiffel_001)` → evaluator: 0.85 (hard constraint satisfied!)
  - **A2** `get_attraction_detail(att_eiffel)` → evaluator: 0.70

**Iterations 16–50:**
- **A1** accumulates visits (best Q=0.85 among all nodes)
- **B** gets explored: deep branch reveals Paris→Rome+Eiffel Tower combo → 0.80

**Final tree state:**
```
Root (visits=50, Q=0.74)
├── A: search_attractions (visits=28, Q=0.78)  ← best subtree
│   ├── A1: book_event/Eiffel (visits=20, Q=0.85) ← BEST PATH
│   └── A2: get_attraction_detail (visits=8, Q=0.70)
├── B: search_flights/Rome (visits=17, Q=0.68)
│   └── B1: select_flight+Eiffel combo (visits=10, Q=0.80)
└── C: search_restaurants (visits=5, Q=0.50)
```

**CompressedState produced by first compression:**
```
## HARD CONSTRAINT LEDGER ##
  UNKNOWN: MUST_VISIT_EIFFEL_TOWER | MUST_VISIT_ROME | BUDGET_USD_5000

## DECISIONS MADE ##
  - NYC→Paris flight FLT-001 ($450, June 1)
  - Hotel Marais Paris HT-002 ($350, June 1–3)

## OPEN QUESTIONS ##
  ⚠ PRIORITY: Eiffel Tower NOT BOOKED — hard constraint unsatisfied
  - Rome accommodation not searched
  - Return flight not planned

## KEY DISCOVERIES ##
  - Paris city_id=par_001 | Rome city_id=rom_001
  - 4-star Paris hotels: $175–$220/night

## CURRENT ITINERARY SKETCH ##
  Days 1–2: Paris (flight + hotel, $800). Days 3–5: TBD. Remaining: $4,200.

## TOP CANDIDATES (from MCTS) ##
  1. search_attractions(par_001, "landmark") → book_event(Eiffel Tower) — satisfies hard constraint
  2. After Eiffel: search_flights(par_001→rom_001, June 3)
  3. Rome hotel: search_hotels(rom_001, June 3–5, ≤$200/night, ★★★★)

## TRADEOFFS ##
  Eiffel Tower immediately vs. Rome planning first: Eiffel Tower is a hard constraint.
  Prioritize Eiffel Tower.
```

#### Steps 6–10 (with CompressedState in context)

The agent reads the memory. The `OPEN QUESTIONS` and `TOP CANDIDATES` sections dominate its reasoning:

```
Step 6 — Thought: Memory says PRIORITY: Eiffel Tower unsatisfied hard constraint.
         Action:  search_attractions({"city_id":"par_001","category":"landmark"})
         Obs:     [{attraction_id:"att_eiffel", name:"Eiffel Tower", price:30}]

Step 7 — Thought: Found Eiffel Tower. Must book it — hard constraint.
         Action:  book_event({"event_id":"evt_eiffel_001","quantity":1})
         Obs:     {booking_id:"EVT-003", cost:$30}
         → HARD CONSTRAINT EIFFEL TOWER: ✓ SATISFIED

Step 8 — Thought: Per memory candidate 2: plan Paris→Rome flight.
         Action:  search_flights({"origin":"par_001","dest":"rom_001","date":"2025-06-03"})
         Obs:     [{edge_id:"e_PR", price:$180, dep:09:00, arr:11:00}]

Step 9 — Action:  select_flight({"edge_id":"e_PR"})
         Obs:     {booking_id:"FLT-004", cost:$180}

Step 10 — Thought: Per memory candidate 3: Rome hotel.
          Action:  search_hotels({"city_id":"rom_001","check_in":"2025-06-03",
                                   "check_out":"2025-06-05","max_price":200,"min_stars":4})
          Obs:     [{hotel_id:"h_rom", name:"Hotel Colosseo", price:$190/night, ★★★★}]
```

#### Second MCTS Trigger (after step 10)

State: $1,060 spent. Hard constraints: Eiffel Tower ✓, Paris ✓. Still open: Rome confirmed, return flight, budget adherence.

MCTS now focuses its simulations on:
- Rome hotel booking
- Rome activities
- Return flight NYC

The new `CompressedState` updates `hard_constraint_ledger.satisfied` and refines `top_candidates` for Rome completion.

---

## 6. What MCTS Adds (and What It Can't Do)

### What MCTS provides

| Benefit | Mechanism |
|---|---|
| Constraint prioritization | Simulations that satisfy hard constraints score higher → their paths rise in UCB1 |
| Avoid premature commitment | Candidate paths that lead to booking-cancellation cycles score lower |
| Forward-looking agenda | `top_candidates` encode a ready-made plan for next $N$ steps |
| Tradeoff awareness | Multiple high-scoring paths are surfaced; tradeoffs explained in text |

### What standalone MCTS cannot provide

1. **Calibrated value estimation.** The LLM evaluator (gpt-4o-mini) reasons in natural language about constraint satisfaction. It cannot reliably distinguish "trajectory with 3/4 hard constraints satisfied" from "trajectory with 2/4 hard constraints satisfied" — especially for numeric constraints like budget adherence.

2. **Cross-episode learning.** Each episode starts fresh. MCTS cannot exploit the fact that "in 47 prior trips, booking the morning flight before the evening hotel was always suboptimal." It samples candidate actions from the same LLM prior regardless of history.

3. **Cost control.** 50 simulations × 1 evaluator call each = 50 LLM API calls per compression event. For a 30-step episode with 6 compression events, that's 300 evaluator calls per episode. This is expensive at scale.

4. **Policy improvement.** The LLM used for action sampling is fixed. Even after seeing 1,000 episodes, the candidate actions it generates for "Paris hotel booked, Eiffel Tower unsatisfied" look the same as on episode 1.

These limitations motivate adding RL: a **learned value function** replaces the LLM evaluator, and a **trained compressor policy** improves with every episode. See [`04_rl_training.md`](04_rl_training.md) for the full picture.

---

## 7. Big Picture Tie-In

The ReAct + MCTS combination addresses the research question *"can context compression alone outperform baselines?"* from the lookahead direction: instead of better summarizing what happened, it encodes *what should happen next*, derived from systematic exploration of future trajectories.

However, MCTS alone is not a trained system — it cannot improve with data. The RL-trained compressors in [`02_gat_distiller.md`](02_gat_distiller.md) and [`03_section_cross_attention.md`](03_section_cross_attention.md) close this gap by learning, from episode outcomes, which compressions actually lead to better planning.
