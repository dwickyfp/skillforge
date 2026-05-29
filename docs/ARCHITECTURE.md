# SkillForge Architecture

A deep dive into SkillForge's design, data flow, algorithms, and research foundations.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [The Evolution Loop](#the-evolution-loop)
5. [Q-Value Algorithm (TD-λ)](#q-value-algorithm-td-λ)
6. [Progressive Loading Strategy](#progressive-loading-strategy)
7. [Dependency Graph Engine](#dependency-graph-engine)
8. [Self-Diagnosis Engine](#self-diagnosis-engine)
9. [Storage Layer](#storage-layer)
10. [Research Foundations](#research-foundations)

---

## System Overview

SkillForge is a **self-improving skill management system** for AI agents. It addresses a fundamental problem: as agents accumulate skills (prompts, tools, workflows), they need a principled way to decide *which* skills to use, *how well* they work, and *when* to improve or retire them.

```
                    ┌─────────────────────────────────────┐
                    │           Agent Runtime              │
                    │                                     │
                    │   "Help me set up CI/CD"            │
                    │        │                            │
                    │        ▼                            │
                    │   ┌──────────┐                      │
                    │   │ SkillForge│◀── outcomes ──┐     │
                    │   └────┬─────┘               │     │
                    │        │ skills               │     │
                    │        ▼                      │     │
                    │   Execute task ───────────────┘     │
                    └─────────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
     ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
     │   Registry   │   │   Tracker    │   │  Evolution   │
     │  (SQLite)    │   │  (SQLite)    │   │   (in-mem)   │
     │             │   │              │   │              │
     │  Skills     │   │  Outcomes    │   │  Health      │
     │  Metadata   │   │  Q-values    │   │  Diagnosis   │
     │  Versions   │   │  Statistics  │   │  Pruning     │
     └─────────────┘   └──────────────┘   └──────────────┘
```

### Design Principles

1. **Progressive Disclosure** — Load minimal metadata first, full instructions only when needed
2. **Reinforcement Learning** — Use TD(λ) to learn which skills actually work
3. **Self-Healing** — Automatically diagnose failures and suggest patches
4. **Dependency Awareness** — Understand how skills relate to each other
5. **Zero External Dependencies** — Core works with Python stdlib only (SQLite, no numpy/torch)

---

## Component Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     SkillForge (orchestrator)                 │
│                     forge.py                                 │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│          │          │          │          │                 │
│ Registry │ Tracker  │  Loader  │  Graph   │  Evolution Loop │
│ (SQLite) │ (SQLite) │ (in-mem) │ (in-mem) │  (in-mem)      │
│          │          │          │          │       │         │
│ - CRUD   │ - Q-val  │ - Tier   │ - BFS    │       ▼        │
│ - Search │ - TD(λ)  │ - Route  │ - Impact │  Self-Diagnosis│
│ - Version│ - Stats  │ - Sticky │ - Prop   │  Engine        │
│          │          │          │ - Topo   │                │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│                    Benchmark Framework                        │
│  Runner │ Tasks │ Metrics │ Report                           │
├─────────────────────────────────────────────────────────────┤
│                    Integrations                               │
│  Hermes Adapter │ (OpenClaw, MCP - via API)                  │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Location | State | Purpose |
|-----------|----------|-------|---------|
| **SkillForge** | `forge.py` | Orchestrator | Wires everything together, provides unified API |
| **SkillRegistry** | `core/registry.py` | SQLite | CRUD, search, versioning of skills |
| **QValueTracker** | `core/tracker.py` | SQLite | Outcome recording, Q-value computation |
| **ProgressiveLoader** | `core/loader.py` | In-memory | Tier-based retrieval with routing |
| **SkillDependencyGraph** | `core/graph.py` | In-memory | Dependency DAG, impact analysis, propagation |
| **EvolutionLoop** | `core/evolution.py` | In-memory | Health monitoring, evolution, pruning |
| **SelfDiagnosisEngine** | `core/diagnosis.py` | In-memory | Failure analysis, pattern matching, auto-patching |
| **BenchmarkRunner** | `benchmark/runner.py` | In-memory | A/B comparison framework |
| **ReportGenerator** | `benchmark/report.py` | In-memory | Text and chart report generation |
| **HermesAdapter** | `integrations/hermes/adapter.py` | In-memory | Bidirectional SKILL.md import/export |

---

## Data Flow

### 1. Skill Registration

```
User/Agent                    SkillForge                    SQLite
    │                             │                            │
    │ register_skill(             │                            │
    │   name, tier1, tier2,       │                            │
    │   tier3, tags)              │                            │
    │────────────────────────────▶│                            │
    │                             │ register_skill(...)        │
    │                             │───────────────────────────▶│
    │                             │                            │
    │                             │ add_skill(id, metadata)    │
    │                             │──── Graph ──────────────── │
    │                             │                            │
    │◀──── Skill object ─────────│                            │
```

### 2. Skill Loading (Progressive)

```
Agent                        ProgressiveLoader              Registry/Tracker
  │                              │                               │
  │ load_skill("code review")    │                               │
  │─────────────────────────────▶│                               │
  │                              │ search_skills(query, limit*4) │
  │                              │──────────────────────────────▶│
  │                              │◀────── candidates ────────────│
  │                              │                               │
  │                              │ [filter active skills]        │
  │                              │                               │
  │                              │ get_stats(id) for each        │
  │                              │──────────────────────────────▶│
  │                              │◀────── stats ─────────────────│
  │                              │                               │
  │                              │ [compute relevance scores]    │
  │                              │ [sort by routing strategy]    │
  │                              │ [take top N]                  │
  │                              │                               │
  │                              │ get_skill(id, tier) for top N │
  │                              │──────────────────────────────▶│
  │                              │◀────── full skills ───────────│
  │                              │                               │
  │◀──── ranked skills ─────────│                               │
```

### 3. Outcome Recording

```
Agent                        SkillForge                    SQLite
  │                              │                            │
  │ record_outcome(              │                            │
  │   skill_id, success,         │                            │
  │   latency, tokens)           │                            │
  │─────────────────────────────▶│                            │
  │                              │ record_outcome(outcome)    │
  │                              │───────────────────────────▶│
  │                              │                            │
  │                              │ get_stats(skill_id)        │
  │                              │───────────────────────────▶│
  │                              │◀────── stats ──────────────│
  │                              │                            │
  │                              │ update_skill({              │
  │                              │   usage_count,              │
  │                              │   success_rate,             │
  │                              │   q_value})                 │
  │                              │───────────────────────────▶│
```

### 4. Evolution Cycle

```
EvolutionLoop
  │
  ├─▶ For each skill:
  │     ├─▶ check_skill_health(skill_id)
  │     │     ├─▶ tracker.get_q_value()
  │     │     ├─▶ tracker.get_failures()
  │     │     └─▶ Return HEALTHY | WARNING | CRITICAL
  │     │
  │     ├─▶ If WARNING or CRITICAL:
  │     │     ├─▶ evolve_skill(skill_id)
  │     │     │     ├─▶ diagnosis.analyze_failures()
  │     │     │     │     ├─▶ [LLM analysis] OR
  │     │     │     │     └─▶ [rule-based pattern matching]
  │     │     │     ├─▶ Select best insight by confidence
  │     │     │     └─▶ diagnosis.auto_patch_skill()
  │     │     │           └─▶ registry.update_skill({patches})
  │     │     └─▶ If evolution failed:
  │     │           └─▶ graph.downstream_impact()
  │     │
  │     └─▶ Log action to report
  │
  └─▶ prune_dead_skills()
        ├─▶ For each skill:
        │     ├─▶ Check Q < threshold
        │     ├─▶ Check usage < minimum
        │     ├─▶ Check age > max_days
        │     ├─▶ Check no downstream dependents
        │     └─▶ Deprecate if all criteria met
        └─▶ Return list of pruned IDs
```

---

## The Evolution Loop

The evolution loop is the heart of SkillForge's self-improvement capability. It runs periodically (on a schedule or triggered manually) and performs three phases:

### Phase 1: Health Assessment

Each skill is classified into one of three health states based on its Q-value and failure rate:

```
                    HEALTHY           WARNING           CRITICAL
                  ┌──────────┐     ┌──────────┐     ┌──────────┐
  Q-value:        │ ≥ 0.5    │     │ [0.3,0.5)│     │ < 0.3    │
  Failure rate:   │ ≤ 20%    │     │ (20%,50%]│     │ > 50%    │
  Action:         │ (none)   │     │ Evolve   │     │ Evolve + │
                  │          │     │          │     │ Consider │
                  │          │     │          │     │ Pruning  │
                  └──────────┘     └──────────┘     └──────────┘
```

### Phase 2: Evolution

For WARNING and CRITICAL skills:

1. **Analyze failures** — The `SelfDiagnosisEngine` examines the last N failure records
2. **Generate insights** — Either via LLM (if `llm_fn` is provided) or rule-based pattern matching
3. **Select best patch** — The insight with highest confidence is chosen
4. **Apply patch** — The skill's metadata is updated with the patch suggestion recorded

### Phase 3: Pruning

A skill is deprecated when ALL conditions are met:
- Q-value < `prune_q_threshold` (default 0.3)
- Usage count < `prune_min_usage` (default 5)
- Last used > `prune_max_age_days` ago (default 90 days)
- **No downstream dependents** in the dependency graph

This ensures we never prune a skill that other skills depend on.

---

## Q-Value Algorithm (TD-λ)

SkillForge uses **Temporal Difference learning with eligibility traces** (TD(λ)) to maintain quality estimates for each skill. This is adapted from reinforcement learning, where each skill use is a "state-action" and the outcome is a "reward."

### The Update Rule

For a single-step update (no history), this reduces to standard Q-learning:

```
Q(s) ← Q(s) + α * (reward - Q(s))
```

With eligibility traces over the last `k` outcomes:

```
Q(s) ← Q(s) + α * Σᵢ₌₀ᵏ⁻¹ (γλ)ⁱ * (reward - Q(s))
```

Where:
- `α` (alpha) = learning rate (default 0.1) — how fast Q-values change
- `γ` (gamma) = discount factor (default 0.9) — how much we value future vs immediate
- `λ` (lambda) = trace decay (default 0.8) — how far back in history to look

### Eligibility Traces

The trace mechanism gives more weight to recent outcomes:

```
Trace weight at step i: (γλ)ⁱ

Step 0 (most recent): weight = 1.0
Step 1:               weight = 0.72  (0.9 * 0.8)
Step 2:               weight = 0.518 (0.72 * 0.72)
Step 3:               weight = 0.373
...
Step 9:               weight = 0.035
```

This creates a natural recency bias: recent outcomes have more impact on Q-values than old ones, while still incorporating historical performance.

### Implementation Detail

```python
# From core/tracker.py
def td_lambda_update(self, skill_id, reward, alpha=0.1, gamma=0.9, lambda_=0.8):
    current_q = self.get_q_value(skill_id)
    
    # Gather last 10 outcomes as trace
    rows = self._conn.execute(
        "SELECT success, latency_ms, tokens_used, user_feedback "
        "FROM outcomes WHERE skill_id = ? ORDER BY id DESC LIMIT 10",
        (skill_id,),
    ).fetchall()
    
    # TD error with eligibility trace
    td_error = reward - current_q
    trace_weight = 1.0
    cumulative_update = 0.0
    
    for i, row in enumerate(rows):
        cumulative_update += trace_weight * td_error
        trace_weight *= gamma * lambda_
        if trace_weight < 1e-6:
            break
    
    new_q = current_q + alpha * cumulative_update
    new_q = max(0.0, min(1.0, new_q))  # clamp to [0, 1]
    
    return new_q
```

### Q-Value Propagation Through Dependencies

When a skill's Q-value changes, the change propagates through the dependency graph:

```
              ┌─────────┐  ΔQ = +0.2
              │ Skill A  │──────────────────┐
              └─────────┘                   │
                 │ γ=0.9, w=0.8            │ γ=0.9, w=0.5
                 ▼                          ▼
           ┌─────────┐               ┌─────────┐
           │ Skill B  │               │ Skill C  │
           │ ΔQ=+0.144│               │ ΔQ=+0.090│
           └─────────┘               └─────────┘
                 │ γ=0.9, w=0.6
                 ▼
           ┌─────────┐
           │ Skill D  │
           │ ΔQ=+0.078│
           └─────────┘
```

**Formula:** `propagated_delta = parent_delta * γ * edge_weight`

The decay factor `γ` ensures influence diminishes with distance. Edge weights encode the strength of the dependency relationship.

---

## Progressive Loading Strategy

Progressive loading is a key optimization that reduces token consumption by loading skill data in tiers.

### The Three Tiers

```
Tier 1 — Metadata (~30 tokens per skill)
├── Skill name
├── Short description (tier1_metadata)
├── Q-value, success rate, usage count
└── Tags

Tier 2 — Core Instructions (~200-500 tokens per skill)
├── Everything in Tier 1
└── Full prompt/instructions (tier2_core)

Tier 3 — Full Resources (~500+ tokens per skill)
├── Everything in Tier 2
└── Auxiliary files, examples, references (tier3_resources)
```

### Loading Strategy

```
Agent request: "help me set up CI/CD"
  │
  ▼
Tier 1: Load metadata for ALL matching skills (cheap)
  │    Evaluate relevance & Q-values
  │    Rank and filter top N
  ▼
Tier 2: Load core instructions for top N only
  │    [Only if tier >= 2 was requested]
  ▼
Tier 3: Load resources for top M (M ≤ N)
  │    [Only if tier >= 3 was requested]
  ▼
Return skills to agent
```

### Routing Strategies

| Strategy | Sorts By | Best For |
|----------|----------|----------|
| `q_value` | Quality estimate | Default; balances success and recency |
| `success_rate` | Raw success ratio | When consistency matters most |
| `usage_count` | Popularity | When experience matters |
| `relevance` | Keyword overlap | When precision matters |

### Sticky Skills

The loader identifies "sticky skills" — skills that should be pre-loaded in context regardless of the query. These are ranked by a composite score:

```
composite = 0.6 * q_value + 0.4 * (usage_count / max_usage)
```

This ensures both high-quality and frequently-used skills remain accessible.

---

## Dependency Graph Engine

The dependency graph is an in-memory directed acyclic graph (DAG) using adjacency lists.

### Data Structures

```python
_adjacency: dict[str, list[tuple[str, float]]]  # forward: skill → [(dependency, weight)]
_reverse:   dict[str, list[tuple[str, float]]]  # reverse: skill → [(dependent, weight)]
_nodes:     dict[str, SkillNode]                 # all nodes with metadata
```

### Operations and Complexity

| Operation | Algorithm | Time Complexity |
|-----------|-----------|----------------|
| `add_skill` | Hash insert | O(1) |
| `add_dependency` | List append | O(1) amortized |
| `find_path` | BFS | O(V + E) |
| `downstream_impact` | BFS on reverse | O(V + E) |
| `upstream_impact` | BFS on forward | O(V + E) |
| `propagate_q_update` | BFS on reverse | O(V + E) |
| `topological_sort` | Kahn's algorithm | O(V + E) |
| `save` / `load` | JSON serialize | O(V + E) |

Where V = number of skills, E = number of dependency edges.

### Graph Validation

- **Self-loops** are rejected at insertion time
- **Duplicate edges** update the weight of the existing edge
- **Cycles** are detected during topological sort (raises `ValueError`)

---

## Self-Diagnosis Engine

The diagnosis engine operates in two modes:

### Rule-Based Mode (Default)

Pattern-matches error messages against 10 built-in regex patterns:

| Pattern | Root Cause | Suggestion |
|---------|------------|------------|
| timeout | Execution timeout | Add timeout handling, break into chunks |
| memory/OOM | Memory exhaustion | Implement streaming/batching |
| connection/network | Network issue | Add retry with backoff |
| permission/auth | Auth failure | Verify credentials |
| not found/404 | Missing resource | Add existence checks |
| parse/json | Format error | Add input validation |
| rate limit/429 | Rate limiting | Implement queuing |
| null/undefined | Missing value | Add null checks |
| recursion | Stack overflow | Convert to iterative |
| type error | Type mismatch | Add type checking |

Confidence is computed as: `match_count / total_failures`

### LLM-Assisted Mode

When `llm_fn` is provided:

1. Build a structured prompt with failure history
2. Call the LLM with the analysis request
3. Parse the structured response (ROOT_CAUSE, HEURISTIC, PATCH, CONFIDENCE)
4. Fall back to rule-based if LLM call fails

### Auto-Patching

When a patch is applied, the record is appended to the skill's metadata:

```json
{
  "patches": [
    {
      "applied_at": "2026-05-29T12:00:00",
      "root_cause": "Network connectivity issue",
      "heuristic": "network_pattern",
      "patch_suggestion": "Add retry logic with exponential backoff",
      "confidence": 0.85
    }
  ],
  "patch_count": 1,
  "status": "patched"
}
```

---

## Storage Layer

### SQLite Configuration

Both the registry and tracker use SQLite with:
- **WAL mode** (`PRAGMA journal_mode=WAL`) — concurrent reads during writes
- **Foreign keys** enabled (registry)
- Automatic directory creation for the database path

### Schema

**Registry (`skills` table):**

```sql
CREATE TABLE skills (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    version        INTEGER NOT NULL DEFAULT 1,
    lifecycle      TEXT NOT NULL DEFAULT 'draft',
    tier1_metadata TEXT NOT NULL DEFAULT '',
    tier2_core     TEXT NOT NULL DEFAULT '',
    tier3_resources TEXT NOT NULL DEFAULT '[]',
    q_value        REAL NOT NULL DEFAULT 0.5,
    success_rate   REAL NOT NULL DEFAULT 0.5,
    usage_count    INTEGER NOT NULL DEFAULT 0,
    tags           TEXT NOT NULL DEFAULT '[]',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX idx_skills_lifecycle ON skills(lifecycle);
CREATE INDEX idx_skills_q_value   ON skills(q_value);
```

**Tracker (`outcomes` table):**

```sql
CREATE TABLE outcomes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id      TEXT NOT NULL,
    success       INTEGER NOT NULL,
    latency_ms    REAL NOT NULL,
    tokens_used   INTEGER NOT NULL,
    user_feedback REAL,
    recorded_at   TEXT NOT NULL
);
CREATE INDEX idx_outcomes_skill ON outcomes(skill_id);
```

**Tracker (`q_values` table):**

```sql
CREATE TABLE q_values (
    skill_id      TEXT PRIMARY KEY,
    q_value       REAL NOT NULL DEFAULT 0.5,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    total_latency REAL NOT NULL DEFAULT 0.0,
    total_tokens  INTEGER NOT NULL DEFAULT 0,
    updated_at    TEXT NOT NULL
);
```

### Shared Database

When using `SkillForge(db_path=...)`, the same database file is used by both the registry and tracker. This allows atomic reads across both components and simplifies backup/migration.

---

## Research Foundations

SkillForge draws on several research areas:

### 1. Temporal Difference Learning

**Sutton, R.S. (1988).** "Learning to predict by the methods of temporal differences." *Machine Learning*, 3(1), 9-44.

The TD(λ) algorithm with eligibility traces provides a principled way to estimate skill quality from sequential outcome data. Unlike simple averaging, TD learning:
- Weights recent outcomes more heavily (recency bias)
- Smoothly adapts to changing performance (non-stationary)
- The λ parameter controls the bias-variance tradeoff

### 2. Multi-Armed Bandits / Contextual Bandits

**Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).** "Finite-time analysis of the multiarmed bandit problem." *Machine Learning*, 47(2-3), 235-256.

Skill selection is modeled as a contextual bandit problem: given a query (context), select the skill (arm) that maximizes expected reward. Q-values provide the exploitation signal, while relevance scoring provides the exploration signal.

### 3. Progressive Disclosure in UX

**Nielsen, J. (1994).** "Progressive Disclosure." *Nielsen Norman Group*.

The three-tier loading strategy applies progressive disclosure to AI skill management: start with the minimum information needed for routing decisions, and load full details only for the most promising candidates.

### 4. Self-Healing Systems

**Kephart, J.O., & Chess, D.M. (2003).** "The vision of autonomic computing." *IEEE Computer*, 36(1), 41-50.

The evolution loop implements a simplified autonomic control loop:
- **Monitor** — Track outcomes and compute Q-values
- **Analyze** — Detect underperformance via health thresholds
- **Plan** — Generate diagnosis insights and patches
- **Execute** — Apply patches and version bumps

### 5. Eligibility Traces in RL

**Sutton, R.S., & Barto, A.G. (2018).** *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

The eligibility trace mechanism (Chapter 7) is adapted for skill quality estimation. Rather than a full MDP, we treat each skill execution as a one-step episode, with traces providing a bridge to historical performance.

### 6. Impact Analysis in Dependency Graphs

**Abadi, M., et al. (2016).** "TensorFlow: A system for large-scale machine learning." *OSDI '16*.

The dependency graph with Q-value propagation is inspired by computational graph backpropagation: changes propagate backward through the graph with geometric decay, allowing the system to understand how a change in one component affects others.

---

## Extension Points

### Custom Routing Strategies

Subclass `ProgressiveLoader` and override the sorting logic:

```python
class MyLoader(ProgressiveLoader):
    def load_skill(self, query, tier=1, routing="q_value", limit=5):
        # Custom pre-processing
        skills = super().load_skill(query, tier=tier, routing=routing, limit=limit)
        # Custom post-processing
        return self._apply_business_rules(skills)
```

### Custom Diagnosis Heuristics

Add patterns to `SelfDiagnosisEngine`:

```python
from skillforge.core.diagnosis import _ERROR_PATTERNS

_ERROR_PATTERNS.append((
    r"(?i)custom.?error|my.?specific.?failure",
    "Custom error pattern detected",
    "custom_pattern",
    "Apply the custom fix: ...",
))
```

### Custom Benchmark Verifiers

```python
def my_verifier(expected, actual) -> float:
    """Custom scoring function for your domain."""
    # Implement domain-specific correctness scoring
    return score  # 0.0 to 1.0

task = Task(..., verifier_fn=my_verifier)
```
