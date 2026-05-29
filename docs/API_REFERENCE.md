# SkillForge API Reference

Complete API documentation for all SkillForge classes, methods, and types.

---

## Table of Contents

- [SkillForge](#skillforge) — Main orchestrator
- [SkillRegistry](#skillregistry) — Persistent skill storage
- [Skill / SkillLifecycle](#skill) — Data types
- [QValueTracker / EffectivenessTracker](#qvaluetracker) — Outcome tracking
- [ProgressiveLoader](#progressiveloader) — Tier-based skill loading
- [SkillDependencyGraph](#skilldependencygraph) — Dependency management
- [EvolutionLoop](#evolutionloop) — Continuous improvement
- [SelfDiagnosisEngine](#selfdiagnosisengine) — Failure analysis
- [BenchmarkRunner](#benchmarkrunner) — A/B benchmarking
- [ReportGenerator](#reportgenerator) — Benchmark reports
- [MetricCollector / MetricResult](#metriccollector) — Benchmark metrics
- [TaskSuite / Task](#tasksuite) — Benchmark tasks

---

## SkillForge

`skillforge.forge.SkillForge`

Main orchestrator that wires together all core components behind a single API.

### Constructor

```python
SkillForge(
    db_path: str | Path = "~/.skillforge/skillforge.db",
    llm_fn: Callable[[str], str] | None = None,
)
```

**Parameters:**
- `db_path` — Path to the shared SQLite database. Parent directories are created automatically.
- `llm_fn` — Optional callable for LLM-assisted failure analysis. Takes a prompt string, returns a response string.

**Example:**
```python
forge = SkillForge()  # Default: ~/.skillforge/skillforge.db
forge = SkillForge(db_path="/tmp/test.db", llm_fn=my_llm_func)
```

### Methods

#### `load_skill`

```python
load_skill(
    query: str,
    tier: int = 1,
    routing: str = "q_value",
    limit: int = 5,
) -> list[Skill]
```

Load skills matching a query at the requested detail tier.

**Parameters:**
- `query` — Natural-language query or keywords
- `tier` — Maximum detail level: 1 (metadata), 2 (core prompt), 3 (full with resources)
- `routing` — Ranking strategy: `"q_value"`, `"success_rate"`, `"usage_count"`, or `"relevance"`
- `limit` — Maximum skills to return

**Returns:** Ranked list of matching `Skill` objects.

---

#### `record_outcome`

```python
record_outcome(
    skill_id: str,
    success: bool,
    latency_ms: float | None = None,
    tokens_used: int | None = None,
    user_feedback: float | None = None,
) -> None
```

Record an execution outcome and update the skill's rolling stats.

**Parameters:**
- `skill_id` — The skill that produced the outcome
- `success` — Whether the execution was successful
- `latency_ms` — Wall-clock latency in milliseconds (default: 0.0)
- `tokens_used` — Number of LLM tokens consumed (default: 0)
- `user_feedback` — Optional user feedback score on a 0–5 scale

---

#### `register_skill`

```python
register_skill(
    name: str,
    tier1_metadata: str,
    tier2_core: str = "",
    tier3_resources: list[str] | None = None,
    tags: list[str] | None = None,
    skill_id: str | None = None,
) -> Skill
```

Register a brand-new skill.

**Parameters:**
- `name` — Human-readable skill name
- `tier1_metadata` — Short summary (~30 tokens) for routing
- `tier2_core` — Full core prompt / instructions
- `tier3_resources` — Auxiliary file paths or URLs
- `tags` — Descriptive tags for search and filtering
- `skill_id` — Explicit ID (UUID generated when None)

**Returns:** The newly registered `Skill`.

---

#### `import_skill`

```python
import_skill(skill_dict: dict[str, Any]) -> Skill
```

Import a skill from a dictionary (e.g. agentskills.io format).

**Accepted keys:** `name` (required), `id`/`skill_id`, `description`/`tier1_metadata`, `instructions`/`tier2_core`, `resources`/`tier3_resources`, `tags`

**Raises:** `ValueError` if `name` is missing.

---

#### `run_evolution_loop`

```python
run_evolution_loop(
    thresholds: dict[str, float] | None = None,
) -> EvolutionReport
```

Run a full evolution cycle across all skills.

**Parameters:**
- `thresholds` — Optional overrides (e.g. `{"q_critical": 0.25}`)

**Returns:** `EvolutionReport` with counts and action details.

---

#### `get_skill_stats`

```python
get_skill_stats(
    skill_id: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]
```

Return performance statistics. If `skill_id` is given, returns stats for that skill. Otherwise returns stats for all skills.

**Returns:** Dict(s) with keys: `id`, `name`, `version`, `lifecycle`, `tags`, `created_at`, `updated_at`, `q_value`, `success_rate`, `total_outcomes`, `avg_latency_ms`, `avg_tokens`.

---

#### `list_skills`

```python
list_skills(
    filters: dict[str, Any] | None = None,
) -> list[Skill]
```

**Filter keys:** `lifecycle` (str or SkillLifecycle), `tags` (list), `limit` (int, default 100), `offset` (int, default 0).

---

#### `search_skills`

```python
search_skills(query: str, limit: int = 20) -> list[Skill]
```

Keyword search across skill names and metadata. Returns skills ordered by Q-value descending.

---

#### `close`

```python
close() -> None
```

Release all resources. The instance must not be used after closing.

**Context manager support:** `with SkillForge() as forge: ...`

---

## SkillRegistry

`skillforge.core.registry.SkillRegistry`

Persistent skill registry backed by SQLite with WAL mode.

### Constructor

```python
SkillRegistry(db_path: str | Path | None = None)
```

Defaults to `~/.skillforge/skills.db`.

### Methods

#### `register_skill`

```python
register_skill(
    name: str,
    tier1_metadata: str,
    tier2_core: str = "",
    tier3_resources: list[str] | None = None,
    tags: list[str] | None = None,
    skill_id: str | None = None,
) -> Skill
```

**Raises:** `ValueError` if `skill_id` already exists.

---

#### `get_skill`

```python
get_skill(skill_id: str, tier: int = 1) -> Skill | None
```

Retrieve a skill. `tier` controls detail: 1 = metadata only, 2 = +core prompt, 3 = +resources. Returns `None` if not found.

---

#### `update_skill`

```python
update_skill(skill_id: str, updates: dict[str, Any]) -> Skill | None
```

Partial update. Accepted keys: `name`, `version`, `lifecycle`, `tier1_metadata`, `tier2_core`, `tier3_resources`, `q_value`, `success_rate`, `usage_count`, `tags`. Returns updated `Skill` or `None`.

---

#### `list_skills`

```python
list_skills(
    lifecycle: SkillLifecycle | None = None,
    tags: list[str] | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Skill]
```

List skills with optional filters, ordered by Q-value descending.

---

#### `search_skills`

```python
search_skills(query: str, limit: int = 20) -> list[Skill]
```

LIKE-based search across `name` and `tier1_metadata`.

---

#### `version_skill`

```python
version_skill(skill_id: str) -> Skill | None
```

Bump the version number. Returns updated skill or `None`.

---

#### `close`

```python
close() -> None
```

Close the database connection.

---

## Skill

`skillforge.core.registry.Skill` (dataclass)

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | — | Unique identifier |
| `name` | `str` | — | Human-readable name |
| `version` | `int` | — | Version number |
| `lifecycle` | `SkillLifecycle` | — | Current lifecycle stage |
| `tier1_metadata` | `str` | — | Short summary for routing |
| `tier2_core` | `str` | — | Full instructions |
| `tier3_resources` | `list[str]` | — | Auxiliary resources |
| `q_value` | `float` | `0.5` | Estimated quality (0–1) |
| `success_rate` | `float` | `0.5` | Rolling success rate (0–1) |
| `usage_count` | `int` | `0` | Total usage count |
| `tags` | `list[str]` | `[]` | Descriptive tags |
| `created_at` | `datetime` | now (UTC) | Creation timestamp |
| `updated_at` | `datetime` | now (UTC) | Last update timestamp |

### SkillLifecycle

`skillforge.core.registry.SkillLifecycle` (Enum)

- `DRAFT` — Initial state, not yet active
- `ACTIVE` — Available for use by the loader
- `DEPRECATED` — Marked for removal
- `ARCHIVED` — Kept for history but excluded from queries

---

## QValueTracker

`skillforge.core.tracker.QValueTracker`

Tracks skill outcomes and maintains Q-values via TD(λ) updates. SQLite-backed.

> **Alias:** `EffectivenessTracker = QValueTracker`

### Constructor

```python
QValueTracker(db_path: str | Path | None = None)
```

Defaults to `~/.skillforge/tracker.db`.

### Methods

#### `record_outcome`

```python
record_outcome(outcome: Outcome) -> None
```

Persist a single execution outcome and update rolling aggregate stats.

---

#### `get_q_value`

```python
get_q_value(skill_id: str) -> float
```

Return current Q-value. Returns `0.5` if skill is unknown.

---

#### `get_success_rate`

```python
get_success_rate(skill_id: str) -> float
```

Return rolling success rate (0–1). Returns `0.5` if unknown.

---

#### `get_stats`

```python
get_stats(skill_id: str) -> dict[str, Any]
```

**Returns:** `{"skill_id", "q_value", "success_rate", "total_outcomes", "avg_latency_ms", "avg_tokens"}`

---

#### `td_lambda_update`

```python
td_lambda_update(
    skill_id: str,
    reward: float,
    alpha: float = 0.1,
    gamma: float = 0.9,
    lambda_: float = 0.8,
) -> float
```

Apply a TD(λ) update to the Q-value. Uses eligibility traces over the last 10 outcomes.

**Parameters:**
- `skill_id` — Skill to update
- `reward` — Observed scalar reward (typically 0–1)
- `alpha` — Learning rate (default 0.1)
- `gamma` — Discount factor (default 0.9)
- `lambda_` — Trace decay (0=TD(0), 1=Monte-Carlo-like, default 0.8)

**Returns:** New Q-value (clamped to [0, 1]).

**Algorithm:**
```
Q ← Q + α * Σᵢ (γλ)ⁱ * (reward - Q)
```
Where `i` iterates over the trace with geometric decay.

---

#### `close`

```python
close() -> None
```

### Outcome

`skillforge.core.tracker.Outcome` (dataclass)

| Field | Type | Default |
|-------|------|---------|
| `skill_id` | `str` | — |
| `success` | `bool` | — |
| `latency_ms` | `float` | — |
| `tokens_used` | `int` | — |
| `user_feedback` | `float \| None` | `None` |

---

## ProgressiveLoader

`skillforge.core.loader.ProgressiveLoader`

Loads skills progressively by tier with multiple routing strategies.

### Constructor

```python
ProgressiveLoader(registry: SkillRegistry, tracker: QValueTracker)
```

### Class Attributes

```python
ROUTING_STRATEGIES = ("q_value", "success_rate", "usage_count", "relevance")
```

### Methods

#### `load_skill`

```python
load_skill(
    query: str,
    tier: int = 1,
    routing: str = "q_value",
    limit: int = 5,
) -> list[Skill]
```

Loads skills tier-by-tier: tier-1 metadata is always evaluated first, then tier-2 core prompts, then tier-3 resources — only for the top-ranked results. Non-active skills are filtered out unless nothing else matches.

**Parameters:**
- `query` — Natural-language query or keywords
- `tier` — Maximum detail level (1, 2, or 3)
- `routing` — Ranking strategy
- `limit` — Maximum results

---

#### `get_sticky_skills`

```python
get_sticky_skills(limit: int = 5) -> list[Skill]
```

Return top skills to keep pre-loaded in context. Ranked by composite: `0.6 * q_value + 0.4 * normalized_usage_count`.

---

## SkillDependencyGraph

`skillforge.core.graph.SkillDependencyGraph`

In-memory directed graph for skill dependencies using adjacency lists. Supports BFS pathfinding, impact analysis, Q-value propagation, and topological ordering.

### Constructor

```python
SkillDependencyGraph()
```

### Methods

#### `add_skill`

```python
add_skill(skill_id: str, metadata: dict[str, Any] | None = None) -> None
```

Add a skill node. Updates metadata if already exists.

**Raises:** `ValueError` if `skill_id` is empty.

---

#### `add_dependency`

```python
add_dependency(from_id: str, to_id: str, weight: float = 1.0) -> None
```

Add a directed edge: `from_id` depends on `to_id`. Updates weight if edge exists.

**Raises:** `ValueError` if either skill is missing or on self-loop.

---

#### `find_path`

```python
find_path(from_id: str, to_id: str) -> list[str] | None
```

BFS shortest path. Returns list of skill IDs or `None`.

---

#### `downstream_impact`

```python
downstream_impact(skill_id: str) -> list[str]
```

All skills that depend on the given skill (via reverse BFS).

---

#### `upstream_impact`

```python
upstream_impact(skill_id: str) -> list[str]
```

All skills that the given skill depends on (via forward BFS).

---

#### `propagate_q_update`

```python
propagate_q_update(
    skill_id: str,
    delta: float,
    gamma: float = 0.9,
) -> dict[str, float]
```

Propagate a Q-value change through the dependency graph with geometric decay.

**Parameters:**
- `skill_id` — Skill whose Q-value changed
- `delta` — The Q-value change
- `gamma` — Decay factor (0.0–1.0)

**Returns:** Dict mapping affected skill IDs to their propagated delta.

**Formula:** `propagated_delta = delta * γ^distance * edge_weight`

---

#### `topological_sort`

```python
topological_sort() -> list[str]
```

Kahn's algorithm topological sort. Dependencies come before dependents.

**Raises:** `ValueError` if the graph contains a cycle.

---

#### `get_skills`

```python
get_skills() -> list[str]
```

Return all skill IDs in the graph.

---

#### `get_dependencies`

```python
get_dependencies(skill_id: str) -> list[tuple[str, float]]
```

Direct dependencies as `(dependency_id, weight)` tuples.

---

#### `get_dependents`

```python
get_dependents(skill_id: str) -> list[tuple[str, float]]
```

Direct dependents as `(dependent_id, weight)` tuples.

---

#### `remove_skill`

```python
remove_skill(skill_id: str) -> None
```

Remove a skill and all its edges.

**Raises:** `ValueError` if not found.

---

#### `save` / `load`

```python
save(path: str | Path) -> None           # Serialize to JSON
load(path: str | Path) -> SkillDependencyGraph  # Deserialize from JSON (classmethod)
```

---

#### Dunder Methods

- `len(graph)` — Number of skills
- `"skill_id" in graph` — Check membership
- `repr(graph)` — Summary string

---

## EvolutionLoop

`skillforge.core.evolution.EvolutionLoop`

Orchestrates continuous improvement: health monitoring, evolution of underperformers, pruning dead skills.

### Constructor

```python
EvolutionLoop(
    registry: SkillRegistry,
    tracker: FailureTracker,
    graph: SkillDependencyGraph,
    diagnosis: SelfDiagnosisEngine,
)
```

### Default Thresholds

```python
DEFAULT_THRESHOLDS = {
    "q_warning": 0.5,
    "q_critical": 0.3,
    "failure_rate_warning": 0.2,
    "failure_rate_critical": 0.5,
    "prune_q_threshold": 0.3,
    "prune_min_usage": 5,
    "prune_max_age_days": 90,
}
```

### Methods

#### `run_evolution_loop`

```python
run_evolution_loop(
    thresholds: dict[str, float] | None = None,
) -> EvolutionReport
```

Execute a full cycle: evaluate health → evolve underperformers → prune dead skills.

**Returns:** `EvolutionReport`

---

#### `check_skill_health`

```python
check_skill_health(skill_id: str) -> HealthStatus
```

**Returns:** `HealthStatus.HEALTHY`, `HealthStatus.WARNING`, or `HealthStatus.CRITICAL`

**Determination logic:**
- CRITICAL: `q_value < q_critical` OR `failure_rate > failure_rate_critical`
- WARNING: `q_value < q_warning` OR `failure_rate > failure_rate_warning`
- HEALTHY: otherwise

---

#### `evolve_skill`

```python
evolve_skill(skill_id: str) -> bool
```

Run diagnosis and apply the best-confidence patch. Returns `True` on success.

---

#### `prune_dead_skills`

```python
prune_dead_skills(
    q_threshold: float = 0.3,
    min_usage: int = 5,
    max_age_days: int = 90,
) -> list[str]
```

Deprecate skills that are ALL of: low Q-value, low usage, old. Skips skills with downstream dependents.

**Returns:** List of deprecated skill IDs.

---

#### `get_evolution_report`

```python
get_evolution_report() -> EvolutionReport | None
```

---

#### `get_history`

```python
get_history(limit: int = 50) -> list[EvolutionAction]
```

---

#### `get_skill_evolution_summary`

```python
get_skill_evolution_summary(skill_id: str) -> dict[str, Any]
```

**Returns:** `{"skill_id", "total_actions", "evolutions", "successful_evolutions", "last_action", "health"}`

### HealthStatus

`skillforge.core.evolution.HealthStatus` (str, Enum)

- `HEALTHY = "healthy"`
- `WARNING = "warning"`
- `CRITICAL = "critical"`

### EvolutionReport

`skillforge.core.evolution.EvolutionReport` (dataclass)

| Field | Type |
|-------|------|
| `cycle_start` | `datetime` |
| `cycle_end` | `datetime` |
| `total_skills_evaluated` | `int` |
| `skills_evolved` | `int` |
| `skills_pruned` | `int` |
| `skills_healthy` | `int` |
| `skills_warning` | `int` |
| `skills_critical` | `int` |
| `actions` | `list[EvolutionAction]` |

**Methods:** `to_dict()`, `summary()`

### EvolutionAction

`skillforge.core.evolution.EvolutionAction` (dataclass)

| Field | Type |
|-------|------|
| `action_type` | `str` |
| `skill_id` | `str` |
| `details` | `dict[str, Any]` |
| `timestamp` | `datetime` |
| `success` | `bool` |

**Methods:** `to_dict()`

---

## SelfDiagnosisEngine

`skillforge.core.diagnosis.SelfDiagnosisEngine`

Analyzes skill failures and generates actionable insights. Supports LLM-assisted and rule-based analysis.

### Constructor

```python
SelfDiagnosisEngine(
    registry: SkillRegistry,
    tracker: FailureTracker,
    llm_fn: Callable[[str], str] | None = None,
)
```

### Methods

#### `analyze_failures`

```python
analyze_failures(skill_id: str, window: int = 10) -> list[Insight]
```

Analyze recent failures and generate insights. Uses LLM if `llm_fn` was provided, otherwise falls back to rule-based pattern matching.

**Built-in patterns:** timeout, memory, network, auth, missing resource, parsing, rate limit, null value, recursion, type error.

---

#### `generate_insight`

```python
generate_insight(failures: list[dict[str, Any]]) -> Insight
```

Generate a single consolidated insight from a failure list.

---

#### `auto_patch_skill`

```python
auto_patch_skill(skill_id: str, insight: Insight) -> dict[str, Any]
```

Apply a patch to a skill based on a diagnostic insight.

**Returns:** `{"success": bool, "skill_id": str, "patch": dict, "total_patches": int}` or error dict.

### Insight

`skillforge.core.diagnosis.Insight` (dataclass)

| Field | Type | Default |
|-------|------|---------|
| `root_cause` | `str` | — |
| `heuristic` | `str` | — |
| `patch_suggestion` | `str` | — |
| `confidence` | `float` | `0.5` |
| `timestamp` | `datetime` | now |

**Methods:** `to_dict()`, `from_dict(data)` (classmethod)

---

## BenchmarkRunner

`skillforge.benchmark.runner.BenchmarkRunner`

Runs A/B benchmarks comparing baseline vs SkillForge performance.

### Constructor

```python
BenchmarkRunner(
    skillforge: Any = None,
    model_pricing: dict[str, float] | None = None,
)
```

Default pricing: `{"input_cost_per_1k": 0.005, "output_cost_per_1k": 0.015}`

### Methods

#### `run_baseline`

```python
run_baseline(tasks: TaskSuite | list[Task]) -> BenchmarkRunResult
```

---

#### `run_skillforge`

```python
run_skillforge(tasks: TaskSuite | list[Task]) -> BenchmarkRunResult
```

---

#### `compare`

```python
compare(
    baseline: BenchmarkRunResult,
    skillforge: BenchmarkRunResult,
) -> ComparisonResult
```

---

#### `run_full_benchmark`

```python
run_full_benchmark(
    tasks: TaskSuite | list[Task],
    n_retries: int = 3,
) -> dict[str, Any]
```

**Returns:** `{"runs": [...], "comparisons": [...], "summary": {...}}`

### BenchmarkRunResult

`skillforge.benchmark.runner.BenchmarkRunResult` (dataclass)

| Property | Type | Description |
|----------|------|-------------|
| `label` | `str` | Run label |
| `metrics` | `list[MetricResult]` | Per-task metrics |
| `avg_correctness` | `float` | Mean correctness |
| `pass_rate` | `float` | Fraction ≥ 0.8 |
| `total_cost` | `float` | Total USD cost |
| `avg_latency` | `float` | Mean latency (ms) |
| `total_tokens` | `int` | Sum of tokens |
| `avg_tool_calls` | `float` | Mean tool calls |

### ComparisonResult

`skillforge.benchmark.runner.ComparisonResult` (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `baseline` | `BenchmarkRunResult` | Baseline results |
| `skillforge` | `BenchmarkRunResult` | SkillForge results |
| `correctness_lift` | `float` | Absolute lift in mean correctness |
| `cost_ratio` | `float` | SkillForge cost / baseline cost |
| `token_ratio` | `float` | SkillForge tokens / baseline tokens |
| `latency_ratio` | `float` | SkillForge latency / baseline latency |
| `tool_call_diff` | `float` | SkillForge - baseline avg tool calls |
| `stat_significance` | `dict` | `{"p_value_correctness", "cohens_d"}` |

---

## ReportGenerator

`skillforge.benchmark.report.ReportGenerator`

Generates text reports and charts from benchmark results.

### Constructor

```python
ReportGenerator(output_dir: str | Path | None = None)
```

### Methods

#### `generate_summary_report`

```python
generate_summary_report(
    baseline: BenchmarkRunResult,
    skillforge: BenchmarkRunResult,
) -> str
```

Returns a formatted text report with per-category breakdown.

---

#### `generate_evolution_curves`

```python
generate_evolution_curves(
    results: list[dict[str, BenchmarkRunResult]],
    save_path: str | Path | None = None,
) -> str | None
```

Rolling correctness and token usage charts. Returns path or `None` if matplotlib unavailable.

---

#### `generate_cost_analysis`

```python
generate_cost_analysis(
    baseline: BenchmarkRunResult,
    skillforge: BenchmarkRunResult,
    save_path: str | Path | None = None,
) -> str | None
```

Three-panel chart: cumulative cost, cost efficiency, token distribution.

---

#### `generate_full_dashboard`

```python
generate_full_dashboard(
    baseline: BenchmarkRunResult,
    skillforge: BenchmarkRunResult,
    save_dir: str | Path | None = None,
) -> dict[str, str | None]
```

**Returns:** `{"report", "evolution_curves", "cost_analysis", "summary_json"}` with file paths.

---

## MetricCollector

`skillforge.benchmark.metrics.MetricCollector`

Collects and aggregates benchmark metrics.

### Methods

#### `collect`

```python
collect(
    task: Any,
    result: Any,
    pricing: dict[str, float] | None = None,
) -> MetricResult
```

`task` must have `.id`. `result` is a dict with keys: `correctness`, `tokens_used`, `tool_calls`, `skills_used`, `latency_ms`.

---

#### `timer`

```python
timer() -> CollectorTimer
```

Context manager that records `elapsed_ms`.

---

#### `aggregate`

```python
aggregate() -> dict[str, float]
```

**Returns:** `{"total_tasks", "avg_correctness", "pass_rate", "total_tokens", "avg_tokens", "total_cost_usd", "avg_cost_usd", "avg_latency_ms", "total_tool_calls", "avg_tool_calls"}`

### MetricResult

`skillforge.benchmark.metrics.MetricResult` (dataclass)

| Field | Type |
|-------|------|
| `task_id` | `str` |
| `correctness` | `float` (0–1) |
| `tokens_used` | `int` |
| `cost_usd` | `float` |
| `latency_ms` | `float` |
| `tool_calls` | `int` |
| `skills_used` | `list[str]` |

**Method:** `is_passing(threshold: float = 0.8) -> bool`

### Standalone Functions

```python
calculate_efficiency_ratio(actual_steps: int, optimal_steps: int) -> float
```
Returns `min(optimal / actual, 1.0)`. Returns `0.0` if actual is 0.

---

## TaskSuite / Task

`skillforge.benchmark.tasks`

### Task

`skillforge.benchmark.tasks.Task` (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier |
| `name` | `str` | Human-readable name |
| `description` | `str` | Task prompt |
| `expected` | `Any` | Expected output/keyword |
| `difficulty` | `TaskDifficulty` | `EASY`, `MEDIUM`, `HARD` |
| `category` | `TaskCategory` | `CODING`, `RESEARCH`, `FILE_OPS`, `SKILL_INTENSIVE` |
| `verifier_fn` | `Callable[[Any, Any], float]` | Scoring function (expected, actual) → 0–1 |
| `optimal_steps` | `int` | Minimum steps expected (default 1) |

**Method:** `verify(actual: Any) -> float`

### TaskSuite

`skillforge.benchmark.tasks.TaskSuite`

#### Constructor

```python
TaskSuite(tasks: list[Task] | None = None)
```

#### Class Methods

```python
generate_standard_suite(count: int = 50) -> TaskSuite   # Built-in 60-task suite
load_from_yaml(path: str | Path) -> TaskSuite            # Load from YAML file
```

#### Instance Methods

```python
filter_by_category(category: TaskCategory) -> TaskSuite
filter_by_difficulty(difficulty: TaskDifficulty) -> TaskSuite
```

**Supports:** `len(suite)`, iteration with `for task in suite`.
