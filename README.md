<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/Tests-373%20passing-brightgreen.svg" alt="373 Tests">
  <img src="https://img.shields.io/badge/Deps-zero-orange.svg" alt="Zero Dependencies">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="v1.0.0">
</p>

<h1 align="center">⚒️ SkillForge</h1>
<p align="center"><strong>Self-Evolving Skill Intelligence Platform for AI Agents</strong></p>
<p align="center">
  Turn static agent skills into living, data-driven assets that learn from every interaction,<br>
  self-diagnose failures, and continuously evolve to stay optimal.
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Intelligence Layer](#intelligence-layer)
- [Advanced Intelligence](#advanced-intelligence)
- [Platform Layer](#platform-layer)
- [Production Layer](#production-layer)
- [Web Dashboard](#web-dashboard)
- [Benchmarking](#benchmarking)
- [Hermes Integration](#hermes-integration)
- [Project Structure](#project-structure)
- [Research Foundations](#research-foundations)
- [License](#license)

---

## Overview

**SkillForge** is a skill intelligence layer for AI agents. Rather than treating skills as static prompt files, SkillForge makes them *living assets* — tracked, ranked, diagnosed, and evolved through real-world usage data.

Most agents ship with a fixed set of skills. When a skill fails, nobody notices. When a better version exists, nobody knows. SkillForge solves this by wrapping every skill in a feedback loop:

1. **Track** every execution outcome (success, latency, tokens, feedback)
2. **Rank** skills dynamically using reinforcement-learning-inspired Q-values
3. **Diagnose** failure patterns automatically (rule-based or LLM-assisted)
4. **Evolve** underperforming skills — patch prompts, adjust metadata, prune dead skills
5. **Load** progressively at 3 detail tiers to minimize context window waste

Built on research from Memento-Skills, AEL, SKILLREDUCER, SEA-Eval, and MemQ, SkillForge is designed as a drop-in skill intelligence layer for any agent framework.

### Key Stats

| Metric | Value |
|--------|-------|
| **Components** | 35+ |
| **Python files** | 58 |
| **Lines of code** | 22,593 |
| **Tests** | 373 passing |
| **Test time** | 1.30s |
| **External deps** | 0 (stdlib-only) |

---

## Features

### Core (6 components)

| Component | Description |
|-----------|-------------|
| **Skill Registry** | SQLite-backed registry with 3-tier progressive loading (metadata → core prompt → full resources). Full CRUD, versioning, lifecycle management (draft → active → deprecated → archived). |
| **Effectiveness Tracker** | Tracks execution outcomes (success, latency, tokens, feedback). Maintains rolling Q-values via TD(λ) temporal-difference learning — skills that consistently succeed get higher priority. |
| **Self-Diagnosis Engine** | Analyzes failure patterns across recent outcomes. Supports rule-based heuristics out of the box, with optional LLM-assisted analysis for deeper insights. Generates patch suggestions with confidence scores. |
| **Skill Dependency Graph** | Directed acyclic graph of skill relationships. Supports topological sorting, downstream impact analysis, and Q-value propagation through dependency chains. |
| **Progressive Loader** | Loads skills at the right detail level for the task. Tier 1 = metadata (~30 tokens), Tier 2 = core prompt, Tier 3 = full resources. Supports multiple routing strategies (Q-value, success rate, relevance, usage count). |
| **Evolution Loop** | Continuous lifecycle management. Evaluates skill health (healthy/warning/critical), triggers diagnosis and patching for underperformers, and prunes dead skills that are low-Q, low-usage, and stale. |

### Intelligence (5 components)

| Component | Description |
|-----------|-------------|
| **Conflict Detector** | Detects skill overlaps, contradictions, and dependency cycles. Uses Jaccard similarity for overlap detection (threshold > 0.6). |
| **Health Monitor** | 4-weight health scoring: Q-value (35%), success rate (30%), recency decay (20%, half-life 14 days), usage frequency (15%, log-scaled saturation). |
| **Skill Creator** | Auto-creates skills from successful execution trajectories. Pattern frequency threshold (>60%), optional LLM-assisted extraction. |
| **Skill Analyzer** | Clustering, pattern detection, and recommendations. Identifies underperforming skill clusters and suggests improvements. |
| **Skill Optimizer** | Compression (deduplicate, reorder), splitting (decompose monolithic skills), and merging (consolidate similar skills). |

### Advanced Intelligence (6 components)

| Component | Description |
|-----------|-------------|
| **Elastic Memory** | Adaptive memory store with importance tracking. Consolidation via Jaccard similarity > 0.7. Auto-compact with composite retention score (importance 0.45, recency 0.30, access frequency 0.25). |
| **Alert Manager** | Threshold, trend, and anomaly detection alerts. Alert lifecycle (active → acknowledged → resolved). Cooldown support, rule targeting, callback registration for webhook/email/Slack notifications. |
| **Skill Generator** | Zero-shot skill generation from natural language descriptions. Template-driven (no LLM required), TF-IDF-lite keyword extraction, domain-aware tag inference. |
| **Enhanced RL Optimizer** | Contextual bandits (epsilon-greedy with decay), replay buffer (fixed-capacity deque), reward model (linear SGD), curriculum scheduler (difficulty-based ordering). |
| **Performance Predictor** | OLS linear regression for performance trend prediction. Predicts skill improvement/decline from last 20 outcomes. Population variance for consistency measurement. |
| **Skill Transfer Engine** | Cross-agent skill export/import. Formats skills for Hermes, OpenClaw, LangChain, CrewAI. Includes metadata, core content, and dependencies. |

### Platform (4 components)

| Component | Description |
|-----------|-------------|
| **REST API Server** | 13 endpoints on port 8742 (stdlib http.server). CRUD for skills, outcomes, evolution, health, graph, metrics. Zero external dependencies. |
| **MCP Server** | 8 tools via stdin/stdout JSON-RPC. Compatible with Claude Desktop, Cursor, and other MCP clients. |
| **CLI Tool** | 8 commands: serve, skills list/health/search, evolve, dashboard, import, export. |
| **Web Dashboard** | React 18 + Vite + Tailwind CSS 4 + shadcn/ui. 5 pages (Dashboard, Skills, Evolution, Graph, Settings). Recharts visualization. Dark theme. Mock data fallback. |

### Marketplace & Observability (4 components)

| Component | Description |
|-----------|-------------|
| **Skill Marketplace** | Publish, install, rate, and deprecate skills. SQLite-backed catalogue with search, filtering, and install tracking. |
| **Observability — Tracing** | OpenTelemetry-inspired span-based distributed tracing. SQLite persistence, nested spans, parent-child relationships. |
| **Observability — Metrics** | Counters, gauges, timings with summarization (min/max/mean/p50/p95/p99), histograms, label-based filtering. |
| **Observability — Logging** | JSON-structured log entries with 5 severity levels, trace/span correlation, component/skill filtering, full-text search. |

### Async & Versioning (4 components)

| Component | Description |
|-----------|-------------|
| **Async Skill Registry** | Async wrapper via `asyncio.to_thread()` for all registry CRUD operations. |
| **Async Q-Value Tracker** | Async outcome recording, Q-value queries, and TD(λ) updates. |
| **Async Progressive Loader** | Async skill loading and sticky skill retrieval. |
| **Version Manager** | Semantic versioning (semver 2.0.0). Full version history with snapshots, rollback to any version, content diff, auto-generated changelogs. |

### Scale (3 components)

| Component | Description |
|-----------|-------------|
| **A/B Testing** | Full experiment lifecycle. 4 assignment strategies (RANDOM, ROUND_ROBIN, WEIGHTED_RANDOM, HASH_BASED). Two-proportion z-test (2 variants) + chi-squared (3+ variants). Normal CDF via Abramowitz & Stegun (~1.5e-7 accuracy). |
| **DB Abstraction** | `DatabaseBackend` Protocol interface. `SQLiteBackend` (WAL mode, threading lock) + `MemoryBackend` (ephemeral for tests). `DatabaseConnection` context manager with auto-commit/rollback. |
| **Integration Adapters** | `GitHubAdapter` (REST API with SHA tracking), `SlackAdapter` (incoming webhooks), `WebhookAdapter` (HTTP POST + HMAC-SHA256), `FileAdapter` (JSON/markdown export). `AdapterRegistry` for dispatch and health checks. |

### Production (6 components)

| Component | Description |
|-----------|-------------|
| **Circuit Breaker** | CLOSED → OPEN → HALF_OPEN state machine. Configurable failure threshold, recovery timeout, half-open probes. Thread-safe via `threading.Lock`. |
| **Retry Policy** | Exponential backoff with jitter. Configurable max attempts, base delay, max delay, backoff factor, retriable exceptions. |
| **Bulkhead** | Concurrency limiter with semaphore. Configurable max concurrent executions and max wait time. |
| **Graceful Degradation** | Cascading fallback chains. Primary function + ordered fallback list. `FallbackChainExhaustedError` on total failure. |
| **Resilient Executor** | Composable wrapper: Bulkhead → Circuit Breaker → Retry → Callable. Single API for all resilience layers. |
| **Caching** | `TTLCache` (time-to-live with lazy expiry + max_size eviction), `LRUCache` (Least Recently Used), `CachedStore` (composable over data source), `@cached` decorator. `CacheStats` for hit/miss tracking. |

---

## Installation

```bash
# From Git
git clone https://github.com/dwickyfp/skillforge.git
cd skillforge
pip install -e .

# Or install directly
pip install git+https://github.com/dwickyfp/skillforge.git
```

**Requirements:** Python 3.10+, SQLite 3.35+ (ships with Python)

**Zero external dependencies** — SkillForge uses only Python stdlib.

---

## Quick Start

```python
from skillforge.forge import SkillForge

# Initialize — creates ~/.skillforge/skillforge.db automatically
forge = SkillForge()

# Register a skill with 3-tier content
skill = forge.register_skill(
    name="code-reviewer",
    tier1_metadata="Review code for bugs, style, and performance issues",
    tier2_core="You are an expert code reviewer. Analyze the given code for...",
    tier3_resources=["examples/review_template.md", "rules/style_guide.md"],
    tags=["code", "review", "quality"],
)

# Load skills by natural-language query (ranked by Q-value)
skills = forge.load_skill("review my code", tier=2, limit=3)
print(f"Best match: {skills[0].name} (Q={skills[0].q_value:.2f})")

# Record execution outcomes
forge.record_outcome(
    skill_id=skill.id,
    success=True,
    latency_ms=340,
    tokens_used=1200,
    user_feedback=4.5,
)

# Run evolution cycle — diagnose failures, patch skills, prune dead ones
report = forge.run_evolution_loop()
print(report.summary())

# Inspect skill stats
stats = forge.get_skill_stats(skill.id)
print(f"Success rate: {stats['success_rate']:.0%}")

forge.close()
```

Or use the context manager:

```python
with SkillForge() as forge:
    forge.register_skill("greeter", "Greet users warmly")
    # ... all operations ...
# automatically closed
```

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         SkillForge Orchestrator                           │
│                             (forge.py)                                    │
├───────────────┬───────────────┬──────────────────┬────────────────────────┤
│               │               │                  │                        │
│  ┌────────────▼──────────┐    │  ┌───────────────▼────────────┐          │
│  │   Skill Registry      │    │  │  Effectiveness Tracker     │          │
│  │   ─────────────────   │    │  │  ──────────────────────    │          │
│  │   • 3-tier loading    │    │  │  • Outcome recording       │          │
│  │   • CRUD + version    │    │  │  • Q-values (TD(λ))       │          │
│  │   • Lifecycle mgmt    │    │  │  • Rolling statistics      │          │
│  │   • SQLite backend    │    │  │  • SQLite backend          │          │
│  └────────────┬──────────┘    │  └───────────────┬────────────┘          │
│               │               │                  │                        │
│  ┌────────────▼───────────────▼──────────────────▼──────────────┐        │
│  │                  Progressive Loader                           │        │
│  │                  ──────────────────                           │        │
│  │   • Tier-by-tier retrieval                                    │        │
│  │   • Q-value routing / relevance / success rate               │        │
│  │   • Sticky skills (recently used get priority)               │        │
│  └──────────────────────────────┬────────────────────────────────┘        │
│                                  │                                        │
│  ┌───────────────────────────────▼────────────────────────────────┐      │
│  │                    Evolution Loop                               │      │
│  │                    ──────────────                               │      │
│  │   • Health assessment (healthy/warning/critical)                │      │
│  │   • Triggers diagnosis → patch → version bump                   │      │
│  │   • Prunes dead skills (low-Q, low-usage, stale)               │      │
│  └───────┬────────────────────────────────────────┬───────────────┘      │
│          │                                        │                       │
│  ┌───────▼──────────┐              ┌──────────────▼──────────────┐       │
│  │  Self-Diagnosis  │              │  Skill Dependency Graph     │       │
│  │  ─────────────── │              │  ──────────────────────    │       │
│  │  • Failure       │              │  • DAG of dependencies      │       │
│  │    pattern       │              │  • Topological sort         │       │
│  │    analysis      │              │  • Impact analysis          │       │
│  │  • Rule-based +  │              │  • Q-value propagation      │       │
│  │    LLM insights  │              │  • In-memory graph          │       │
│  │  • Auto-patching │              │                             │       │
│  └──────────────────┘              └─────────────────────────────┘       │
│                                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│  Intelligence Layer                                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Conflict Detector  │  Health Monitor  │  Skill Creator            │  │
│  │  Skill Analyzer     │  Skill Optimizer │  Elastic Memory           │  │
│  │  Alert Manager      │  Skill Generator │  Enhanced RL Optimizer    │  │
│  │  Performance Predictor │  Skill Transfer Engine                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│  Platform Layer                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  REST API (port 8742)  │  MCP Server  │  CLI  │  Web Dashboard    │  │
│  │  Skill Marketplace     │  Observability (Tracing/Metrics/Logging) │  │
│  │  Async Support         │  Versioning  │  A/B Testing              │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│  Production Layer                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Circuit Breaker │ Retry Policy │ Bulkhead │ Graceful Degradation │  │
│  │  Resilient Executor │ TTL/LRU Caching │ DB Abstraction            │  │
│  │  Integration Adapters (GitHub/Slack/Webhook/File)                  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│  Integrations                                                             │
│  ┌──────────────────────────────────────────┐                            │
│  │  Hermes SkillForge Adapter               │                            │
│  │  • Import SKILL.md → SkillForge          │                            │
│  │  • Export SkillForge → SKILL.md          │                            │
│  │  • Bidirectional sync                    │                            │
│  └──────────────────────────────────────────┘                            │
│  ┌──────────────────────────────────────────┐                            │
│  │  Benchmark Runner                        │                            │
│  │  • A/B comparison (vanilla vs +SF)       │                            │
│  │  • Correctness, cost, latency metrics    │                            │
│  │  • Statistical significance testing      │                            │
│  └──────────────────────────────────────────┘                            │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### Skill Registry

The central store for all skills, backed by SQLite with WAL mode for concurrent access.

```python
from skillforge.core.registry import SkillRegistry, SkillLifecycle

registry = SkillRegistry(db_path="./my_skills.db")

# Register
skill = registry.register_skill(
    name="summarizer",
    tier1_metadata="Summarize long documents into key points",
    tier2_core="You are an expert summarizer. Given the following document...",
    tier3_resources=["examples/summary_format.md"],
    tags=["nlp", "summarization"],
)

# Progressive retrieval — control detail level
s1 = registry.get_skill(skill.id, tier=1)   # metadata only
s2 = registry.get_skill(skill.id, tier=2)   # + core prompt
s3 = registry.get_skill(skill.id, tier=3)   # + resources

# Search and filter
results = registry.search_skills("summarize")
active = registry.list_skills(lifecycle=SkillLifecycle.ACTIVE, tags=["nlp"])

# Update and version
registry.update_skill(skill.id, {"tier2_core": "Updated prompt..."})
registry.version_skill(skill.id)  # bumps version number

registry.close()
```

### Effectiveness Tracker

Records execution outcomes and maintains Q-values using TD(λ) temporal-difference learning.

```python
from skillforge.core.tracker import QValueTracker, Outcome

tracker = QValueTracker(db_path="./tracker.db")

# Record outcomes
tracker.record_outcome(Outcome(
    skill_id="summarizer",
    success=True,
    latency_ms=250.0,
    tokens_used=800,
    user_feedback=4.0,
))

# Get rolling statistics
stats = tracker.get_stats("summarizer")
# → {'q_value': 0.72, 'success_rate': 0.85, 'avg_latency_ms': 250.0, ...}

# TD(λ) update — called automatically or manually
new_q = tracker.td_lambda_update("summarizer", reward=0.85)

tracker.close()
```

### Self-Diagnosis Engine

Analyzes failure patterns and generates patch suggestions.

```python
from skillforge.core.diagnosis import SelfDiagnosisEngine

# Rule-based analysis (no LLM needed)
diagnosis = SelfDiagnosisEngine(registry=registry, tracker=tracker)

insights = diagnosis.analyze_failures("summarizer", window=10)
for insight in insights:
    print(f"Issue: {insight.failure_type}")
    print(f"Patch: {insight.patch_suggestion}")
    print(f"Confidence: {insight.confidence:.0%}")

# With LLM-assisted analysis
def my_llm(prompt: str) -> str:
    return call_my_model(prompt)

diagnosis_llm = SelfDiagnosisEngine(
    registry=registry,
    tracker=tracker,
    llm_fn=my_llm,
)

# Auto-patch
result = diagnosis.auto_patch_skill("summarizer", insights[0])
print(f"Patched: {result['success']}, new version: {result.get('new_version')}")
```

### Skill Dependency Graph

DAG for modeling relationships between skills.

```python
from skillforge.core.graph import SkillDependencyGraph

graph = SkillDependencyGraph()

graph.add_skill("http-client")
graph.add_skill("api-caller")
graph.add_skill("data-pipeline")

# api-caller depends on http-client (weight 0.8)
graph.add_dependency("api-caller", "http-client", weight=0.8)
graph.add_dependency("data-pipeline", "api-caller", weight=0.5)

# Topological sort
order = graph.topological_sort()
# → ["http-client", "api-caller", "data-pipeline"]

# Impact analysis — what breaks if http-client degrades?
impact = graph.downstream_impact("http-client")
# → ["api-caller", "data-pipeline"]

# Q-value propagation
propagated = graph.propagate_q_update("http-client", delta=-0.2, gamma=0.8)
# → {"api-caller": -0.16, "data-pipeline": -0.128}
```

### Progressive Loader

Loads skills at the right detail level with configurable routing strategies.

```python
from skillforge.core.loader import ProgressiveLoader

loader = ProgressiveLoader(registry, tracker)

# Load by query — tier 1 (fast, metadata-only)
quick = loader.load_skill("email management", tier=1, limit=5)

# Load by query — tier 2 (core prompt for execution)
ready = loader.load_skill("email management", tier=2, routing="q_value")

# Different routing strategies
by_quality  = loader.load_skill("code", routing="q_value")
by_success  = loader.load_skill("code", routing="success_rate")
by_usage    = loader.load_skill("code", routing="usage_count")
by_relevance = loader.load_skill("code", routing="relevance")

# Sticky skills — recently and frequently used
sticky = loader.get_sticky_skills(limit=5)
```

### Evolution Loop

Continuous lifecycle management for all skills.

```python
from skillforge.core.evolution import EvolutionLoop

evolution = EvolutionLoop(
    registry=registry,
    tracker=tracker,
    graph=graph,
    diagnosis=diagnosis,
)

# Full evolution cycle
report = evolution.run_evolution_loop()

print(f"Evaluated: {report.total_skills_evaluated}")
print(f"Healthy:   {report.skills_healthy}")
print(f"Warning:   {report.skills_warning}")
print(f"Critical:  {report.skills_critical}")
print(f"Evolved:   {report.skills_evolved}")
print(f"Pruned:    {report.skills_pruned}")

# Custom thresholds
report = evolution.run_evolution_loop(thresholds={
    "q_warning": 0.6,
    "q_critical": 0.35,
    "prune_max_age_days": 60,
})
```

---

## Intelligence Layer

### Elastic Memory

Adaptive memory store with consolidation and auto-compact.

```python
from skillforge.advanced.elastic_memory import ElasticMemory, MemoryEntry

memory = ElasticMemory(db_path="./memory.db")

# Remember a skill execution
entry = memory.remember(
    skill_id="code-reviewer",
    content="Found critical bug in auth module",
    importance=0.9,
    metadata={"severity": "high", "module": "auth"},
)

# Recall relevant memories
results = memory.recall("auth bug critical")
for result in results:
    print(f"[{result.importance:.2f}] {result.content}")

# Consolidate similar memories (Jaccard > 0.7)
memory.consolidate()

# Auto-compact (LRU with composite retention score)
memory.auto_compact(max_memories=1000)

memory.close()
```

### Alert Manager

Threshold, trend, and anomaly detection alerts.

```python
from skillforge.intelligence.alert_manager import AlertManager, AlertRule, AlertRuleType

alerts = AlertManager(registry, tracker, health_monitor)

# Add alert rules
alerts.add_rule(AlertRule(
    name="critical-skill-degradation",
    rule_type=AlertRuleType.THRESHOLD,
    metric="health_score",
    threshold=0.4,
    operator="less_than",
    cooldown_seconds=3600,
))

# Check alerts (evaluates all rules)
fired = alerts.check_alerts()
for alert in fired:
    print(f"🚨 {alert.severity}: {alert.message}")

# Get active alerts
active = alerts.get_active_alerts()

# Acknowledge and resolve
alerts.acknowledge_alert(alert.id)
alerts.resolve_alert(alert.id, resolution="Fixed by updating prompt")
```

### Skill Generator

Zero-shot skill generation from natural language.

```python
from skillforge.advanced.skill_generator import SkillGenerator, GenerationRequest

generator = SkillGenerator(registry)

# Generate a skill from description
skill = generator.generate(GenerationRequest(
    description="A skill for reviewing pull requests on GitHub, checking for bugs, style issues, and performance problems",
    domain="code-review",
    complexity="medium",
    include_examples=True,
))

print(f"Generated: {skill.name}")
print(f"Tags: {skill.tags}")
print(f"Confidence: {skill.confidence:.2f}")

# Generate and register in one step
skill_id = generator.generate_and_register(
    description="Summarize long documents into bullet points",
    domain="nlp",
)

# Batch generation
skills = generator.generate_batch([
    GenerationRequest(description="Debug Python code", domain="debugging"),
    GenerationRequest(description="Write unit tests", domain="testing"),
])
```

### Enhanced RL Optimizer

Contextual bandits, replay buffer, reward model, curriculum scheduling.

```python
from skillforge.advanced.rl_optimizer import RLOptimizer

rl = RLOptimizer(registry, tracker, evolution)

# Contextual bandit: recommend next action
action = rl.recommend_action("code-reviewer")
# → "evolve" | "patch" | "prune" | "keep"

# Predict reward for an action
reward = rl.predict_reward("code-reviewer", action="evolve")

# Curriculum optimization (difficulty-based ordering)
rl.curriculum_optimize(skill_ids=["skill-1", "skill-2", "skill-3"])

# Batch optimize all underperforming skills
rl.batch_optimize(threshold=0.5)

# Get diagnostics
diag = rl.get_diagnostics()
print(f"Bandit epsilon: {diag['epsilon']:.3f}")
print(f"Replay buffer size: {diag['buffer_size']}")
```

### Performance Predictor

OLS linear regression for performance trend prediction.

```python
from skillforge.advanced.predictor import SkillPredictor

predictor = SkillPredictor(registry, tracker, graph)

# Predict performance trend
trend = predictor.predict_performance("code-reviewer")
print(f"Predicted Q in 7 days: {trend.predicted_q:.2f}")
print(f"Confidence: {trend.confidence:.2f}")
print(f"Trend: {trend.direction}")  # "improving" | "stable" | "declining"

# Get recommendations for a task
recommendations = predictor.predict_for_task("review github pr")
for rec in recommendations:
    print(f"  {rec.skill_id}: Q={rec.predicted_q:.2f} (σ²={rec.variance:.3f})")
```

### Skill Transfer Engine

Cross-agent skill export/import.

```python
from skillforge.advanced.transfer import SkillTransferEngine

transfer = SkillTransferEngine(registry)

# Export to Hermes format
hermes_path = transfer.export_to_hermes(
    skill_id="code-reviewer",
    output_dir="~/.hermes/skills/productivity/",
)

# Export to OpenClaw format
openclaw_path = transfer.export_to_openclaw(
    skill_id="code-reviewer",
    output_dir="./openclaw-skills/",
)

# Export to LangChain format
langchain_path = transfer.export_to_langchain(
    skill_id="code-reviewer",
    output_dir="./langchain-skills/",
)

# Import from another agent
imported = transfer.import_skill(
    format="hermes",
    path="~/.hermes/skills/productivity/code-reviewer/SKILL.md",
)
```

---

## Advanced Intelligence

### Conflict Detector

Detects skill overlaps, contradictions, and dependency cycles.

```python
from skillforge.intelligence.conflict_detector import ConflictDetector

detector = ConflictDetector(registry, tracker)

# Detect overlapping skills (Jaccard similarity > 0.6)
overlaps = detector.detect_overlaps()
for overlap in overlaps:
    print(f"Overlap: {overlap.skill_a} ↔ {overlap.skill_b} ({overlap.similarity:.2f})")

# Detect contradictory skills
contradictions = detector.detect_contradictions()

# Detect dependency cycles
cycles = detector.detect_cycles()
```

### Health Monitor

4-weight health scoring with exponential decay.

```python
from skillforge.intelligence.health_monitor import HealthMonitor

health = HealthMonitor(registry, tracker, graph)

# Check all skills
results = health.check_all()
for result in results:
    print(f"{result.skill_id}: {result.status} (score={result.score:.2f})")

# Health formula:
# 0.35 × Q_value
# + 0.30 × Success_Rate
# + 0.20 × Recency_Decay (half-life 14 days)
# + 0.15 × Usage_Frequency (log-scaled, saturates at ~100 uses)
```

---

## Platform Layer

### REST API Server

```bash
# Start the API server
skillforge serve --port 8742

# Endpoints:
# GET  /api/skills              → List all skills
# GET  /api/skills/:id          → Skill detail
# POST /api/skills              → Create skill
# POST /api/skills/:id/outcomes → Record outcome
# POST /api/evolution/run       → Trigger evolution
# GET  /api/health              → Health dashboard
# GET  /api/metrics             → Aggregated metrics
# GET  /api/graph               → Dependency graph
# GET  /api/evolution/history   → Evolution timeline
```

### Python API Client

```python
from skillforge.api.client import SkillForgeClient

client = SkillForgeClient(base_url="http://localhost:8742")

# List skills
skills = client.list_skills()

# Record outcome
client.record_outcome("skill-id", success=True, latency_ms=300, tokens_used=1000)

# Trigger evolution
client.run_evolution()

# Get health dashboard
health = client.get_health()
```

### CLI Tool

```bash
# List all skills
skillforge skills list

# Search skills
skillforge skills search "code review"

# Check health
skillforge health

# Run evolution
skillforge evolve

# Start API server
skillforge serve --port 8742

# Open dashboard
skillforge dashboard

# Import from Hermes
skillforge import --source ~/.hermes/skills

# Export to Hermes
skillforge export --skill code-reviewer --target ~/.hermes/skills/
```

### MCP Server

```bash
# Start MCP server (stdin/stdout JSON-RPC)
skillforge-mcp

# 8 tools available:
# - list_skills
# - get_skill
# - search_skills
# - record_outcome
# - run_evolution
# - get_health
# - get_graph
# - get_metrics
```

### A/B Testing

```python
from skillforge.advanced.ab_testing import ABTestRunner, ExperimentConfig, Variant

runner = ABTestRunner(db_path="./ab_tests.db")

# Create experiment
config = ExperimentConfig(
    name="prompt-optimization",
    variants=[
        Variant(id="control", skill_id="code-reviewer-v1", weight=0.5),
        Variant(id="treatment", skill_id="code-reviewer-v2", weight=0.5),
    ],
    significance_level=0.05,
    min_sample_size=50,
)

exp_id = runner.create_experiment(config)
runner.start_experiment(exp_id)

# Assign users to variants (4 strategies: RANDOM, ROUND_ROBIN, WEIGHTED_RANDOM, HASH_BASED)
variant = runner.assign_variant(exp_id, user_id="user-123")

# Record outcomes
runner.record_outcome(exp_id, variant_id="control", success=True, latency_ms=300)
runner.record_outcome(exp_id, variant_id="treatment", success=True, latency_ms=250)

# Evaluate statistical significance
result = runner.evaluate(exp_id)
print(f"Significant: {result.is_significant}")
print(f"p-value: {result.p_value:.4f}")
print(f"Winner: {result.winner}")
```

### Skill Marketplace

```python
from skillforge.marketplace.registry import MarketplaceRegistry
from skillforge.marketplace.publisher import SkillPublisher
from skillforge.marketplace.installer import SkillInstaller

marketplace = MarketplaceRegistry(db_path="./marketplace.db")
publisher = SkillPublisher(marketplace, registry)
installer = SkillInstaller(marketplace, registry)

# Publish a skill
publisher.publish(
    skill_id="code-reviewer",
    version="1.2.0",
    description="Expert code review for Python projects",
    tags=["code", "review", "python"],
    visibility="public",
)

# Search marketplace
results = marketplace.search("code review", tags=["python"])

# Install a skill
installed = installer.install(
    skill_id="code-reviewer",
    version="1.2.0",
    target_dir="~/.hermes/skills/productivity/",
)

# Rate a skill
marketplace.rate("code-reviewer", rating=5, review="Excellent skill!")
```

### Observability

```python
from skillforge.observability.tracer import SkillTracer
from skillforge.observability.metrics import MetricsCollector
from skillforge.observability.logger import StructuredLogger

# Tracing
tracer = SkillTracer(db_path="./traces.db")
with tracer.start_span("skill-execution", skill_id="code-reviewer") as span:
    # ... execute skill ...
    span.set_attribute("tokens_used", 1200)
    span.set_status("ok")

# Metrics
metrics = MetricsCollector(db_path="./metrics.db")
metrics.counter("skill_loads", labels={"skill": "code-reviewer"}).inc()
metrics.gauge("q_value", labels={"skill": "code-reviewer"}).set(0.85)
metrics.timing("execution_time", labels={"skill": "code-reviewer"}).observe(340.0)

# Get summary
summary = metrics.summarize("execution_time")
print(f"p50: {summary['p50']:.1f}ms, p95: {summary['p95']:.1f}ms")

# Structured logging
logger = StructuredLogger(db_path="./logs.db")
logger.info("Skill loaded", component="loader", skill="code-reviewer")
logger.warning("Low Q-value", component="evolution", skill="old-skill", q_value=0.3)
```

### Versioning

```python
from skillforge.versioning.version_manager import VersionManager, SemanticVersion

vm = VersionManager(db_path="./versions.db")

# Save a version snapshot
vm.save_version("code-reviewer", version="1.2.0", data={
    "tier1_metadata": "Review code for bugs...",
    "tier2_core": "You are an expert code reviewer...",
})

# List version history
history = vm.list_versions("code-reviewer")
for v in history:
    print(f"  v{v.version} — {v.created_at}")

# Diff between versions
diff = vm.diff("code-reviewer", "1.1.0", "1.2.0")
print(f"Added: {diff['added']}")
print(f"Modified: {diff['modified']}")
print(f"Removed: {diff['removed']}")

# Rollback to a previous version
vm.rollback("code-reviewer", version="1.1.0")
```

---

## Production Layer

### Circuit Breaker

```python
from skillforge.core.resilience import CircuitBreaker, CircuitState

cb = CircuitBreaker(
    name="llm-api",
    failure_threshold=5,        # Open after 5 consecutive failures
    recovery_timeout=30.0,      # Wait 30s before half-open
    half_open_max=2,            # Allow 2 probe calls
    excluded_exceptions=(ValueError,),  # Don't count validation errors
)

# Use the circuit breaker
try:
    result = cb.call(lambda: call_llm_api(prompt))
except CircuitOpenError as e:
    print(f"Circuit open, retry in {e.remaining_seconds:.1f}s")

# Check state
stats = cb.get_stats()
print(f"State: {stats.state}")  # CLOSED / OPEN / HALF_OPEN
print(f"Failures: {stats.consecutive_failures}")
```

### Retry Policy

```python
from skillforge.core.resilience import RetryPolicy

rp = RetryPolicy(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    jitter=True,
    retriable_exceptions=(ConnectionError, TimeoutError),
)

# Execute with retries
result = rp.execute(lambda: fetch_remote_data(url))
```

### Resilient Executor

Composable wrapper combining circuit breaker, retry, and bulkhead.

```python
from skillforge.core.resilience import ResilientExecutor, CircuitBreaker, RetryPolicy, Bulkhead

executor = ResilientExecutor(
    name="llm-call",
    circuit_breaker=CircuitBreaker("llm", failure_threshold=5),
    retry_policy=RetryPolicy(max_attempts=3, base_delay=1.0),
    bulkhead=Bulkhead("llm-pool", max_concurrent=10),
)

# Execute through all resilience layers
result = executor.execute(lambda: call_llm(prompt))

# Get aggregated stats
stats = executor.get_stats()
print(f"Total calls: {stats.total_attempts}")
print(f"Successes: {stats.total_successes}")
```

### Caching

```python
from skillforge.core.cache import TTLCache, LRUCache, CachedStore, cached

# TTL Cache
cache = TTLCache(ttl_seconds=300, max_size=1024)
cache.put("skill-stats:code-reviewer", {"q_value": 0.85, "usage": 42})
stats = cache.get("skill-stats:code-reviewer")

# LRU Cache
lru = LRUCache(max_size=256)
lru.put("key", "value")

# CachedStore — wraps a data source with TTL
store = CachedStore(
    fetch_fn=lambda key: expensive_query(key),
    ttl_seconds=60,
)
result = store.get("query-key")  # cached after first call

# @cached decorator
@cached(ttl_seconds=120)
def get_skill_stats(skill_id: str) -> dict:
    return forge.get_skill_stats(skill_id)

# Cache stats
print(f"Hit rate: {cache.stats.hit_rate:.1%}")
```

### DB Abstraction

```python
from skillforge.core.db import create_backend, SQLiteBackend, MemoryBackend

# SQLite backend (production)
db = create_backend("sqlite", path="./skillforge.db")

# Memory backend (testing)
db = create_backend("memory")

# Use the connection
with db.connect() as conn:
    result = conn.execute("SELECT * FROM skills WHERE q_value > ?", (0.7,))
    for row in result:
        print(row)
```

### Integration Adapters

```python
from skillforge.integrations.adapters import (
    GitHubAdapter, SlackAdapter, WebhookAdapter, FileAdapter, AdapterRegistry
)

registry = AdapterRegistry()

# Register adapters
registry.register("github", GitHubAdapter(
    repo="user/skills-repo",
    token="ghp_xxx",
))

registry.register("slack", SlackAdapter(
    webhook_url="https://hooks.slack.com/services/xxx",
))

registry.register("webhook", WebhookAdapter(
    url="https://api.example.com/skills",
    secret="hmac-secret",  # HMAC-SHA256 signature
))

# Dispatch events to all adapters
registry.dispatch(event_type="skill_evolved", payload={
    "skill_id": "code-reviewer",
    "old_q": 0.45,
    "new_q": 0.72,
})

# Health check all adapters
health = registry.health_check()
for name, status in health.items():
    print(f"  {name}: {'✅' if status else '❌'}")
```

---

## Web Dashboard

React 18 + Vite + Tailwind CSS 4 + shadcn/ui dashboard with 5 pages.

### Setup

```bash
cd dashboard
npm install
npm run dev    # Dev server at http://localhost:5173
npm run build  # Production build in dist/
```

### Pages

| Page | Features |
|------|----------|
| **Dashboard** | KPI cards (Total Skills, Avg Q-Value, Success Rate, Outcomes), health pie chart, token usage trend, top 5 skills, recent evolution events |
| **Skills** | Searchable skill list, expandable rows (tier1/2/3), health badges, Q-value progress bars, per-skill evolution |
| **Evolution** | Timeline visualization, stat cards, global evolution trigger, event cards with success/failure |
| **Graph** | SVG dependency visualization, Q-value color-coded nodes, hover highlighting, click-to-select |
| **Settings** | API endpoint config, auto-evolution toggle, alert thresholds, import/export |

### Connect to API

```bash
# Terminal 1: Start SkillForge API server
python -m skillforge.api --port 8742

# Terminal 2: Start dashboard dev server
cd dashboard && npm run dev
# Dashboard fetches from localhost:8742 (Vite proxy)
```

Dashboard automatically falls back to mock data when the API server is offline.

---

## Benchmarking

SkillForge ships with a benchmark runner for A/B comparison: vanilla agent vs. agent + SkillForge.

```python
from skillforge.benchmark.runner import BenchmarkRunner
from skillforge.benchmark.tasks import Task, TaskSuite, TaskCategory

# Define a task suite
tasks = TaskSuite(
    name="coding-benchmark",
    tasks=[
        Task(
            id="task-001",
            description="Write a Python function to parse CSV files",
            category=TaskCategory.SKILL_INTENSIVE,
            expected_output="Working CSV parser",
            optimal_steps=3,
        ),
        Task(
            id="task-002",
            description="Debug a segmentation fault in C code",
            category=TaskCategory.REASONING,
            expected_output="Root cause identified",
            optimal_steps=5,
        ),
    ],
)

# Run the benchmark
runner = BenchmarkRunner(skillforge=forge)
results = runner.run_full_benchmark(tasks, n_retries=3)

# View summary
summary = results["summary"]
print(f"Correctness lift: {summary['avg_correctness_lift']:+.1%}")
print(f"Cost ratio:       {summary['avg_cost_ratio']:.2f}x")
print(f"Token ratio:      {summary['avg_token_ratio']:.2f}x")
print(f"Stat. significant: {summary['significant']} (p={summary['avg_p_value']:.4f})")
```

---

## Hermes Integration

SkillForge provides a bidirectional adapter for [Hermes Agent](https://hermes-agent.nousresearch.com) skills.

### Import Hermes Skills → SkillForge

```python
from skillforge.forge import SkillForge
from skillforge.integrations.hermes.adapter import HermesSkillForgeAdapter

forge = SkillForge()
adapter = HermesSkillForgeAdapter(
    skillforge=forge._registry,
    hermes_skills_dir="~/.hermes/skills",
)

# Import all SKILL.md files
imported = adapter.import_hermes_skills()
for skill in imported:
    print(f"Imported: {skill.name} (v{skill.version}, Q={skill.q_value})")
```

### Export SkillForge Skills → Hermes

```python
# Export a SkillForge skill as a SKILL.md file
path = adapter.export_skill_to_hermes(
    skill_id="code-reviewer",
    category="productivity",
)
print(f"Written to: {path}")
# → ~/.hermes/skills/productivity/code-reviewer/SKILL.md
```

### Bidirectional Sync

```python
# Sync in both directions
summary = adapter.sync()
print(f"Imported: {summary['imported']}")
print(f"Exported: {summary['exported']}")
print(f"Errors:   {len(summary['errors'])}")
```

---

## Project Structure

```
skillforge/
├── skillforge/
│   ├── __init__.py
│   ├── forge.py                  # Main SkillForge orchestrator
│   ├── core/
│   │   ├── registry.py           # SkillRegistry (SQLite)
│   │   ├── tracker.py            # EffectivenessTracker + QValueTracker
│   │   ├── loader.py             # ProgressiveLoader
│   │   ├── graph.py              # SkillDependencyGraph
│   │   ├── diagnosis.py          # SelfDiagnosisEngine
│   │   ├── evolution.py          # EvolutionLoop
│   │   ├── db.py                 # Database abstraction (SQLite/Memory)
│   │   ├── resilience.py         # CircuitBreaker, Retry, Bulkhead, ResilientExecutor
│   │   └── cache.py              # TTLCache, LRUCache, @cached
│   ├── intelligence/
│   │   ├── conflict_detector.py  # Overlap/contradiction/cycle detection
│   │   ├── health_monitor.py     # 4-weight health scoring
│   │   ├── skill_creator.py      # Auto-create from trajectories
│   │   ├── analyzer.py           # Clustering + recommendations
│   │   ├── optimizer.py          # Compression, splitting, merging
│   │   └── alert_manager.py      # Threshold/trend/anomaly alerts
│   ├── advanced/
│   │   ├── elastic_memory.py     # Adaptive memory store + consolidation
│   │   ├── skill_generator.py    # Zero-shot skill generation
│   │   ├── rl_optimizer.py       # Bandits + replay + curriculum
│   │   ├── predictor.py          # OLS performance prediction
│   │   ├── transfer.py           # Cross-agent skill export/import
│   │   ├── multi_agent.py        # Shared skill pool + access control
│   │   └── ab_testing.py         # A/B testing with statistical tests
│   ├── marketplace/
│   │   ├── registry.py           # Marketplace catalogue
│   │   ├── publisher.py          # Skill validation + packaging
│   │   └── installer.py          # Install + update tracking
│   ├── observability/
│   │   ├── tracer.py             # Distributed tracing (spans)
│   │   ├── metrics.py            # Counters, gauges, timings
│   │   └── logger.py             # Structured JSON logging
│   ├── async_support/
│   │   ├── async_registry.py     # Async SkillRegistry wrapper
│   │   ├── async_tracker.py      # Async QValueTracker wrapper
│   │   └── async_loader.py       # Async ProgressiveLoader wrapper
│   ├── versioning/
│   │   └── version_manager.py    # Semver + history + rollback + diff
│   ├── integrations/
│   │   ├── hermes/
│   │   │   └── adapter.py        # HermesSkillForgeAdapter
│   │   └── adapters.py           # GitHub/Slack/Webhook/File adapters
│   ├── api/
│   │   ├── server.py             # REST API server (stdlib http.server)
│   │   ├── client.py             # Python API client
│   │   └── cli.py                # CLI tool
│   ├── mcp/
│   │   └── server.py             # MCP server (stdin/stdout JSON-RPC)
│   └── benchmark/
│       ├── runner.py             # BenchmarkRunner
│       ├── tasks.py              # Task + TaskSuite definitions
│       ├── metrics.py            # MetricCollector
│       └── report.py             # Report generation
├── dashboard/                    # React + Vite + Tailwind + shadcn/ui
│   ├── src/
│   │   ├── pages/                # Dashboard, Skills, Evolution, Graph, Settings
│   │   ├── components/ui/        # shadcn components (Radix UI)
│   │   ├── components/layout/    # Sidebar, Header, ThemeToggle
│   │   └── lib/                  # API client, utils, mock data
│   ├── package.json
│   └── vite.config.ts
├── tests/                        # 373 tests
│   ├── test_registry.py
│   ├── test_tracker.py
│   ├── test_graph.py
│   ├── test_forge.py
│   ├── test_phase5a.py
│   ├── test_phase6a.py
│   └── test_phase8.py
├── docs/
│   ├── ARTICLE.md                # Medium article draft
│   ├── INTEGRATION_GUIDE.md
│   └── BENCHMARK_GUIDE.md
├── examples/
│   └── hermes_integration/
│       └── example.py
└── README.md
```

---

## Running Tests

```bash
# Run all 373 tests
pytest

# With coverage
pytest --cov=skillforge --cov-report=term-missing

# Run specific test module
pytest tests/test_registry.py -v
```

---

## Research References

SkillForge is built on a comprehensive survey of 2026's most important research in agent skill systems. Below are the papers that directly inspired our architecture:

### Self-Evolution & Skill Learning

| Paper | Key Finding | SkillForge Usage |
|-------|------------|------------------|
| [Memento-Skills](https://arxiv.org/abs/2603.18743) | Self-evolving skill library: +13.7pp GAIA, +20.8pp HLE | EvolutionLoop, SkillCreator |
| [Skill-Pro](https://arxiv.org/abs/2602.01869) | Non-parametric PPO for skill evolution | Q-value update mechanism |
| [AutoSkill](https://arxiv.org/abs/2603.01145) | Version-controlled skill lifecycle | SkillRegistry lifecycle management |
| [AgentFactory](https://arxiv.org/abs/2603.18000) | Skills as executable Python subagent code | SkillTransferEngine |
| [MUSE-Autoskill](https://arxiv.org/abs/2605.27366) | 5-stage lifecycle with per-skill memory | Skill lifecycle stages |
| [XSkill](https://arxiv.org/abs/2603.12056) | Dual-stream: skills (task-level) + experiences (action-level) | Tier architecture (L1/L2/L3) |
| [MemSkill](https://github.com/ViktorAxelsen/MemSkill) | Memory operations as learnable meta-skills | SelfDiagnosisEngine |
| [SkillFlow](https://skillflow.dev) | Opus 4.6 improves from 62.65% to 71.08% (+8.43pp) | Benchmark targets |

### Learning to Self-Evolve

| Paper | Key Finding | SkillForge Usage |
|-------|------------|------------------|
| [LSE — Learning to Self-Evolve](https://openreview.net/pdf?id=zedEdPhmsA) | 4B model beats GPT-5 via learned self-evolution | Core philosophy |
| [Evolving-RL](https://arxiv.org/abs/2605.10663) | 98.7% improvement on ALFWorld unseen tasks | RLOptimizer |
| [AEL — Agent Evolving Learning](https://arxiv.org/abs/2604.21725) | "Less is more" — self-diagnosis > more mechanisms (Sharpe 2.13) | SelfDiagnosisEngine (minimal mechanism) |
| [Native Evolution](https://arxiv.org/abs/2604.18131) | 14B model outperforms unassisted Gemini-2.5-Flash | Evolution philosophy |
| [AutoAgent](https://arxiv.org/abs/2603.09716) | Evolving cognition + elastic memory, closed-loop | Architecture design |

### Context Efficiency

| Paper | Key Finding | SkillForge Usage |
|-------|------------|------------------|
| [Anthropic Progressive Disclosure](https://www.anthropic.com/engineering/code-execution-with-mcp) | 98.7% token reduction (150K → 2K) | ProgressiveLoader 3-tier |
| [SKILLREDUCER](https://arxiv.org/abs/2603.29919) | Compression *improves* quality (48% desc + 39% body) | SkillOptimizer |
| [GenericAgent](https://arxiv.org/abs/2604.17091) | Context density maximization, 4-tier memory | Architecture design |
| [Cloudflare Code Mode](https://nevo.systems) | 99.9% token reduction (1.17M → 1K) | Token efficiency targets |

### Experience-Based Learning & Tool Memory

| Paper | Key Finding | SkillForge Usage |
|-------|------------|------------------|
| [SEARL — Tool Graph Memory](https://github.com/circles-post/SEARL) | 23% higher completion, 68% tool reuse rate | SkillDependencyGraph |
| [MemQ — Provenance DAG](https://github.com/jwliao-ai/MemQ) | Q-learning on provenance DAG, TD(λ) traces | EffectivenessTracker Q-values |
| [ERL — Experiential Reflective Learning](https://arxiv.org/abs/2603.24639) | Heuristics > raw trajectories, +7.8% over ReAct | SelfDiagnosisEngine insights |
| [DeepAgent](https://github.com/RUC-NLPIR/DeepAgent) | Autonomous memory folding, brain-inspired | Architecture design |

### Evaluation & Benchmarking

| Paper | Key Finding | SkillForge Usage |
|-------|------------|------------------|
| [SEA-Eval](https://arxiv.org/abs/2605.04848) | SR + T convergence detects genuine vs pseudo-evolution | BenchmarkRunner evolution metrics |
| [SWE-Bench](https://swe-bench.github.io/) | Standard for coding task evaluation | Benchmark task suite |
| [GAIA Benchmark](https://huggingface.co/gaia-benchmark) | General AI assistant benchmark | Benchmark correctness metric |

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>SkillForge</strong> — Making every agent interaction a training signal.<br>
  <sub>Built with research. Shaped by usage. Evolved by intelligence.</sub>
</p>
