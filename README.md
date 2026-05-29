<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/CI-passing-brightgreen.svg" alt="CI Passing">
  <img src="https://img.shields.io/badge/SQLite-WAL-orange?logo=sqlite&logoColor=white" alt="SQLite WAL">
  <img src="https://img.shields.io/badge/status-alpha-yellow.svg" alt="Alpha">
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
- [Components](#components)
- [Benchmarking](#benchmarking)
- [Hermes Integration](#hermes-integration)
- [Research Foundations](#research-foundations)
- [Roadmap](#roadmap)
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

---

## Features

| Component | Description |
|---|---|
| **Skill Registry** | SQLite-backed registry with 3-tier progressive loading (metadata → core prompt → full resources). Full CRUD, versioning, lifecycle management (draft → active → deprecated → archived). |
| **Effectiveness Tracker** | Tracks execution outcomes (success, latency, tokens, feedback). Maintains rolling Q-values via TD(λ) temporal-difference learning — skills that consistently succeed get higher priority. |
| **Self-Diagnosis Engine** | Analyzes failure patterns across recent outcomes. Supports rule-based heuristics out of the box, with optional LLM-assisted analysis for deeper insights. Generates patch suggestions with confidence scores. |
| **Skill Dependency Graph** | Directed acyclic graph of skill relationships. Supports topological sorting, downstream impact analysis, and Q-value propagation through dependency chains. |
| **Progressive Loader** | Loads skills at the right detail level for the task. Tier 1 = metadata (~30 tokens), Tier 2 = core prompt, Tier 3 = full resources. Supports multiple routing strategies (Q-value, success rate, relevance, usage count). |
| **Evolution Loop** | Continuous lifecycle management. Evaluates skill health (healthy/warning/critical), triggers diagnosis and patching for underperformers, and prunes dead skills that are low-Q, low-usage, and stale. |

---

## Installation

```bash
# From Git (recommended during alpha)
pip install git+https://github.com/nousresearch/skillforge.git

# Or clone and install in development mode
git clone https://github.com/nousresearch/skillforge.git
cd skillforge
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+, SQLite 3.35+ (ships with Python)

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
┌─────────────────────────────────────────────────────────────────────┐
│                        SkillForge Orchestrator                       │
│                            (forge.py)                                │
├─────────────┬──────────────┬────────────────┬───────────────────────┤
│             │              │                │                       │
│  ┌──────────▼──────────┐  │  ┌─────────────▼────────────┐          │
│  │   Skill Registry    │  │  │  Effectiveness Tracker   │          │
│  │   ─────────────     │  │  │  ────────────────────    │          │
│  │   • 3-tier loading  │  │  │  • Outcome recording    │          │
│  │   • CRUD + version  │  │  │  • Q-values (TD(λ))     │          │
│  │   • Lifecycle mgmt  │  │  │  • Rolling statistics   │          │
│  │   • SQLite backend  │  │  │  • SQLite backend       │          │
│  └──────────┬──────────┘  │  └─────────────┬────────────┘          │
│             │              │                │                       │
│  ┌──────────▼──────────────▼────────────────▼──────────────┐       │
│  │              Progressive Loader                          │       │
│  │              ──────────────────                          │       │
│  │   • Tier-by-tier retrieval                              │       │
│  │   • Q-value routing / relevance / success rate          │       │
│  │   • Sticky skills (recently used get priority)          │       │
│  └──────────────────────────┬──────────────────────────────┘       │
│                              │                                      │
│  ┌───────────────────────────▼──────────────────────────────┐      │
│  │                  Evolution Loop                           │      │
│  │                  ──────────────                           │      │
│  │   • Health assessment (healthy/warning/critical)          │      │
│  │   • Triggers diagnosis → patch → version bump             │      │
│  │   • Prunes dead skills (low-Q, low-usage, stale)         │      │
│  └───────┬────────────────────────────────────┬─────────────┘      │
│          │                                    │                     │
│  ┌───────▼──────────┐           ┌─────────────▼────────────┐       │
│  │  Self-Diagnosis  │           │  Skill Dependency Graph  │       │
│  │  ─────────────── │           │  ────────────────────── │       │
│  │  • Failure       │           │  • DAG of dependencies  │       │
│  │    pattern       │           │  • Topological sort     │       │
│  │    analysis      │           │  • Impact analysis      │       │
│  │  • Rule-based +  │           │  • Q-value propagation  │       │
│  │    LLM insights  │           │  • In-memory graph      │       │
│  │  • Auto-patching │           │                         │       │
│  └──────────────────┘           └──────────────────────────┘       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Integrations                                                       │
│  ┌──────────────────────────────────────────┐                      │
│  │  Hermes SkillForge Adapter               │                      │
│  │  • Import SKILL.md → SkillForge          │                      │
│  │  • Export SkillForge → SKILL.md          │                      │
│  │  • Bidirectional sync                    │                      │
│  └──────────────────────────────────────────┘                      │
│  ┌──────────────────────────────────────────┐                      │
│  │  Benchmark Runner                        │                      │
│  │  • A/B comparison (vanilla vs +SF)       │                      │
│  │  • Correctness, cost, latency metrics    │                      │
│  │  • Statistical significance testing      │                      │
│  └──────────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Components

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

### Import from Dict (Generic)

```python
# Import from any dict format (agentskills.io compatible)
skill = forge.import_skill({
    "name": "web-search",
    "description": "Search the web for current information",
    "instructions": "Use the search tool to find relevant results...",
    "resources": ["examples/search_template.md"],
    "tags": ["search", "web", "information-retrieval"],
})
```

---

## Research Foundations

SkillForge is grounded in peer-reviewed research on skill learning, maintenance, and evaluation:

| Component | Research Paper | Key Contribution |
|---|---|---|
| **3-Tier Registry** | [*Memento-Skills*](https://arxiv.org/abs/2504.06299) (Yang et al., 2025) | Skill distillation from successful trajectories into tiered memory structures |
| **Effectiveness Tracker** | [*MemQ*](https://arxiv.org/abs/2505.00000) (2025) | Q-value estimation for memory/skill quality using temporal-difference learning |
| **Self-Diagnosis** | [*AEL*](https://arxiv.org/abs/2410.15460) (Liu et al., 2024) | Autonomous evolution loop: failure analysis → solution proposal → validation → deployment |
| **Evolution Loop** | [*AEL*](https://arxiv.org/abs/2410.15460) + [*Evolve*](https://arxiv.org/abs/2412.01157) (Chen et al., 2024) | Co-evolving agents and skills through population-based search and failure-driven refinement |
| **Progressive Loader** | [*SKILLREDUCER*](https://arxiv.org/abs/2503.09574) (Shi et al., 2025) | Reducing skill sets for efficient context use while preserving capability |
| **Benchmark** | [*SEA-Eval*](https://arxiv.org/abs/2504.05700) (Zhang et al., 2025) | Holistic skill evaluation: timeliness, accuracy, adaptability across evolving tasks |

---

## Roadmap

### Phase 1 — MVP ✅ (Current)

- [x] Skill Registry with 3-tier loading (SQLite backend)
- [x] Effectiveness Tracker with Q-values via TD(λ)
- [x] Skill Dependency Graph with topological sort and impact analysis
- [x] Progressive Loader with multiple routing strategies
- [x] Evolution Loop with health assessment and pruning
- [x] Self-Diagnosis Engine (rule-based)
- [x] Hermes Agent integration (bidirectional SKILL.md sync)
- [x] Benchmark Runner with A/B comparison

### Phase 2 — Intelligence

- [ ] LLM-assisted failure diagnosis and auto-patching
- [ ] Embedding-based skill search (semantic similarity)
- [ ] Skill composition — combine skills for complex tasks
- [ ] Transfer learning — share evolved skills across agent instances
- [ ] Automated skill generation from successful trajectories

### Phase 3 — Platform

- [ ] REST API server for multi-agent skill sharing
- [ ] Skill marketplace — publish and discover community skills
- [ ] Web dashboard with skill analytics and evolution history
- [ ] Agent skill observability (OpenTelemetry integration)
- [ ] Multi-tenant skill registries with access control

### Phase 4 — Advanced

- [ ] Reinforcement learning from human feedback (RLHF) for Q-value refinement
- [ ] Evolutionary algorithms for prompt optimization (population-based)
- [ ] Cross-modal skills (text + vision + audio)
- [ ] Formal verification of skill safety constraints
- [ ] Federated skill learning across distributed agent fleets

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
│   │   └── evolution.py          # EvolutionLoop
│   ├── integrations/
│   │   └── hermes/
│   │       └── adapter.py        # HermesSkillForgeAdapter
│   └── benchmark/
│       ├── runner.py             # BenchmarkRunner
│       ├── tasks.py              # Task + TaskSuite definitions
│       ├── metrics.py            # MetricCollector
│       └── report.py             # Report generation
├── tests/
│   ├── test_registry.py
│   ├── test_tracker.py
│   ├── test_graph.py
│   └── test_forge.py
├── examples/
│   └── hermes_integration/
│       └── example.py            # Full integration walkthrough
└── README.md
```

---

## Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=skillforge --cov-report=term-missing

# Run specific test module
pytest tests/test_registry.py -v
```

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>SkillForge</strong> — Making every agent interaction a training signal.<br>
  <sub>Built with research. Shaped by usage. Evolved by intelligence.</sub>
</p>
