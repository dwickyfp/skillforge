# SkillForge Benchmark Guide

How to measure, compare, and optimize SkillForge's impact on agent performance.

---

## Table of Contents

1. [Overview](#overview)
2. [Setting Up the Benchmark Environment](#setting-up-the-benchmark-environment)
3. [Running an A/B Comparison](#running-an-ab-comparison)
4. [Understanding Metrics](#understanding-metrics)
5. [Custom Task Suites](#custom-task-suites)
6. [Interpreting Results](#interpreting-results)
7. [Report Generation](#report-generation)
8. [CI/CD Integration](#cicd-integration)

---

## Overview

SkillForge includes a full benchmarking framework that measures the impact of skill-augmented agents versus baseline agents. The benchmark:

- Runs identical tasks with and without SkillForge
- Measures correctness, cost, latency, and efficiency
- Computes statistical significance (Cohen's d, approximate p-value)
- Generates text reports and visualization charts

```
┌────────────┐     ┌────────────┐
│  Baseline   │     │ SkillForge │
│  (no skills)│     │ (w/ skills)│
└──────┬─────┘     └──────┬─────┘
       │                  │
       ▼                  ▼
   ┌────────────────────────┐
   │   MetricCollector      │
   │   - correctness        │
   │   - tokens / cost      │
   │   - latency            │
   │   - tool calls         │
   └───────────┬────────────┘
               ▼
   ┌────────────────────────┐
   │   ComparisonResult     │
   │   - lift, ratios       │
   │   - significance       │
   └───────────┬────────────┘
               ▼
   ┌────────────────────────┐
   │   ReportGenerator      │
   │   - text summary       │
   │   - charts (png)       │
   │   - JSON summary       │
   └────────────────────────┘
```

---

## Setting Up the Benchmark Environment

### Install Dependencies

```bash
# Core (required)
pip install -e .

# Optional: for chart generation
pip install matplotlib
```

### Quick Environment Check

```python
from skillforge.benchmark.runner import BenchmarkRunner
from skillforge.benchmark.tasks import TaskSuite

# Generate the standard 60-task suite
suite = TaskSuite.generate_standard_suite(count=50)
print(f"Task suite: {len(suite)} tasks")

# Verify task distribution
from collections import Counter
categories = Counter(t.category.value for t in suite)
difficulties = Counter(t.difficulty.value for t in suite)
print(f"Categories:   {dict(categories)}")
print(f"Difficulties: {dict(difficulties)}")
```

Expected output:
```
Task suite: 50 tasks
Categories:   {'coding': 12, 'research': 12, 'file_ops': 12, 'skill_intensive': 14}
Difficulties: {'easy': 17, 'medium': 18, 'hard': 15}
```

---

## Running an A/B Comparison

### Basic A/B Run

```python
from skillforge.benchmark.runner import BenchmarkRunner
from skillforge.benchmark.tasks import TaskSuite

# Create a task suite
suite = TaskSuite.generate_standard_suite(count=20)

# Initialize runner (uses stub executor by default)
runner = BenchmarkRunner()

# Run baseline (no SkillForge)
baseline = runner.run_baseline(suite)
print(f"Baseline pass rate: {baseline.pass_rate:.1%}")
print(f"Baseline avg correctness: {baseline.avg_correctness:.3f}")

# Run with SkillForge
skillforge = runner.run_skillforge(suite)
print(f"SkillForge pass rate: {skillforge.pass_rate:.1%}")
print(f"SkillForge avg correctness: {skillforge.avg_correctness:.3f}")

# Compare
comparison = runner.compare(baseline, skillforge)
print(f"\nCorrectness lift: {comparison.correctness_lift:+.3f}")
print(f"Cost ratio:       {comparison.cost_ratio:.2f}x")
print(f"Token ratio:      {comparison.token_ratio:.2f}x")
print(f"Latency ratio:    {comparison.latency_ratio:.2f}x")
print(f"Tool call diff:   {comparison.tool_call_diff:+.1f}")
print(f"p-value:          {comparison.stat_significance['p_value_correctness']:.6f}")
print(f"Cohen's d:        {comparison.stat_significance['cohens_d']:.4f}")
```

### Full Benchmark with Retries

```python
# Run the complete protocol with 3 retries for statistical robustness
result = runner.run_full_benchmark(suite, n_retries=3)

summary = result["summary"]
print(f"Runs: {summary['n_runs']}")
print(f"Avg correctness lift: {summary['avg_correctness_lift']:+.4f}")
print(f"Avg cost ratio:       {summary['avg_cost_ratio']:.4f}")
print(f"Avg token ratio:      {summary['avg_token_ratio']:.4f}")
print(f"Avg p-value:          {summary['avg_p_value']:.6f}")
print(f"Statistically significant: {summary['significant']}")
```

### Custom Executor

Override the default stub executor with your real agent:

```python
from skillforge.benchmark.tasks import Task

def my_real_executor(task: Task, use_skillforge: bool) -> dict:
    """Your actual agent execution logic."""
    if use_skillforge:
        # Load skills and execute with SkillForge
        skills = forge.load_skill(task.description, tier=2)
        result = agent.execute(task.description, skills=skills)
    else:
        # Execute without SkillForge (baseline)
        result = agent.execute(task.description)
    
    return {
        "correctness": task.verify(result.output),
        "tokens_used": result.tokens,
        "tool_calls": result.tool_calls,
        "skills_used": [s.name for s in skills] if use_skillforge else [],
        "latency_ms": result.elapsed_ms,
    }

runner = BenchmarkRunner(skillforge=forge)
runner._executor = my_real_executor
```

---

## Understanding Metrics

### The Six Core Metrics

| Metric | Description | Good Direction |
|--------|-------------|----------------|
| **Correctness** | Task output quality (0.0–1.0) | Higher is better |
| **Efficiency** | optimal_steps / actual_steps | Higher is better (1.0 = perfect) |
| **Cost (USD)** | LLM API cost per task | Lower is better |
| **Latency (ms)** | Wall-clock execution time | Lower is better |
| **Evolution** | Q-value improvement over time | Higher is better |
| **Reliability** | Consistency of success rate | Higher is better |

### MetricResult Structure

```python
from skillforge.benchmark.metrics import MetricResult

# Each task produces a MetricResult:
metric = MetricResult(
    task_id="coding_easy_reverse",
    correctness=0.95,       # 0.0–1.0 from verifier function
    tokens_used=1200,       # LLM tokens consumed
    cost_usd=0.018,         # Computed from pricing model
    latency_ms=2340.5,      # Wall-clock milliseconds
    tool_calls=3,           # Number of tool invocations
    skills_used=["code-writer", "python-helper"],
)

print(f"Passing: {metric.is_passing(threshold=0.8)}")  # True
```

### Efficiency Ratio

```python
from skillforge.benchmark.metrics import calculate_efficiency_ratio

# Perfect execution
eff = calculate_efficiency_ratio(actual_steps=3, optimal_steps=3)
print(f"Efficiency: {eff:.2f}")  # 1.0

# Over-iteration
eff = calculate_efficiency_ratio(actual_steps=6, optimal_steps=3)
print(f"Efficiency: {eff:.2f}")  # 0.5
```

### Aggregate Metrics

```python
from skillforge.benchmark.metrics import MetricCollector

collector = MetricCollector()
# ... collect metrics from multiple tasks ...

agg = collector.aggregate()
print(f"Total tasks:     {agg['total_tasks']}")
print(f"Avg correctness: {agg['avg_correctness']:.3f}")
print(f"Pass rate:       {agg['pass_rate']:.1%}")
print(f"Total cost:      ${agg['total_cost_usd']:.4f}")
print(f"Avg latency:     {agg['avg_latency_ms']:.0f}ms")
```

---

## Custom Task Suites

### Creating Individual Tasks

```python
from skillforge.benchmark.tasks import (
    Task, TaskDifficulty, TaskCategory,
    TaskSuite,
)

# Custom verifier function
def check_python_code(expected: str, actual: str) -> float:
    """Verify Python code output."""
    if not actual:
        return 0.0
    score = 0.0
    # Check for function definition
    if "def " in actual:
        score += 0.3
    # Check for expected keyword
    if expected.lower() in actual.lower():
        score += 0.3
    # Check for docstring
    if '"""' in actual or "'''" in actual:
        score += 0.2
    # Check for return statement
    if "return " in actual:
        score += 0.2
    return min(score, 1.0)

task = Task(
    id="custom_sorting",
    name="Implement Merge Sort",
    description="Implement a merge sort algorithm in Python",
    expected="merge_sort",
    difficulty=TaskDifficulty.MEDIUM,
    category=TaskCategory.CODING,
    verifier_fn=check_python_code,
    optimal_steps=5,
)
```

### Building a Suite from YAML

Create a YAML file `my_tasks.yaml`:

```yaml
tasks:
  - id: custom_001
    name: "REST API Design"
    description: "Design a REST API for a todo application"
    expected: "REST API design with CRUD endpoints"
    difficulty: medium
    category: coding
    optimal_steps: 4

  - id: custom_002
    name: "Database Schema"
    description: "Design a PostgreSQL schema for user management"
    expected: "SQL schema with users, roles, and permissions tables"
    difficulty: medium
    category: coding
    optimal_steps: 3
```

Load it:

```python
suite = TaskSuite.load_from_yaml("my_tasks.yaml")
print(f"Loaded {len(suite)} custom tasks")
```

### Programmatic Suite Construction

```python
tasks = [
    Task(
        id="api_001",
        name="Build REST endpoint",
        description="Create a FastAPI endpoint for user registration",
        expected="fastapi",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.CODING,
        verifier_fn=lambda e, a: 1.0 if e in str(a).lower() else 0.0,
        optimal_steps=4,
    ),
    Task(
        id="research_001",
        name="Compare databases",
        description="Compare PostgreSQL vs MongoDB for a social media app",
        expected="database comparison",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.RESEARCH,
        verifier_fn=lambda e, a: 0.8 if e in str(a).lower() else 0.2,
        optimal_steps=3,
    ),
]

suite = TaskSuite(tasks)
```

### Filtering Suites

```python
full_suite = TaskSuite.generate_standard_suite(count=60)

# Run only coding tasks
coding_suite = full_suite.filter_by_category(TaskCategory.CODING)

# Run only hard tasks
hard_suite = full_suite.filter_by_difficulty(TaskDifficulty.HARD)

# Combine filters
hard_coding = hard_suite.filter_by_category(TaskCategory.CODING)
```

---

## Interpreting Results

### Reading the Comparison

```python
comparison = runner.compare(baseline, skillforge_result)

# Correctness Lift
# Positive = SkillForge improved correctness
# The lift is the absolute difference in mean correctness
print(f"Lift: {comparison.correctness_lift:+.3f}")
# +0.150 means SkillForge scored 15 percentage points higher

# Cost Ratio
# < 1.0 = SkillForge is cheaper
# > 1.0 = SkillForge costs more
print(f"Cost: {comparison.cost_ratio:.2f}x")
# 0.75x means SkillForge costs 25% less

# Token Ratio
# Same interpretation as cost ratio
print(f"Tokens: {comparison.token_ratio:.2f}x")

# Latency Ratio
print(f"Latency: {comparison.latency_ratio:.2f}x")
```

### Statistical Significance

```python
sig = comparison.stat_significance

# p-value: probability the difference is due to chance
# p < 0.05 typically considered significant
print(f"p-value: {sig['p_value_correctness']:.6f}")

# Cohen's d: effect size
# 0.2 = small, 0.5 = medium, 0.8 = large
print(f"Cohen's d: {sig['cohens_d']:.4f}")
# d > 0.8 with p < 0.05 = strong evidence of improvement
```

### Interpreting by Category

The report generator breaks down performance by task category. Look for:

- **skill_intensive**: Should show the largest lift (this is SkillForge's sweet spot)
- **coding**: Typically shows moderate improvement
- **research**: May show smaller improvements
- **file_ops**: Depends on whether file operation skills exist

---

## Report Generation

### Text Report

```python
from skillforge.benchmark.report import ReportGenerator

reporter = ReportGenerator(output_dir="./benchmark_results")

# Generate text summary
text = reporter.generate_summary_report(baseline, skillforge_result)
print(text)
```

Expected output:
```
============================================================
SKILLFORGE BENCHMARK REPORT
Generated: 2026-05-29 12:00:00 UTC
============================================================

Metric                             Baseline   SkillForge        Delta
------------------------------------------------------------------
Pass Rate                            60.0%       85.0%       +25.0%
Avg Correctness                      0.580       0.780      +0.200
Total Tokens                        48,000      36,000     -12,000
Total Cost (USD)                   $0.7200     $0.5400    -$0.1800
Avg Latency (ms)                      2500        1800        -700
Avg Tool Calls                         4.2         3.1        -1.1

Tasks Run: 50 (baseline), 50 (skillforge)

Per-Category Breakdown:
Category               Baseline   SkillForge
--------------------------------------------
coding                     0.550        0.720
file_ops                   0.600        0.780
research                   0.580        0.690
skill_intensive            0.590        0.880

============================================================
```

### Charts

```python
# Generate evolution curves (correctness and tokens over time)
results = [{"Baseline": baseline, "SkillForge": skillforge_result}]
chart_path = reporter.generate_evolution_curves(results, "evolution.png")

# Generate cost analysis chart
cost_path = reporter.generate_cost_analysis(baseline, skillforge_result, "cost.png")
```

### Full Dashboard

```python
# Generate everything at once
artifacts = reporter.generate_full_dashboard(
    baseline, skillforge_result, save_dir="./benchmark_results"
)

# Returns paths to all generated files:
print(f"Report:     {artifacts['report']}")
print(f"Evolution:  {artifacts['evolution_curves']}")
print(f"Cost:       {artifacts['cost_analysis']}")
print(f"JSON:       {artifacts['summary_json']}")
```

The `generate_full_dashboard()` method creates:
- `benchmark_report.txt` — Human-readable text report
- `evolution_curves.png` — Rolling correctness and token usage charts
- `cost_analysis.png` — Cumulative cost, cost efficiency, token distribution
- `benchmark_summary.json` — Machine-readable summary

### Programmatic Access to JSON

```python
import json

with open(artifacts["summary_json"]) as f:
    data = json.load(f)

# Use in CI pipelines
assert data["skillforge"]["pass_rate"] > 0.7, "Pass rate below threshold"
assert data["skillforge"]["avg_correctness"] > data["baseline"]["avg_correctness"]
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: SkillForge Benchmarks
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -e ".[benchmark]"
      
      - name: Run benchmarks
        run: |
          python -c "
          from skillforge.benchmark.runner import BenchmarkRunner
          from skillforge.benchmark.tasks import TaskSuite
          from skillforge.benchmark.report import ReportGenerator
          
          suite = TaskSuite.generate_standard_suite(count=50)
          runner = BenchmarkRunner()
          result = runner.run_full_benchmark(suite, n_retries=3)
          
          reporter = ReportGenerator(output_dir='benchmark_results')
          baseline = result['runs'][0]['baseline']
          skillforge = result['runs'][0]['skillforge']
          reporter.generate_full_dashboard(baseline, skillforge)
          
          summary = result['summary']
          print(f'Lift: {summary[\"avg_correctness_lift\"]:+.4f}')
          print(f'Significant: {summary[\"significant\"]}')
          
          import sys
          if summary['avg_correctness_lift'] < 0.05:
              print('ERROR: Correctness lift below threshold')
              sys.exit(1)
          "
      
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: benchmark_results/
```

### Benchmarking Custom Tasks in CI

```python
# benchmark_ci.py
import sys
from skillforge.benchmark.runner import BenchmarkRunner
from skillforge.benchmark.tasks import TaskSuite

# Load your project-specific tasks
suite = TaskSuite.load_from_yaml("tests/benchmark_tasks.yaml")

runner = BenchmarkRunner()
result = runner.run_full_benchmark(suite, n_retries=3)

summary = result["summary"]
print(f"Correctness lift: {summary['avg_correctness_lift']:+.4f}")
print(f"p-value: {summary['avg_p_value']:.6f}")

# Gate on quality
THRESHOLD = 0.10  # Minimum 10% lift
if summary["avg_correctness_lift"] < THRESHOLD:
    print(f"FAIL: Lift {summary['avg_correctness_lift']:.4f} < {THRESHOLD}")
    sys.exit(1)

print("PASS: Benchmark meets quality threshold")
```

---

## API Quick Reference

| Class | Purpose |
|-------|---------|
| `BenchmarkRunner` | Orchestrates A/B execution |
| `TaskSuite` | Collection of benchmark tasks |
| `Task` | Single benchmark task with verifier |
| `MetricCollector` | Collects and aggregates metrics |
| `MetricResult` | Metrics for one task execution |
| `BenchmarkRunResult` | Aggregated results for one run |
| `ComparisonResult` | Statistical comparison between runs |
| `ReportGenerator` | Text reports and charts |

See [API_REFERENCE.md](API_REFERENCE.md) for detailed method signatures.
