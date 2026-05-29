"""Benchmark runner: A/B comparison between baseline and SkillForge."""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

from .metrics import MetricCollector, MetricResult
from .tasks import Task, TaskSuite

logger = logging.getLogger(__name__)

_DEFAULT_PRICING: dict[str, float] = {
    "input_cost_per_1k": 0.005,
    "output_cost_per_1k": 0.015,
}


@dataclass
class BenchmarkRunResult:
    """Outcome of a single benchmark run (baseline or SkillForge)."""

    label: str
    metrics: list[MetricResult] = field(default_factory=list)
    outcomes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def avg_correctness(self) -> float:
        if not self.metrics:
            return 0.0
        return statistics.mean(m.correctness for m in self.metrics)

    @property
    def pass_rate(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(1 for m in self.metrics if m.is_passing()) / len(self.metrics)

    @property
    def total_cost(self) -> float:
        return sum(m.cost_usd for m in self.metrics)

    @property
    def avg_latency(self) -> float:
        if not self.metrics:
            return 0.0
        return statistics.mean(m.latency_ms for m in self.metrics)

    @property
    def total_tokens(self) -> int:
        return sum(m.tokens_used for m in self.metrics)

    @property
    def avg_tool_calls(self) -> float:
        if not self.metrics:
            return 0.0
        return statistics.mean(m.tool_calls for m in self.metrics)


@dataclass
class ComparisonResult:
    """Statistical comparison between baseline and SkillForge runs."""

    baseline: BenchmarkRunResult
    skillforge: BenchmarkRunResult
    correctness_lift: float = 0.0
    cost_ratio: float = 0.0
    token_ratio: float = 0.0
    latency_ratio: float = 0.0
    tool_call_diff: float = 0.0
    stat_significance: dict[str, float] = field(default_factory=dict)


def _execute_task_stub(task: Task, use_skillforge: bool) -> dict[str, Any]:
    """Stub executor for a task. In production this calls the real agent/SkillForge."""
    import random

    random.seed(hash(task.id) + (1 if use_skillforge else 0))
    base_correctness = 0.5 if not use_skillforge else 0.7
    bonus = 0.15 if use_skillforge and task.category.value == "skill_intensive" else 0.0
    correctness = min(base_correctness + bonus + random.uniform(-0.1, 0.2), 1.0)
    correctness = max(correctness, 0.0)
    tokens = task.optimal_steps * random.randint(200, 800)
    if use_skillforge:
        tokens = int(tokens * random.uniform(0.7, 0.9))
    tool_calls = task.optimal_steps + random.randint(0, 3)
    skills_used: list[str] = []
    if use_skillforge:
        skills_used = [f"skill_{i}" for i in range(random.randint(1, 3))]
    return {
        "correctness": round(correctness, 4),
        "tokens_used": tokens,
        "tool_calls": tool_calls,
        "skills_used": skills_used,
        "latency_ms": random.uniform(500, 5000),
    }


class BenchmarkRunner:
    """Runs benchmarks comparing baseline vs SkillForge performance."""

    def __init__(
        self,
        skillforge: Any = None,
        model_pricing: dict[str, float] | None = None,
    ) -> None:
        self.skillforge = skillforge
        self.pricing = model_pricing or _DEFAULT_PRICING
        self._executor = self._default_executor

    def _default_executor(
        self, task: Task, use_skillforge: bool
    ) -> dict[str, Any]:
        """Default executor using stub. Override for real execution."""
        if use_skillforge and self.skillforge is not None:
            # In production: call self.skillforge.execute(task.description)
            pass
        return _execute_task_stub(task, use_skillforge)

    def run_baseline(self, tasks: TaskSuite | list[Task]) -> BenchmarkRunResult:
        """Run tasks WITHOUT SkillForge (baseline)."""
        task_list = tasks.tasks if isinstance(tasks, TaskSuite) else tasks
        collector = MetricCollector()
        result = BenchmarkRunResult(label="Baseline")
        for task in task_list:
            logger.info("Baseline: running %s", task.id)
            raw = self._executor(task, use_skillforge=False)
            metric = collector.collect(task, raw, self.pricing)
            result.metrics.append(metric)
            result.outcomes.append(raw)
        return result

    def run_skillforge(self, tasks: TaskSuite | list[Task]) -> BenchmarkRunResult:
        """Run tasks WITH SkillForge."""
        task_list = tasks.tasks if isinstance(tasks, TaskSuite) else tasks
        collector = MetricCollector()
        result = BenchmarkRunResult(label="SkillForge")
        for task in task_list:
            logger.info("SkillForge: running %s", task.id)
            raw = self._executor(task, use_skillforge=True)
            metric = collector.collect(task, raw, self.pricing)
            result.metrics.append(metric)
            result.outcomes.append(raw)
        return result

    def compare(
        self,
        baseline: BenchmarkRunResult,
        skillforge: BenchmarkRunResult,
    ) -> ComparisonResult:
        """Statistical comparison between baseline and SkillForge results."""
        comp = ComparisonResult(baseline=baseline, skillforge=skillforge)

        bl_corr = [m.correctness for m in baseline.metrics]
        sf_corr = [m.correctness for m in skillforge.metrics]

        if bl_corr:
            bl_mean = statistics.mean(bl_corr)
            sf_mean = statistics.mean(sf_corr)
            comp.correctness_lift = sf_mean - bl_mean if bl_mean > 0 else 0.0

        bl_cost = baseline.total_cost
        sf_cost = skillforge.total_cost
        comp.cost_ratio = sf_cost / bl_cost if bl_cost > 0 else 0.0

        bl_tok = baseline.total_tokens
        sf_tok = skillforge.total_tokens
        comp.token_ratio = sf_tok / bl_tok if bl_tok > 0 else 0.0

        bl_lat = baseline.avg_latency
        sf_lat = skillforge.avg_latency
        comp.latency_ratio = sf_lat / bl_lat if bl_lat > 0 else 0.0

        comp.tool_call_diff = skillforge.avg_tool_calls - baseline.avg_tool_calls

        # Simple statistical significance via bootstrap-style p-value
        comp.stat_significance = self._compute_significance(bl_corr, sf_corr)
        return comp

    def _compute_significance(
        self, bl: list[float], sf: list[float]
    ) -> dict[str, float]:
        """Compute a simple significance estimate. Returns approximate p-value."""
        if len(bl) < 2 or len(sf) < 2:
            return {"p_value_correctness": 1.0, "cohens_d": 0.0}
        bl_mean, sf_mean = statistics.mean(bl), statistics.mean(sf)
        bl_std = statistics.stdev(bl) if len(bl) > 1 else 0.0
        sf_std = statistics.stdev(sf) if len(sf) > 1 else 0.0
        pooled_std = ((bl_std**2 + sf_std**2) / 2) ** 0.5
        cohens_d = (sf_mean - bl_mean) / pooled_std if pooled_std > 0 else 0.0
        # Rough p-value approximation from Cohen's d
        import math

        z = abs(cohens_d) * math.sqrt(len(bl) * len(sf) / (len(bl) + len(sf)))
        p_approx = max(0.0, min(1.0, math.exp(-0.717 * z - 0.416 * z**2)))
        return {"p_value_correctness": round(p_approx, 6), "cohens_d": round(cohens_d, 4)}

    def run_full_benchmark(
        self,
        tasks: TaskSuite | list[Task],
        n_retries: int = 3,
    ) -> dict[str, Any]:
        """Full A/B benchmark protocol with retries.

        Returns dict with 'runs', 'comparisons', and 'summary' keys.
        """
        runs: list[dict[str, BenchmarkRunResult]] = []
        comparisons: list[ComparisonResult] = []

        for i in range(n_retries):
            logger.info("Benchmark run %d/%d", i + 1, n_retries)
            baseline = self.run_baseline(tasks)
            skillforge = self.run_skillforge(tasks)
            comp = self.compare(baseline, skillforge)
            runs.append({"baseline": baseline, "skillforge": skillforge})
            comparisons.append(comp)

        # Summary statistics across retries
        avg_lift = statistics.mean(c.correctness_lift for c in comparisons)
        avg_cost_ratio = statistics.mean(c.cost_ratio for c in comparisons)
        avg_token_ratio = statistics.mean(c.token_ratio for c in comparisons)
        avg_p = statistics.mean(
            c.stat_significance.get("p_value_correctness", 1.0)
            for c in comparisons
        )

        return {
            "runs": runs,
            "comparisons": comparisons,
            "summary": {
                "n_runs": n_retries,
                "avg_correctness_lift": round(avg_lift, 4),
                "avg_cost_ratio": round(avg_cost_ratio, 4),
                "avg_token_ratio": round(avg_token_ratio, 4),
                "avg_p_value": round(avg_p, 6),
                "significant": avg_p < 0.05,
            },
        }
