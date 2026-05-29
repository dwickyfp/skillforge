"""Benchmark metrics collection and analysis."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """Result metrics for a single benchmark task execution."""

    task_id: str
    correctness: float  # 0.0 to 1.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    tool_calls: int = 0
    skills_used: list[str] = field(default_factory=list)

    def is_passing(self, threshold: float = 0.8) -> bool:
        return self.correctness >= threshold


@dataclass
class CollectorTimer:
    """Context manager for timing task execution."""

    _start: float = 0.0
    elapsed_ms: float = 0.0

    def __enter__(self) -> CollectorTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


class MetricCollector:
    """Collects and aggregates benchmark metrics."""

    def __init__(self) -> None:
        self.results: list[MetricResult] = []

    def collect(
        self,
        task: Any,
        result: Any,
        pricing: dict[str, float] | None = None,
    ) -> MetricResult:
        """Collect metrics from a task execution.

        Args:
            task: The benchmark task (must have .id attribute).
            result: Execution result dict with keys: correctness, tokens_used,
                    tool_calls, skills_used, latency_ms.
            pricing: Optional dict with 'input_cost_per_1k' and 'output_cost_per_1k'.
        """
        pricing = pricing or {}
        tokens = result.get("tokens_used", 0)
        cost = (
            tokens / 1000.0 * pricing.get("output_cost_per_1k", 0.0)
            if tokens > 0
            else 0.0
        )

        metric = MetricResult(
            task_id=task.id,
            correctness=result.get("correctness", 0.0),
            tokens_used=tokens,
            cost_usd=cost,
            latency_ms=result.get("latency_ms", 0.0),
            tool_calls=result.get("tool_calls", 0),
            skills_used=result.get("skills_used", []),
        )
        self.results.append(metric)
        return metric

    def timer(self) -> CollectorTimer:
        return CollectorTimer()

    def aggregate(self) -> dict[str, float]:
        """Return aggregate statistics across all collected results."""
        if not self.results:
            return {}
        n = len(self.results)
        return {
            "total_tasks": n,
            "avg_correctness": sum(r.correctness for r in self.results) / n,
            "pass_rate": sum(1 for r in self.results if r.is_passing()) / n,
            "total_tokens": sum(r.tokens_used for r in self.results),
            "avg_tokens": sum(r.tokens_used for r in self.results) / n,
            "total_cost_usd": sum(r.cost_usd for r in self.results),
            "avg_cost_usd": sum(r.cost_usd for r in self.results) / n,
            "avg_latency_ms": sum(r.latency_ms for r in self.results) / n,
            "total_tool_calls": sum(r.tool_calls for r in self.results),
            "avg_tool_calls": sum(r.tool_calls for r in self.results) / n,
        }

    def clear(self) -> None:
        self.results.clear()


def calculate_efficiency_ratio(actual_steps: int, optimal_steps: int) -> float:
    """Calculate efficiency ratio: optimal / actual (1.0 = perfect).

    Returns 0.0 if actual_steps is 0.
    """
    if actual_steps <= 0:
        return 0.0
    return min(optimal_steps / actual_steps, 1.0)
