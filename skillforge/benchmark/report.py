"""Benchmark report generation with optional matplotlib charts."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .runner import BenchmarkRunResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates text reports and charts from benchmark results."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path(".")

    def generate_summary_report(
        self,
        baseline: BenchmarkRunResult,
        skillforge: BenchmarkRunResult,
    ) -> str:
        """Generate a text summary report comparing baseline and SkillForge."""
        lines = [
            "=" * 60,
            "SKILLFORGE BENCHMARK REPORT",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "=" * 60,
            "",
            f"{'Metric':<30} {'Baseline':>12} {'SkillForge':>12} {'Delta':>12}",
            "-" * 66,
        ]

        rows = [
            (
                "Pass Rate",
                f"{baseline.pass_rate:.1%}",
                f"{skillforge.pass_rate:.1%}",
                f"{skillforge.pass_rate - baseline.pass_rate:+.1%}",
            ),
            (
                "Avg Correctness",
                f"{baseline.avg_correctness:.3f}",
                f"{skillforge.avg_correctness:.3f}",
                f"{skillforge.avg_correctness - baseline.avg_correctness:+.3f}",
            ),
            (
                "Total Tokens",
                f"{baseline.total_tokens:,}",
                f"{skillforge.total_tokens:,}",
                f"{skillforge.total_tokens - baseline.total_tokens:+,}",
            ),
            (
                "Total Cost (USD)",
                f"${baseline.total_cost:.4f}",
                f"${skillforge.total_cost:.4f}",
                f"${skillforge.total_cost - baseline.total_cost:+.4f}",
            ),
            (
                "Avg Latency (ms)",
                f"{baseline.avg_latency:.0f}",
                f"{skillforge.avg_latency:.0f}",
                f"{skillforge.avg_latency - baseline.avg_latency:+.0f}",
            ),
            (
                "Avg Tool Calls",
                f"{baseline.avg_tool_calls:.1f}",
                f"{skillforge.avg_tool_calls:.1f}",
                f"{skillforge.avg_tool_calls - baseline.avg_tool_calls:+.1f}",
            ),
        ]

        for label, bl, sf, delta in rows:
            lines.append(f"{label:<30} {bl:>12} {sf:>12} {delta:>12}")

        lines.extend(
            [
                "",
                f"Tasks Run: {len(baseline.metrics)} (baseline), {len(skillforge.metrics)} (skillforge)",
                "",
            ]
        )

        # Per-category breakdown
        categories: dict[str, dict[str, list[float]]] = {}
        for m in baseline.metrics:
            cat = m.task_id.split("_")[0] if "_" in m.task_id else "other"
            categories.setdefault(cat, {"bl": [], "sf": []})
            categories[cat]["bl"].append(m.correctness)
        for m in skillforge.metrics:
            cat = m.task_id.split("_")[0] if "_" in m.task_id else "other"
            categories.setdefault(cat, {"bl": [], "sf": []})
            categories[cat]["sf"].append(m.correctness)

        lines.append("Per-Category Breakdown:")
        lines.append(f"{'Category':<20} {'Baseline':>12} {'SkillForge':>12}")
        lines.append("-" * 44)
        for cat, vals in sorted(categories.items()):
            import statistics

            bl_avg = statistics.mean(vals["bl"]) if vals["bl"] else 0.0
            sf_avg = statistics.mean(vals["sf"]) if vals["sf"] else 0.0
            lines.append(f"{cat:<20} {bl_avg:>12.3f} {sf_avg:>12.3f}")

        lines.extend(["", "=" * 60])
        report = "\n".join(lines)
        return report

    def generate_evolution_curves(
        self,
        results: list[dict[str, BenchmarkRunResult]],
        save_path: str | Path | None = None,
    ) -> str | None:
        """Generate chart showing success rate and tokens over sequential tasks."""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available; skipping chart generation")
            return None
        save_path = Path(save_path or self.output_dir / "evolution_curves.png")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for entry in results:
            for label, run in entry.items():
                correctances = [m.correctness for m in run.metrics]
                tokens = [m.tokens_used for m in run.metrics]
                x = list(range(1, len(correctances) + 1))

                # Rolling average correctness
                window = min(5, len(correctances))
                if window > 1:
                    rolling = [
                        sum(correctances[max(0, i - window) : i + 1])
                        / min(i + 1, window)
                        for i in range(len(correctances))
                    ]
                else:
                    rolling = correctances

                ax1.plot(x, rolling, label=f"{label}", alpha=0.8, linewidth=2)
                ax2.plot(x, tokens, label=f"{label}", alpha=0.8, linewidth=1.5)

        ax1.set_xlabel("Task #")
        ax1.set_ylabel("Correctness (rolling avg)")
        ax1.set_title("Success Rate Over Time")
        ax1.legend()
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Task #")
        ax2.set_ylabel("Tokens Used")
        ax2.set_title("Token Usage Per Task")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        fig.suptitle("SkillForge Benchmark: Evolution Curves", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)

    def generate_cost_analysis(
        self,
        baseline: BenchmarkRunResult,
        skillforge: BenchmarkRunResult,
        save_path: str | Path | None = None,
    ) -> str | None:
        """Generate cost comparison chart."""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available; skipping chart generation")
            return None
        save_path = Path(save_path or self.output_dir / "cost_analysis.png")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Cumulative cost
        ax = axes[0]
        for run in [baseline, skillforge]:
            cum_cost = []
            total = 0.0
            for m in run.metrics:
                total += m.cost_usd
                cum_cost.append(total)
            ax.plot(range(1, len(cum_cost) + 1), cum_cost, label=run.label, linewidth=2)
        ax.set_xlabel("Task #")
        ax.set_ylabel("Cumulative Cost (USD)")
        ax.set_title("Cumulative Cost")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.3f}"))

        # 2. Cost per correctness unit
        ax = axes[1]
        labels = ["Baseline", "SkillForge"]
        bl_cpc = baseline.total_cost / max(baseline.avg_correctness * len(baseline.metrics), 0.01)
        sf_cpc = skillforge.total_cost / max(skillforge.avg_correctness * len(skillforge.metrics), 0.01)
        colors = ["#e74c3c", "#2ecc71"]
        ax.bar(labels, [bl_cpc, sf_cpc], color=colors, width=0.5)
        ax.set_ylabel("Cost per Correctness Unit")
        ax.set_title("Cost Efficiency")
        ax.grid(True, alpha=0.3, axis="y")

        # 3. Token distribution
        ax = axes[2]
        bl_tokens = [m.tokens_used for m in baseline.metrics]
        sf_tokens = [m.tokens_used for m in skillforge.metrics]
        ax.boxplot([bl_tokens, sf_tokens], labels=labels, patch_artist=True,
                   boxprops=dict(facecolor="#3498db", alpha=0.5))
        ax.set_ylabel("Tokens")
        ax.set_title("Token Distribution")
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle("SkillForge Benchmark: Cost Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)

    def generate_full_dashboard(
        self,
        baseline: BenchmarkRunResult,
        skillforge: BenchmarkRunResult,
        save_dir: str | Path | None = None,
    ) -> dict[str, str | None]:
        """Generate all charts and a text report, saving to disk.

        Returns dict mapping artifact name to file path (or None if skipped).
        """
        save_dir = Path(save_dir or self.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Text report
        report_text = self.generate_summary_report(baseline, skillforge)
        report_path = save_dir / "benchmark_report.txt"
        report_path.write_text(report_text)
        logger.info("Report written to %s", report_path)

        # Charts
        results = [{"Baseline": baseline, "SkillForge": skillforge}]
        evolution_path = self.generate_evolution_curves(results, save_dir / "evolution_curves.png")
        cost_path = self.generate_cost_analysis(baseline, skillforge, save_dir / "cost_analysis.png")

        # JSON summary for programmatic access
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "baseline": {
                "pass_rate": baseline.pass_rate,
                "avg_correctness": baseline.avg_correctness,
                "total_tokens": baseline.total_tokens,
                "total_cost_usd": baseline.total_cost,
                "avg_latency_ms": baseline.avg_latency,
                "avg_tool_calls": baseline.avg_tool_calls,
            },
            "skillforge": {
                "pass_rate": skillforge.pass_rate,
                "avg_correctness": skillforge.avg_correctness,
                "total_tokens": skillforge.total_tokens,
                "total_cost_usd": skillforge.total_cost,
                "avg_latency_ms": skillforge.avg_latency,
                "avg_tool_calls": skillforge.avg_tool_calls,
            },
        }
        json_path = save_dir / "benchmark_summary.json"
        json_path.write_text(json.dumps(summary, indent=2))

        return {
            "report": str(report_path),
            "evolution_curves": evolution_path,
            "cost_analysis": cost_path,
            "summary_json": str(json_path),
        }
