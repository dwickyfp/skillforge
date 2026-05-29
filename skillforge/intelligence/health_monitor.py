"""Real-time skill health monitoring and diagnostics.

Computes a 0–1 health score for each active skill based on Q-value,
success rate, recency of last use, and usage frequency.  Skills are
classified into :class:`HealthStatus` buckets and a dashboard summary
is available for quick overviews.

No additional database tables are needed — all metrics are computed
on-the-fly from the existing registry and tracker data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from skillforge.core.registry import Skill, SkillLifecycle, SkillRegistry
from skillforge.core.tracker import QValueTracker
from skillforge.core.graph import SkillDependencyGraph


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class HealthStatus(Enum):
    """Health classification buckets for a skill."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEAD = "dead"


@dataclass
class SkillHealth:
    """Comprehensive health report for a single skill.

    Attributes:
        skill_id: The skill's registry ID.
        status: Health classification bucket.
        q_value: Current Q-value from the tracker (0–1).
        success_rate: Rolling success rate (0–1).
        usage_count: Total recorded outcomes.
        last_used: ISO timestamp of the most recent outcome, or ``None``.
        days_since_last_use: Days elapsed since the last outcome.
        failure_rate: ``1 - success_rate``.
        health_score: Composite score in [0, 1].
        recommendations: Actionable suggestions to improve health.
    """

    skill_id: str = ""
    status: HealthStatus = HealthStatus.WARNING
    q_value: float = 0.5
    success_rate: float = 0.5
    usage_count: int = 0
    last_used: Optional[str] = None
    days_since_last_use: float = float("inf")
    failure_rate: float = 0.5
    health_score: float = 0.5
    recommendations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HealthMonitor
# ---------------------------------------------------------------------------

class HealthMonitor:
    """Monitors skill health in real time by combining registry and tracker data.

    Parameters:
        registry: The skill registry to query for active skills.
        tracker: The :class:`~skillforge.core.tracker.QValueTracker`
            providing Q-values, success rates, and outcome history.
        graph: The :class:`~skillforge.core.graph.SkillDependencyGraph`
            used for dependency-aware diagnostics (optional context).
    """

    # Health-score thresholds for status classification.
    _HEALTHY_THRESHOLD: float = 0.7
    _WARNING_THRESHOLD: float = 0.4
    _CRITICAL_THRESHOLD: float = 0.2

    def __init__(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        self._registry = registry
        self._tracker = tracker
        self._graph = graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_health(self, skill_id: str) -> SkillHealth:
        """Compute a full health report for a single skill.

        The health score is a weighted combination of:
        - Q-value (weight 0.35)
        - Success rate (weight 0.30)
        - Recency of use (weight 0.20) — exponential decay
        - Usage frequency (weight 0.15) — log-scaled

        Health status is then classified:
        - ``HEALTHY`` if score > 0.7
        - ``WARNING`` if score > 0.4
        - ``CRITICAL`` if score > 0.2
        - ``DEAD`` if score <= 0.2

        Parameters:
            skill_id: The ID of the skill to evaluate.

        Returns:
            A :class:`SkillHealth` report.

        Raises:
            ValueError: If *skill_id* is not found in the registry.
        """
        skill = self._registry.get_skill(skill_id, tier=2)
        if skill is None:
            raise ValueError(f"Skill '{skill_id}' not found in registry")

        stats = self._tracker.get_stats(skill_id)
        q_value: float = stats.get("q_value", 0.5)
        success_rate: float = stats.get("success_rate", 0.5)
        total_outcomes: int = stats.get("total_outcomes", 0)

        # Recency: find the most recent outcome timestamp.
        last_used, days_since = self._get_recency(skill_id)

        health_score = self._compute_health_score(
            q_value, success_rate, days_since, total_outcomes
        )
        status = self._classify(health_score)
        failure_rate = 1.0 - success_rate
        recommendations = self._build_recommendations(
            q_value, success_rate, days_since, total_outcomes, status
        )

        return SkillHealth(
            skill_id=skill_id,
            status=status,
            q_value=q_value,
            success_rate=success_rate,
            usage_count=total_outcomes,
            last_used=last_used,
            days_since_last_use=days_since,
            failure_rate=failure_rate,
            health_score=health_score,
            recommendations=recommendations,
        )

    def check_all(self) -> list[SkillHealth]:
        """Run health checks on every active skill.

        Returns:
            A list of :class:`SkillHealth` reports sorted by health score
            (worst first).
        """
        active_skills = self._registry.list_skills(lifecycle=SkillLifecycle.ACTIVE)
        reports: list[SkillHealth] = []
        for skill in active_skills:
            try:
                reports.append(self.check_health(skill.id))
            except ValueError:
                # Skip skills that are not in the registry (shouldn't happen).
                continue
        reports.sort(key=lambda r: r.health_score)
        return reports

    def get_dashboard_summary(self) -> dict[str, Any]:
        """Produce a high-level dashboard summary of all active skills.

        Returns:
            A dict with keys:
            - ``total_skills``: count of active skills.
            - ``by_status``: dict mapping status name → count.
            - ``avg_health``: mean health score across all skills.
            - ``worst_skills``: list of (skill_id, health_score) tuples for
              the bottom 5 skills.
            - ``best_skills``: list of (skill_id, health_score) tuples for
              the top 5 skills.
        """
        reports = self.check_all()

        if not reports:
            return {
                "total_skills": 0,
                "by_status": {s.value: 0 for s in HealthStatus},
                "avg_health": 0.0,
                "worst_skills": [],
                "best_skills": [],
            }

        by_status: dict[str, int] = {s.value: 0 for s in HealthStatus}
        for report in reports:
            by_status[report.status.value] += 1

        avg_health = sum(r.health_score for r in reports) / len(reports)

        # reports is already sorted worst-first.
        worst = [(r.skill_id, round(r.health_score, 4)) for r in reports[:5]]
        best = [(r.skill_id, round(r.health_score, 4)) for r in reports[-5:]][::-1]

        return {
            "total_skills": len(reports),
            "by_status": by_status,
            "avg_health": round(avg_health, 4),
            "worst_skills": worst,
            "best_skills": best,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_health_score(
        self,
        q_value: float,
        success_rate: float,
        recency_days: float,
        usage_count: int,
    ) -> float:
        """Compute composite health score from individual signals.

        Weights:
        - Q-value: 0.35
        - Success rate: 0.30
        - Recency (exponential decay): 0.20
        - Usage frequency (log-scaled): 0.15

        Parameters:
            q_value: Q-value in [0, 1].
            success_rate: Success rate in [0, 1].
            recency_days: Days since last use (inf if never used).
            usage_count: Total number of recorded outcomes.

        Returns:
            A float in [0, 1] representing the composite health score.
        """
        # Q-value component (already 0-1).
        q_component = max(0.0, min(1.0, q_value))

        # Success rate component.
        sr_component = max(0.0, min(1.0, success_rate))

        # When a skill has no recorded outcomes, use neutral defaults
        # for recency and usage to avoid unfairly penalising new or
        # untested skills (which would otherwise score ≤ 0.35 and
        # land in the CRITICAL bucket regardless of Q / success rate).
        if usage_count == 0 and recency_days == float("inf"):
            recency_component = 0.5
            usage_component = 0.5
        else:
            # Recency component: exponential decay with half-life of 14 days.
            # score = exp(-ln(2) * days / 14)
            if recency_days == float("inf"):
                recency_component = 0.0
            else:
                import math
                recency_component = math.exp(-0.6931 * recency_days / 14.0)

            # Usage frequency: log-scaled, saturating around 100 uses.
            # score = min(1.0, log(1 + count) / log(101))
            if usage_count <= 0:
                usage_component = 0.0
            else:
                import math
                usage_component = min(1.0, math.log(1 + usage_count) / math.log(101))

        health = (
            0.35 * q_component
            + 0.30 * sr_component
            + 0.20 * recency_component
            + 0.15 * usage_component
        )
        return max(0.0, min(1.0, health))

    def _classify(self, health_score: float) -> HealthStatus:
        """Map a health score to a :class:`HealthStatus` bucket."""
        if health_score > self._HEALTHY_THRESHOLD:
            return HealthStatus.HEALTHY
        if health_score > self._WARNING_THRESHOLD:
            return HealthStatus.WARNING
        if health_score > self._CRITICAL_THRESHOLD:
            return HealthStatus.CRITICAL
        return HealthStatus.DEAD

    def _build_recommendations(
        self,
        q_value: float,
        success_rate: float,
        recency_days: float,
        usage_count: int,
        status: HealthStatus,
    ) -> list[str]:
        """Generate actionable recommendations based on health signals."""
        recs: list[str] = []

        if q_value < 0.4:
            recs.append(
                "Low Q-value ({:.2f}). Review skill instructions for "
                "accuracy and consider re-training.".format(q_value)
            )

        if success_rate < 0.5:
            recs.append(
                "Success rate is below 50% ({:.1%}). Investigate failure "
                "patterns and add edge-case handling.".format(success_rate)
            )

        if recency_days == float("inf"):
            recs.append("Skill has never been used. Consider promoting it or archiving.")
        elif recency_days > 30:
            recs.append(
                f"Not used in {recency_days:.0f} days. "
                "Evaluate whether this skill is still relevant."
            )

        if usage_count < 5:
            recs.append(
                f"Very few executions ({usage_count}). "
                "More usage data is needed for reliable health assessment."
            )

        if status == HealthStatus.DEAD:
            recs.append(
                "Health is critically low. Consider deprecating or "
                "replacing this skill entirely."
            )

        if not recs:
            recs.append("Skill is healthy. No action required.")

        return recs

    def _get_recency(self, skill_id: str) -> tuple[Optional[str], float]:
        """Determine the last-used timestamp and days elapsed for a skill.

        Queries the tracker's outcome table directly for the most recent
        ``recorded_at`` value.

        Returns:
            A tuple of (iso_timestamp_or_None, days_since_last_use).
            If no outcomes exist, returns ``(None, inf)``.
        """
        try:
            row = self._tracker._conn.execute(
                "SELECT recorded_at FROM outcomes WHERE skill_id = ? "
                "ORDER BY id DESC LIMIT 1",
                (skill_id,),
            ).fetchone()
        except Exception:
            return None, float("inf")

        if row is None:
            return None, float("inf")

        ts_str: str = row["recorded_at"] if "recorded_at" in row.keys() else row[0]
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            return ts_str, float("inf")

        now = datetime.now(timezone.utc)
        # Handle naive datetimes by assuming UTC.
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = now - ts
        days = delta.total_seconds() / 86400.0

        return ts_str, max(0.0, days)
