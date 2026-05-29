"""Advanced skill analytics — clustering, usage patterns, and recommendations."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from skillforge.core.graph import SkillDependencyGraph
from skillforge.core.registry import Skill, SkillLifecycle, SkillRegistry
from skillforge.core.tracker import QValueTracker


@dataclass
class SkillCluster:
    """A group of skills clustered by a common dimension.

    Attributes:
        cluster_id: Unique identifier for the cluster.
        skill_ids: List of skill IDs belonging to this cluster.
        domain: The clustering key (e.g., tag name, Q-value bucket).
        centroid_description: Human-readable description of the cluster.
        avg_q_value: Average Q-value of skills in this cluster.
    """

    cluster_id: str
    skill_ids: list[str]
    domain: str
    centroid_description: str
    avg_q_value: float = 0.0


@dataclass
class UsagePattern:
    """Analysis of a skill's usage patterns.

    Attributes:
        skill_id: The analysed skill.
        peak_hours: Hours of the day with the most usage (UTC).
        common_predecessors: Skills commonly used before this one.
        common_successors: Skills commonly used after this one.
        avg_session_length: Average gap between first and last usage
            in a clustered time-window (seconds).
    """

    skill_id: str
    peak_hours: list[int]
    common_predecessors: list[str]
    common_successors: list[str]
    avg_session_length: float


class SkillAnalyzer:
    """Provides advanced analytics for the skill ecosystem.

    Parameters
    ----------
    registry : SkillRegistry
        Skill metadata store.
    tracker : QValueTracker
        Outcome / Q-value tracker.
    graph : SkillDependencyGraph
        Dependency graph for co-occurrence analysis.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Initialise the analyzer.

        Args:
            registry: Skill registry instance.
            tracker: Q-value tracker instance.
            graph: Skill dependency graph instance.
        """
        self._registry = registry
        self._tracker = tracker
        self._graph = graph

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster_skills(
        self, method: str = "domain"
    ) -> list[SkillCluster]:
        """Group skills into clusters.

        Parameters
        ----------
        method : str
            Clustering strategy.  ``'domain'`` groups by the primary tag,
            ``'q_value'`` groups into high (>0.7) / medium (>0.4) / low
            buckets, ``'tags'`` is an alias for ``'domain'``.

        Returns
        -------
        list[SkillCluster]
            Discovered clusters.

        Raises
        ------
        ValueError
            If *method* is not recognised.
        """
        if method not in ("domain", "q_value", "tags"):
            raise ValueError(
                f"Unknown clustering method '{method}'. "
                "Use 'domain', 'q_value', or 'tags'."
            )

        skills = self._registry.list_skills()
        if not skills:
            return []

        if method in ("domain", "tags"):
            return self._cluster_by_tags(skills)
        else:
            return self._cluster_by_q_value(skills)

    def _cluster_by_tags(self, skills: list[Skill]) -> list[SkillCluster]:
        """Cluster skills by their primary tag (first tag, or 'untagged')."""
        buckets: dict[str, list[Skill]] = defaultdict(list)
        for skill in skills:
            primary = skill.tags[0] if skill.tags else "untagged"
            buckets[primary].append(skill)

        clusters: list[SkillCluster] = []
        for idx, (domain, group) in enumerate(sorted(buckets.items())):
            avg_q = sum(s.q_value for s in group) / len(group) if group else 0.0
            clusters.append(
                SkillCluster(
                    cluster_id=f"domain_{idx}",
                    skill_ids=[s.id for s in group],
                    domain=domain,
                    centroid_description=(
                        f"Domain '{domain}' containing {len(group)} skill(s)"
                    ),
                    avg_q_value=round(avg_q, 4),
                )
            )
        return clusters

    def _cluster_by_q_value(self, skills: list[Skill]) -> list[SkillCluster]:
        """Cluster skills into high / medium / low Q-value buckets."""
        high: list[Skill] = []
        medium: list[Skill] = []
        low: list[Skill] = []

        for skill in skills:
            if skill.q_value > 0.7:
                high.append(skill)
            elif skill.q_value > 0.4:
                medium.append(skill)
            else:
                low.append(skill)

        buckets = [
            ("high", high, "> 0.7 Q-value"),
            ("medium", medium, "0.4 – 0.7 Q-value"),
            ("low", low, "≤ 0.4 Q-value"),
        ]

        clusters: list[SkillCluster] = []
        for idx, (name, group, desc) in enumerate(buckets):
            if not group:
                continue
            avg_q = sum(s.q_value for s in group) / len(group)
            clusters.append(
                SkillCluster(
                    cluster_id=f"q_{name}",
                    skill_ids=[s.id for s in group],
                    domain=name,
                    centroid_description=f"{desc} — {len(group)} skill(s)",
                    avg_q_value=round(avg_q, 4),
                )
            )
        return clusters

    # ------------------------------------------------------------------
    # Usage patterns
    # ------------------------------------------------------------------

    def get_usage_patterns(
        self, skill_id: str, window_days: int = 30
    ) -> UsagePattern:
        """Analyse outcome timestamps to derive usage patterns.

        Parameters
        ----------
        skill_id : str
            The skill to analyse.
        window_days : int
            Number of past days to consider.

        Returns
        -------
        UsagePattern
            Peak hours, predecessor / successor lists, and average
            session length.
        """
        conn = self._tracker._conn  # noqa: SLF001 — private access required
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=window_days)
        ).isoformat()

        rows = conn.execute(
            "SELECT recorded_at FROM outcomes WHERE skill_id = ? "
            "AND recorded_at >= ? ORDER BY recorded_at",
            (skill_id, cutoff),
        ).fetchall()

        peak_hours = self._extract_peak_hours(rows)
        predecessors, successors = self._extract_predecessor_successor(
            skill_id, cutoff
        )
        avg_session = self._compute_avg_session_length(rows)

        return UsagePattern(
            skill_id=skill_id,
            peak_hours=peak_hours,
            common_predecessors=predecessors,
            common_successors=successors,
            avg_session_length=avg_session,
        )

    def _extract_peak_hours(
        self, rows: list[sqlite3.Row]
    ) -> list[int]:
        """Return the hour(s) with the most activity (top 3)."""
        hour_counts: dict[int, int] = defaultdict(int)
        for row in rows:
            try:
                dt = datetime.fromisoformat(row["recorded_at"])
                hour_counts[dt.hour] += 1
            except (ValueError, KeyError):
                continue
        sorted_hours = sorted(hour_counts.items(), key=lambda kv: kv[1], reverse=True)
        return [h for h, _ in sorted_hours[:3]]

    def _extract_predecessor_successor(
        self, skill_id: str, cutoff: str
    ) -> tuple[list[str], list[str]]:
        """Find skills executed before/after the target within a 5-min window."""
        conn = self._tracker._conn  # noqa: SLF001
        # Get all outcomes in window, ordered by time
        rows = conn.execute(
            "SELECT skill_id, recorded_at FROM outcomes "
            "WHERE recorded_at >= ? ORDER BY recorded_at",
            (cutoff,),
        ).fetchall()

        predecessor_counts: dict[str, int] = defaultdict(int)
        successor_counts: dict[str, int] = defaultdict(int)

        for i, row in enumerate(rows):
            if row["skill_id"] != skill_id:
                continue
            ts = datetime.fromisoformat(row["recorded_at"])
            # Look for outcomes within ±5 minutes
            for j in range(max(0, i - 10), min(len(rows), i + 10)):
                if j == i:
                    continue
                other = rows[j]
                try:
                    other_ts = datetime.fromisoformat(other["recorded_at"])
                except ValueError:
                    continue
                delta = (other_ts - ts).total_seconds()
                if abs(delta) > 300:
                    continue
                if delta < 0:
                    predecessor_counts[other["skill_id"]] += 1
                elif delta > 0:
                    successor_counts[other["skill_id"]] += 1

        pred_sorted = sorted(predecessor_counts.items(), key=lambda kv: kv[1], reverse=True)
        succ_sorted = sorted(successor_counts.items(), key=lambda kv: kv[1], reverse=True)

        return (
            [s for s, _ in pred_sorted[:5]],
            [s for s, _ in succ_sorted[:5]],
        )

    def _compute_avg_session_length(
        self, rows: list[sqlite3.Row]
    ) -> float:
        """Compute average session length (seconds) from timestamp clusters."""
        if len(rows) < 2:
            return 0.0

        timestamps: list[datetime] = []
        for row in rows:
            try:
                timestamps.append(datetime.fromisoformat(row["recorded_at"]))
            except (ValueError, KeyError):
                continue

        if len(timestamps) < 2:
            return 0.0

        timestamps.sort()
        # Sessions are separated by gaps > 30 min
        session_gaps: list[float] = []
        session_start = timestamps[0]

        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i - 1]).total_seconds()
            if gap > 1800:  # 30 min gap = new session
                session_length = (timestamps[i - 1] - session_start).total_seconds()
                session_gaps.append(session_length)
                session_start = timestamps[i]

        # Final session
        session_gaps.append((timestamps[-1] - session_start).total_seconds())

        return sum(session_gaps) / len(session_gaps) if session_gaps else 0.0

    # ------------------------------------------------------------------
    # Co-occurrence matrix
    # ------------------------------------------------------------------

    def get_skill_matrix(self) -> dict[str, dict[str, int]]:
        """Build an N×N co-occurrence matrix of skills used together.

        Skills are considered co-occurring if they were both executed
        within a 5-minute window.

        Returns
        -------
        dict[str, dict[str, int]]
            Nested dict ``matrix[a][b] = count`` of co-occurrences.
        """
        conn = self._tracker._conn  # noqa: SLF001
        rows = conn.execute(
            "SELECT skill_id, recorded_at FROM outcomes ORDER BY recorded_at"
        ).fetchall()

        matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for i, row_a in enumerate(rows):
            try:
                ts_a = datetime.fromisoformat(row_a["recorded_at"])
            except (ValueError, KeyError):
                continue
            for j in range(i + 1, min(len(rows), i + 50)):
                row_b = rows[j]
                try:
                    ts_b = datetime.fromisoformat(row_b["recorded_at"])
                except (ValueError, KeyError):
                    continue
                delta = (ts_b - ts_a).total_seconds()
                if delta > 300:
                    break
                a, b = row_a["skill_id"], row_b["skill_id"]
                if a != b:
                    matrix[a][b] += 1
                    matrix[b][a] += 1

        # Convert defaultdicts to plain dicts for clean serialization
        return {k: dict(v) for k, v in matrix.items()}

    # ------------------------------------------------------------------
    # Under/over-utilized skills
    # ------------------------------------------------------------------

    def find_underutilized_skills(self, threshold: int = 5) -> list[str]:
        """Find skills whose usage_count is below *threshold*.

        Parameters
        ----------
        threshold : int
            Minimum usage count to be considered utilised.

        Returns
        -------
        list[str]
            Skill IDs with usage_count < threshold.
        """
        return [
            s.id
            for s in self._registry.list_skills()
            if s.usage_count < threshold
        ]

    def find_overloaded_skills(self, threshold: int = 100) -> list[str]:
        """Find skills whose usage_count exceeds *threshold*.

        Parameters
        ----------
        threshold : int
            Maximum usage count before considered overloaded.

        Returns
        -------
        list[str]
            Skill IDs with usage_count > threshold.
        """
        return [
            s.id
            for s in self._registry.list_skills()
            if s.usage_count > threshold
        ]

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def get_recommendations(self) -> list[dict[str, Any]]:
        """Generate actionable recommendations for the skill ecosystem.

        Returns
        -------
        list[dict[str, Any]]
            Each recommendation has keys: ``skill_id``, ``action``,
            ``reason``, ``priority`` (1 = highest).
        """
        recs: list[dict[str, Any]] = []
        skills = self._registry.list_skills()

        for skill in skills:
            # Deprecate dead skills
            if skill.usage_count == 0 and skill.q_value < 0.3:
                recs.append({
                    "skill_id": skill.id,
                    "action": "deprecate",
                    "reason": (
                        f"Zero usage and low Q-value ({skill.q_value:.2f})"
                    ),
                    "priority": 1,
                })

            # Consolidate low-quality skills
            elif skill.q_value < 0.4 and skill.usage_count > 10:
                recs.append({
                    "skill_id": skill.id,
                    "action": "consolidate",
                    "reason": (
                        f"Low Q-value ({skill.q_value:.2f}) despite "
                        f"{skill.usage_count} uses — needs rework"
                    ),
                    "priority": 2,
                })

            # Promote high-performing skills
            elif skill.q_value > 0.8 and skill.usage_count > 20:
                recs.append({
                    "skill_id": skill.id,
                    "action": "promote",
                    "reason": (
                        f"High Q-value ({skill.q_value:.2f}) with "
                        f"{skill.usage_count} uses — ready for wider use"
                    ),
                    "priority": 3,
                })

            # Create new related skills when overloaded
            elif skill.usage_count > 100:
                recs.append({
                    "skill_id": skill.id,
                    "action": "split",
                    "reason": (
                        f"Overloaded: {skill.usage_count} uses — "
                        f"consider splitting into focused sub-skills"
                    ),
                    "priority": 2,
                })

        # Suggest creating new skills for untagged domains
        domains: dict[str, int] = defaultdict(int)
        for s in skills:
            for tag in s.tags:
                domains[tag] += 1

        for domain, count in domains.items():
            if count == 1:
                # Lone skill in a domain — might benefit from a companion
                for s in skills:
                    if domain in s.tags:
                        recs.append({
                            "skill_id": s.id,
                            "action": "create_new",
                            "reason": (
                                f"Lone skill in domain '{domain}' — "
                                f"consider adding complementary skills"
                            ),
                            "priority": 4,
                        })
                        break

        return sorted(recs, key=lambda r: r["priority"])

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        """Generate a formatted text report with all analytics.

        Returns
        -------
        str
            Multi-line formatted report.
        """
        skills = self._registry.list_skills()
        clusters = self.cluster_skills(method="q_value")
        underutilized = self.find_underutilized_skills()
        overloaded = self.find_overloaded_skills()
        recs = self.get_recommendations()

        lines: list[str] = [
            "=" * 60,
            "  SkillForge Intelligence Report",
            "=" * 60,
            "",
            f"  Total Skills: {len(skills)}",
            "",
        ]

        # Q-value clusters
        lines.append("  Q-Value Distribution:")
        for cluster in clusters:
            lines.append(
                f"    {cluster.domain}: {len(cluster.skill_ids)} skills "
                f"(avg Q={cluster.avg_q_value:.2f})"
            )
        lines.append("")

        # Utilisation
        lines.append(f"  Underutilised (< 5 uses): {len(underutilized)}")
        for sid in underutilized[:5]:
            skill = self._registry.get_skill(sid)
            if skill:
                lines.append(f"    - {skill.name} ({skill.usage_count} uses)")
        lines.append(f"  Overloaded (> 100 uses): {len(overloaded)}")
        for sid in overloaded[:5]:
            skill = self._registry.get_skill(sid)
            if skill:
                lines.append(f"    - {skill.name} ({skill.usage_count} uses)")
        lines.append("")

        # Recommendations
        lines.append(f"  Recommendations ({len(recs)}):")
        for r in recs[:10]:
            lines.append(
                f"    [{r['priority']}] {r['action']}: "
                f"{r.get('reason', 'N/A')}"
            )

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
