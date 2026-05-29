"""
Evolution Loop Module.

Manages the continuous improvement lifecycle of skills: monitoring health,
triggering evolution for underperforming skills, pruning dead skills, and
reporting on system-wide evolution activity.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status of a skill."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class EvolutionAction:
    """Record of an action taken during an evolution cycle."""

    action_type: str
    skill_id: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "action_type": self.action_type,
            "skill_id": self.skill_id,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
        }


@dataclass
class EvolutionReport:
    """Summary report of an evolution cycle."""

    cycle_start: datetime
    cycle_end: datetime = field(default_factory=datetime.now)
    total_skills_evaluated: int = 0
    skills_evolved: int = 0
    skills_pruned: int = 0
    skills_healthy: int = 0
    skills_warning: int = 0
    skills_critical: int = 0
    actions: list[EvolutionAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cycle_start": self.cycle_start.isoformat(),
            "cycle_end": self.cycle_end.isoformat(),
            "total_skills_evaluated": self.total_skills_evaluated,
            "skills_evolved": self.skills_evolved,
            "skills_pruned": self.skills_pruned,
            "skills_healthy": self.skills_healthy,
            "skills_warning": self.skills_warning,
            "skills_critical": self.skills_critical,
            "actions": [a.to_dict() for a in self.actions],
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Evolution Cycle: {self.cycle_start.isoformat()} -> {self.cycle_end.isoformat()}",
            f"  Skills evaluated: {self.total_skills_evaluated}",
            f"  Healthy: {self.skills_healthy} | Warning: {self.skills_warning} | Critical: {self.skills_critical}",
            f"  Evolved: {self.skills_evolved} | Pruned: {self.skills_pruned}",
            f"  Total actions: {len(self.actions)}",
        ]
        return "\n".join(lines)


class SkillRegistry(Protocol):
    """Protocol for skill registry interface."""

    def get_skill(self, skill_id: str) -> Optional[dict[str, Any]]:
        """Retrieve skill metadata by ID."""
        ...

    def update_skill(self, skill_id: str, updates: dict[str, Any]) -> bool:
        """Update skill metadata."""
        ...

    def list_skills(self) -> list[str]:
        """List all skill IDs."""
        ...


class FailureTracker(Protocol):
    """Protocol for failure tracking interface."""

    def get_q_value(self, skill_id: str) -> Optional[float]:
        """Get current Q-value for a skill."""
        ...

    def get_usage_count(self, skill_id: str) -> int:
        """Get total usage count for a skill."""
        ...

    def get_failures(
        self, skill_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Retrieve recent failures for a skill."""
        ...

    def get_last_used(self, skill_id: str) -> Optional[datetime]:
        """Get last usage timestamp for a skill."""
        ...


class SkillDependencyGraph(Protocol):
    """Protocol for dependency graph interface."""

    def downstream_impact(self, skill_id: str) -> list[str]:
        """Find all downstream dependent skills."""
        ...

    def get_skills(self) -> list[str]:
        """List all skill IDs in the graph."""
        ...


class SelfDiagnosisEngine(Protocol):
    """Protocol for diagnosis engine interface."""

    def analyze_failures(
        self, skill_id: str, window: int = 10
    ) -> list[Any]:
        """Analyze failures and return insights."""
        ...

    def auto_patch_skill(self, skill_id: str, insight: Any) -> dict[str, Any]:
        """Apply auto-patch based on insight."""
        ...


class EvolutionLoop:
    """
    Orchestrates the continuous evolution and improvement of skills.

    Monitors skill health, triggers evolution for underperforming skills,
    prunes unused/dead skills, and maintains evolution reports.

    Attributes:
        _registry: Skill registry for metadata access.
        _tracker: Failure/performance tracker.
        _graph: Dependency graph for impact analysis.
        _diagnosis: Self-diagnosis engine for failure analysis.
        _history: Record of past evolution actions.
    """

    # Default thresholds for evolution decisions
    DEFAULT_THRESHOLDS: dict[str, float] = {
        "q_warning": 0.5,
        "q_critical": 0.3,
        "failure_rate_warning": 0.2,
        "failure_rate_critical": 0.5,
        "prune_q_threshold": 0.3,
        "prune_min_usage": 5,
        "prune_max_age_days": 90,
    }

    def __init__(
        self,
        registry: SkillRegistry,
        tracker: FailureTracker,
        graph: SkillDependencyGraph,
        diagnosis: SelfDiagnosisEngine,
    ) -> None:
        """
        Initialize the evolution loop.

        Args:
            registry: Skill registry instance.
            tracker: Failure/performance tracker instance.
            graph: Skill dependency graph instance.
            diagnosis: Self-diagnosis engine instance.
        """
        self._registry = registry
        self._tracker = tracker
        self._graph = graph
        self._diagnosis = diagnosis
        self._history: list[EvolutionAction] = []
        self._last_report: Optional[EvolutionReport] = None

    def run_evolution_loop(
        self, thresholds: Optional[dict[str, float]] = None
    ) -> EvolutionReport:
        """
        Execute a full evolution cycle across all active skills.

        Evaluates each skill's health, evolves underperforming ones,
        and generates a comprehensive report.

        Args:
            thresholds: Optional custom thresholds. Merged with defaults
                        for any keys not provided.

        Returns:
            EvolutionReport summarizing all actions taken.
        """
        merged_thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        report = EvolutionReport(cycle_start=datetime.now())

        try:
            skill_ids = self._registry.list_skills()
        except Exception as e:
            logger.error("Failed to list skills: %s", str(e))
            report.cycle_end = datetime.now()
            report.actions.append(
                EvolutionAction(
                    action_type="error",
                    skill_id="system",
                    details={"error": f"Failed to list skills: {str(e)}"},
                    success=False,
                )
            )
            self._last_report = report
            return report

        report.total_skills_evaluated = len(skill_ids)

        for skill_id in skill_ids:
            try:
                health = self.check_skill_health(skill_id)

                if health == HealthStatus.HEALTHY:
                    report.skills_healthy += 1
                elif health == HealthStatus.WARNING:
                    report.skills_warning += 1
                    # Evolve warning-level skills
                    evolved = self.evolve_skill(skill_id)
                    if evolved:
                        report.skills_evolved += 1
                        report.actions.append(
                            EvolutionAction(
                                action_type="evolve",
                                skill_id=skill_id,
                                details={"health": health.value, "reason": "warning"},
                            )
                        )
                elif health == HealthStatus.CRITICAL:
                    report.skills_critical += 1
                    # Evolve critical skills
                    evolved = self.evolve_skill(skill_id)
                    if evolved:
                        report.skills_evolved += 1
                        report.actions.append(
                            EvolutionAction(
                                action_type="evolve",
                                skill_id=skill_id,
                                details={"health": health.value, "reason": "critical"},
                            )
                        )
                    else:
                        # Check downstream impact before considering deprecation
                        downstream = self._graph.downstream_impact(skill_id)
                        report.actions.append(
                            EvolutionAction(
                                action_type="evolve_failed",
                                skill_id=skill_id,
                                details={
                                    "health": health.value,
                                    "downstream_impact_count": len(downstream),
                                },
                                success=False,
                            )
                        )

            except Exception as e:
                logger.error(
                    "Error processing skill '%s' in evolution loop: %s",
                    skill_id,
                    str(e),
                )
                report.actions.append(
                    EvolutionAction(
                        action_type="error",
                        skill_id=skill_id,
                        details={"error": str(e)},
                        success=False,
                    )
                )

        # Prune dead skills
        pruned = self.prune_dead_skills(
            q_threshold=merged_thresholds.get(
                "prune_q_threshold", 0.3
            ),
            min_usage=int(merged_thresholds.get("prune_min_usage", 5)),
            max_age_days=int(merged_thresholds.get("prune_max_age_days", 90)),
        )
        report.skills_pruned = len(pruned)
        for pruned_id in pruned:
            report.actions.append(
                EvolutionAction(
                    action_type="prune",
                    skill_id=pruned_id,
                    details={"reason": "dead_skill"},
                )
            )

        # Record actions in history
        self._history.extend(report.actions)
        report.cycle_end = datetime.now()
        self._last_report = report

        logger.info(
            "Evolution cycle complete: %d evaluated, %d evolved, %d pruned",
            report.total_skills_evaluated,
            report.skills_evolved,
            report.skills_pruned,
        )

        return report

    def check_skill_health(self, skill_id: str) -> HealthStatus:
        """
        Evaluate the health status of a skill based on Q-value and failure rate.

        Health is determined by:
        - Q-value thresholds (lower Q = worse health)
        - Recent failure rate

        Args:
            skill_id: The skill to evaluate.

        Returns:
            HealthStatus enum value (healthy, warning, or critical).

        Raises:
            ValueError: If skill_id is empty.
        """
        if not skill_id:
            raise ValueError("skill_id must be a non-empty string")

        # Get Q-value (default to 0.5 if unavailable)
        q_value = self._tracker.get_q_value(skill_id)
        if q_value is None:
            # No data yet - assume healthy (new skill)
            return HealthStatus.HEALTHY

        # Get failure rate from recent failures
        failures = self._tracker.get_failures(skill_id, limit=10)
        usage_count = self._tracker.get_usage_count(skill_id)

        if usage_count > 0:
            failure_rate = len(failures) / max(usage_count, 1)
        else:
            failure_rate = 0.0

        # Determine health status
        if (
            q_value < self.DEFAULT_THRESHOLDS["q_critical"]
            or failure_rate > self.DEFAULT_THRESHOLDS["failure_rate_critical"]
        ):
            return HealthStatus.CRITICAL
        elif (
            q_value < self.DEFAULT_THRESHOLDS["q_warning"]
            or failure_rate > self.DEFAULT_THRESHOLDS["failure_rate_warning"]
        ):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

    def evolve_skill(self, skill_id: str) -> bool:
        """
        Run diagnosis and apply patches to improve a skill.

        Analyzes recent failures, generates insights, and applies the
        highest-confidence patch suggestion.

        Args:
            skill_id: The skill to evolve.

        Returns:
            True if evolution was successful, False otherwise.
        """
        if not skill_id:
            logger.warning("Cannot evolve: empty skill_id")
            return False

        try:
            # Analyze failures
            insights = self._diagnosis.analyze_failures(skill_id, window=10)

            if not insights:
                logger.info(
                    "No insights generated for skill '%s' - nothing to evolve",
                    skill_id,
                )
                self._history.append(
                    EvolutionAction(
                        action_type="evolve_skip",
                        skill_id=skill_id,
                        details={"reason": "no_insights"},
                    )
                )
                return False

            # Sort by confidence and apply the best insight
            best_insight = max(insights, key=lambda i: i.confidence)

            result = self._diagnosis.auto_patch_skill(skill_id, best_insight)

            if result.get("success"):
                logger.info(
                    "Successfully evolved skill '%s': %s",
                    skill_id,
                    best_insight.patch_suggestion,
                )
                self._history.append(
                    EvolutionAction(
                        action_type="evolve_success",
                        skill_id=skill_id,
                        details={
                            "insight": best_insight.to_dict()
                            if hasattr(best_insight, "to_dict")
                            else str(best_insight),
                            "patch_result": result,
                        },
                    )
                )
                return True
            else:
                logger.warning(
                    "Failed to evolve skill '%s': %s",
                    skill_id,
                    result.get("error", "unknown error"),
                )
                self._history.append(
                    EvolutionAction(
                        action_type="evolve_failure",
                        skill_id=skill_id,
                        details={"error": result.get("error", "unknown")},
                        success=False,
                    )
                )
                return False

        except Exception as e:
            logger.error(
                "Error evolving skill '%s': %s", skill_id, str(e)
            )
            self._history.append(
                EvolutionAction(
                    action_type="evolve_error",
                    skill_id=skill_id,
                    details={"error": str(e)},
                    success=False,
                )
            )
            return False

    def prune_dead_skills(
        self,
        q_threshold: float = 0.3,
        min_usage: int = 5,
        max_age_days: int = 90,
    ) -> list[str]:
        """
        Identify and deprecate skills that are no longer useful.

        A skill is considered "dead" if ALL of the following are true:
        - Q-value is below the threshold
        - Usage count is below the minimum
        - Last used more than max_age_days ago (or never used)

        Args:
            q_threshold: Maximum Q-value to consider for pruning. Default 0.3.
            min_usage: Minimum usage count to keep alive. Default 5.
            max_age_days: Maximum days since last use. Default 90.

        Returns:
            List of skill IDs that were deprecated.
        """
        pruned: list[str] = []
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        try:
            skill_ids = self._registry.list_skills()
        except Exception as e:
            logger.error("Failed to list skills for pruning: %s", str(e))
            return pruned

        for skill_id in skill_ids:
            try:
                skill = self._registry.get_skill(skill_id)
                if skill is None:
                    continue

                # Skip already deprecated skills
                if skill.get("status") == "deprecated":
                    continue

                # Check Q-value
                q_value = self._tracker.get_q_value(skill_id)
                if q_value is not None and q_value >= q_threshold:
                    continue

                # Check usage count
                usage_count = self._tracker.get_usage_count(skill_id)
                if usage_count >= min_usage:
                    continue

                # Check last used date
                last_used = self._tracker.get_last_used(skill_id)
                if last_used is not None and last_used > cutoff_date:
                    continue

                # All criteria met - check downstream impact before pruning
                downstream = []
                try:
                    downstream = self._graph.downstream_impact(skill_id)
                except (ValueError, Exception):
                    pass

                if downstream:
                    logger.warning(
                        "Skipping prune of '%s' - has %d downstream dependents",
                        skill_id,
                        len(downstream),
                    )
                    continue

                # Deprecate the skill
                success = self._registry.update_skill(
                    skill_id,
                    {
                        "status": "deprecated",
                        "deprecated_at": datetime.now().isoformat(),
                        "deprecation_reason": "dead_skill",
                        "final_q_value": q_value,
                        "final_usage_count": usage_count,
                    },
                )

                if success:
                    pruned.append(skill_id)
                    logger.info(
                        "Deprecated dead skill '%s' (q=%.2f, usage=%d)",
                        skill_id,
                        q_value or 0.0,
                        usage_count,
                    )

            except Exception as e:
                logger.error(
                    "Error evaluating skill '%s' for pruning: %s",
                    skill_id,
                    str(e),
                )

        return pruned

    def get_evolution_report(self) -> Optional[EvolutionReport]:
        """
        Get the most recent evolution report.

        Returns:
            The last EvolutionReport generated, or None if no cycle has run.
        """
        return self._last_report

    def get_history(self, limit: int = 50) -> list[EvolutionAction]:
        """
        Get recent evolution history.

        Args:
            limit: Maximum number of actions to return. Default 50.

        Returns:
            List of most recent EvolutionAction objects.
        """
        return self._history[-limit:]

    def get_skill_evolution_summary(self, skill_id: str) -> dict[str, Any]:
        """
        Get an evolution summary for a specific skill.

        Args:
            skill_id: The skill to summarize.

        Returns:
            Dictionary with evolution history and current status for the skill.
        """
        skill_actions = [
            a for a in self._history if a.skill_id == skill_id
        ]

        return {
            "skill_id": skill_id,
            "total_actions": len(skill_actions),
            "evolutions": len(
                [a for a in skill_actions if a.action_type.startswith("evolve")]
            ),
            "successful_evolutions": len(
                [
                    a
                    for a in skill_actions
                    if a.action_type == "evolve_success"
                ]
            ),
            "last_action": skill_actions[-1].to_dict() if skill_actions else None,
            "health": self.check_skill_health(skill_id).value,
        }
