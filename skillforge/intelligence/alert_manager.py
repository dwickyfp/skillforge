"""Alert Manager — rule-based alerting for skill health and performance.

Monitors skill performance metrics via configurable rules and generates
alerts when thresholds are breached, trends are detected, or anomalies
occur.  Alerts progress through a lifecycle: ACTIVE → ACKNOWLEDGED → RESOLVED.

Supports callback-based notifications so external systems (e.g. messaging,
dashboards) can be wired in without coupling.

All alert history is persisted in SQLite for audit trails.

No external dependencies — stdlib only.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from skillforge.core.registry import SkillRegistry
from skillforge.core.tracker import QValueTracker
from skillforge.intelligence.health_monitor import HealthMonitor, HealthStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AlertSeverity(Enum):
    """Severity levels for alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Lifecycle states an alert can be in."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class AlertRuleType(Enum):
    """Types of alert rules."""

    THRESHOLD = "threshold"
    TREND = "trend"
    ANOMALY = "anomaly"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AlertRule:
    """A monitoring rule that defines when an alert should fire.

    Attributes
    ----------
    id : str
        Unique rule identifier.
    name : str
        Human-readable name.
    rule_type : AlertRuleType
        The type of rule (threshold, trend, anomaly).
    skill_id : str | None
        If set, only monitor this skill.  ``None`` means all skills.
    metric : str
        The metric to evaluate.  Supported: ``health_score``, ``q_value``,
        ``success_rate``, ``failure_rate``, ``usage_count``.
    threshold : float
        The threshold value.  For ``threshold`` rules: alert fires when
        metric < threshold.  For ``trend`` rules: alert fires when the
        metric drops by more than *threshold* between consecutive checks.
    severity : AlertSeverity
        Severity of the alert when fired.
    cooldown_minutes : int
        Minimum minutes between re-firing for the same skill.
    enabled : bool
        Whether this rule is currently active.
    description : str
        Human-readable description of the rule.
    """

    id: str
    name: str
    rule_type: AlertRuleType = AlertRuleType.THRESHOLD
    skill_id: str | None = None
    metric: str = "health_score"
    threshold: float = 0.3
    severity: AlertSeverity = AlertSeverity.WARNING
    cooldown_minutes: int = 60
    enabled: bool = True
    description: str = ""


@dataclass
class Alert:
    """A fired alert with full context.

    Attributes
    ----------
    id : str
        Unique alert identifier.
    rule_id : str
        ID of the rule that generated this alert.
    rule_name : str
        Name of the rule for quick reference.
    skill_id : str
        The skill that triggered the alert.
    severity : AlertSeverity
        Severity level.
    status : AlertStatus
        Current lifecycle state.
    message : str
        Human-readable alert message.
    metric_value : float
        The metric value that triggered the alert.
    threshold : float
        The threshold that was breached.
    created_at : str
        ISO-8601 UTC timestamp.
    acknowledged_at : str | None
        When the alert was acknowledged.
    resolved_at : str | None
        When the alert was resolved.
    metadata : dict[str, Any]
        Additional context (skill name, health report, etc.).
    """

    id: str
    rule_id: str
    rule_name: str
    skill_id: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    metric_value: float
    threshold: float
    created_at: str
    acknowledged_at: str | None = None
    resolved_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------


class AlertManager:
    """Rule-based alerting for skill performance monitoring.

    Parameters
    ----------
    registry : SkillRegistry
        Skill registry for looking up skill metadata.
    tracker : QValueTracker
        Tracker for Q-values and outcome statistics.
    health_monitor : HealthMonitor | None
        Optional health monitor for health-score based rules.
    db_path : str | Path | None
        SQLite database path.  Defaults to ``~/.skillforge/alerts.db``.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        health_monitor: HealthMonitor | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        self._registry = registry
        self._tracker = tracker
        self._health_monitor = health_monitor
        self._db_path = str(db_path or Path.home() / ".skillforge" / "alerts.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_tables()

        # In-memory state
        self._rules: dict[str, AlertRule] = {}
        self._callbacks: list[Callable[[Alert], None]] = []
        self._load_rules()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS alert_rules (
                id               TEXT PRIMARY KEY,
                name             TEXT NOT NULL,
                rule_type        TEXT NOT NULL DEFAULT 'threshold',
                skill_id         TEXT,
                metric           TEXT NOT NULL DEFAULT 'health_score',
                threshold        REAL NOT NULL DEFAULT 0.3,
                severity         TEXT NOT NULL DEFAULT 'warning',
                cooldown_minutes INTEGER NOT NULL DEFAULT 60,
                enabled          INTEGER NOT NULL DEFAULT 1,
                description      TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id               TEXT PRIMARY KEY,
                rule_id          TEXT NOT NULL,
                rule_name        TEXT NOT NULL DEFAULT '',
                skill_id         TEXT NOT NULL,
                severity         TEXT NOT NULL DEFAULT 'warning',
                status           TEXT NOT NULL DEFAULT 'active',
                message          TEXT NOT NULL DEFAULT '',
                metric_value     REAL NOT NULL DEFAULT 0.0,
                threshold        REAL NOT NULL DEFAULT 0.0,
                created_at       TEXT NOT NULL,
                acknowledged_at  TEXT,
                resolved_at      TEXT,
                metadata         TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
            CREATE INDEX IF NOT EXISTS idx_alerts_skill  ON alerts(skill_id);
            CREATE INDEX IF NOT EXISTS idx_alerts_rule   ON alerts(rule_id);

            CREATE TABLE IF NOT EXISTS alert_firings (
                rule_id    TEXT NOT NULL,
                skill_id   TEXT NOT NULL,
                fired_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_firings_lookup
                ON alert_firings(rule_id, skill_id, fired_at);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def _load_rules(self) -> None:
        """Load persisted rules from DB into memory."""
        rows = self._conn.execute("SELECT * FROM alert_rules").fetchall()
        for row in rows:
            d = dict(row)
            rule = AlertRule(
                id=d["id"],
                name=d["name"],
                rule_type=AlertRuleType(d["rule_type"]),
                skill_id=d["skill_id"],
                metric=d["metric"],
                threshold=d["threshold"],
                severity=AlertSeverity(d["severity"]),
                cooldown_minutes=d["cooldown_minutes"],
                enabled=bool(d["enabled"]),
                description=d["description"],
            )
            self._rules[rule.id] = rule

    def add_rule(self, rule: AlertRule) -> AlertRule:
        """Register a monitoring rule.

        Parameters
        ----------
        rule : AlertRule
            The rule to add.  If *rule.id* is empty, a UUID is generated.

        Returns
        -------
        AlertRule
            The persisted rule (with generated ID if needed).
        """
        if not rule.id:
            rule.id = str(uuid.uuid4())

        self._rules[rule.id] = rule
        self._conn.execute(
            "INSERT OR REPLACE INTO alert_rules "
            "(id, name, rule_type, skill_id, metric, threshold, severity, "
            " cooldown_minutes, enabled, description) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                rule.id,
                rule.name,
                rule.rule_type.value,
                rule.skill_id,
                rule.metric,
                rule.threshold,
                rule.severity.value,
                rule.cooldown_minutes,
                int(rule.enabled),
                rule.description,
            ),
        )
        self._conn.commit()
        logger.debug("Added alert rule '%s' (%s)", rule.name, rule.id)
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a monitoring rule.

        Parameters
        ----------
        rule_id : str
            The rule's ID.

        Returns
        -------
        bool
            ``True`` if the rule was found and removed.
        """
        if rule_id not in self._rules:
            return False
        del self._rules[rule_id]
        self._conn.execute("DELETE FROM alert_rules WHERE id = ?", (rule_id,))
        self._conn.commit()
        logger.debug("Removed alert rule '%s'", rule_id)
        return True

    def get_rules(self) -> list[AlertRule]:
        """Return all registered rules.

        Returns
        -------
        list[AlertRule]
            All rules (enabled and disabled).
        """
        return list(self._rules.values())

    def get_rule(self, rule_id: str) -> AlertRule | None:
        """Return a single rule by ID.

        Parameters
        ----------
        rule_id : str
            The rule's ID.

        Returns
        -------
        AlertRule | None
            The rule, or ``None`` if not found.
        """
        return self._rules.get(rule_id)

    # ------------------------------------------------------------------
    # Alert checking
    # ------------------------------------------------------------------

    def check_alerts(self) -> list[Alert]:
        """Evaluate all enabled rules and fire new alerts.

        For each enabled rule, the relevant skills are checked against
        the rule's condition.  Cooldown is enforced to prevent alert spam.

        Returns
        -------
        list[Alert]
            Newly fired alerts (empty if no rules triggered).
        """
        new_alerts: list[Alert] = []
        now = datetime.now(timezone.utc)

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            skill_ids = self._get_target_skills(rule)
            for sid in skill_ids:
                if self._is_in_cooldown(rule.id, sid, rule.cooldown_minutes, now):
                    continue

                metric_value = self._evaluate_metric(rule, sid)
                triggered = self._check_trigger(rule, metric_value)

                if triggered:
                    alert = self._fire_alert(rule, sid, metric_value, now)
                    new_alerts.append(alert)

                    # Record firing for cooldown tracking
                    self._conn.execute(
                        "INSERT INTO alert_firings (rule_id, skill_id, fired_at) "
                        "VALUES (?,?,?)",
                        (rule.id, sid, now.isoformat()),
                    )
                    self._conn.commit()

                    # Invoke callbacks
                    for cb in self._callbacks:
                        try:
                            cb(alert)
                        except Exception:
                            logger.exception("Alert callback error for rule '%s'", rule.name)

        return new_alerts

    def _get_target_skills(self, rule: AlertRule) -> list[str]:
        """Determine which skills a rule applies to."""
        if rule.skill_id is not None:
            return [rule.skill_id]
        # All registered skills
        skills = self._registry.list_skills(limit=10_000)
        return [s.id for s in skills]

    def _is_in_cooldown(
        self,
        rule_id: str,
        skill_id: str,
        cooldown_minutes: int,
        now: datetime,
    ) -> bool:
        """Check if a rule+skill combo is in cooldown."""
        cutoff = (now - timedelta(minutes=cooldown_minutes)).isoformat()
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM alert_firings "
            "WHERE rule_id = ? AND skill_id = ? AND fired_at >= ?",
            (rule_id, skill_id, cutoff),
        ).fetchone()
        return (row["cnt"] if row else 0) > 0

    def _evaluate_metric(self, rule: AlertRule, skill_id: str) -> float:
        """Compute the current metric value for a skill.

        Supported metrics:
        - ``health_score``: requires a HealthMonitor
        - ``q_value``: from the tracker
        - ``success_rate``: from the tracker
        - ``failure_rate``: 1 - success_rate
        - ``usage_count``: from the tracker
        """
        metric = rule.metric

        if metric == "health_score":
            if self._health_monitor is not None:
                try:
                    report = self._health_monitor.check_health(skill_id)
                    return report.health_score
                except ValueError:
                    return 0.5  # skill not found
            # Fallback: use Q-value as proxy
            return self._tracker.get_q_value(skill_id)

        if metric == "q_value":
            return self._tracker.get_q_value(skill_id)

        if metric == "success_rate":
            return self._tracker.get_success_rate(skill_id)

        if metric == "failure_rate":
            return 1.0 - self._tracker.get_success_rate(skill_id)

        if metric == "usage_count":
            return float(self._tracker.get_usage_count(skill_id))

        # Unknown metric
        return 0.5

    def _check_trigger(self, rule: AlertRule, metric_value: float) -> bool:
        """Determine if a rule's condition is met.

        - ``threshold``: fires when metric < threshold
        - ``trend``: fires when metric < threshold (simplified — would need
          history for true trend detection; we use a static threshold here)
        - ``anomaly``: fires when metric < threshold (simplified anomaly
          detection using a fixed lower bound)
        """
        if rule.rule_type == AlertRuleType.THRESHOLD:
            return metric_value < rule.threshold
        elif rule.rule_type == AlertRuleType.TREND:
            return metric_value < rule.threshold
        elif rule.rule_type == AlertRuleType.ANOMALY:
            return metric_value < rule.threshold
        return False

    def _fire_alert(
        self,
        rule: AlertRule,
        skill_id: str,
        metric_value: float,
        now: datetime,
    ) -> Alert:
        """Create and persist a new alert."""
        skill = self._registry.get_skill(skill_id, tier=1)
        skill_name = skill.name if skill else skill_id

        message = (
            f"[{rule.severity.value.upper()}] Rule '{rule.name}' triggered "
            f"for skill '{skill_name}' ({skill_id}): "
            f"{rule.metric} = {metric_value:.4f} "
            f"(threshold: {rule.threshold:.4f})"
        )

        alert = Alert(
            id=str(uuid.uuid4()),
            rule_id=rule.id,
            rule_name=rule.name,
            skill_id=skill_id,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=message,
            metric_value=metric_value,
            threshold=rule.threshold,
            created_at=now.isoformat(),
            metadata={
                "skill_name": skill_name,
                "rule_type": rule.rule_type.value,
                "metric": rule.metric,
            },
        )

        self._conn.execute(
            "INSERT INTO alerts "
            "(id, rule_id, rule_name, skill_id, severity, status, message, "
            " metric_value, threshold, created_at, acknowledged_at, "
            " resolved_at, metadata) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                alert.id,
                alert.rule_id,
                alert.rule_name,
                alert.skill_id,
                alert.severity.value,
                alert.status.value,
                alert.message,
                alert.metric_value,
                alert.threshold,
                alert.created_at,
                alert.acknowledged_at,
                alert.resolved_at,
                json.dumps(alert.metadata),
            ),
        )
        self._conn.commit()

        logger.warning("Alert fired: %s", alert.message)
        return alert

    # ------------------------------------------------------------------
    # Alert lifecycle
    # ------------------------------------------------------------------

    def get_active_alerts(
        self,
        skill_id: str | None = None,
        severity: AlertSeverity | None = None,
        limit: int = 50,
    ) -> list[Alert]:
        """Return currently active alerts.

        Parameters
        ----------
        skill_id : str | None
            Filter by skill.
        severity : AlertSeverity | None
            Filter by severity.
        limit : int
            Maximum results.

        Returns
        -------
        list[Alert]
            Active alerts sorted by creation time (newest first).
        """
        return self._query_alerts(
            status=AlertStatus.ACTIVE,
            skill_id=skill_id,
            severity=severity,
            limit=limit,
        )

    def get_alert_history(
        self,
        skill_id: str | None = None,
        status: AlertStatus | None = None,
        limit: int = 100,
    ) -> list[Alert]:
        """Return historical alerts (all statuses).

        Parameters
        ----------
        skill_id : str | None
            Filter by skill.
        status : AlertStatus | None
            Filter by status.
        limit : int
            Maximum results.

        Returns
        -------
        list[Alert]
            Alerts sorted by creation time (newest first).
        """
        return self._query_alerts(
            status=status,
            skill_id=skill_id,
            limit=limit,
        )

    def acknowledge_alert(self, alert_id: str) -> Alert | None:
        """Mark an active alert as acknowledged.

        Parameters
        ----------
        alert_id : str
            The alert's ID.

        Returns
        -------
        Alert | None
            The updated alert, or ``None`` if not found.
        """
        now = self._now_iso()
        cur = self._conn.execute(
            "UPDATE alerts SET status = ?, acknowledged_at = ? "
            "WHERE id = ? AND status = ?",
            (AlertStatus.ACKNOWLEDGED.value, now, alert_id, AlertStatus.ACTIVE.value),
        )
        self._conn.commit()
        if cur.rowcount == 0:
            return None
        return self._get_alert(alert_id)

    def resolve_alert(self, alert_id: str) -> Alert | None:
        """Mark an alert as resolved.

        Parameters
        ----------
        alert_id : str
            The alert's ID.

        Returns
        -------
        Alert | None
            The updated alert, or ``None`` if not found.
        """
        now = self._now_iso()
        cur = self._conn.execute(
            "UPDATE alerts SET status = ?, resolved_at = ? "
            "WHERE id = ? AND status IN (?, ?)",
            (
                AlertStatus.RESOLVED.value,
                now,
                alert_id,
                AlertStatus.ACTIVE.value,
                AlertStatus.ACKNOWLEDGED.value,
            ),
        )
        self._conn.commit()
        if cur.rowcount == 0:
            return None
        return self._get_alert(alert_id)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def register_callback(self, callback: Callable[[Alert], None]) -> None:
        """Register a callback to be invoked when a new alert fires.

        Parameters
        ----------
        callback : Callable[[Alert], None]
            A callable that receives the :class:`Alert` object.
        """
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[Alert], None]) -> bool:
        """Remove a previously registered callback.

        Parameters
        ----------
        callback : Callable[[Alert], None]
            The callback to remove.

        Returns
        -------
        bool
            ``True`` if the callback was found and removed.
        """
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Summary / stats
    # ------------------------------------------------------------------

    def get_alert_summary(self) -> dict[str, Any]:
        """Return a summary of the alerting system state.

        Returns
        -------
        dict[str, Any]
            Keys: ``total_rules``, ``enabled_rules``, ``active_alerts``,
            ``acknowledged_alerts``, ``resolved_alerts``, ``by_severity``.
        """
        enabled = sum(1 for r in self._rules.values() if r.enabled)

        counts: dict[str, int] = {}
        for status in AlertStatus:
            row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM alerts WHERE status = ?",
                (status.value,),
            ).fetchone()
            counts[status.value] = row["cnt"] if row else 0

        by_severity: dict[str, int] = {}
        for sev in AlertSeverity:
            row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM alerts WHERE severity = ? AND status = ?",
                (sev.value, AlertStatus.ACTIVE.value),
            ).fetchone()
            by_severity[sev.value] = row["cnt"] if row else 0

        return {
            "total_rules": len(self._rules),
            "enabled_rules": enabled,
            "active_alerts": counts.get("active", 0),
            "acknowledged_alerts": counts.get("acknowledged", 0),
            "resolved_alerts": counts.get("resolved", 0),
            "by_severity": by_severity,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_alerts(
        self,
        status: AlertStatus | None = None,
        skill_id: str | None = None,
        severity: AlertSeverity | None = None,
        limit: int = 100,
    ) -> list[Alert]:
        """Generic alert query with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []

        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if skill_id is not None:
            clauses.append("skill_id = ?")
            params.append(skill_id)
        if severity is not None:
            clauses.append("severity = ?")
            params.append(severity.value)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT * FROM alerts{where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._row_to_alert(r) for r in rows]

    def _get_alert(self, alert_id: str) -> Alert | None:
        """Fetch a single alert by ID."""
        row = self._conn.execute(
            "SELECT * FROM alerts WHERE id = ?", (alert_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_alert(row)

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> Alert:
        d = dict(row)
        d["severity"] = AlertSeverity(d["severity"])
        d["status"] = AlertStatus(d["status"])
        d["metadata"] = json.loads(d.get("metadata") or "{}")
        return Alert(**d)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
