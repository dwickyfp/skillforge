"""Tests for Phase 5a modules: ElasticMemory and AlertManager.

Uses pytest with tmp_path for DB isolation.  Tests cover:
- Memory CRUD, recall, consolidation, compaction
- Alert rule management, firing, lifecycle, callbacks, summaries
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from skillforge.core.graph import SkillDependencyGraph
from skillforge.core.registry import Skill, SkillLifecycle, SkillRegistry
from skillforge.core.tracker import Outcome, QValueTracker
from skillforge.intelligence.health_monitor import HealthMonitor
from skillforge.advanced.elastic_memory import ElasticMemory, MemoryEntry
from skillforge.intelligence.alert_manager import (
    Alert,
    AlertManager,
    AlertRule,
    AlertRuleType,
    AlertSeverity,
    AlertStatus,
)


# -----------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def db_dir(tmp_path: Path) -> Path:
    """Create a temp directory for DB files."""
    d = tmp_path / "db"
    d.mkdir()
    return d


@pytest.fixture
def registry(db_dir: Path) -> SkillRegistry:
    """Fresh registry backed by a temp DB."""
    reg = SkillRegistry(db_path=db_dir / "skills.db")
    yield reg
    reg.close()


@pytest.fixture
def tracker(db_dir: Path) -> QValueTracker:
    """Fresh tracker backed by a temp DB."""
    tr = QValueTracker(db_path=db_dir / "tracker.db")
    yield tr
    tr.close()


@pytest.fixture
def graph() -> SkillDependencyGraph:
    """Empty dependency graph."""
    return SkillDependencyGraph()


@pytest.fixture
def memory(db_dir: Path) -> ElasticMemory:
    """Fresh ElasticMemory backed by a temp DB."""
    mem = ElasticMemory(db_path=db_dir / "memory.db", max_entries=100)
    yield mem
    mem.close()


@pytest.fixture
def health_monitor(
    registry: SkillRegistry, tracker: QValueTracker, graph: SkillDependencyGraph
) -> HealthMonitor:
    """Health monitor using the test registry/tracker."""
    return HealthMonitor(registry, tracker, graph)


@pytest.fixture
def alert_manager(
    registry: SkillRegistry,
    tracker: QValueTracker,
    health_monitor: HealthMonitor,
    db_dir: Path,
) -> AlertManager:
    """Fresh AlertManager backed by a temp DB."""
    am = AlertManager(
        registry, tracker, health_monitor,
        db_path=str(db_dir / "alerts.db"),
    )
    yield am
    am.close()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _register_sample_skills(registry: SkillRegistry) -> list[Skill]:
    """Create sample skills for testing."""
    s1 = registry.register_skill(
        name="web-scraper",
        tier1_metadata="Scrapes web content",
        tier2_core="1. Fetch URL\n2. Parse HTML\n3. Extract data",
        tags=["web", "scraping"],
        skill_id="skill-1",
    )
    s2 = registry.register_skill(
        name="api-fetcher",
        tier1_metadata="Fetches API data",
        tier2_core="1. Authenticate\n2. Send request\n3. Parse JSON",
        tags=["api", "data"],
        skill_id="skill-2",
    )
    registry.update_skill("skill-1", {"lifecycle": SkillLifecycle.ACTIVE})
    registry.update_skill("skill-2", {"lifecycle": SkillLifecycle.ACTIVE})
    return [s1, s2]


def _record_outcomes(
    tracker: QValueTracker,
    skill_id: str,
    count: int = 10,
    success_rate: float = 0.7,
) -> None:
    """Record a series of outcomes for testing."""
    successes = int(count * success_rate)
    for i in range(count):
        success = i < successes
        tracker.record_outcome(Outcome(
            skill_id=skill_id,
            success=success,
            latency_ms=100.0 + i * 10,
            tokens_used=500 + i * 50,
        ))


# =======================================================================
# TestElasticMemory
# =======================================================================


class TestElasticMemory:
    """Tests for the ElasticMemory module."""

    def test_remember_and_recall(
        self, memory: ElasticMemory
    ) -> None:
        """Remembering a memory and recalling by skill_id should return it."""
        entry = memory.remember(
            skill_id="skill-1",
            context="Scraped example.com for product data",
            outcome="Successfully extracted 50 products",
            importance=0.8,
            metadata={"tokens": 1200, "latency_ms": 340.5},
        )

        assert isinstance(entry, MemoryEntry)
        assert entry.skill_id == "skill-1"
        assert entry.importance == 0.8
        assert entry.metadata["tokens"] == 1200

        results = memory.recall(skill_id="skill-1")
        assert len(results) == 1
        assert results[0].id == entry.id
        # Access count should have been incremented
        assert results[0].access_count == 1

    def test_recall_with_query(
        self, memory: ElasticMemory
    ) -> None:
        """Recall with keywords should rank matching memories higher."""
        memory.remember(
            skill_id="skill-1",
            context="Scraped weather data from API",
            outcome="Got temperature and humidity readings",
            importance=0.5,
        )
        memory.remember(
            skill_id="skill-1",
            context="Parsed email headers",
            outcome="Extracted sender and subject lines",
            importance=0.5,
        )
        memory.remember(
            skill_id="skill-1",
            context="Scraped product prices from e-commerce site",
            outcome="Extracted 100 product entries",
            importance=0.5,
        )

        # Query for "scraped" — should match 1st and 3rd memories
        results = memory.recall(skill_id="skill-1", query="scraped products")
        assert len(results) >= 1
        # The most relevant (product scraping) should be first
        assert "product" in results[0].context.lower() or "scraped" in results[0].context.lower()

    def test_recall_all_skills(
        self, memory: ElasticMemory
    ) -> None:
        """Recall without skill_id should search across all skills."""
        memory.remember(skill_id="a", context="alpha task", outcome="ok", importance=0.3)
        memory.remember(skill_id="b", context="beta task", outcome="ok", importance=0.7)
        memory.remember(skill_id="c", context="gamma task", outcome="ok", importance=0.5)

        results = memory.recall(limit=10)
        assert len(results) == 3
        # Default ordering: importance descending
        assert results[0].skill_id == "b"

    def test_recall_min_importance(
        self, memory: ElasticMemory
    ) -> None:
        """Recall with min_importance should filter out low-importance entries."""
        memory.remember(skill_id="x", context="c", outcome="o", importance=0.1)
        memory.remember(skill_id="x", context="c", outcome="o", importance=0.9)

        results = memory.recall(min_importance=0.5)
        assert all(r.importance >= 0.5 for r in results)

    def test_get_memory(self, memory: ElasticMemory) -> None:
        """get_memory should return a single entry by ID."""
        entry = memory.remember(skill_id="s", context="c", outcome="o")
        fetched = memory.get_memory(entry.id)
        assert fetched is not None
        assert fetched.id == entry.id

        assert memory.get_memory("nonexistent") is None

    def test_forget_by_skill(
        self, memory: ElasticMemory
    ) -> None:
        """Forget should remove memories matching criteria."""
        memory.remember(skill_id="a", context="c", outcome="o", importance=0.2)
        memory.remember(skill_id="a", context="c", outcome="o", importance=0.9)
        memory.remember(skill_id="b", context="c", outcome="o", importance=0.2)

        # Forget low-importance memories for skill "a" only
        deleted = memory.forget(skill_id="a", max_importance=0.5)
        assert deleted == 1
        assert memory.count(skill_id="a") == 1

    def test_forget_by_age(
        self, memory: ElasticMemory
    ) -> None:
        """Forget with older_than_days should remove old memories."""
        entry = memory.remember(skill_id="s", context="c", outcome="o", importance=0.5)
        # Manually backdate the entry
        old_time = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        memory._conn.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?",
            (old_time, entry.id),
        )
        memory._conn.commit()

        deleted = memory.forget(older_than_days=30, max_importance=0.5)
        assert deleted == 1
        assert memory.count() == 0

    def test_forget_returns_zero_when_no_match(
        self, memory: ElasticMemory
    ) -> None:
        """Forget should return 0 when nothing matches."""
        memory.remember(skill_id="s", context="c", outcome="o", importance=0.9)
        deleted = memory.forget(max_importance=0.1)
        assert deleted == 0

    def test_consolidate_merges_similar(
        self, memory: ElasticMemory
    ) -> None:
        """Consolidation should merge memories with high keyword overlap."""
        memory.remember(
            skill_id="s",
            context="Fetch weather data from API endpoint",
            outcome="Got temperature humidity data successfully",
            importance=0.3,
        )
        memory.remember(
            skill_id="s",
            context="Fetch weather data from API endpoint",
            outcome="Got temperature humidity readings successfully",
            importance=0.7,
        )
        memory.remember(
            skill_id="s",
            context="Parse email headers for spam detection",
            outcome="Identified 5 spam emails in inbox",
            importance=0.5,
        )

        removed = memory.consolidate("s")
        assert removed >= 1
        remaining = memory.count(skill_id="s")
        assert remaining < 3

    def test_consolidate_no_similar(
        self, memory: ElasticMemory
    ) -> None:
        """Consolidation should return 0 when memories are distinct."""
        memory.remember(skill_id="s", context="alpha beta gamma", outcome="ok")
        memory.remember(skill_id="s", context="delta epsilon zeta", outcome="ok")

        removed = memory.consolidate("s")
        assert removed == 0

    def test_consolidate_single_entry(
        self, memory: ElasticMemory
    ) -> None:
        """Consolidation with a single memory should return 0."""
        memory.remember(skill_id="s", context="only one", outcome="ok")
        assert memory.consolidate("s") == 0

    def test_auto_compact_evicts_low_score(
        self, db_dir: Path
    ) -> None:
        """auto_compact should evict lowest-retention entries."""
        mem = ElasticMemory(db_path=db_dir / "compact.db", max_entries=5)

        for i in range(8):
            mem.remember(
                skill_id=f"skill-{i}",
                context=f"task number {i}",
                outcome=f"outcome {i}",
                importance=i / 10.0,
            )

        evicted = mem.auto_compact()
        assert evicted == 3
        assert mem.count() == 5

        mem.close()

    def test_auto_compact_no_eviction(
        self, memory: ElasticMemory
    ) -> None:
        """auto_compact should return 0 when under cap."""
        memory.remember(skill_id="s", context="c", outcome="o")
        evicted = memory.auto_compact(max_entries=1000)
        assert evicted == 0

    def test_get_memory_stats_global(
        self, memory: ElasticMemory
    ) -> None:
        """get_memory_stats should return global stats without skill_id."""
        memory.remember(skill_id="a", context="c", outcome="o", importance=0.6)
        memory.remember(skill_id="b", context="c", outcome="o", importance=0.8)

        stats = memory.get_memory_stats()
        assert stats["total_memories"] == 2
        assert stats["avg_importance"] == pytest.approx(0.7, abs=0.01)
        assert len(stats["skills"]) == 2

    def test_get_memory_stats_per_skill(
        self, memory: ElasticMemory
    ) -> None:
        """get_memory_stats with skill_id should scope to that skill."""
        memory.remember(skill_id="a", context="c", outcome="o", importance=0.5)
        memory.remember(skill_id="b", context="c", outcome="o", importance=0.9)

        stats = memory.get_memory_stats(skill_id="a")
        assert stats["total_memories"] == 1
        assert stats["skills"] == ["a"]

    def test_count(self, memory: ElasticMemory) -> None:
        """Count should return accurate totals."""
        assert memory.count() == 0
        memory.remember(skill_id="s", context="c", outcome="o")
        assert memory.count() == 1
        assert memory.count(skill_id="s") == 1
        assert memory.count(skill_id="other") == 0

    def test_importance_clamped(
        self, memory: ElasticMemory
    ) -> None:
        """Importance should be clamped to [0, 1]."""
        entry = memory.remember(skill_id="s", context="c", outcome="o", importance=1.5)
        assert entry.importance == 1.0

        entry = memory.remember(skill_id="s", context="c", outcome="o", importance=-0.5)
        assert entry.importance == 0.0


# =======================================================================
# TestAlertManager
# =======================================================================


class TestAlertManager:
    """Tests for the AlertManager module."""

    def test_add_and_get_rules(
        self, alert_manager: AlertManager
    ) -> None:
        """Adding a rule should make it retrievable."""
        rule = AlertRule(
            id="rule-1",
            name="Low Q-value",
            rule_type=AlertRuleType.THRESHOLD,
            metric="q_value",
            threshold=0.3,
            severity=AlertSeverity.WARNING,
            description="Alert when Q-value drops below 0.3",
        )
        alert_manager.add_rule(rule)

        rules = alert_manager.get_rules()
        assert len(rules) == 1
        assert rules[0].name == "Low Q-value"

        fetched = alert_manager.get_rule("rule-1")
        assert fetched is not None
        assert fetched.threshold == 0.3

    def test_add_rule_generates_id(
        self, alert_manager: AlertManager
    ) -> None:
        """add_rule should generate an ID if one is empty."""
        rule = AlertRule(
            id="",
            name="Auto-ID rule",
            metric="success_rate",
            threshold=0.5,
        )
        result = alert_manager.add_rule(rule)
        assert result.id != ""

    def test_remove_rule(
        self, alert_manager: AlertManager
    ) -> None:
        """Removing a rule should return True and remove it from memory."""
        rule = AlertRule(id="r1", name="test", metric="q_value", threshold=0.3)
        alert_manager.add_rule(rule)

        assert alert_manager.remove_rule("r1") is True
        assert len(alert_manager.get_rules()) == 0

    def test_remove_nonexistent_rule(
        self, alert_manager: AlertManager
    ) -> None:
        """Removing a nonexistent rule should return False."""
        assert alert_manager.remove_rule("nonexistent") is False

    def test_check_alerts_fires_on_low_q(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        """A threshold rule should fire when Q-value is below threshold."""
        _register_sample_skills(registry)

        # Record failures to lower Q-value
        for _ in range(10):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=False, latency_ms=500, tokens_used=200
            ))
        tracker.td_lambda_update("skill-1", reward=0.0, alpha=0.5)

        # Add a rule: Q-value < 0.4
        alert_manager.add_rule(AlertRule(
            id="q-rule",
            name="Low Q-value alert",
            rule_type=AlertRuleType.THRESHOLD,
            metric="q_value",
            threshold=0.4,
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=0,
        ))

        new_alerts = alert_manager.check_alerts()

        # At least one alert should have fired for skill-1
        skill_alerts = [a for a in new_alerts if a.skill_id == "skill-1"]
        assert len(skill_alerts) >= 1
        assert skill_alerts[0].severity == AlertSeverity.CRITICAL
        assert skill_alerts[0].status == AlertStatus.ACTIVE

    def test_check_alerts_no_fire_above_threshold(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        """No alert should fire when metric is above threshold."""
        _register_sample_skills(registry)

        # Record successes to boost Q-value
        for _ in range(10):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=True, latency_ms=100, tokens_used=50
            ))
        tracker.td_lambda_update("skill-1", reward=1.0, alpha=0.5)

        alert_manager.add_rule(AlertRule(
            id="q-rule",
            name="Low Q-value",
            metric="q_value",
            threshold=0.1,  # Very low threshold — won't fire
            cooldown_minutes=0,
        ))

        new_alerts = alert_manager.check_alerts()
        assert len(new_alerts) == 0

    def test_check_alerts_respects_cooldown(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        """Alerts should not re-fire within the cooldown period."""
        _register_sample_skills(registry)

        for _ in range(10):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=False, latency_ms=500, tokens_used=200
            ))
        tracker.td_lambda_update("skill-1", reward=0.0, alpha=0.5)

        alert_manager.add_rule(AlertRule(
            id="cd-rule",
            name="Cooldown test",
            metric="q_value",
            threshold=0.9,  # High threshold so it always fires
            cooldown_minutes=60,
        ))

        # First check should fire
        first = alert_manager.check_alerts()
        assert len([a for a in first if a.skill_id == "skill-1"]) >= 1

        # Second check immediately — should be in cooldown
        second = alert_manager.check_alerts()
        assert len([a for a in second if a.skill_id == "skill-1"]) == 0

    def test_check_alerts_disabled_rule(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
    ) -> None:
        """Disabled rules should not fire."""
        _register_sample_skills(registry)

        alert_manager.add_rule(AlertRule(
            id="disabled",
            name="Disabled rule",
            metric="q_value",
            threshold=0.99,
            enabled=False,
            cooldown_minutes=0,
        ))

        alerts = alert_manager.check_alerts()
        assert len(alerts) == 0

    def test_check_alerts_specific_skill(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        """A rule with skill_id set should only monitor that skill."""
        _register_sample_skills(registry)

        # Lower Q-value for skill-1 only
        for _ in range(10):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=False, latency_ms=500, tokens_used=200
            ))
        tracker.td_lambda_update("skill-1", reward=0.0, alpha=0.5)

        alert_manager.add_rule(AlertRule(
            id="specific",
            name="Skill-1 only",
            skill_id="skill-1",
            metric="q_value",
            threshold=0.4,
            cooldown_minutes=0,
        ))

        alerts = alert_manager.check_alerts()
        skill_alerts = [a for a in alerts if a.skill_id == "skill-1"]
        assert len(skill_alerts) >= 1
        # Should NOT fire for skill-2
        s2_alerts = [a for a in alerts if a.skill_id == "skill-2"]
        assert len(s2_alerts) == 0

    def test_alert_lifecycle(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        """Alerts should progress through ACTIVE → ACKNOWLEDGED → RESOLVED."""
        _register_sample_skills(registry)

        for _ in range(10):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=False, latency_ms=500, tokens_used=200
            ))
        tracker.td_lambda_update("skill-1", reward=0.0, alpha=0.5)

        alert_manager.add_rule(AlertRule(
            id="lifecycle",
            name="Lifecycle test",
            metric="q_value",
            threshold=0.4,
            cooldown_minutes=0,
        ))

        alerts = alert_manager.check_alerts()
        assert len(alerts) >= 1
        alert_id = alerts[0].id

        # Acknowledge
        acked = alert_manager.acknowledge_alert(alert_id)
        assert acked is not None
        assert acked.status == AlertStatus.ACKNOWLEDGED
        assert acked.acknowledged_at is not None

        # Resolve
        resolved = alert_manager.resolve_alert(alert_id)
        assert resolved is not None
        assert resolved.status == AlertStatus.RESOLVED
        assert resolved.resolved_at is not None

    def test_acknowledge_nonexistent(
        self, alert_manager: AlertManager
    ) -> None:
        """Acknowledging a nonexistent alert should return None."""
        assert alert_manager.acknowledge_alert("nope") is None

    def test_resolve_nonexistent(
        self, alert_manager: AlertManager
    ) -> None:
        """Resolving a nonexistent alert should return None."""
        assert alert_manager.resolve_alert("nope") is None

    def test_get_active_alerts(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        """get_active_alerts should return only active alerts."""
        _register_sample_skills(registry)

        for _ in range(10):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=False, latency_ms=500, tokens_used=200
            ))
        tracker.td_lambda_update("skill-1", reward=0.0, alpha=0.5)

        alert_manager.add_rule(AlertRule(
            id="active-test",
            name="Active test",
            metric="q_value",
            threshold=0.4,
            cooldown_minutes=0,
        ))

        alerts = alert_manager.check_alerts()
        assert len(alerts) >= 1

        active = alert_manager.get_active_alerts()
        assert len(active) >= 1
        assert all(a.status == AlertStatus.ACTIVE for a in active)

        # Resolve one and check it disappears from active
        alert_manager.resolve_alert(active[0].id)
        active_after = alert_manager.get_active_alerts()
        assert len(active_after) == len(active) - 1

    def test_get_alert_history(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        """get_alert_history should return all alerts regardless of status."""
        _register_sample_skills(registry)

        for _ in range(10):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=False, latency_ms=500, tokens_used=200
            ))
        tracker.td_lambda_update("skill-1", reward=0.0, alpha=0.5)

        alert_manager.add_rule(AlertRule(
            id="history-test",
            name="History test",
            metric="q_value",
            threshold=0.4,
            cooldown_minutes=0,
        ))

        alerts = alert_manager.check_alerts()
        assert len(alerts) >= 1

        history = alert_manager.get_alert_history()
        assert len(history) >= 1

    def test_callback_invoked(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        """Registered callbacks should be invoked when alerts fire."""
        _register_sample_skills(registry)

        for _ in range(10):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=False, latency_ms=500, tokens_used=200
            ))
        tracker.td_lambda_update("skill-1", reward=0.0, alpha=0.5)

        captured: list[Alert] = []

        def _on_alert(alert: Alert) -> None:
            captured.append(alert)

        alert_manager.register_callback(_on_alert)

        alert_manager.add_rule(AlertRule(
            id="cb-test",
            name="Callback test",
            metric="q_value",
            threshold=0.4,
            cooldown_minutes=0,
        ))

        alert_manager.check_alerts()
        assert len(captured) >= 1
        assert isinstance(captured[0], Alert)

    def test_unregister_callback(
        self, alert_manager: AlertManager
    ) -> None:
        """Unregistering a callback should return True."""
        def _cb(a: Alert) -> None:
            pass

        alert_manager.register_callback(_cb)
        assert alert_manager.unregister_callback(_cb) is True
        assert alert_manager.unregister_callback(_cb) is False

    def test_alert_summary(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        """get_alert_summary should return a complete summary dict."""
        _register_sample_skills(registry)

        for _ in range(10):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=False, latency_ms=500, tokens_used=200
            ))
        tracker.td_lambda_update("skill-1", reward=0.0, alpha=0.5)

        alert_manager.add_rule(AlertRule(
            id="sum-1", name="rule 1", metric="q_value", threshold=0.4, cooldown_minutes=0,
        ))
        alert_manager.add_rule(AlertRule(
            id="sum-2", name="rule 2", metric="success_rate", threshold=0.1,
            enabled=False, cooldown_minutes=0,
        ))

        alert_manager.check_alerts()

        summary = alert_manager.get_alert_summary()
        assert summary["total_rules"] == 2
        assert summary["enabled_rules"] == 1
        assert summary["active_alerts"] >= 0
        assert "by_severity" in summary

    def test_success_rate_rule(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        """A success_rate rule should fire when success rate is low."""
        _register_sample_skills(registry)

        for _ in range(20):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=False, latency_ms=500, tokens_used=200
            ))

        alert_manager.add_rule(AlertRule(
            id="sr-rule",
            name="Low success rate",
            metric="success_rate",
            threshold=0.5,
            cooldown_minutes=0,
        ))

        alerts = alert_manager.check_alerts()
        skill_alerts = [a for a in alerts if a.skill_id == "skill-1"]
        assert len(skill_alerts) >= 1

    def test_failure_rate_rule(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        """A failure_rate rule should fire when failure_rate < threshold (not high failure rate)."""
        _register_sample_skills(registry)

        # All failures → failure_rate = 1.0
        for _ in range(10):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=False, latency_ms=500, tokens_used=200
            ))

        alert_manager.add_rule(AlertRule(
            id="fr-rule",
            name="High failure rate",
            skill_id="skill-1",
            metric="failure_rate",
            threshold=0.9,  # fires when failure_rate < 0.9 → 1.0 is NOT < 0.9
            cooldown_minutes=0,
        ))

        # The threshold rule fires when metric < threshold
        # failure_rate = 1.0, threshold = 0.9 → 1.0 < 0.9 is False → won't fire
        alerts = alert_manager.check_alerts()
        assert len(alerts) == 0

    def test_rule_persistence(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        health_monitor: HealthMonitor,
        db_dir: Path,
    ) -> None:
        """Rules should survive creating a new AlertManager instance."""
        db_path = str(db_dir / "persist.db")

        am1 = AlertManager(registry, tracker, health_monitor, db_path=db_path)
        am1.add_rule(AlertRule(
            id="persist-1",
            name="Persistent rule",
            metric="q_value",
            threshold=0.3,
        ))
        am1.close()

        am2 = AlertManager(registry, tracker, health_monitor, db_path=db_path)
        rules = am2.get_rules()
        assert len(rules) == 1
        assert rules[0].id == "persist-1"
        assert rules[0].threshold == 0.3
        am2.close()

    def test_health_score_rule(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        """A health_score rule should use the HealthMonitor."""
        _register_sample_skills(registry)

        # Record failures to create a critical health state
        for _ in range(10):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=False, latency_ms=500, tokens_used=200
            ))
        tracker.td_lambda_update("skill-1", reward=0.0, alpha=0.5)

        alert_manager.add_rule(AlertRule(
            id="health-rule",
            name="Low health",
            metric="health_score",
            threshold=0.5,
            severity=AlertSeverity.WARNING,
            cooldown_minutes=0,
        ))

        alerts = alert_manager.check_alerts()
        skill_alerts = [a for a in alerts if a.skill_id == "skill-1"]
        assert len(skill_alerts) >= 1

    def test_alert_message_format(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        """Alert messages should contain key information."""
        _register_sample_skills(registry)

        for _ in range(10):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=False, latency_ms=500, tokens_used=200
            ))
        tracker.td_lambda_update("skill-1", reward=0.0, alpha=0.5)

        alert_manager.add_rule(AlertRule(
            id="msg-test",
            name="Message format test",
            metric="q_value",
            threshold=0.4,
            severity=AlertSeverity.WARNING,
            cooldown_minutes=0,
        ))

        alerts = alert_manager.check_alerts()
        assert len(alerts) >= 1
        msg = alerts[0].message
        assert "WARNING" in msg
        assert "Message format test" in msg
        assert "q_value" in msg

    def test_alert_metadata(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        """Alert metadata should contain skill_name and rule info."""
        _register_sample_skills(registry)

        for _ in range(10):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=False, latency_ms=500, tokens_used=200
            ))
        tracker.td_lambda_update("skill-1", reward=0.0, alpha=0.5)

        alert_manager.add_rule(AlertRule(
            id="meta-test",
            name="Metadata test",
            metric="q_value",
            threshold=0.4,
            cooldown_minutes=0,
        ))

        alerts = alert_manager.check_alerts()
        assert len(alerts) >= 1
        meta = alerts[0].metadata
        assert "skill_name" in meta
        assert meta["metric"] == "q_value"
        assert meta["rule_type"] == "threshold"

    def test_trend_rule_type(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        """Trend-type rules should fire when metric < threshold."""
        _register_sample_skills(registry)

        for _ in range(10):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=False, latency_ms=500, tokens_used=200
            ))
        tracker.td_lambda_update("skill-1", reward=0.0, alpha=0.5)

        alert_manager.add_rule(AlertRule(
            id="trend-rule",
            name="Declining trend",
            rule_type=AlertRuleType.TREND,
            metric="q_value",
            threshold=0.5,
            cooldown_minutes=0,
        ))

        alerts = alert_manager.check_alerts()
        skill_alerts = [a for a in alerts if a.skill_id == "skill-1"]
        assert len(skill_alerts) >= 1

    def test_anomaly_rule_type(
        self,
        alert_manager: AlertManager,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        """Anomaly-type rules should fire when metric < threshold."""
        _register_sample_skills(registry)

        for _ in range(10):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=False, latency_ms=500, tokens_used=200
            ))
        tracker.td_lambda_update("skill-1", reward=0.0, alpha=0.5)

        alert_manager.add_rule(AlertRule(
            id="anomaly-rule",
            name="Anomaly detected",
            rule_type=AlertRuleType.ANOMALY,
            metric="q_value",
            threshold=0.5,
            cooldown_minutes=0,
        ))

        alerts = alert_manager.check_alerts()
        skill_alerts = [a for a in alerts if a.skill_id == "skill-1"]
        assert len(skill_alerts) >= 1


# =======================================================================
# Integration tests
# =======================================================================


class TestPhase5aIntegration:
    """Integration tests combining ElasticMemory and AlertManager."""

    def test_memory_and_alerts_together(
        self,
        db_dir: Path,
        registry: SkillRegistry,
        tracker: QValueTracker,
        health_monitor: HealthMonitor,
    ) -> None:
        """ElasticMemory and AlertManager should work side by side."""
        memory = ElasticMemory(db_path=db_dir / "mem.db")
        alert_mgr = AlertManager(
            registry, tracker, health_monitor,
            db_path=str(db_dir / "alert.db"),
        )

        # Register a skill
        _register_sample_skills(registry)

        # Record outcomes that will trigger alerts
        for _ in range(10):
            tracker.record_outcome(Outcome(
                skill_id="skill-1", success=False, latency_ms=500, tokens_used=200
            ))
        tracker.td_lambda_update("skill-1", reward=0.0, alpha=0.5)

        # Store memory of the failure
        mem = memory.remember(
            skill_id="skill-1",
            context="Attempted web scraping of example.com",
            outcome="Failed: connection timeout after 500ms",
            importance=0.6,
            metadata={"latency_ms": 500, "success": False},
        )
        assert mem.skill_id == "skill-1"

        # Set up an alert rule
        alert_mgr.add_rule(AlertRule(
            id="integ-1",
            name="Integration test alert",
            metric="q_value",
            threshold=0.4,
            cooldown_minutes=0,
        ))

        # Check alerts
        alerts = alert_mgr.check_alerts()
        assert len(alerts) >= 1

        # Recall the memory
        memories = memory.recall(skill_id="skill-1")
        assert len(memories) == 1
        assert "timeout" in memories[0].outcome

        # Both systems should have their data
        assert memory.count() == 1
        assert alert_mgr.get_alert_summary()["active_alerts"] >= 1

        memory.close()
        alert_mgr.close()
