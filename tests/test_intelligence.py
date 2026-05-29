"""Tests for SkillForge intelligence modules.

Covers ConflictDetector, HealthMonitor, SkillCreator, SkillAnalyzer,
and SkillOptimizer.  Uses pytest fixtures with tmpdir for DB isolation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from skillforge.core.graph import SkillDependencyGraph
from skillforge.core.registry import Skill, SkillLifecycle, SkillRegistry
from skillforge.core.tracker import Outcome, QValueTracker
from skillforge.intelligence.analyzer import SkillAnalyzer, SkillCluster, UsagePattern
from skillforge.intelligence.conflict_detector import SkillConflict, ConflictDetector
from skillforge.intelligence.health_monitor import HealthMonitor, SkillHealth, HealthStatus
from skillforge.intelligence.optimizer import OptimizationAction, SkillOptimizer
from skillforge.intelligence.skill_creator import (
    SkillCreator,
    Trajectory,
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


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _register_sample_skills(registry: SkillRegistry) -> list[Skill]:
    """Create a set of sample skills for testing."""
    s1 = registry.register_skill(
        name="web-scraper",
        tier1_metadata="Scrapes web content",
        tier2_core="1. Fetch the URL\n2. Parse HTML\n3. Extract data\n4. Store results",
        tags=["web", "scraping", "data"],
        skill_id="skill-1",
    )
    s2 = registry.register_skill(
        name="api-fetcher",
        tier1_metadata="Fetches API data",
        tier2_core="1. Authenticate\n2. Send request\n3. Parse JSON\n4. Store results",
        tags=["api", "data", "networking"],
        skill_id="skill-2",
    )
    s3 = registry.register_skill(
        name="email-sender",
        tier1_metadata="Sends emails",
        tier2_core="1. Connect SMTP\n2. Compose message\n3. Send email\n4. Log result",
        tags=["email", "communication"],
        skill_id="skill-3",
    )
    return [s1, s2, s3]


# =======================================================================
# ConflictDetector tests
# =======================================================================

class TestConflictDetector:
    """Tests for the ConflictDetector module."""

    def test_detect_overlapping_skills(
        self, registry: SkillRegistry, graph: SkillDependencyGraph, db_dir: Path
    ) -> None:
        """Skills with identical tier1_metadata should be detected as overlapping."""
        registry.register_skill(
            name="skill-a",
            tier1_metadata="meta description for overlap test",
            tags=["web", "scraping", "data"],
            skill_id="a",
        )
        registry.register_skill(
            name="skill-b",
            tier1_metadata="meta description for overlap test",
            tags=["web", "scraping", "api"],
            skill_id="b",
        )
        # Skills must be ACTIVE for detect_conflicts to scan them
        registry.update_skill("a", {"lifecycle": SkillLifecycle.ACTIVE})
        registry.update_skill("b", {"lifecycle": SkillLifecycle.ACTIVE})

        detector = ConflictDetector(registry, graph, db_path=str(db_dir / "conflicts.db"))
        conflicts = detector.detect_conflicts()

        overlap_conflicts = [c for c in conflicts if c.conflict_type == "overlap"]
        assert len(overlap_conflicts) == 1
        assert overlap_conflicts[0].severity in ("low", "medium", "high", "critical")
        assert {overlap_conflicts[0].skill_a_id, overlap_conflicts[0].skill_b_id} == {"a", "b"}

    def test_no_overlap_for_distinct_skills(
        self, registry: SkillRegistry, graph: SkillDependencyGraph, db_dir: Path
    ) -> None:
        """Skills with completely different metadata should not overlap."""
        registry.register_skill(
            name="x", tier1_metadata="alpha beta gamma", tags=["a", "b"], skill_id="x"
        )
        registry.register_skill(
            name="y", tier1_metadata="delta epsilon zeta", tags=["c", "d"], skill_id="y"
        )
        registry.update_skill("x", {"lifecycle": SkillLifecycle.ACTIVE})
        registry.update_skill("y", {"lifecycle": SkillLifecycle.ACTIVE})

        detector = ConflictDetector(registry, graph, db_path=str(db_dir / "conflicts.db"))
        conflicts = detector.detect_conflicts()
        overlap_conflicts = [c for c in conflicts if c.conflict_type == "overlap"]
        assert len(overlap_conflicts) == 0

    def test_detect_contradictory_instructions(
        self, registry: SkillRegistry, graph: SkillDependencyGraph, db_dir: Path
    ) -> None:
        """Skills with opposing keywords in tier2_core should be flagged."""
        registry.register_skill(
            name="enable-feature",
            tier1_metadata="Enable things",
            tier2_core="Always enable the feature. Include all modules.",
            tags=["config"],
            skill_id="c1",
        )
        registry.register_skill(
            name="disable-feature",
            tier1_metadata="Disable things",
            tier2_core="Never enable the feature. Exclude optional modules.",
            tags=["config"],
            skill_id="c2",
        )
        registry.update_skill("c1", {"lifecycle": SkillLifecycle.ACTIVE})
        registry.update_skill("c2", {"lifecycle": SkillLifecycle.ACTIVE})

        detector = ConflictDetector(registry, graph, db_path=str(db_dir / "conflicts.db"))
        conflicts = detector.detect_conflicts()
        contradictions = [c for c in conflicts if c.conflict_type == "contradiction"]

        assert len(contradictions) >= 1
        assert contradictions[0].severity in ("low", "medium", "high", "critical")

    def test_no_contradiction_different_domains(
        self, registry: SkillRegistry, graph: SkillDependencyGraph, db_dir: Path
    ) -> None:
        """Skills without contradictory instructions should not be flagged."""
        registry.register_skill(
            name="x",
            tier1_metadata="x metadata about web",
            tier2_core="Always enable everything properly.",
            tags=["domain-a"],
            skill_id="d1",
        )
        registry.register_skill(
            name="y",
            tier1_metadata="y metadata about processing",
            tier2_core="Process the data efficiently.",
            tags=["domain-b"],
            skill_id="d2",
        )
        registry.update_skill("d1", {"lifecycle": SkillLifecycle.ACTIVE})
        registry.update_skill("d2", {"lifecycle": SkillLifecycle.ACTIVE})

        detector = ConflictDetector(registry, graph, db_path=str(db_dir / "conflicts.db"))
        conflicts = detector.detect_conflicts()
        contradictions = [c for c in conflicts if c.conflict_type == "contradiction"]
        assert len(contradictions) == 0

    def test_resolve_by_deprecation(
        self, registry: SkillRegistry, graph: SkillDependencyGraph, db_dir: Path
    ) -> None:
        """Resolution deprecates the lower-quality skill."""
        registry.register_skill(
            name="better", tier1_metadata="meta",
            tags=["x", "y"], skill_id="good"
        )
        registry.register_skill(
            name="worse", tier1_metadata="meta",
            tags=["x", "y"], skill_id="bad"
        )
        registry.update_skill("good", {"lifecycle": SkillLifecycle.ACTIVE, "q_value": 0.9})
        registry.update_skill("bad", {"lifecycle": SkillLifecycle.ACTIVE, "q_value": 0.3})

        detector = ConflictDetector(registry, graph, db_path=str(db_dir / "conflicts.db"))
        conflicts = detector.detect_conflicts()
        assert len(conflicts) >= 1

        conflict = conflicts[0]
        result = detector.resolve_conflict(conflict.conflict_id, "deprecate")

        assert result is not None
        assert result.resolved is True

        # Lower-Q skill should be deprecated
        updated = registry.get_skill("bad")
        assert updated is not None
        assert updated.lifecycle == SkillLifecycle.DEPRECATED

        # Better skill should remain active
        kept = registry.get_skill("good")
        assert kept is not None
        assert kept.lifecycle != SkillLifecycle.DEPRECATED

    def test_resolve_equal_q_value_deprecates_skill_b(
        self, registry: SkillRegistry, graph: SkillDependencyGraph, db_dir: Path
    ) -> None:
        """When Q-values are equal, the code deprecates skill_b."""
        registry.register_skill(
            name="popular", tier1_metadata="meta",
            tags=["t"], skill_id="popular"
        )
        registry.register_skill(
            name="unpopular", tier1_metadata="meta",
            tags=["t"], skill_id="unpopular"
        )
        registry.update_skill("popular", {"lifecycle": SkillLifecycle.ACTIVE, "q_value": 0.5})
        registry.update_skill("unpopular", {"lifecycle": SkillLifecycle.ACTIVE, "q_value": 0.5})

        detector = ConflictDetector(registry, graph, db_path=str(db_dir / "conflicts.db"))
        conflicts = detector.detect_conflicts()
        assert len(conflicts) >= 1

        conflict = conflicts[0]
        result = detector.resolve_conflict(conflict.conflict_id, "deprecate")

        assert result is not None
        assert result.resolved is True

        # Verify exactly one skill was deprecated
        good = registry.get_skill("popular")
        bad = registry.get_skill("unpopular")
        deprecated_count = sum(
            1 for s in [good, bad] if s.lifecycle == SkillLifecycle.DEPRECATED
        )
        assert deprecated_count == 1


# =======================================================================
# HealthMonitor tests
# =======================================================================

class TestHealthMonitor:
    """Tests for the HealthMonitor module."""

    def test_compute_health_score_healthy(
        self, registry: SkillRegistry, tracker: QValueTracker, graph: SkillDependencyGraph
    ) -> None:
        """A high Q-value, high success rate skill should be healthy."""
        registry.register_skill(
            name="healthy-skill", tier1_metadata="good", skill_id="h1"
        )
        # Record successful outcomes in the tracker
        for _ in range(5):
            tracker.record_outcome(Outcome(
                skill_id="h1", success=True, latency_ms=100.0, tokens_used=50
            ))
        # Boost Q-value via TD update
        tracker.td_lambda_update("h1", reward=1.0, alpha=0.5)

        monitor = HealthMonitor(registry, tracker, graph)
        report = monitor.check_health("h1")

        assert isinstance(report, SkillHealth)
        assert report.status == HealthStatus.HEALTHY
        assert report.health_score > 0.7

    def test_compute_health_score_critical(
        self, registry: SkillRegistry, tracker: QValueTracker, graph: SkillDependencyGraph
    ) -> None:
        """A low Q-value skill with failures should be critical."""
        registry.register_skill(
            name="bad-skill", tier1_metadata="bad", skill_id="c1"
        )
        # Record failing outcomes
        for _ in range(3):
            tracker.record_outcome(Outcome(
                skill_id="c1", success=False, latency_ms=500.0, tokens_used=200
            ))
        # Lower Q-value via TD update
        tracker.td_lambda_update("c1", reward=0.0, alpha=0.5)

        monitor = HealthMonitor(registry, tracker, graph)
        report = monitor.check_health("c1")

        assert report.status == HealthStatus.CRITICAL
        assert report.health_score < 0.4
        assert len(report.recommendations) > 0

    def test_compute_health_score_missing_skill(
        self, registry: SkillRegistry, tracker: QValueTracker, graph: SkillDependencyGraph
    ) -> None:
        """A non-existent skill should raise ValueError."""
        monitor = HealthMonitor(registry, tracker, graph)
        with pytest.raises(ValueError, match="not found"):
            monitor.check_health("nonexistent")

    def test_check_all_skills(
        self, registry: SkillRegistry, tracker: QValueTracker, graph: SkillDependencyGraph
    ) -> None:
        """check_all returns a report for every active registered skill."""
        _register_sample_skills(registry)
        # Skills must be ACTIVE for check_all to find them
        for sid in ["skill-1", "skill-2", "skill-3"]:
            registry.update_skill(sid, {"lifecycle": SkillLifecycle.ACTIVE})

        monitor = HealthMonitor(registry, tracker, graph)
        reports = monitor.check_all()

        assert len(reports) == 3
        assert all(isinstance(r, SkillHealth) for r in reports)
        assert {r.skill_id for r in reports} == {"skill-1", "skill-2", "skill-3"}

    def test_dashboard_summary(
        self, registry: SkillRegistry, tracker: QValueTracker, graph: SkillDependencyGraph
    ) -> None:
        """Dashboard summary returns a dict with stats."""
        _register_sample_skills(registry)
        for sid in ["skill-1", "skill-2", "skill-3"]:
            registry.update_skill(sid, {"lifecycle": SkillLifecycle.ACTIVE})

        monitor = HealthMonitor(registry, tracker, graph)
        dashboard = monitor.get_dashboard_summary()

        assert isinstance(dashboard, dict)
        assert dashboard["total_skills"] == 3
        assert "by_status" in dashboard
        assert "avg_health" in dashboard

    def test_dashboard_empty_registry(
        self, registry: SkillRegistry, tracker: QValueTracker, graph: SkillDependencyGraph
    ) -> None:
        """Empty registry produces a dashboard with zero skills."""
        monitor = HealthMonitor(registry, tracker, graph)
        dashboard = monitor.get_dashboard_summary()
        assert dashboard["total_skills"] == 0

    def test_skill_with_no_usage_has_low_health(
        self, registry: SkillRegistry, tracker: QValueTracker, graph: SkillDependencyGraph
    ) -> None:
        """A skill with no usage data should have low health (CRITICAL)."""
        registry.register_skill(
            name="unused", tier1_metadata="unused", skill_id="unused1"
        )

        monitor = HealthMonitor(registry, tracker, graph)
        report = monitor.check_health("unused1")
        # With no outcomes: q=0.5, sr=0.5, recency=inf→0, usage=0→0
        # Health: 0.35*0.5 + 0.30*0.5 = 0.325 → CRITICAL
        assert report.status == HealthStatus.CRITICAL
        assert report.health_score < 0.4


# =======================================================================
# SkillCreator tests
# =======================================================================

class TestSkillCreator:
    """Tests for the SkillCreator module."""

    def test_create_from_trajectories(
        self, registry: SkillRegistry, tracker: QValueTracker
    ) -> None:
        """Should create a skill from successful trajectories."""
        trajectories = [
            Trajectory(
                task_description="x",
                steps=["step 1", "step 2", "step 3"],
                outcome="success",
            ),
            Trajectory(
                task_description="x",
                steps=["step 1", "step 2", "step 4"],
                outcome="success",
            ),
            Trajectory(
                task_description="x",
                steps=["step 1", "step 2"],
                outcome="success",
            ),
        ]

        creator = SkillCreator(registry, tracker)
        skill_id = creator.create_from_trajectory("test-skill", trajectories)

        assert isinstance(skill_id, str)
        assert skill_id

        # Verify skill was registered
        skill = registry.get_skill(skill_id, tier=2)
        assert skill is not None
        assert skill.name == "test-skill"
        # Common steps should appear in tier2_core
        assert "step 1" in skill.tier2_core
        assert "step 2" in skill.tier2_core

    def test_create_from_no_success_raises(
        self, registry: SkillRegistry, tracker: QValueTracker
    ) -> None:
        """Should raise if fewer than 2 successful trajectories."""
        trajectories = [
            Trajectory(
                task_description="x",
                steps=["fail"],
                outcome="failure",
            ),
        ]

        creator = SkillCreator(registry, tracker)
        with pytest.raises(ValueError, match="successful trajectories"):
            creator.create_from_trajectory("fail-skill", trajectories)

    def test_extract_common_steps(
        self, registry: SkillRegistry, tracker: QValueTracker
    ) -> None:
        """Should extract steps appearing in majority of trajectories."""
        trajectories = [
            Trajectory(
                task_description="a",
                steps=["init", "fetch", "parse", "store"],
                outcome="success",
            ),
            Trajectory(
                task_description="a",
                steps=["init", "fetch", "validate", "store"],
                outcome="success",
            ),
            Trajectory(
                task_description="a",
                steps=["init", "fetch", "store"],
                outcome="success",
            ),
        ]

        creator = SkillCreator(registry, tracker)
        common = creator._extract_common_steps(trajectories)

        assert "init" in common
        assert "fetch" in common
        assert "store" in common
        # "parse" and "validate" appear only once each — below threshold (60%)
        assert "parse" not in common
        assert "validate" not in common

    def test_extract_common_steps_empty(
        self, registry: SkillRegistry, tracker: QValueTracker
    ) -> None:
        """Empty trajectories return empty list."""
        creator = SkillCreator(registry, tracker)
        assert creator._extract_common_steps([]) == []

    def test_evaluate_skill(
        self, registry: SkillRegistry, tracker: QValueTracker
    ) -> None:
        """Evaluation should return a dict with stats."""
        skill = registry.register_skill(
            name="test", tier1_metadata="test", skill_id="eval1"
        )
        test_trajectories = [
            Trajectory(
                task_description="a",
                steps=["step 1", "step 2", "step 3"],
                outcome="success",
                tokens_used=100,
                latency_ms=200.0,
            ),
            Trajectory(
                task_description="a",
                steps=["step 1", "step 2", "step 3"],
                outcome="success",
                tokens_used=120,
                latency_ms=250.0,
            ),
            Trajectory(
                task_description="a",
                steps=["step 1", "step 2"],
                outcome="failure",
                tokens_used=80,
                latency_ms=300.0,
            ),
        ]

        creator = SkillCreator(registry, tracker)
        result = creator.evaluate_skill("eval1", test_trajectories)

        assert isinstance(result, dict)
        assert result["skill_id"] == "eval1"
        assert result["total_tested"] == 3
        assert result["success_count"] == 2
        assert 0.0 <= result["success_rate"] <= 1.0
        assert result["success_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_evaluate_skill_empty(
        self, registry: SkillRegistry, tracker: QValueTracker
    ) -> None:
        """Empty trajectories yield a dict with zero stats."""
        registry.register_skill(
            name="test", tier1_metadata="test", skill_id="eval2"
        )
        creator = SkillCreator(registry, tracker)
        result = creator.evaluate_skill("eval2", [])
        assert result["total_tested"] == 0
        assert result["success_rate"] == 0.0

    def test_synthesized_instructions(
        self, registry: SkillRegistry, tracker: QValueTracker
    ) -> None:
        """Created skill's tier2_core should contain extracted steps."""
        trajectories = [
            Trajectory(
                task_description="x",
                steps=["connect to db", "run query", "close conn"],
                outcome="success",
            ),
            Trajectory(
                task_description="x",
                steps=["connect to db", "run query", "close conn"],
                outcome="success",
            ),
        ]

        creator = SkillCreator(registry, tracker)
        skill_id = creator.create_from_trajectory("db-query", trajectories)

        skill = registry.get_skill(skill_id, tier=2)
        assert skill is not None
        assert "connect to db" in skill.tier2_core
        assert "run query" in skill.tier2_core


# =======================================================================
# SkillAnalyzer tests
# =======================================================================

class TestSkillAnalyzer:
    """Tests for the SkillAnalyzer module."""

    def test_cluster_by_domain(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Clustering by domain groups skills by primary (first) tag."""
        # Register skills with shared primary tag
        registry.register_skill(
            name="s1", tier1_metadata="s1", tags=["data", "web"], skill_id="cd1"
        )
        registry.register_skill(
            name="s2", tier1_metadata="s2", tags=["data", "api"], skill_id="cd2"
        )
        registry.register_skill(
            name="s3", tier1_metadata="s3", tags=["email"], skill_id="cd3"
        )

        analyzer = SkillAnalyzer(registry, tracker, graph)
        clusters = analyzer.cluster_skills(method="domain")

        assert len(clusters) > 0
        assert all(isinstance(c, SkillCluster) for c in clusters)

        # "data" is the primary (first) tag of both cd1 and cd2
        data_cluster = [c for c in clusters if c.domain == "data"]
        assert len(data_cluster) == 1
        assert len(data_cluster[0].skill_ids) == 2

    def test_cluster_by_q_value(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Clustering by Q-value produces high/medium/low buckets."""
        s1 = registry.register_skill(
            name="high", tier1_metadata="h", skill_id="qh"
        )
        s2 = registry.register_skill(
            name="medium", tier1_metadata="m", skill_id="qm"
        )
        s3 = registry.register_skill(
            name="low", tier1_metadata="l", skill_id="ql"
        )
        registry.update_skill("qh", {"q_value": 0.9})
        registry.update_skill("qm", {"q_value": 0.5})
        registry.update_skill("ql", {"q_value": 0.2})

        analyzer = SkillAnalyzer(registry, tracker, graph)
        clusters = analyzer.cluster_skills(method="q_value")

        domains = {c.domain for c in clusters}
        assert "high" in domains
        assert "low" in domains

    def test_cluster_invalid_method_raises(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Invalid method raises ValueError."""
        analyzer = SkillAnalyzer(registry, tracker, graph)
        with pytest.raises(ValueError, match="Unknown clustering method"):
            analyzer.cluster_skills(method="invalid")

    def test_cluster_empty_registry(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Empty registry returns empty clusters."""
        analyzer = SkillAnalyzer(registry, tracker, graph)
        clusters = analyzer.cluster_skills()
        assert clusters == []

    def test_get_usage_patterns(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Usage patterns should return a UsagePattern object."""
        registry.register_skill(
            name="test", tier1_metadata="test", skill_id="up1"
        )

        # Record some outcomes
        for _ in range(5):
            tracker.record_outcome(Outcome(
                skill_id="up1", success=True,
                latency_ms=100.0, tokens_used=50
            ))

        analyzer = SkillAnalyzer(registry, tracker, graph)
        pattern = analyzer.get_usage_patterns("up1")

        assert isinstance(pattern, UsagePattern)
        assert pattern.skill_id == "up1"
        assert isinstance(pattern.peak_hours, list)
        assert isinstance(pattern.common_predecessors, list)
        assert isinstance(pattern.common_successors, list)

    def test_usage_patterns_no_data(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Skills with no usage data return empty patterns."""
        registry.register_skill(
            name="unused", tier1_metadata="u", skill_id="unused"
        )

        analyzer = SkillAnalyzer(registry, tracker, graph)
        pattern = analyzer.get_usage_patterns("unused")

        assert pattern.peak_hours == []
        assert pattern.common_predecessors == []
        assert pattern.common_successors == []

    def test_find_underutilized_skills(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Skills with low usage count should be flagged."""
        registry.register_skill(
            name="popular", tier1_metadata="p", skill_id="pop"
        )
        registry.register_skill(
            name="rare", tier1_metadata="r", skill_id="rare"
        )
        registry.update_skill("pop", {"usage_count": 50})
        registry.update_skill("rare", {"usage_count": 2})

        analyzer = SkillAnalyzer(registry, tracker, graph)
        underutilized = analyzer.find_underutilized_skills(threshold=5)

        assert "rare" in underutilized
        assert "pop" not in underutilized

    def test_find_overloaded_skills(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Skills with high usage count should be flagged."""
        registry.register_skill(
            name="overworked", tier1_metadata="o", skill_id="over"
        )
        registry.register_skill(
            name="normal", tier1_metadata="n", skill_id="norm"
        )
        registry.update_skill("over", {"usage_count": 200})
        registry.update_skill("norm", {"usage_count": 30})

        analyzer = SkillAnalyzer(registry, tracker, graph)
        overloaded = analyzer.find_overloaded_skills(threshold=100)

        assert "over" in overloaded
        assert "norm" not in overloaded

    def test_get_recommendations(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Recommendations should include actions for problematic skills."""
        # Zero usage + low Q-value → deprecate
        registry.register_skill(
            name="dead", tier1_metadata="dead", skill_id="dead1"
        )
        registry.update_skill("dead1", {"q_value": 0.1, "usage_count": 0})

        # High Q-value + high usage → promote
        registry.register_skill(
            name="star", tier1_metadata="star", skill_id="star1"
        )
        registry.update_skill("star1", {"q_value": 0.9, "usage_count": 50})

        analyzer = SkillAnalyzer(registry, tracker, graph)
        recs = analyzer.get_recommendations()

        assert isinstance(recs, list)
        assert len(recs) >= 1

        actions = {r["action"] for r in recs}
        # Dead skill should have deprecate action
        assert "deprecate" in actions or "promote" in actions

    def test_generate_report(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Report should be a formatted string."""
        _register_sample_skills(registry)

        analyzer = SkillAnalyzer(registry, tracker, graph)
        report = analyzer.generate_report()

        assert isinstance(report, str)
        assert "Intelligence Report" in report
        assert "Total Skills: 3" in report

    def test_skill_matrix_empty(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Empty tracker yields empty matrix."""
        analyzer = SkillAnalyzer(registry, tracker, graph)
        matrix = analyzer.get_skill_matrix()
        assert isinstance(matrix, dict)
        assert len(matrix) == 0


# =======================================================================
# SkillOptimizer tests
# =======================================================================

class TestSkillOptimizer:
    """Tests for the SkillOptimizer module."""

    def test_compress_skill(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Compression should reduce tier2_core size."""
        content = (
            "1. Please note that you should initialize the system\n"
            "2. In order to connect, ensure that credentials are set\n"
            "3. In the event that data is missing, fetch defaults\n"
            "1. Please note that you should initialize the system\n"  # duplicate
            "4. For the purpose of logging, at this point in time, use stdout\n"
        )
        registry.register_skill(
            name="verbose", tier1_metadata="verbose",
            tier2_core=content, skill_id="comp1"
        )

        optimizer = SkillOptimizer(registry, tracker, graph)
        action = optimizer.compress_skill("comp1")

        assert isinstance(action, OptimizationAction)
        assert action.action_type == "compress"
        assert action.applied is True
        assert action.expected_improvement > 0.0

        # Verify the skill was actually compressed
        updated = registry.get_skill("comp1", tier=2)
        assert updated is not None
        assert len(updated.tier2_core) < len(content)

    def test_compress_no_content(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Compressing an empty skill should return a no-op action."""
        registry.register_skill(
            name="empty", tier1_metadata="e",
            tier2_core="", skill_id="empty1"
        )

        optimizer = SkillOptimizer(registry, tracker, graph)
        action = optimizer.compress_skill("empty1")

        assert action.applied is False
        assert action.expected_improvement == 0.0

    def test_compress_nonexistent_raises(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Compressing a nonexistent skill raises ValueError."""
        optimizer = SkillOptimizer(registry, tracker, graph)
        with pytest.raises(ValueError, match="not found"):
            optimizer.compress_skill("nope")

    def test_compress_removes_duplicates(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Duplicate lines should be removed."""
        content = "Step A\nStep B\nStep A\nStep C\nStep B"
        registry.register_skill(
            name="dup", tier1_metadata="d",
            tier2_core=content, skill_id="dup1"
        )

        optimizer = SkillOptimizer(registry, tracker, graph)
        action = optimizer.compress_skill("dup1")

        updated = registry.get_skill("dup1", tier=2)
        assert updated is not None
        # Count unique lines
        lines = [l for l in updated.tier2_core.split("\n") if l.strip()]
        assert len(lines) == 3  # Step A, B, C

    def test_merge_skills(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Merging combines two skills and deprecates originals."""
        registry.register_skill(
            name="alpha", tier1_metadata="a",
            tier2_core="1. Init\n2. Process\n3. Finish",
            tags=["common"], skill_id="merge-a",
        )
        registry.register_skill(
            name="beta", tier1_metadata="b",
            tier2_core="1. Init\n2. Validate\n3. Process",
            tags=["common", "extra"], skill_id="merge-b",
        )

        optimizer = SkillOptimizer(registry, tracker, graph)
        merged_id = optimizer.merge_skills(["merge-a", "merge-b"])

        # Merged skill should exist
        merged = registry.get_skill(merged_id, tier=2)
        assert merged is not None
        assert "Merged:" in merged.name
        # Combined unique steps
        assert "Init" in merged.tier2_core
        assert "Validate" in merged.tier2_core
        assert "Finish" in merged.tier2_core
        # Tags merged
        assert "common" in merged.tags
        assert "extra" in merged.tags

        # Originals should be deprecated
        orig_a = registry.get_skill("merge-a")
        orig_b = registry.get_skill("merge-b")
        assert orig_a is not None and orig_a.lifecycle == SkillLifecycle.DEPRECATED
        assert orig_b is not None and orig_b.lifecycle == SkillLifecycle.DEPRECATED

    def test_merge_fewer_than_two_raises(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Merging fewer than 2 skills raises ValueError."""
        optimizer = SkillOptimizer(registry, tracker, graph)
        with pytest.raises(ValueError, match="At least 2"):
            optimizer.merge_skills(["only-one"])

    def test_optimize_all(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """optimize_all should process all registered skills."""
        registry.register_skill(
            name="s1", tier1_metadata="s1",
            tier2_core="In order to do X, ensure that Y is set\n"
                       "Make sure to validate inputs\n"
                       "Please note that results are cached",
            skill_id="opt1",
        )
        registry.register_skill(
            name="s2", tier1_metadata="s2",
            tier2_core="Short skill",
            skill_id="opt2",
        )

        optimizer = SkillOptimizer(registry, tracker, graph)
        actions = optimizer.optimize_all()

        assert isinstance(actions, list)
        assert len(actions) >= 2  # At least compress for each skill
        assert all(isinstance(a, OptimizationAction) for a in actions)

    def test_reorder_steps(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Reorder should re-number steps by priority."""
        content = (
            "1. This is a very long step with lots of text and details\n"
            "2. Short\n"
            "3. Medium length step here"
        )
        registry.register_skill(
            name="reorder-test", tier1_metadata="r",
            tier2_core=content, skill_id="re1"
        )

        optimizer = SkillOptimizer(registry, tracker, graph)
        action = optimizer.reorder_steps("re1")

        assert isinstance(action, OptimizationAction)
        assert action.action_type == "reorder"

    def test_reorder_no_steps(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Reordering a skill without numbered steps is a no-op."""
        registry.register_skill(
            name="no-steps", tier1_metadata="n",
            tier2_core="Just a paragraph with no numbered steps.",
            skill_id="nosteps"
        )

        optimizer = SkillOptimizer(registry, tracker, graph)
        action = optimizer.reorder_steps("nosteps")

        assert action.applied is False

    def test_split_small_skill_no_split(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """A small skill (< 500 tokens) should not be split."""
        registry.register_skill(
            name="small", tier1_metadata="s",
            tier2_core="Short content", skill_id="small1"
        )

        optimizer = SkillOptimizer(registry, tracker, graph)
        new_ids = optimizer.split_skill("small1")

        assert new_ids == []

    def test_split_nonexistent_raises(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Splitting a nonexistent skill raises ValueError."""
        optimizer = SkillOptimizer(registry, tracker, graph)
        with pytest.raises(ValueError, match="not found"):
            optimizer.split_skill("nope")

    def test_estimate_improvement(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Improvement estimation returns reasonable values."""
        optimizer = SkillOptimizer(registry, tracker, graph)

        compress_action = OptimizationAction(
            action_type="compress", skill_id="x", description=""
        )
        assert optimizer._estimate_improvement(compress_action) > 0.0

        merge_action = OptimizationAction(
            action_type="merge", skill_id="x", description=""
        )
        assert optimizer._estimate_improvement(merge_action) > 0.0

        reorder_action = OptimizationAction(
            action_type="reorder", skill_id="x", description=""
        )
        assert optimizer._estimate_improvement(reorder_action) == 0.0
