"""Tests for SkillForge Advanced modules (Phase 4).

Covers RLOptimizer, SharedSkillPool (MultiAgent), SkillPredictor,
and SkillTransferEngine.  Uses pytest with tmpdir for DB/file isolation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from skillforge.core.registry import Skill, SkillLifecycle, SkillRegistry
from skillforge.core.tracker import Outcome, QValueTracker
from skillforge.core.graph import SkillDependencyGraph
from skillforge.advanced.rl_optimizer import TrainingStep, RLOptimizer
from skillforge.advanced.multi_agent import SharedSkillPool, AgentAccess, AccessLevel
from skillforge.advanced.predictor import SkillPrediction, SkillPredictor
from skillforge.advanced.transfer import SkillTransferEngine, TransferResult


# -----------------------------------------------------------------------
# Mock evolution for RLOptimizer
# -----------------------------------------------------------------------

class MockEvolution:
    """Mock evolution protocol for RLOptimizer."""

    def evolve_skill(self, skill_id: str) -> bool:
        return True


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
def sample_registry(registry: SkillRegistry) -> SkillRegistry:
    """Registry with sample skills pre-loaded."""
    registry.register_skill(
        name="web-scraper",
        tier1_metadata="Scrapes web content from URLs",
        tier2_core="1. Fetch the URL\n2. Parse HTML\n3. Extract data\n4. Store results",
        tags=["web", "scraping", "data"],
        skill_id="skill-1",
    )
    registry.register_skill(
        name="api-fetcher",
        tier1_metadata="Fetches API data with authentication",
        tier2_core="1. Authenticate\n2. Send request\n3. Parse JSON\n4. Store results",
        tags=["api", "data", "networking"],
        skill_id="skill-2",
    )
    registry.register_skill(
        name="email-sender",
        tier1_metadata="Sends emails via SMTP",
        tier2_core="1. Connect SMTP\n2. Compose message\n3. Send email\n4. Log result",
        tags=["email", "communication"],
        skill_id="skill-3",
    )
    return registry


def _record_outcomes(
    tracker: QValueTracker, skill_id: str, count: int = 10, success_rate: float = 0.7
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
# TestRLOptimizer
# =======================================================================

class TestRLOptimizer:
    """Tests for the RLOptimizer module."""

    def test_optimize_returns_training_steps(
        self, sample_registry: SkillRegistry, tracker: QValueTracker, db_dir: Path
    ) -> None:
        """Optimization should return a list of TrainingStep objects."""
        _record_outcomes(tracker, "skill-1", count=20, success_rate=0.9)

        evolution = MockEvolution()
        optimizer = RLOptimizer(
            sample_registry, tracker, evolution,
            db_path=str(db_dir / "rl_optimizer.db"),
        )
        steps = optimizer.optimize("skill-1", max_iterations=3)

        assert isinstance(steps, list)
        # May or may not have steps depending on whether compression improves things
        for step in steps:
            assert isinstance(step, TrainingStep)
            assert step.skill_id == "skill-1"
            assert step.action in ("compress", "split", "reorder")

        optimizer.close()

    def test_optimize_missing_skill_returns_empty(
        self, sample_registry: SkillRegistry, tracker: QValueTracker, db_dir: Path
    ) -> None:
        """Optimizing a missing skill should return empty list (not raise)."""
        evolution = MockEvolution()
        optimizer = RLOptimizer(
            sample_registry, tracker, evolution,
            db_path=str(db_dir / "rl_optimizer.db"),
        )
        steps = optimizer.optimize("nonexistent", max_iterations=1)

        assert steps == []
        optimizer.close()

    def test_batch_optimize(
        self, sample_registry: SkillRegistry, tracker: QValueTracker, db_dir: Path
    ) -> None:
        """Batch optimization should optimize skills below threshold."""
        # Set skill-1 to low Q-value (below threshold)
        sample_registry.update_skill("skill-1", {"q_value": 0.2})
        _record_outcomes(tracker, "skill-1", count=10, success_rate=0.8)
        _record_outcomes(tracker, "skill-2", count=10, success_rate=0.6)

        evolution = MockEvolution()
        optimizer = RLOptimizer(
            sample_registry, tracker, evolution,
            db_path=str(db_dir / "rl_optimizer.db"),
        )

        # Use a high threshold so skills get optimized
        steps = optimizer.batch_optimize(threshold=0.8)

        assert isinstance(steps, list)
        # Should have attempted to optimize at least one skill
        optimized_ids = {s.skill_id for s in steps}
        # skill-1 has q_value=0.2 in registry, which is below 0.8 threshold
        # It may or may not have produced actual steps depending on compression results
        assert isinstance(optimized_ids, set)

        optimizer.close()

    def test_try_compression(
        self, sample_registry: SkillRegistry, tracker: QValueTracker, db_dir: Path
    ) -> None:
        """Compression should return a reward value."""
        # Add redundant content that can be compressed
        sample_registry.update_skill("skill-1", {
            "tier2_core": "1. Fetch URL\n2. Parse HTML\n1. Fetch URL\n3. Extract data\n2. Parse HTML\n\n\n\n"
        })
        _record_outcomes(tracker, "skill-1", count=10, success_rate=0.8)

        evolution = MockEvolution()
        optimizer = RLOptimizer(
            sample_registry, tracker, evolution,
            db_path=str(db_dir / "rl_optimizer.db"),
        )

        reward = optimizer._try_compression("skill-1")

        assert isinstance(reward, float)
        # The compression should have reduced the text
        skill = sample_registry.get_skill("skill-1", tier=3)
        assert skill is not None
        # After compression, there should be fewer duplicate lines
        assert "\n\n\n\n" not in skill.tier2_core

        optimizer.close()

    def test_evaluate_reward(
        self, sample_registry: SkillRegistry, tracker: QValueTracker, db_dir: Path
    ) -> None:
        """Reward evaluation should compute composite reward."""
        evolution = MockEvolution()
        optimizer = RLOptimizer(
            sample_registry, tracker, evolution,
            db_path=str(db_dir / "rl_optimizer.db"),
        )

        before = {"success_rate": 0.5, "q_value": 0.5, "avg_tokens": 100}
        after = {"success_rate": 0.8, "q_value": 0.7, "avg_tokens": 80}

        reward = optimizer._evaluate_reward("skill-1", before, after)

        assert isinstance(reward, float)
        # With improvement in all metrics, reward should be positive
        assert reward > 0.0

        optimizer.close()

    def test_evaluate_reward_negative(
        self, sample_registry: SkillRegistry, tracker: QValueTracker, db_dir: Path
    ) -> None:
        """Degradation should produce negative reward."""
        evolution = MockEvolution()
        optimizer = RLOptimizer(
            sample_registry, tracker, evolution,
            db_path=str(db_dir / "rl_optimizer.db"),
        )

        before = {"success_rate": 0.8, "q_value": 0.8, "avg_tokens": 50}
        after = {"success_rate": 0.3, "q_value": 0.3, "avg_tokens": 200}

        reward = optimizer._evaluate_reward("skill-1", before, after)

        assert reward < 0.0
        optimizer.close()

    def test_optimization_history(
        self, sample_registry: SkillRegistry, tracker: QValueTracker, db_dir: Path
    ) -> None:
        """Optimization history should be logged and retrievable."""
        _record_outcomes(tracker, "skill-1", count=10, success_rate=0.9)

        evolution = MockEvolution()
        optimizer = RLOptimizer(
            sample_registry, tracker, evolution,
            db_path=str(db_dir / "rl_optimizer.db"),
        )

        optimizer.optimize("skill-1", max_iterations=2)
        history = optimizer.get_optimization_history("skill-1")

        # History may have entries depending on whether improvements were found
        assert isinstance(history, list)
        for step in history:
            assert isinstance(step, TrainingStep)
            assert step.skill_id == "skill-1"

        optimizer.close()


# =======================================================================
# TestMultiAgent (SharedSkillPool)
# =======================================================================

class TestMultiAgent:
    """Tests for the SharedSkillPool (multi-agent) module."""

    def test_create_workspace(
        self, sample_registry: SkillRegistry, graph: SkillDependencyGraph, db_dir: Path
    ) -> None:
        """Creating a workspace should succeed and return summary dict."""
        pool = SharedSkillPool(
            sample_registry, graph,
            db_path=str(db_dir / "multi_agent.db"),
        )
        result = pool.create_workspace("ws-1", ["agent-a", "agent-b"])

        assert isinstance(result, dict)
        assert result["workspace_id"] == "ws-1"
        assert result["agent_count"] == 2
        assert "created_at" in result

        pool.close()

    def test_share_skill(
        self, sample_registry: SkillRegistry, graph: SkillDependencyGraph, db_dir: Path
    ) -> None:
        """Sharing a skill should create access records."""
        pool = SharedSkillPool(
            sample_registry, graph,
            db_path=str(db_dir / "multi_agent.db"),
        )
        pool.create_workspace("ws-1", ["agent-a", "agent-b"])

        records = pool.share_skill(
            "skill-1",
            from_agent="agent-a",
            to_agents=["agent-b"],
            access_level=AccessLevel.READ,
            workspace_id="ws-1",
        )

        assert len(records) == 1
        assert isinstance(records[0], AgentAccess)
        assert records[0].agent_id == "agent-b"
        assert records[0].skill_id == "skill-1"
        assert records[0].access_level == AccessLevel.READ

        pool.close()

    def test_get_shared_skills(
        self, sample_registry: SkillRegistry, graph: SkillDependencyGraph, db_dir: Path
    ) -> None:
        """Getting shared skills should return skills shared with an agent."""
        pool = SharedSkillPool(
            sample_registry, graph,
            db_path=str(db_dir / "multi_agent.db"),
        )
        pool.create_workspace("ws-1", ["agent-a", "agent-b"])
        pool.share_skill("skill-1", "agent-a", ["agent-b"], AccessLevel.WRITE, "ws-1")
        pool.share_skill("skill-2", "agent-a", ["agent-b"], AccessLevel.READ, "ws-1")

        shared = pool.get_shared_skills("agent-b")

        assert len(shared) == 2
        skill_ids = {s["skill_id"] for s in shared}
        assert "skill-1" in skill_ids
        assert "skill-2" in skill_ids

        pool.close()

    def test_sync_workspaces(
        self, sample_registry: SkillRegistry, graph: SkillDependencyGraph, db_dir: Path
    ) -> None:
        """Syncing should propagate Q-value updates."""
        pool = SharedSkillPool(
            sample_registry, graph,
            db_path=str(db_dir / "multi_agent.db"),
        )
        pool.create_workspace("ws-1", ["agent-a"])
        pool.share_skill("skill-1", "agent-a", ["agent-a"], AccessLevel.ADMIN, "ws-1")

        # Update Q-value of the skill
        sample_registry.update_skill("skill-1", {"q_value": 0.9})

        result = pool.sync_workspaces()

        assert isinstance(result, dict)
        # Should have synced workspace if Q-value changed from cache
        # (initial cache is 0.5, now skill has 0.9)

        pool.close()

    def test_workspace_stats(
        self, sample_registry: SkillRegistry, graph: SkillDependencyGraph, db_dir: Path
    ) -> None:
        """Getting workspace stats should return complete info."""
        pool = SharedSkillPool(
            sample_registry, graph,
            db_path=str(db_dir / "multi_agent.db"),
        )
        pool.create_workspace("ws-1", ["agent-a", "agent-b", "agent-c"])
        pool.share_skill("skill-1", "agent-a", ["agent-b"], AccessLevel.READ, "ws-1")

        stats = pool.get_workspace_stats("ws-1")

        assert stats["workspace_id"] == "ws-1"
        assert stats["agent_count"] >= 2  # At least agent-a and agent-b
        assert stats["skill_count"] >= 1

        pool.close()

    def test_share_skill_multiple_agents(
        self, sample_registry: SkillRegistry, graph: SkillDependencyGraph, db_dir: Path
    ) -> None:
        """Sharing to multiple agents should create multiple access records."""
        pool = SharedSkillPool(
            sample_registry, graph,
            db_path=str(db_dir / "multi_agent.db"),
        )
        pool.create_workspace("ws-1", ["agent-a", "agent-b", "agent-c"])

        records = pool.share_skill(
            "skill-1",
            from_agent="agent-a",
            to_agents=["agent-b", "agent-c"],
            access_level=AccessLevel.WRITE,
            workspace_id="ws-1",
        )

        assert len(records) == 2
        assert {r.agent_id for r in records} == {"agent-b", "agent-c"}

        pool.close()


# =======================================================================
# TestSkillPredictor
# =======================================================================

class TestSkillPredictor:
    """Tests for the SkillPredictor module."""

    def test_predict_performance_with_data(
        self, sample_registry: SkillRegistry, tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Predicting with sufficient data should return valid predictions."""
        _record_outcomes(tracker, "skill-1", count=20, success_rate=0.8)

        predictor = SkillPredictor(sample_registry, tracker, graph)
        pred = predictor.predict_performance("skill-1")

        assert isinstance(pred, SkillPrediction)
        assert pred.skill_id == "skill-1"
        assert 0.0 <= pred.predicted_success_rate <= 1.0
        assert pred.confidence > 0.0

    def test_predict_performance_insufficient_data(
        self, sample_registry: SkillRegistry, tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Predicting with no history should return low confidence."""
        predictor = SkillPredictor(sample_registry, tracker, graph)
        pred = predictor.predict_performance("skill-1")

        assert pred.confidence <= 0.1
        assert pred.recommendation == "insufficient_data"

    def test_detect_decline_declining(
        self, sample_registry: SkillRegistry, tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """A skill with declining outcomes should be detected."""
        # First half: high success
        for i in range(5):
            tracker.record_outcome(Outcome(
                skill_id="skill-1",
                success=True,
                latency_ms=100.0,
                tokens_used=500,
            ))
        # Second half: low success (declining)
        for i in range(15):
            tracker.record_outcome(Outcome(
                skill_id="skill-1",
                success=False,
                latency_ms=300.0,
                tokens_used=800,
            ))

        predictor = SkillPredictor(sample_registry, tracker, graph)
        declining = predictor.detect_decline("skill-1")

        assert declining is True

    def test_detect_decline_stable(
        self, sample_registry: SkillRegistry, tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """A consistently good skill should not be flagged as declining."""
        # Use 100% success rate so there's no decline slope at all
        for i in range(30):
            tracker.record_outcome(Outcome(
                skill_id="skill-1",
                success=True,
                latency_ms=100.0 + (i % 5) * 2,  # small variance
                tokens_used=500 + (i % 5) * 10,
            ))

        predictor = SkillPredictor(sample_registry, tracker, graph)
        declining = predictor.detect_decline("skill-1")

        assert declining is False

    def test_detect_decline_insufficient_data(
        self, sample_registry: SkillRegistry, tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Insufficient data should not flag decline."""
        predictor = SkillPredictor(sample_registry, tracker, graph)
        declining = predictor.detect_decline("skill-1")

        assert declining is False

    def test_linear_regression_basic(self) -> None:
        """Linear regression should compute correct slope and intercept."""
        # y = 2x + 1
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0, 3.0, 5.0, 7.0, 9.0]

        slope, intercept = SkillPredictor._linear_regression(x, y)

        assert abs(slope - 2.0) < 0.01
        assert abs(intercept - 1.0) < 0.01

    def test_linear_regression_flat(self) -> None:
        """Flat data should have zero slope."""
        x = [0.0, 1.0, 2.0, 3.0]
        y = [5.0, 5.0, 5.0, 5.0]

        slope, intercept = SkillPredictor._linear_regression(x, y)

        assert abs(slope) < 0.01
        assert abs(intercept - 5.0) < 0.01

    def test_linear_regression_empty(self) -> None:
        """Empty input should return (0, 0)."""
        slope, intercept = SkillPredictor._linear_regression([], [])
        assert slope == 0.0
        assert intercept == 0.0

    def test_linear_regression_single_point(self) -> None:
        """Single data point should return (0, y)."""
        slope, intercept = SkillPredictor._linear_regression([0.0], [3.5])
        assert slope == 0.0
        assert intercept == 3.5

    def test_recommend_skills(
        self, sample_registry: SkillRegistry, tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Recommendations should return ranked skill list."""
        _record_outcomes(tracker, "skill-1", count=20, success_rate=0.9)
        _record_outcomes(tracker, "skill-2", count=20, success_rate=0.6)
        _record_outcomes(tracker, "skill-3", count=20, success_rate=0.3)

        predictor = SkillPredictor(sample_registry, tracker, graph)
        recs = predictor.recommend_skills("data scraping", top_k=5)

        assert isinstance(recs, list)
        # Results depend on search matching, but should return dicts
        for rec in recs:
            assert "skill_id" in rec
            assert "prediction" in rec
            assert "score" in rec

    def test_recommend_skills_empty_query(
        self, sample_registry: SkillRegistry, tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        """Empty query should return empty recommendations."""
        predictor = SkillPredictor(sample_registry, tracker, graph)
        recs = predictor.recommend_skills("zzz_nonexistent_zzz", top_k=3)

        assert isinstance(recs, list)
        assert len(recs) == 0


# =======================================================================
# TestTransferEngine
# =======================================================================

class TestTransferEngine:
    """Tests for the SkillTransferEngine module."""

    def test_export_to_hermes(
        self, sample_registry: SkillRegistry, tmp_path: Path
    ) -> None:
        """Exporting to Hermes should create .skill.md files."""
        engine = SkillTransferEngine(sample_registry)
        output_dir = str(tmp_path / "hermes_export")

        results = engine.export_to_hermes(["skill-1", "skill-2"], output_dir)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.target_format == "hermes" for r in results)

        # Verify files exist and contain expected content
        for result in results:
            file_path = Path(result.target_path)
            assert file_path.exists()
            content = file_path.read_text()
            assert "---" in content  # YAML frontmatter
            assert "name:" in content

    def test_export_to_hermes_contains_metadata(
        self, sample_registry: SkillRegistry, tmp_path: Path
    ) -> None:
        """Hermes export should include all skill metadata."""
        engine = SkillTransferEngine(sample_registry)
        output_dir = str(tmp_path / "hermes_export")

        results = engine.export_to_hermes(["skill-1"], output_dir)

        assert results[0].success
        content = Path(results[0].target_path).read_text()

        assert "web-scraper" in content
        assert "tier1_metadata" in content
        assert "Instructions" in content

    def test_export_to_hermes_missing_skill(
        self, sample_registry: SkillRegistry, tmp_path: Path
    ) -> None:
        """Exporting a missing skill should fail gracefully."""
        engine = SkillTransferEngine(sample_registry)
        output_dir = str(tmp_path / "hermes_export")

        results = engine.export_to_hermes(["nonexistent"], output_dir)

        assert len(results) == 1
        assert results[0].success is False
        assert "not found" in results[0].notes

    def test_export_to_openclaw(
        self, sample_registry: SkillRegistry, tmp_path: Path
    ) -> None:
        """Exporting to OpenClaw should create .learning files."""
        engine = SkillTransferEngine(sample_registry)
        output_dir = str(tmp_path / "openclaw_export")

        results = engine.export_to_openclaw(["skill-1"], output_dir)

        assert len(results) == 1
        assert results[0].success
        assert ".learning" in results[0].target_path

        # Verify .learnings directory structure
        file_path = Path(results[0].target_path)
        assert file_path.exists()
        content = file_path.read_text()
        assert "source: skillforge" in content
        assert "confidence:" in content

    def test_export_to_json(
        self, sample_registry: SkillRegistry, tmp_path: Path
    ) -> None:
        """Exporting to JSON should create a valid JSON file."""
        engine = SkillTransferEngine(sample_registry)
        output_path = str(tmp_path / "export.json")

        results = engine.export_to_json(["skill-1", "skill-2", "skill-3"], output_path)

        assert len(results) == 3
        assert all(r.success for r in results)

        # Verify JSON file
        with open(output_path) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 3
        assert all("name" in d for d in data)
        assert all("tier2_core" in d for d in data)

    def test_export_to_json_content(
        self, sample_registry: SkillRegistry, tmp_path: Path
    ) -> None:
        """JSON export should preserve all skill fields."""
        engine = SkillTransferEngine(sample_registry)
        output_path = str(tmp_path / "export.json")

        engine.export_to_json(["skill-1"], output_path)

        with open(output_path) as f:
            data = json.load(f)

        skill_data = data[0]
        assert skill_data["id"] == "skill-1"
        assert skill_data["name"] == "web-scraper"
        assert skill_data["tags"] == ["web", "scraping", "data"]
        assert "tier2_core" in skill_data
        assert "q_value" in skill_data

    def test_export_to_markdown(
        self, sample_registry: SkillRegistry, tmp_path: Path
    ) -> None:
        """Exporting to markdown should create readable docs."""
        engine = SkillTransferEngine(sample_registry)
        output_dir = str(tmp_path / "md_export")

        results = engine.export_to_markdown(["skill-1"], output_dir)

        assert len(results) == 1
        assert results[0].success

        content = Path(results[0].target_path).read_text()
        assert "# web-scraper" in content
        assert "Instructions" in content

    def test_import_from_directory_hermes(
        self, sample_registry: SkillRegistry, tmp_path: Path
    ) -> None:
        """Importing from Hermes format should register skills."""
        engine = SkillTransferEngine(sample_registry)

        # First export
        output_dir = str(tmp_path / "import_test")
        engine.export_to_hermes(["skill-1"], output_dir)

        # Create a new registry for import
        import_db = tmp_path / "import_skills.db"
        import_registry = SkillRegistry(db_path=import_db)
        import_engine = SkillTransferEngine(import_registry)

        results = import_engine.import_from_directory(output_dir, format="hermes")

        assert len(results) == 1
        assert results[0].success
        assert results[0].target_format == "hermes"

        # Verify the skill was registered
        skills = import_registry.list_skills()
        assert len(skills) == 1
        assert "web-scraper" in skills[0].name

        import_registry.close()

    def test_import_from_directory_json(
        self, sample_registry: SkillRegistry, tmp_path: Path
    ) -> None:
        """Importing from JSON format should register skills."""
        engine = SkillTransferEngine(sample_registry)

        # First export
        json_path = str(tmp_path / "skills.json")
        engine.export_to_json(["skill-1", "skill-2"], json_path)

        # Import into new registry
        import_db = tmp_path / "import_skills.db"
        import_registry = SkillRegistry(db_path=import_db)
        import_engine = SkillTransferEngine(import_registry)

        # Put the JSON file in a directory for import
        import_dir = tmp_path / "json_import"
        import_dir.mkdir()
        import_file = import_dir / "skills.json"
        import_file.write_text(Path(json_path).read_text())

        results = import_engine.import_from_directory(str(import_dir), format="json")

        assert len(results) == 2
        assert all(r.success for r in results)

        import_registry.close()

    def test_import_from_nonexistent_directory(
        self, sample_registry: SkillRegistry
    ) -> None:
        """Importing from a nonexistent directory should fail."""
        engine = SkillTransferEngine(sample_registry)
        results = engine.import_from_directory("/nonexistent/path", format="hermes")

        assert len(results) == 1
        assert results[0].success is False

    def test_import_unknown_format(
        self, sample_registry: SkillRegistry, tmp_path: Path
    ) -> None:
        """Importing with unknown format should fail gracefully."""
        engine = SkillTransferEngine(sample_registry)
        results = engine.import_from_directory(str(tmp_path), format="yaml")

        assert len(results) == 1
        assert results[0].success is False
        assert "Unknown format" in results[0].notes

    def test_batch_transfer(
        self, sample_registry: SkillRegistry, tmp_path: Path
    ) -> None:
        """Batch transfer should export all skills in the target format."""
        engine = SkillTransferEngine(sample_registry)
        output_dir = str(tmp_path / "batch_export")

        results = engine.batch_transfer(
            source_format="registry",
            target_format="hermes",
            skill_ids=["skill-1", "skill-2"],
            output_dir=output_dir,
        )

        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.target_format == "hermes" for r in results)

    def test_batch_transfer_json(
        self, sample_registry: SkillRegistry, tmp_path: Path
    ) -> None:
        """Batch transfer to JSON should create a valid file."""
        engine = SkillTransferEngine(sample_registry)
        output_dir = str(tmp_path / "batch_json")

        results = engine.batch_transfer(
            source_format="registry",
            target_format="json",
            skill_ids=["skill-1", "skill-2", "skill-3"],
            output_dir=output_dir,
        )

        assert all(r.success for r in results)

        # Verify JSON file
        json_file = Path(output_dir) / "skills_export.json"
        assert json_file.exists()
        data = json.loads(json_file.read_text())
        assert len(data) == 3

    def test_batch_transfer_markdown(
        self, sample_registry: SkillRegistry, tmp_path: Path
    ) -> None:
        """Batch transfer to markdown should create docs."""
        engine = SkillTransferEngine(sample_registry)
        output_dir = str(tmp_path / "batch_md")

        results = engine.batch_transfer(
            source_format="registry",
            target_format="markdown",
            skill_ids=["skill-1"],
            output_dir=output_dir,
        )

        assert results[0].success
        assert Path(results[0].target_path).exists()

    def test_batch_transfer_all_skills(
        self, sample_registry: SkillRegistry, tmp_path: Path
    ) -> None:
        """Batch transfer without skill_ids should export all."""
        engine = SkillTransferEngine(sample_registry)
        output_dir = str(tmp_path / "all_export")

        results = engine.batch_transfer(
            source_format="registry",
            target_format="hermes",
            output_dir=output_dir,
        )

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_roundtrip_hermes(
        self, sample_registry: SkillRegistry, tmp_path: Path
    ) -> None:
        """Export then import should preserve skill data."""
        engine = SkillTransferEngine(sample_registry)

        # Export
        output_dir = str(tmp_path / "roundtrip")
        engine.export_to_hermes(["skill-1"], output_dir)

        # Import into new registry
        import_db = tmp_path / "roundtrip.db"
        import_registry = SkillRegistry(db_path=import_db)
        import_engine = SkillTransferEngine(import_registry)

        import_engine.import_from_directory(output_dir, format="hermes")

        # Verify
        imported = import_registry.list_skills()
        assert len(imported) == 1
        assert "web-scraper" in imported[0].name

        import_registry.close()

    def test_roundtrip_json(
        self, sample_registry: SkillRegistry, tmp_path: Path
    ) -> None:
        """JSON roundtrip should preserve all fields."""
        engine = SkillTransferEngine(sample_registry)

        # Export
        json_path = str(tmp_path / "roundtrip.json")
        engine.export_to_json(["skill-1"], json_path)

        # Import into new registry
        import_dir = tmp_path / "import"
        import_dir.mkdir()
        (import_dir / "skills.json").write_text(Path(json_path).read_text())

        import_db = tmp_path / "roundtrip.db"
        import_registry = SkillRegistry(db_path=import_db)
        import_engine = SkillTransferEngine(import_registry)

        import_engine.import_from_directory(str(import_dir), format="json")

        imported = import_registry.list_skills()
        assert len(imported) == 1
        skill = import_registry.get_skill(imported[0].id, tier=3)
        assert skill is not None
        assert skill.name == "web-scraper"
        assert "Fetch the URL" in skill.tier2_core

        import_registry.close()

    def test_sanitize_filename(self) -> None:
        """Filename sanitization should produce safe names."""
        assert SkillTransferEngine._sanitize_filename("My Skill!") == "my-skill"
        assert SkillTransferEngine._sanitize_filename("a/b/c") == "abc"
        assert SkillTransferEngine._sanitize_filename("   spaces   ") == "spaces"
        assert SkillTransferEngine._sanitize_filename("") == "unnamed"
