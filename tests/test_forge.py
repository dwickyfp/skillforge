"""Integration tests for SkillForge orchestrator — full end-to-end scenarios.

These tests exercise the interaction between SkillRegistry, QValueTracker,
SkillDependencyGraph, and the Hermes adapter working together.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from skillforge.core.registry import Skill, SkillLifecycle, SkillRegistry
from skillforge.core.tracker import Outcome, QValueTracker
from skillforge.core.graph import SkillDependencyGraph
from skillforge.core.loader import ProgressiveLoader
from skillforge.integrations.hermes.adapter import HermesSkillForgeAdapter


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def registry(tmp_dir: Path) -> Iterator[SkillRegistry]:
    reg = SkillRegistry(db_path=tmp_dir / "forge_test.db")
    yield reg
    reg.close()


@pytest.fixture
def tracker(tmp_dir: Path) -> Iterator[QValueTracker]:
    t = QValueTracker(db_path=tmp_dir / "forge_tracker.db")
    yield t
    t.close()


@pytest.fixture
def graph() -> SkillDependencyGraph:
    return SkillDependencyGraph()


@pytest.fixture
def hermes_skills_dir(tmp_dir: Path) -> Path:
    d = tmp_dir / "hermes_skills"
    d.mkdir()
    return d


@pytest.fixture
def adapter(
    registry: SkillRegistry, hermes_skills_dir: Path
) -> HermesSkillForgeAdapter:
    return HermesSkillForgeAdapter(
        skillforge=registry, hermes_skills_dir=str(hermes_skills_dir)
    )


# -----------------------------------------------------------------------
# Scenario: Register → Track → Route
# -----------------------------------------------------------------------

class TestRegisterTrackRoute:
    """End-to-end: register skills, record outcomes, route a query."""

    def test_full_pipeline(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        # 1. Register skills
        s1 = registry.register_skill(
            name="email-sender",
            tier1_metadata="Sends emails via SMTP",
            tier2_core="Use SMTP to send emails",
            tags=["email", "communication"],
        )
        s2 = registry.register_skill(
            name="web-scraper",
            tier1_metadata="Scrapes web content",
            tier2_core="Use HTTP to scrape web content",
            tags=["web", "data"],
        )

        # 2. Record outcomes
        for _ in range(10):
            tracker.record_outcome(
                Outcome(skill_id=s1.id, success=True, latency_ms=50, tokens_used=100)
            )
            tracker.record_outcome(
                Outcome(skill_id=s2.id, success=False, latency_ms=200, tokens_used=300)
            )

        # 3. Update registry with Q-values from tracker
        for skill in registry.list_skills():
            stats = tracker.get_stats(skill.id)
            registry.update_skill(
                skill.id,
                {"q_value": stats["q_value"], "success_rate": stats["success_rate"]},
            )

        # 4. Route a query
        loader = ProgressiveLoader(registry, tracker)
        results = loader.load_skill("email", routing="q_value")
        assert len(results) >= 1
        # email-sender should rank higher (all successes vs all failures)
        assert any(s.name == "email-sender" for s in results)


# -----------------------------------------------------------------------
# Scenario: Dependency-aware skill management
# -----------------------------------------------------------------------

class TestDependencyAware:
    """End-to-end: manage skills with dependency graph + registry."""

    def test_dependency_propagation(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
    ) -> None:
        # Register base skill
        base = registry.register_skill(
            name="http-client",
            tier1_metadata="Base HTTP client",
            tier2_core="Makes HTTP requests",
        )
        # Register dependent skill
        scraper = registry.register_skill(
            name="web-scraper",
            tier1_metadata="Scrapes websites",
            tier2_core="Uses HTTP client to scrape",
        )

        # Build graph: scraper depends on base
        graph.add_skill(base.id)
        graph.add_skill(scraper.id)
        graph.add_dependency(scraper.id, base.id, weight=0.8)

        # Record successful outcomes for base
        for _ in range(10):
            tracker.record_outcome(
                Outcome(
                    skill_id=base.id,
                    success=True,
                    latency_ms=30,
                    tokens_used=50,
                )
            )

        # Apply TD(λ) update and propagate
        new_q = tracker.td_lambda_update(base.id, reward=1.0)
        delta = new_q - 0.5
        propagated = graph.propagate_q_update(base.id, delta=delta)

        # Scraper depends on base, so scraper should receive propagated update
        assert scraper.id in propagated

        # Topological sort: scraper depends on base → base comes first (lower index)
        order = graph.topological_sort()
        assert order.index(base.id) < order.index(scraper.id)


# -----------------------------------------------------------------------
# Scenario: Hermes Import/Export cycle
# -----------------------------------------------------------------------

class TestHermesIntegration:
    """End-to-end: Hermes skill import, usage tracking, export."""

    def test_import_hermes_skills(
        self,
        adapter: HermesSkillForgeAdapter,
        registry: SkillRegistry,
        hermes_skills_dir: Path,
    ) -> None:
        # Create a fake Hermes skill
        skill_dir = hermes_skills_dir / "tools" / "test-tool"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            '---\n'
            'name: test-tool\n'
            'description: "A test tool for testing."\n'
            'version: 1.0.0\n'
            'author: tester\n'
            'metadata:\n'
            '  hermes:\n'
            '    tags: [test, tool]\n'
            '---\n'
            '\n'
            '# Test Tool\n\n'
            'This is a test skill for integration testing.\n',
            encoding="utf-8",
        )

        # Import
        imported = adapter.import_hermes_skills()
        assert len(imported) == 1
        assert imported[0].name == "test-tool"
        assert "hermes_source" in imported[0].tags

    def test_export_skill_to_hermes(
        self,
        adapter: HermesSkillForgeAdapter,
        registry: SkillRegistry,
        hermes_skills_dir: Path,
    ) -> None:
        # Create a SkillForge skill
        skill = registry.register_skill(
            name="my-custom",
            tier1_metadata="Custom skill for export test",
            tier2_core="# Custom Skill\n\nCustom instructions here.",
            tags=["custom"],
        )

        # Export
        path = adapter.export_skill_to_hermes(skill.id, category="custom")
        assert path is not None
        assert path.exists()
        assert path.name == "SKILL.md"

        # Verify content
        content = path.read_text(encoding="utf-8")
        assert "my-custom" in content
        assert "Custom skill for export test" in content
        assert "# Custom Skill" in content

    def test_import_then_export_roundtrip(
        self,
        adapter: HermesSkillForgeAdapter,
        registry: SkillRegistry,
        hermes_skills_dir: Path,
    ) -> None:
        # Create original Hermes skill
        orig_dir = hermes_skills_dir / "tools" / "roundtrip"
        orig_dir.mkdir(parents=True)
        orig_file = orig_dir / "SKILL.md"
        orig_file.write_text(
            '---\n'
            'name: roundtrip\n'
            'description: "Roundtrip test skill."\n'
            'version: 2.0.0\n'
            'author: tester\n'
            'metadata:\n'
            '  hermes:\n'
            '    tags: [roundtrip, test]\n'
            '---\n'
            '\n'
            '# Roundtrip Skill\n\n'
            'Full instructions for roundtrip testing.\n',
            encoding="utf-8",
        )

        # Import → Export
        adapter.import_hermes_skills()
        skill = registry.get_skill("hermes-roundtrip", tier=3)
        assert skill is not None

        export_path = adapter.export_skill_to_hermes(
            skill.id, category="export-test"
        )
        assert export_path is not None

        exported_content = export_path.read_text(encoding="utf-8")
        assert "roundtrip" in exported_content
        assert "Roundtrip" in exported_content

    def test_bidirectional_sync(
        self,
        adapter: HermesSkillForgeAdapter,
        registry: SkillRegistry,
        hermes_skills_dir: Path,
    ) -> None:
        # Create a Hermes skill
        skill_dir = hermes_skills_dir / "tools" / "sync-test"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            '---\n'
            'name: sync-test\n'
            'description: "Sync test skill."\n'
            'version: 1.0.0\n'
            'metadata:\n'
            '  hermes:\n'
            '    tags: [sync]\n'
            '---\n'
            '\n'
            '# Sync Test\n\n'
            'Testing bidirectional sync.\n',
            encoding="utf-8",
        )

        # Run sync
        summary = adapter.sync()
        assert summary["imported"] == 1
        assert summary["errors"] == []

    def test_import_multiple_skills(
        self,
        adapter: HermesSkillForgeAdapter,
        registry: SkillRegistry,
        hermes_skills_dir: Path,
    ) -> None:
        # Create multiple skills
        for name in ("alpha", "beta", "gamma"):
            d = hermes_skills_dir / "category" / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(
                f'---\n'
                f'name: {name}\n'
                f'description: "The {name} skill."\n'
                f'version: 1.0.0\n'
                f'metadata:\n'
                f'  hermes:\n'
                f'    tags: [{name}]\n'
                f'---\n'
                f'\n'
                f'# {name.title()}\n\n'
                f'Instructions for {name}.\n',
                encoding="utf-8",
            )

        imported = adapter.import_hermes_skills()
        assert len(imported) == 3

        # All should be in registry
        all_skills = registry.list_skills()
        assert len(all_skills) == 3


# -----------------------------------------------------------------------
# Scenario: Skill lifecycle management
# -----------------------------------------------------------------------

class TestSkillLifecycle:
    """End-to-end: skill creation → active → deprecated."""

    def test_full_lifecycle(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        # Create
        skill = registry.register_skill(
            name="lifecycle-test",
            tier1_metadata="Testing lifecycle",
        )
        assert skill.lifecycle == SkillLifecycle.DRAFT

        # Activate
        registry.update_skill(skill.id, {"lifecycle": SkillLifecycle.ACTIVE})
        skill = registry.get_skill(skill.id, tier=3)
        assert skill is not None
        assert skill.lifecycle == SkillLifecycle.ACTIVE

        # Use and track
        for _ in range(5):
            tracker.record_outcome(
                Outcome(skill_id=skill.id, success=True, latency_ms=100, tokens_used=50)
            )

        # Version bump
        v2 = registry.version_skill(skill.id)
        assert v2 is not None
        assert v2.version == 2

        # Deprecate
        registry.update_skill(skill.id, {"lifecycle": SkillLifecycle.DEPRECATED})
        skill = registry.get_skill(skill.id, tier=3)
        assert skill is not None
        assert skill.lifecycle == SkillLifecycle.DEPRECATED


# -----------------------------------------------------------------------
# Scenario: Sticky skills and progressive loading
# -----------------------------------------------------------------------

class TestProgressiveLoading:
    """End-to-end: progressive loading with sticky skills."""

    def test_sticky_skills(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        # Register several skills
        for i in range(5):
            s = registry.register_skill(
                name=f"skill-{i}",
                tier1_metadata=f"Skill number {i}",
                tags=["test"],
            )
            registry.update_skill(s.id, {"lifecycle": SkillLifecycle.ACTIVE})

        # Record more usage for skill-0
        for _ in range(20):
            tracker.record_outcome(
                Outcome(skill_id="skill-0", success=True, latency_ms=10, tokens_used=5)
            )

        loader = ProgressiveLoader(registry, tracker)
        sticky = loader.get_sticky_skills(limit=3)
        assert len(sticky) == 3

    def test_search_and_load(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        registry.register_skill(
            name="email-sender",
            tier1_metadata="Sends emails",
            tier2_core="Full email sending instructions",
            tags=["email"],
        )
        registry.register_skill(
            name="sms-sender",
            tier1_metadata="Sends SMS messages",
            tier2_core="Full SMS sending instructions",
            tags=["sms"],
        )

        loader = ProgressiveLoader(registry, tracker)
        results = loader.load_skill("email", tier=2, routing="relevance")
        assert len(results) >= 1
        assert any("email" in s.name for s in results)


# -----------------------------------------------------------------------
# Scenario: Empty/edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases across the full system."""

    def test_import_from_empty_dir(
        self,
        adapter: HermesSkillForgeAdapter,
    ) -> None:
        imported = adapter.import_hermes_skills()
        assert imported == []

    def test_import_from_nonexistent_dir(
        self,
        registry: SkillRegistry,
    ) -> None:
        adapter = HermesSkillForgeAdapter(
            skillforge=registry, hermes_skills_dir="/nonexistent/path"
        )
        imported = adapter.import_hermes_skills()
        assert imported == []

    def test_export_nonexistent_skill(
        self,
        adapter: HermesSkillForgeAdapter,
    ) -> None:
        result = adapter.export_skill_to_hermes("nonexistent-id")
        assert result is None

    def test_search_empty_registry(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
    ) -> None:
        loader = ProgressiveLoader(registry, tracker)
        results = loader.load_skill("anything")
        assert results == []
