"""Tests for SkillRegistry (skillforge.core.registry)."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from skillforge.core.registry import Skill, SkillLifecycle, SkillRegistry


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Return a temporary SQLite database path."""
    return tmp_path / "test_skills.db"


@pytest.fixture
def registry(tmp_db: Path) -> Iterator[SkillRegistry]:
    """Create a fresh SkillRegistry backed by a temp DB."""
    reg = SkillRegistry(db_path=tmp_db)
    yield reg
    reg.close()


# -----------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------

class TestRegisterSkill:
    """Tests for SkillRegistry.register_skill."""

    def test_register_returns_skill(self, registry: SkillRegistry) -> None:
        skill = registry.register_skill(
            name="test-skill",
            tier1_metadata="A test skill.",
        )
        assert isinstance(skill, Skill)
        assert skill.name == "test-skill"
        assert skill.tier1_metadata == "A test skill."
        assert skill.version == 1
        assert skill.lifecycle == SkillLifecycle.DRAFT

    def test_register_with_custom_id(self, registry: SkillRegistry) -> None:
        skill = registry.register_skill(
            name="custom-id",
            tier1_metadata="Custom.",
            skill_id="my-custom-id",
        )
        assert skill.id == "my-custom-id"

    def test_register_duplicate_raises(self, registry: SkillRegistry) -> None:
        registry.register_skill(
            name="dup",
            tier1_metadata="dup",
            skill_id="dup-id",
        )
        with pytest.raises(ValueError, match="already exists"):
            registry.register_skill(
                name="dup2",
                tier1_metadata="dup2",
                skill_id="dup-id",
            )

    def test_register_default_tags_and_resources(self, registry: SkillRegistry) -> None:
        skill = registry.register_skill(name="bare", tier1_metadata="bare")
        assert skill.tags == []
        assert skill.tier3_resources == []

    def test_register_with_tags_and_resources(self, registry: SkillRegistry) -> None:
        skill = registry.register_skill(
            name="tagged",
            tier1_metadata="tagged",
            tags=["a", "b"],
            tier3_resources=["file1.md", "file2.md"],
        )
        assert "a" in skill.tags
        assert "b" in skill.tags
        assert len(skill.tier3_resources) == 2

    def test_register_generates_uuid(self, registry: SkillRegistry) -> None:
        skill = registry.register_skill(name="uuid", tier1_metadata="uuid")
        assert len(skill.id) > 0
        assert "-" in skill.id  # UUID format


# -----------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------

class TestGetSkill:
    """Tests for SkillRegistry.get_skill."""

    def test_get_existing(self, registry: SkillRegistry) -> None:
        created = registry.register_skill(name="findme", tier1_metadata="here")
        found = registry.get_skill(created.id, tier=1)
        assert found is not None
        assert found.name == "findme"

    def test_get_nonexistent_returns_none(self, registry: SkillRegistry) -> None:
        assert registry.get_skill("nonexistent") is None

    def test_tier1_hides_core_and_resources(self, registry: SkillRegistry) -> None:
        skill = registry.register_skill(
            name="t",
            tier1_metadata="meta",
            tier2_core="core content",
            tier3_resources=["r1"],
        )
        t1 = registry.get_skill(skill.id, tier=1)
        assert t1 is not None
        assert t1.tier2_core == ""
        assert t1.tier3_resources == []

    def test_tier2_shows_core_hides_resources(self, registry: SkillRegistry) -> None:
        skill = registry.register_skill(
            name="t2",
            tier1_metadata="meta",
            tier2_core="core content",
            tier3_resources=["r1"],
        )
        t2 = registry.get_skill(skill.id, tier=2)
        assert t2 is not None
        assert t2.tier2_core == "core content"
        assert t2.tier3_resources == []

    def test_tier3_shows_everything(self, registry: SkillRegistry) -> None:
        skill = registry.register_skill(
            name="t3",
            tier1_metadata="meta",
            tier2_core="core",
            tier3_resources=["r1", "r2"],
        )
        t3 = registry.get_skill(skill.id, tier=3)
        assert t3 is not None
        assert t3.tier2_core == "core"
        assert len(t3.tier3_resources) == 2


# -----------------------------------------------------------------------
# Update
# -----------------------------------------------------------------------

class TestUpdateSkill:
    """Tests for SkillRegistry.update_skill."""

    def test_update_name(self, registry: SkillRegistry) -> None:
        skill = registry.register_skill(name="old", tier1_metadata="meta")
        updated = registry.update_skill(skill.id, {"name": "new"})
        assert updated is not None
        assert updated.name == "new"

    def test_update_tags(self, registry: SkillRegistry) -> None:
        skill = registry.register_skill(name="tag", tier1_metadata="meta")
        updated = registry.update_skill(skill.id, {"tags": ["x", "y"]})
        assert updated is not None
        assert set(updated.tags) == {"x", "y"}

    def test_update_lifecycle(self, registry: SkillRegistry) -> None:
        skill = registry.register_skill(name="lc", tier1_metadata="meta")
        updated = registry.update_skill(
            skill.id, {"lifecycle": SkillLifecycle.ACTIVE}
        )
        assert updated is not None
        assert updated.lifecycle == SkillLifecycle.ACTIVE

    def test_update_nonexistent_returns_none(self, registry: SkillRegistry) -> None:
        result = registry.update_skill("nope", {"name": "x"})
        assert result is None

    def test_update_ignored_keys(self, registry: SkillRegistry) -> None:
        skill = registry.register_skill(name="ig", tier1_metadata="meta")
        updated = registry.update_skill(skill.id, {"id": "new-id", "name": "ok"})
        assert updated is not None
        assert updated.id == skill.id  # id cannot be changed
        assert updated.name == "ok"

    def test_update_updates_timestamp(self, registry: SkillRegistry) -> None:
        skill = registry.register_skill(name="ts", tier1_metadata="meta")
        original_ts = skill.updated_at
        import time
        time.sleep(0.01)
        updated = registry.update_skill(skill.id, {"name": "ts2"})
        assert updated is not None
        assert updated.updated_at >= original_ts


# -----------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------

class TestSearchSkills:
    """Tests for SkillRegistry.search_skills."""

    def test_search_by_name(self, registry: SkillRegistry) -> None:
        registry.register_skill(name="email-helper", tier1_metadata="helps with email")
        registry.register_skill(name="web-scraper", tier1_metadata="scrapes web")
        results = registry.search_skills("email")
        assert len(results) >= 1
        assert any("email" in s.name for s in results)

    def test_search_by_metadata(self, registry: SkillRegistry) -> None:
        registry.register_skill(name="foo", tier1_metadata="does twitter things")
        results = registry.search_skills("twitter")
        assert len(results) >= 1

    def test_search_no_results(self, registry: SkillRegistry) -> None:
        registry.register_skill(name="foo", tier1_metadata="bar")
        results = registry.search_skills("zzzznonexistent")
        assert len(results) == 0

    def test_search_limit(self, registry: SkillRegistry) -> None:
        for i in range(10):
            registry.register_skill(
                name=f"skill-{i}", tier1_metadata="common keyword"
            )
        results = registry.search_skills("common", limit=3)
        assert len(results) == 3


# -----------------------------------------------------------------------
# List
# -----------------------------------------------------------------------

class TestListSkills:
    """Tests for SkillRegistry.list_skills."""

    def test_list_all(self, registry: SkillRegistry) -> None:
        registry.register_skill(name="a", tier1_metadata="a")
        registry.register_skill(name="b", tier1_metadata="b")
        registry.register_skill(name="c", tier1_metadata="c")
        all_skills = registry.list_skills()
        assert len(all_skills) == 3

    def test_list_filter_by_lifecycle(self, registry: SkillRegistry) -> None:
        s1 = registry.register_skill(name="draft", tier1_metadata="d")
        s2 = registry.register_skill(name="active", tier1_metadata="a")
        registry.update_skill(s2.id, {"lifecycle": SkillLifecycle.ACTIVE})
        active = registry.list_skills(lifecycle=SkillLifecycle.ACTIVE)
        assert len(active) == 1
        assert active[0].name == "active"

    def test_list_limit_and_offset(self, registry: SkillRegistry) -> None:
        for i in range(5):
            registry.register_skill(name=f"s{i}", tier1_metadata=str(i))
        page = registry.list_skills(limit=2, offset=1)
        assert len(page) == 2

    def test_list_sorted_by_q_value(self, registry: SkillRegistry) -> None:
        s1 = registry.register_skill(name="low", tier1_metadata="l")
        s2 = registry.register_skill(name="high", tier1_metadata="h")
        registry.update_skill(s1.id, {"q_value": 0.1})
        registry.update_skill(s2.id, {"q_value": 0.9})
        skills = registry.list_skills()
        assert skills[0].name == "high"


# -----------------------------------------------------------------------
# Versioning
# -----------------------------------------------------------------------

class TestVersionSkill:
    """Tests for SkillRegistry.version_skill."""

    def test_version_bumps(self, registry: SkillRegistry) -> None:
        skill = registry.register_skill(name="v", tier1_metadata="v")
        assert skill.version == 1
        v2 = registry.version_skill(skill.id)
        assert v2 is not None
        assert v2.version == 2
        v3 = registry.version_skill(skill.id)
        assert v3 is not None
        assert v3.version == 3

    def test_version_nonexistent_returns_none(self, registry: SkillRegistry) -> None:
        assert registry.version_skill("nope") is None


# -----------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------

class TestPersistence:
    """Verify data survives close/reopen."""

    def test_data_persists(self, tmp_db: Path) -> None:
        reg = SkillRegistry(db_path=tmp_db)
        reg.register_skill(name="persist", tier1_metadata="test", skill_id="p1")
        reg.close()

        reg2 = SkillRegistry(db_path=tmp_db)
        found = reg2.get_skill("p1")
        assert found is not None
        assert found.name == "persist"
        reg2.close()
