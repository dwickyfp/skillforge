"""Skill registry with SQLite backend for persistent skill management."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class SkillLifecycle(Enum):
    """Lifecycle stages a skill can be in."""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class Skill:
    """Represents a registered skill with metadata and performance signals."""

    id: str
    name: str
    version: int
    lifecycle: SkillLifecycle
    tier1_metadata: str  # ~30 tokens — quick summary for routing
    tier2_core: str  # core prompt / instructions
    tier3_resources: list[str]  # auxiliary files, examples, etc.
    q_value: float = 0.5  # 0-1 estimated quality
    success_rate: float = 0.5  # 0-1 rolling success rate
    usage_count: int = 0
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SkillRegistry:
    """Persistent skill registry backed by SQLite.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the SQLite database file.  Defaults to ``~/.skillforge/skills.db``.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or Path.home() / ".skillforge" / "skills.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS skills (
                id            TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                version       INTEGER NOT NULL DEFAULT 1,
                lifecycle     TEXT NOT NULL DEFAULT 'draft',
                tier1_metadata TEXT NOT NULL DEFAULT '',
                tier2_core    TEXT NOT NULL DEFAULT '',
                tier3_resources TEXT NOT NULL DEFAULT '[]',
                q_value       REAL NOT NULL DEFAULT 0.5,
                success_rate  REAL NOT NULL DEFAULT 0.5,
                usage_count   INTEGER NOT NULL DEFAULT 0,
                tags          TEXT NOT NULL DEFAULT '[]',
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_skills_lifecycle ON skills(lifecycle);
            CREATE INDEX IF NOT EXISTS idx_skills_q_value   ON skills(q_value);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_skill(row: sqlite3.Row) -> Skill:
        d = dict(row)
        d["lifecycle"] = SkillLifecycle(d["lifecycle"])
        d["tier3_resources"] = json.loads(d["tier3_resources"])
        d["tags"] = json.loads(d["tags"])
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        d["updated_at"] = datetime.fromisoformat(d["updated_at"])
        return Skill(**d)

    def _skill_to_params(self, s: Skill) -> dict[str, Any]:
        return {
            "id": s.id,
            "name": s.name,
            "version": s.version,
            "lifecycle": s.lifecycle.value,
            "tier1_metadata": s.tier1_metadata,
            "tier2_core": s.tier2_core,
            "tier3_resources": json.dumps(s.tier3_resources),
            "q_value": s.q_value,
            "success_rate": s.success_rate,
            "usage_count": s.usage_count,
            "tags": json.dumps(s.tags),
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat(),
        }

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register_skill(
        self,
        name: str,
        tier1_metadata: str,
        tier2_core: str = "",
        tier3_resources: list[str] | None = None,
        tags: list[str] | None = None,
        skill_id: str | None = None,
    ) -> Skill:
        """Register a new skill and return it.

        Raises
        ------
        ValueError
            If a skill with the given *skill_id* already exists.
        """
        sid = skill_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        skill = Skill(
            id=sid,
            name=name,
            version=1,
            lifecycle=SkillLifecycle.DRAFT,
            tier1_metadata=tier1_metadata,
            tier2_core=tier2_core,
            tier3_resources=tier3_resources or [],
            tags=tags or [],
            created_at=now,
            updated_at=now,
        )
        try:
            self._conn.execute(
                "INSERT INTO skills VALUES (:id,:name,:version,:lifecycle,"
                ":tier1_metadata,:tier2_core,:tier3_resources,:q_value,"
                ":success_rate,:usage_count,:tags,:created_at,:updated_at)",
                self._skill_to_params(skill),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Skill '{sid}' already exists") from exc
        return skill

    def get_skill(self, skill_id: str, tier: int = 1) -> Skill | None:
        """Retrieve a skill.  *tier* controls how much detail is returned.

        tier=1 → metadata only, tier=2 → core prompt included, tier=3 → full.
        """
        row = self._conn.execute(
            "SELECT * FROM skills WHERE id = ?", (skill_id,)
        ).fetchone()
        if row is None:
            return None
        skill = self._row_to_skill(row)
        if tier < 3:
            skill.tier3_resources = []
        if tier < 2:
            skill.tier2_core = ""
        return skill

    def update_skill(self, skill_id: str, updates: dict[str, Any]) -> Skill | None:
        """Apply partial updates to a skill.

        Accepted keys mirror :class:`Skill` fields.  Nested objects (tags,
        tier3_resources) should be passed as Python lists.
        """
        allowed = {
            "name",
            "version",
            "lifecycle",
            "tier1_metadata",
            "tier2_core",
            "tier3_resources",
            "q_value",
            "success_rate",
            "usage_count",
            "tags",
        }
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            return self.get_skill(skill_id, tier=3)

        # Serialise complex fields
        if "tags" in filtered:
            filtered["tags"] = json.dumps(filtered["tags"])
        if "tier3_resources" in filtered:
            filtered["tier3_resources"] = json.dumps(filtered["tier3_resources"])
        if "lifecycle" in filtered and isinstance(filtered["lifecycle"], SkillLifecycle):
            filtered["lifecycle"] = filtered["lifecycle"].value

        filtered["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{k} = :{k}" for k in filtered)
        filtered["id"] = skill_id
        cur = self._conn.execute(
            f"UPDATE skills SET {set_clause} WHERE id = :id", filtered
        )
        self._conn.commit()
        if cur.rowcount == 0:
            return None
        return self.get_skill(skill_id, tier=3)

    def list_skills(
        self,
        lifecycle: SkillLifecycle | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Skill]:
        """List skills with optional filters."""
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if lifecycle is not None:
            clauses.append("lifecycle = :lifecycle")
            params["lifecycle"] = lifecycle.value
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params["limit"] = limit
        params["offset"] = offset
        rows = self._conn.execute(
            f"SELECT * FROM skills{where} ORDER BY q_value DESC LIMIT :limit OFFSET :offset",
            params,
        ).fetchall()
        skills = [self._row_to_skill(r) for r in rows]
        # Post-filter by tags if requested (simpler than JSON SQL gymnastics)
        if tags:
            tag_set = set(tags)
            skills = [s for s in skills if tag_set.intersection(s.tags)]
        return skills

    def search_skills(self, query: str, limit: int = 20) -> list[Skill]:
        """Simple keyword search across name and tier1_metadata."""
        pattern = f"%{query}%"
        rows = self._conn.execute(
            "SELECT * FROM skills WHERE name LIKE :q OR tier1_metadata LIKE :q "
            "ORDER BY q_value DESC LIMIT :limit",
            {"q": pattern, "limit": limit},
        ).fetchall()
        return [self._row_to_skill(r) for r in rows]

    def version_skill(self, skill_id: str) -> Skill | None:
        """Create a new version of an existing skill (bumps version number)."""
        skill = self.get_skill(skill_id, tier=3)
        if skill is None:
            return None
        return self.update_skill(skill_id, {"version": skill.version + 1})

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
