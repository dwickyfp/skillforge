"""Version manager — semantic versioning, history tracking, rollback, and diff.

Provides comprehensive version management for skills including:
- Semantic version parsing, comparison, and bumping
- Full version history with snapshots and change logs
- Rollback to any previous version
- Content diff between versions
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# Semantic version pattern
_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<pre>[0-9A-Za-z\-.]+))?(?:\+(?P<build>[0-9A-Za-z\-.]+))?$"
)


class ChangeType(Enum):
    """Type of change that triggered a version bump."""

    MAJOR = "major"  # Breaking changes
    MINOR = "minor"  # New features
    PATCH = "patch"  # Bug fixes
    PRERELEASE = "pre"
    BUILD = "build"


@dataclass
class SemanticVersion:
    """A parsed semantic version.

    Supports parsing, comparison, and bumping per semver 2.0.0 spec.

    Attributes:
        major: Major version number.
        minor: Minor version number.
        patch: Patch version number.
        pre: Pre-release identifier (e.g. ``"alpha.1"``).
        build: Build metadata (e.g. ``"abc123"``).
    """

    major: int = 0
    minor: int = 0
    patch: int = 0
    pre: str = ""
    build: str = ""

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @classmethod
    def parse(cls, version_string: str) -> "SemanticVersion":
        """Parse a semantic version string.

        Parameters
        ----------
        version_string : str
            Version string (e.g. ``"1.2.3-beta.1+build.42"``).

        Returns
        -------
        SemanticVersion
            Parsed version object.

        Raises
        ------
        ValueError
            If the string is not a valid semantic version.
        """
        m = _SEMVER_RE.match(version_string.strip())
        if m is None:
            raise ValueError(
                f"Invalid semantic version: '{version_string}'. "
                "Expected format: MAJOR.MINOR.PATCH[-pre][+build]"
            )
        return cls(
            major=int(m.group("major")),
            minor=int(m.group("minor")),
            patch=int(m.group("patch")),
            pre=m.group("pre") or "",
            build=m.group("build") or "",
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Return the canonical version string.

        Returns
        -------
        str
            Version string in ``MAJOR.MINOR.PATCH[-pre][+build]`` format.
        """
        v = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre:
            v += f"-{self.pre}"
        if self.build:
            v += f"+{self.build}"
        return v

    def __repr__(self) -> str:
        return f"SemanticVersion('{self}')"

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _cmp_key(self) -> tuple[int, int, int, list[str]]:
        """Build a comparison key (pre-release excluded from numeric comparison)."""
        pre_parts = (
            [int(p) if p.isdigit() else p for p in self.pre.split(".")]
            if self.pre
            else []
        )
        # A pre-release version has lower precedence than the normal version
        # We signal this by adding a sentinel: no-pre < empty-string < actual-pre
        pre_key: list[str] = []
        if self.pre:
            pre_key = [str(p) for p in pre_parts]
        return (self.major, self.minor, self.patch, pre_key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return self._cmp_key() == other._cmp_key()

    def __lt__(self, other: "SemanticVersion") -> bool:
        return self._cmp_key() < other._cmp_key()

    def __le__(self, other: "SemanticVersion") -> bool:
        return self._cmp_key() <= other._cmp_key()

    def __gt__(self, other: "SemanticVersion") -> bool:
        return self._cmp_key() > other._cmp_key()

    def __ge__(self, other: "SemanticVersion") -> bool:
        return self._cmp_key() >= other._cmp_key()

    def __hash__(self) -> int:
        return hash(self._cmp_key())

    # ------------------------------------------------------------------
    # Bumping
    # ------------------------------------------------------------------

    def bump(self, change_type: ChangeType) -> "SemanticVersion":
        """Create a new version by bumping the specified component.

        Parameters
        ----------
        change_type : ChangeType
            Which component to bump.

        Returns
        -------
        SemanticVersion
            The bumped version.
        """
        if change_type == ChangeType.MAJOR:
            return SemanticVersion(major=self.major + 1, minor=0, patch=0)
        elif change_type == ChangeType.MINOR:
            return SemanticVersion(major=self.major, minor=self.minor + 1, patch=0)
        elif change_type == ChangeType.PATCH:
            return SemanticVersion(major=self.major, minor=self.minor, patch=self.patch + 1)
        else:
            return SemanticVersion(
                major=self.major, minor=self.minor, patch=self.patch
            )


@dataclass
class VersionRecord:
    """A version history record.

    Attributes:
        id: Unique record identifier.
        skill_id: The skill this version belongs to.
        version: Semantic version string.
        change_type: What kind of change triggered this version.
        changelog: Human-readable description of changes.
        snapshot: Full skill state at this version (JSON-serializable).
        author: Who made the change.
        created_at: When this version was created.
    """

    id: str
    skill_id: str
    version: str
    change_type: ChangeType = ChangeType.PATCH
    changelog: str = ""
    snapshot: dict[str, Any] = field(default_factory=dict)
    author: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class VersionDiff:
    """Diff between two versions.

    Attributes:
        from_version: Source version.
        to_version: Target version.
        added: Keys added in the new version.
        removed: Keys removed in the new version.
        modified: Keys that changed value.
        summary: Human-readable diff summary.
    """

    from_version: str = ""
    to_version: str = ""
    added: dict[str, Any] = field(default_factory=dict)
    removed: dict[str, Any] = field(default_factory=dict)
    modified: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary: str = ""

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes.

        Returns
        -------
        bool
            True if there are additions, removals, or modifications.
        """
        return bool(self.added or self.removed or self.modified)


class VersionManager:
    """Manages semantic versioning and version history for skills.

    Stores version history in SQLite, supports rollback to any previous
            Path to the SQLite database.
        Defaults to ``~/.skillforge/versions.db``.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or Path.home() / ".skillforge" / "versions.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create version tables if they do not exist."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS version_history (
                id          TEXT PRIMARY KEY,
                skill_id    TEXT NOT NULL,
                version     TEXT NOT NULL,
                change_type TEXT NOT NULL DEFAULT 'patch',
                changelog   TEXT NOT NULL DEFAULT '',
                snapshot    TEXT NOT NULL DEFAULT '{}',
                author      TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vh_skill
                ON version_history(skill_id);
            CREATE INDEX IF NOT EXISTS idx_vh_version
                ON version_history(version);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Version recording
    # ------------------------------------------------------------------

    def record_version(
        self,
        skill_id: str,
        version: str,
        change_type: ChangeType = ChangeType.PATCH,
        changelog: str = "",
        snapshot: dict[str, Any] | None = None,
        author: str = "",
    ) -> VersionRecord:
        """Record a new version in the history.

        Parameters
        ----------
        skill_id : str
            The skill being versioned.
        version : str
            Semantic version string.
        change_type : ChangeType
            Type of change.
        changelog : str
            Description of changes.
        snapshot : dict[str, Any] | None
            Full skill state at this version.
        author : str
            Who made the change.

        Returns
        -------
        VersionRecord
            The created version record.

        Raises
        ------
        ValueError
            If the version string is invalid.
        """
        # Validate version string
        sv = SemanticVersion.parse(version)

        record_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        record = VersionRecord(
            id=record_id,
            skill_id=skill_id,
            version=str(sv),
            change_type=change_type,
            changelog=changelog,
            snapshot=snapshot or {},
            author=author,
            created_at=now,
        )

        self._conn.execute(
            "INSERT INTO version_history VALUES "
            "(:id,:skill_id,:version,:change_type,:changelog,:snapshot,"
            ":author,:created_at)",
            {
                "id": record.id,
                "skill_id": record.skill_id,
                "version": record.version,
                "change_type": record.change_type.value,
                "changelog": record.changelog,
                "snapshot": json.dumps(record.snapshot),
                "author": record.author,
                "created_at": record.created_at.isoformat(),
            },
        )
        self._conn.commit()
        return record

    # ------------------------------------------------------------------
    # Bumping
    # ------------------------------------------------------------------

    def bump_version(
        self,
        skill_id: str,
        change_type: ChangeType,
        changelog: str = "",
        snapshot: dict[str, Any] | None = None,
        author: str = "",
    ) -> VersionRecord:
        """Auto-bump the version for a skill based on change type.

        Looks up the latest version, increments the appropriate component,
        and records the new version.

        Parameters
        ----------
        skill_id : str
            Skill to bump.
        change_type : ChangeType
            Type of change (MAJOR, MINOR, PATCH).
        changelog : str
            Description of changes.
        snapshot : dict[str, Any] | None
            Full skill state.
        author : str
            Who made the change.

        Returns
        -------
        VersionRecord
            New version record.
        """
        latest = self.get_latest_version(skill_id)
        if latest is None:
            current = SemanticVersion(0, 0, 0)
        else:
            current = SemanticVersion.parse(latest.version)

        new_version = current.bump(change_type)

        return self.record_version(
            skill_id=skill_id,
            version=str(new_version),
            change_type=change_type,
            changelog=changelog,
            snapshot=snapshot,
            author=author,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_latest_version(self, skill_id: str) -> VersionRecord | None:
        """Get the latest version record for a skill.

        Parameters
        ----------
        skill_id : str
            Skill to query.

        Returns
        -------
        VersionRecord | None
            The latest version record, or *None* if no versions exist.
        """
        row = self._conn.execute(
            "SELECT * FROM version_history WHERE skill_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (skill_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_version(
        self, skill_id: str, version: str
    ) -> VersionRecord | None:
        """Get a specific version record.

        Parameters
        ----------
        skill_id : str
            Skill to query.
        version : str
            Version string to look up.

        Returns
        -------
        VersionRecord | None
            The version record, or *None* if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM version_history WHERE skill_id = ? AND version = ?",
            (skill_id, version),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_history(
        self, skill_id: str, limit: int = 100
    ) -> list[VersionRecord]:
        """Get the full version history for a skill.

        Parameters
        ----------
        skill_id : str
            Skill to query.
        limit : int
            Maximum records to return.

        Returns
        -------
        list[VersionRecord]
            Version records, most recent first.
        """
        rows = self._conn.execute(
            "SELECT * FROM version_history WHERE skill_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (skill_id, limit),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(
        self, skill_id: str, target_version: str
    ) -> VersionRecord | None:
        """Rollback a skill to a previous version.

        Creates a *new* version record with the snapshot from the
        target version (doesn't delete history).

        Parameters
        ----------
        skill_id : str
            Skill to rollback.
        target_version : str
            Version to restore.

        Returns
        -------
        VersionRecord | None
            The new rollback version record, or *None* if the target
            version was not found.
        """
        target = self.get_version(skill_id, target_version)
        if target is None:
            return None

        # Create a new version record with the old snapshot
        return self.record_version(
            skill_id=skill_id,
            version=target.version,
            change_type=ChangeType.PATCH,
            changelog=f"Rollback to version {target_version}",
            snapshot=target.snapshot,
            author="system:rollback",
        )

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def diff(
        self, skill_id: str, from_version: str, to_version: str
    ) -> VersionDiff:
        """Compute a diff between two versions.

        Compares the snapshot dictionaries and reports additions,
        removals, and modifications.

        Parameters
        ----------
        skill_id : str
            Skill to diff.
        from_version : str
            Source version.
        to_version : str
            Target version.

        Returns
        -------
        VersionDiff
            Detailed diff report.

        Raises
        ------
        ValueError
            If either version is not found.
        """
        from_rec = self.get_version(skill_id, from_version)
        to_rec = self.get_version(skill_id, to_version)

        if from_rec is None:
            raise ValueError(f"Version '{from_version}' not found for skill '{skill_id}'")
        if to_rec is None:
            raise ValueError(f"Version '{to_version}' not found for skill '{skill_id}'")

        from_snap = from_rec.snapshot
        to_snap = to_rec.snapshot

        added: dict[str, Any] = {}
        removed: dict[str, Any] = {}
        modified: dict[str, dict[str, Any]] = {}

        all_keys = set(from_snap.keys()) | set(to_snap.keys())
        for key in sorted(all_keys):
            if key not in from_snap:
                added[key] = to_snap[key]
            elif key not in to_snap:
                removed[key] = from_snap[key]
            elif from_snap[key] != to_snap[key]:
                modified[key] = {"from": from_snap[key], "to": to_snap[key]}

        n_added = len(added)
        n_removed = len(removed)
        n_modified = len(modified)

        parts: list[str] = []
        if n_added:
            parts.append(f"{n_added} additions")
        if n_removed:
            parts.append(f"{n_removed} removals")
        if n_modified:
            parts.append(f"{n_modified} modifications")
        if not parts:
            parts.append("no changes")

        summary = (
            f"Diff from {from_version} to {to_version}: "
            + ", ".join(parts)
        )

        return VersionDiff(
            from_version=from_version,
            to_version=to_version,
            added=added,
            removed=removed,
            modified=modified,
            summary=summary,
        )

    def diff_latest(
        self, skill_id: str, against_version: str
    ) -> VersionDiff:
        """Diff the latest version against a specified version.

        Parameters
        ----------
        skill_id : str
            Skill to diff.
        against_version : str
            Version to compare the latest against.

        Returns
        -------
        VersionDiff
            Diff report.

        Raises
        ------
        ValueError
            If no versions exist or the specified version is not found.
        """
        latest = self.get_latest_version(skill_id)
        if latest is None:
            raise ValueError(f"No versions found for skill '{skill_id}'")

        return self.diff(skill_id, against_version, latest.version)

    # ------------------------------------------------------------------
    # Changelog
    # ------------------------------------------------------------------

    def get_changelog(
        self, skill_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Generate a formatted changelog for a skill.

        Parameters
        ----------
        skill_id : str
            Skill to query.
        limit : int
            Maximum entries.

        Returns
        -------
        list[dict[str, Any]]
            Changelog entries with version, change_type, changelog, and
            timestamp.
        """
        records = self.get_history(skill_id, limit=limit)
        return [
            {
                "version": r.version,
                "change_type": r.change_type.value,
                "changelog": r.changelog,
                "author": r.author,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> VersionRecord:
        """Convert a database row to a VersionRecord."""
        d = dict(row)
        d["change_type"] = ChangeType(d["change_type"])
        d["snapshot"] = json.loads(d["snapshot"])
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        return VersionRecord(**d)

    def count(self, skill_id: str | None = None) -> int:
        """Count version records.

        Parameters
        ----------
        skill_id : str | None
            If given, count only versions for this skill.

        Returns
        -------
        int
            Number of version records.
        """
        if skill_id is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM version_history WHERE skill_id = ?",
                (skill_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM version_history"
            ).fetchone()
        return row["cnt"] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
