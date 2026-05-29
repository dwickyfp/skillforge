"""
Multi-agent skill sharing with isolated workspaces.

Provides a shared skill pool that enables teams of agents to collaborate
on skill development while maintaining fine-grained access control.  Each
workspace isolates a team's skill set, and the sync mechanism propagates
Q-value updates across shared skills.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes & enums
# ---------------------------------------------------------------------------


class AccessLevel(str, Enum):
    """Access levels for shared skills."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


@dataclass
class AgentAccess:
    """Record of an agent's access to a shared skill within a workspace.

    Attributes
    ----------
    agent_id : str
        The agent that has been granted access.
    workspace_id : str
        The workspace this access record belongs to.
    skill_id : str
        The shared skill.
    access_level : AccessLevel
        Permission level (read, write, admin).
    shared_at : str
        ISO-8601 UTC timestamp of when the share was created.
    shared_by : str
        The agent that created this access grant.
    """

    agent_id: str
    workspace_id: str
    skill_id: str
    access_level: AccessLevel
    shared_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    shared_by: str = ""


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class SkillRegistryProto(Protocol):
    """Minimal registry interface required by :class:`SharedSkillPool`."""

    def get_skill(self, skill_id: str, tier: int = 1) -> Any: ...  # pragma: no cover
    def update_skill(self, skill_id: str, updates: dict[str, Any]) -> Any: ...  # pragma: no cover
    def list_skills(self, **kwargs: Any) -> list[Any]: ...  # pragma: no cover


class GraphProto(Protocol):
    """Minimal dependency-graph interface required by :class:`SharedSkillPool`."""

    def get_skills(self) -> list[str]: ...  # pragma: no cover
    def propagate_q_update(self, skill_id: str, delta: float, gamma: float = 0.9) -> dict[str, float]: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# SharedSkillPool
# ---------------------------------------------------------------------------


class SharedSkillPool:
    """Multi-agent skill sharing with isolated workspaces.

    Workspaces group agents into teams that share access to a curated set
    of skills.  Each skill share carries an access-level (read/write/admin)
    and an audit trail.

    Parameters
    ----------
    registry : SkillRegistryProto
        Skill registry for reading and updating skill definitions.
    graph : GraphProto
        Skill dependency graph for propagation during sync.
    db_path : str | Path | None
        SQLite database path.  Defaults to ``~/.skillforge/multi_agent.db``.
    """

    def __init__(
        self,
        registry: SkillRegistryProto,
        graph: GraphProto,
        db_path: str | Path | None = None,
    ) -> None:
        self._registry = registry
        self._graph = graph
        self._db_path = str(
            db_path or Path.home() / ".skillforge" / "multi_agent.db"
        )
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create the ``agent_access`` and ``shared_skills`` tables."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS agent_access (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id      TEXT NOT NULL,
                workspace_id  TEXT NOT NULL,
                skill_id      TEXT NOT NULL,
                access_level  TEXT NOT NULL DEFAULT 'read',
                shared_at     TEXT NOT NULL,
                shared_by     TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_agent_access_agent
                ON agent_access(agent_id);
            CREATE INDEX IF NOT EXISTS idx_agent_access_workspace
                ON agent_access(workspace_id);
            CREATE INDEX IF NOT EXISTS idx_agent_access_skill
                ON agent_access(skill_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_access_unique
                ON agent_access(agent_id, workspace_id, skill_id);

            CREATE TABLE IF NOT EXISTS shared_skills (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id  TEXT NOT NULL,
                skill_id      TEXT NOT NULL,
                owner_agent   TEXT NOT NULL DEFAULT '',
                shared_at     TEXT NOT NULL,
                q_value_cache REAL NOT NULL DEFAULT 0.5
            );
            CREATE INDEX IF NOT EXISTS idx_shared_skills_workspace
                ON shared_skills(workspace_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_shared_skills_unique
                ON shared_skills(workspace_id, skill_id);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_workspace(
        self, workspace_id: str, agent_ids: list[str]
    ) -> dict[str, Any]:
        """Create an isolated skill workspace for a team of agents.

        All listed agents are granted **admin** access to the new workspace
        (but no skills are shared yet — use :meth:`share_skill` for that).

        Parameters
        ----------
        workspace_id : str
            Unique identifier for the workspace.
        agent_ids : list[str]
            Agent identifiers that belong to this workspace.

        Returns
        -------
        dict[str, Any]
            Summary including ``workspace_id``, ``agent_count``, and
            ``created_at``.
        """
        now = datetime.now(timezone.utc).isoformat()
        # Insert placeholder rows to register agents in the workspace
        for agent_id in agent_ids:
            # We use a sentinel skill_id '__workspace__' to mark membership
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO agent_access "
                    "(agent_id, workspace_id, skill_id, access_level, shared_at, shared_by) "
                    "VALUES (?, ?, '__workspace__', 'admin', ?, 'system')",
                    (agent_id, workspace_id, now),
                )
            except sqlite3.IntegrityError:
                pass  # already a member

        self._conn.commit()
        logger.info(
            "Created workspace '%s' with %d agents", workspace_id, len(agent_ids)
        )
        return {
            "workspace_id": workspace_id,
            "agent_count": len(agent_ids),
            "agents": list(agent_ids),
            "created_at": now,
        }

    def share_skill(
        self,
        skill_id: str,
        from_agent: str,
        to_agents: list[str],
        access_level: AccessLevel = AccessLevel.READ,
        workspace_id: str | None = None,
    ) -> list[AgentAccess]:
        """Share a skill across agents with access control.

        Parameters
        ----------
        skill_id : str
            The skill to share.
        from_agent : str
            The agent performing the share.
        to_agents : list[str]
            Agents to grant access to.
        access_level : AccessLevel
            Permission level (default ``READ``).
        workspace_id : str | None
            Workspace context.  If ``None``, a default workspace
            ``"__default__"`` is used.

        Returns
        -------
        list[AgentAccess]
            Access records that were created.
        """
        ws = workspace_id or "__default__"
        now = datetime.now(timezone.utc).isoformat()
        records: list[AgentAccess] = []

        # Ensure skill is in the shared_skills table for this workspace
        skill = self._registry.get_skill(skill_id, tier=1)
        q_cache = 0.5
        if skill and hasattr(skill, "q_value"):
            q_cache = skill.q_value

        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO shared_skills "
                "(workspace_id, skill_id, owner_agent, shared_at, q_value_cache) "
                "VALUES (?, ?, ?, ?, ?)",
                (ws, skill_id, from_agent, now, q_cache),
            )
        except sqlite3.IntegrityError:
            pass

        for agent_id in to_agents:
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO agent_access "
                    "(agent_id, workspace_id, skill_id, access_level, shared_at, shared_by) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (agent_id, ws, skill_id, access_level.value, now, from_agent),
                )
                records.append(
                    AgentAccess(
                        agent_id=agent_id,
                        workspace_id=ws,
                        skill_id=skill_id,
                        access_level=access_level,
                        shared_at=now,
                        shared_by=from_agent,
                    )
                )
            except sqlite3.IntegrityError as exc:
                logger.warning(
                    "Could not grant '%s' access to agent '%s': %s",
                    skill_id,
                    agent_id,
                    exc,
                )

        self._conn.commit()
        logger.info(
            "Shared skill '%s' from '%s' to %d agents in workspace '%s'",
            skill_id,
            from_agent,
            len(records),
            ws,
        )
        return records

    def get_shared_skills(self, agent_id: str) -> list[dict[str, Any]]:
        """Get all skills shared with a given agent.

        Parameters
        ----------
        agent_id : str
            The agent to query.

        Returns
        -------
        list[dict[str, Any]]
            Each dict contains ``skill_id``, ``workspace_id``,
            ``access_level``, ``shared_at``, and ``shared_by``.
        """
        rows = self._conn.execute(
            "SELECT * FROM agent_access "
            "WHERE agent_id = ? AND skill_id != '__workspace__' "
            "ORDER BY shared_at DESC",
            (agent_id,),
        ).fetchall()
        return [
            {
                "skill_id": row["skill_id"],
                "workspace_id": row["workspace_id"],
                "access_level": row["access_level"],
                "shared_at": row["shared_at"],
                "shared_by": row["shared_by"],
            }
            for row in rows
        ]

    def get_workspace_stats(self, workspace_id: str) -> dict[str, Any]:
        """Get usage statistics across all agents in a workspace.

        Parameters
        ----------
        workspace_id : str
            The workspace to query.

        Returns
        -------
        dict[str, Any]
            Includes ``workspace_id``, ``agent_count``, ``skill_count``,
            ``skills`` (list of skill dicts), and ``agents`` (list of agent
            IDs).
        """
        # Agents in workspace
        agent_rows = self._conn.execute(
            "SELECT DISTINCT agent_id FROM agent_access WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchall()
        agent_ids = [r["agent_id"] for r in agent_rows]

        # Shared skills in workspace
        skill_rows = self._conn.execute(
            "SELECT * FROM shared_skills WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchall()
        skills = [
            {
                "skill_id": row["skill_id"],
                "owner_agent": row["owner_agent"],
                "shared_at": row["shared_at"],
                "q_value_cache": row["q_value_cache"],
            }
            for row in skill_rows
        ]

        return {
            "workspace_id": workspace_id,
            "agent_count": len(agent_ids),
            "skill_count": len(skills),
            "agents": agent_ids,
            "skills": skills,
        }

    def sync_workspaces(self) -> dict[str, int]:
        """Propagate Q-value updates across all shared skills.

        For each shared skill, reads the current Q-value from the registry
        and updates the cached value.  If the Q-value changed, triggers
        graph-based propagation.

        Returns
        -------
        dict[str, int]
            Mapping of ``workspace_id`` → number of skills synced.
        """
        rows = self._conn.execute("SELECT * FROM shared_skills").fetchall()
        sync_counts: dict[str, int] = {}

        for row in rows:
            ws_id: str = row["workspace_id"]
            sid: str = row["skill_id"]
            cached_q: float = row["q_value_cache"]

            # Get live Q-value from registry
            skill = self._registry.get_skill(sid, tier=1)
            if skill is None:
                continue
            live_q: float = skill.q_value if hasattr(skill, "q_value") else 0.5

            if abs(live_q - cached_q) > 1e-6:
                delta = live_q - cached_q
                # Propagate through graph
                try:
                    self._graph.propagate_q_update(sid, delta)
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Graph propagation failed for '%s': %s", sid, exc
                    )

                # Update cache
                self._conn.execute(
                    "UPDATE shared_skills SET q_value_cache = ? "
                    "WHERE workspace_id = ? AND skill_id = ?",
                    (live_q, ws_id, sid),
                )
                sync_counts[ws_id] = sync_counts.get(ws_id, 0) + 1

        self._conn.commit()
        logger.info("Synced workspaces: %s", sync_counts)
        return sync_counts

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
