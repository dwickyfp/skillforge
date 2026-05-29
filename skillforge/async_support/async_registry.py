"""Async wrapper for SkillRegistry.

Provides non-blocking async versions of all :class:`SkillRegistry`
methods by delegating synchronous calls to a thread pool via
:func:`asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..core.registry import Skill, SkillLifecycle, SkillRegistry


class AsyncSkillRegistry:
    """Async-friendly wrapper around :class:`SkillRegistry`.

    All synchronous database operations are executed in a thread pool
    via :func:`asyncio.to_thread` so they don't block the event loop.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the SQLite database file.
        Defaults to ``~/.skillforge/skills.db``.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._sync = SkillRegistry(db_path=db_path)

    @property
    def sync(self) -> SkillRegistry:
        """Access the underlying synchronous registry.

        Returns
        -------
        SkillRegistry
            The wrapped registry instance.
        """
        return self._sync

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def register_skill(
        self,
        name: str,
        tier1_metadata: str,
        tier2_core: str = "",
        tier3_resources: list[str] | None = None,
        tags: list[str] | None = None,
        skill_id: str | None = None,
    ) -> Skill:
        """Register a new skill (async).

        Parameters
        ----------
        name : str
            Skill name.
        tier1_metadata : str
            Short routing summary.
        tier2_core : str
            Core prompt / instructions.
        tier3_resources : list[str] | None
            Auxiliary resource list.
        tags : list[str] | None
            Tags for filtering.
        skill_id : str | None
            Explicit ID.  UUID is generated when *None*.

        Returns
        -------
        Skill
            The newly registered skill.
        """
        return await asyncio.to_thread(
            self._sync.register_skill,
            name=name,
            tier1_metadata=tier1_metadata,
            tier2_core=tier2_core,
            tier3_resources=tier3_resources,
            tags=tags,
            skill_id=skill_id,
        )

    async def get_skill(
        self, skill_id: str, tier: int = 1
    ) -> Skill | None:
        """Retrieve a skill by ID (async).

        Parameters
        ----------
        skill_id : str
            Skill identifier.
        tier : int
            Detail level (1–3).

        Returns
        -------
        Skill | None
            The skill, or *None* if not found.
        """
        return await asyncio.to_thread(
            self._sync.get_skill, skill_id, tier=tier
        )

    async def update_skill(
        self, skill_id: str, updates: dict[str, Any]
    ) -> Skill | None:
        """Apply partial updates to a skill (async).

        Parameters
        ----------
        skill_id : str
            Skill to update.
        updates : dict[str, Any]
            Fields to change.

        Returns
        -------
        Skill | None
            Updated skill, or *None* if not found.
        """
        return await asyncio.to_thread(
            self._sync.update_skill, skill_id, updates
        )

    async def list_skills(
        self,
        lifecycle: SkillLifecycle | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Skill]:
        """List skills with optional filters (async).

        Parameters
        ----------
        lifecycle : SkillLifecycle | None
            Filter by lifecycle stage.
        tags : list[str] | None
            Filter by tags (OR logic).
        limit : int
            Maximum results.
        offset : int
            Pagination offset.

        Returns
        -------
        list[Skill]
            Matching skills sorted by Q-value.
        """
        return await asyncio.to_thread(
            self._sync.list_skills,
            lifecycle=lifecycle,
            tags=tags,
            limit=limit,
            offset=offset,
        )

    async def search_skills(
        self, query: str, limit: int = 20
    ) -> list[Skill]:
        """Keyword search (async).

        Parameters
        ----------
        query : str
            Search string.
        limit : int
            Maximum results.

        Returns
        -------
        list[Skill]
            Matching skills.
        """
        return await asyncio.to_thread(
            self._sync.search_skills, query, limit=limit
        )

    async def version_skill(self, skill_id: str) -> Skill | None:
        """Bump the version number of a skill (async).

        Parameters
        ----------
        skill_id : str
            Skill to version.

        Returns
        -------
        Skill | None
            Updated skill, or *None* if not found.
        """
        return await asyncio.to_thread(self._sync.version_skill, skill_id)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the database connection (async)."""
        await asyncio.to_thread(self._sync.close)
