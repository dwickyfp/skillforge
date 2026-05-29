"""Async wrapper for ProgressiveLoader.

Provides non-blocking async versions of :class:`ProgressiveLoader`
methods by delegating synchronous calls to a thread pool via
:func:`asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.loader import ProgressiveLoader
from ..core.registry import Skill


class AsyncProgressiveLoader:
    """Async-friendly wrapper around :class:`ProgressiveLoader`.

    All synchronous operations are executed in a thread pool
    via :func:`asyncio.to_thread`.

    Parameters
    ----------
    registry : AsyncSkillRegistry
        Async wrapper around the skill registry.
    tracker : AsyncQValueTracker
        Async wrapper around the outcome tracker.
    """

    def __init__(
        self,
        registry: Any,
        tracker: Any,
    ) -> None:
        self._sync = ProgressiveLoader(registry.sync, tracker.sync)

    @property
    def sync(self) -> ProgressiveLoader:
        """Access the underlying synchronous loader.

        Returns
        -------
        ProgressiveLoader
            The wrapped loader instance.
        """
        return self._sync

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    async def load_skill(
        self,
        query: str,
        tier: int = 1,
        routing: str = "q_value",
        limit: int = 5,
    ) -> list[Skill]:
        """Load skills matching a query (async).

        Parameters
        ----------
        query : str
            Natural-language query or keywords.
        tier : int
            Maximum detail level (1–3).
        routing : str
            Ranking strategy.
        limit : int
            Maximum skills to return.

        Returns
        -------
        list[Skill]
            Ranked and trimmed list of skills.
        """
        return await asyncio.to_thread(
            self._sync.load_skill,
            query,
            tier=tier,
            routing=routing,
            limit=limit,
        )

    async def get_sticky_skills(self, limit: int = 5) -> list[Skill]:
        """Return the top skills to keep pre-loaded (async).

        Parameters
        ----------
        limit : int
            Maximum skills to return.

        Returns
        -------
        list[Skill]
            Top skills ranked by a composite score.
        """
        return await asyncio.to_thread(
            self._sync.get_sticky_skills, limit=limit
        )
