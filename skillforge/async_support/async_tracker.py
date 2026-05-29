"""Async wrapper for QValueTracker.

Provides non-blocking async versions of all :class:`QValueTracker`
methods by delegating synchronous calls to a thread pool via
:func:`asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..core.tracker import Outcome, QValueTracker


class AsyncQValueTracker:
    """Async-friendly wrapper around :class:`QValueTracker`.

    All synchronous database operations are executed in a thread pool
    via :func:`asyncio.to_thread` so they don't block the event loop.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the SQLite database file.
        Defaults to ``~/.skillforge/tracker.db``.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._sync = QValueTracker(db_path=db_path)

    @property
    def sync(self) -> QValueTracker:
        """Access the underlying synchronous tracker.

        Returns
        -------
        QValueTracker
            The wrapped tracker instance.
        """
        return self._sync

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    async def record_outcome(self, outcome: Outcome) -> None:
        """Persist a single execution outcome (async).

        Parameters
        ----------
        outcome : Outcome
            The outcome to record.
        """
        await asyncio.to_thread(self._sync.record_outcome, outcome)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get_q_value(self, skill_id: str) -> float:
        """Return current Q-value for a skill (async).

        Parameters
        ----------
        skill_id : str
            Skill to query.

        Returns
        -------
        float
            Current Q-value (0.5 if unknown).
        """
        return await asyncio.to_thread(self._sync.get_q_value, skill_id)

    async def get_success_rate(self, skill_id: str) -> float:
        """Return rolling success rate for a skill (async).

        Parameters
        ----------
        skill_id : str
            Skill to query.

        Returns
        -------
        float
            Success rate (0.0–1.0).
        """
        return await asyncio.to_thread(
            self._sync.get_success_rate, skill_id
        )

    async def get_stats(self, skill_id: str) -> dict[str, Any]:
        """Return aggregate statistics for a skill (async).

        Parameters
        ----------
        skill_id : str
            Skill to query.

        Returns
        -------
        dict[str, Any]
            Statistics dictionary.
        """
        return await asyncio.to_thread(self._sync.get_stats, skill_id)

    async def get_failures(
        self, skill_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Retrieve recent failed outcomes (async).

        Parameters
        ----------
        skill_id : str
            Skill to query.
        limit : int
            Maximum failures to return.

        Returns
        -------
        list[dict[str, Any]]
            List of failure dictionaries.
        """
        return await asyncio.to_thread(
            self._sync.get_failures, skill_id, limit=limit
        )

    async def get_usage_count(self, skill_id: str) -> int:
        """Return total number of recorded outcomes (async).

        Parameters
        ----------
        skill_id : str
            Skill to query.

        Returns
        -------
        int
            Total outcome count.
        """
        return await asyncio.to_thread(
            self._sync.get_usage_count, skill_id
        )

    # ------------------------------------------------------------------
    # TD(λ) Update
    # ------------------------------------------------------------------

    async def td_lambda_update(
        self,
        skill_id: str,
        reward: float,
        alpha: float = 0.1,
        gamma: float = 0.9,
        lambda_: float = 0.8,
    ) -> float:
        """Apply a TD(λ) update (async).

        Parameters
        ----------
        skill_id : str
            Skill to update.
        reward : float
            Observed reward.
        alpha : float
            Learning rate.
        gamma : float
            Discount factor.
        lambda_ : float
            Trace decay.

        Returns
        -------
        float
            The new Q-value.
        """
        return await asyncio.to_thread(
            self._sync.td_lambda_update,
            skill_id,
            reward,
            alpha=alpha,
            gamma=gamma,
            lambda_=lambda_,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the database connection (async)."""
        await asyncio.to_thread(self._sync.close)
