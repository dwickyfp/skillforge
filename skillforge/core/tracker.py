"""Outcome tracking and Q-value updates for skills via TD(λ) learning."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Outcome:
    """A single execution outcome for a skill."""

    skill_id: str
    success: bool
    latency_ms: float
    tokens_used: int
    user_feedback: float | None = None  # 0-5 scale, optional


class QValueTracker:
    """Tracks skill outcomes and maintains Q-values via TD(λ) updates.

    Parameters
    ----------
    db_path : str | Path | None
        SQLite database path. Defaults to ``~/.skillforge/tracker.db``.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or Path.home() / ".skillforge" / "tracker.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS outcomes (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_id      TEXT NOT NULL,
                success       INTEGER NOT NULL,
                latency_ms    REAL NOT NULL,
                tokens_used   INTEGER NOT NULL,
                user_feedback REAL,
                recorded_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_outcomes_skill ON outcomes(skill_id);

            CREATE TABLE IF NOT EXISTS q_values (
                skill_id      TEXT PRIMARY KEY,
                q_value       REAL NOT NULL DEFAULT 0.5,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                total_latency REAL NOT NULL DEFAULT 0.0,
                total_tokens  INTEGER NOT NULL DEFAULT 0,
                updated_at    TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_outcome(self, outcome: Outcome) -> None:
        """Persist a single execution outcome and update rolling stats."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO outcomes (skill_id, success, latency_ms, tokens_used, "
            "user_feedback, recorded_at) VALUES (?,?,?,?,?,?)",
            (
                outcome.skill_id,
                int(outcome.success),
                outcome.latency_ms,
                outcome.tokens_used,
                outcome.user_feedback,
                now,
            ),
        )
        # Upsert aggregate row
        self._conn.execute(
            """
            INSERT INTO q_values (skill_id, q_value, success_count, failure_count,
                total_latency, total_tokens, updated_at)
            VALUES (:sid, 0.5, :sc, :fc, :lat, :tok, :now)
            ON CONFLICT(skill_id) DO UPDATE SET
                success_count = success_count + :sc,
                failure_count = failure_count + :fc,
                total_latency = total_latency + :lat,
                total_tokens  = total_tokens + :tok,
                updated_at    = :now
            """,
            {
                "sid": outcome.skill_id,
                "sc": int(outcome.success),
                "fc": int(not outcome.success),
                "lat": outcome.latency_ms,
                "tok": outcome.tokens_used,
                "now": now,
            },
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_q_value(self, skill_id: str) -> float:
        """Return current Q-value for *skill_id* (0.5 if unknown)."""
        row = self._conn.execute(
            "SELECT q_value FROM q_values WHERE skill_id = ?", (skill_id,)
        ).fetchone()
        return row["q_value"] if row else 0.5

    def get_success_rate(self, skill_id: str) -> float:
        """Return rolling success rate (0-1) for *skill_id*."""
        row = self._conn.execute(
            "SELECT success_count, failure_count FROM q_values WHERE skill_id = ?",
            (skill_id,),
        ).fetchone()
        if row is None:
            return 0.5
        total = row["success_count"] + row["failure_count"]
        if total == 0:
            return 0.5
        return row["success_count"] / total

    def get_stats(self, skill_id: str) -> dict[str, Any]:
        """Return a dict of aggregate statistics for *skill_id*."""
        row = self._conn.execute(
            "SELECT * FROM q_values WHERE skill_id = ?", (skill_id,)
        ).fetchone()
        if row is None:
            return {
                "skill_id": skill_id,
                "q_value": 0.5,
                "success_rate": 0.5,
                "total_outcomes": 0,
                "avg_latency_ms": 0.0,
                "avg_tokens": 0,
            }
        total = row["success_count"] + row["failure_count"]
        return {
            "skill_id": skill_id,
            "q_value": row["q_value"],
            "success_rate": row["success_count"] / total if total else 0.5,
            "total_outcomes": total,
            "avg_latency_ms": row["total_latency"] / total if total else 0.0,
            "avg_tokens": row["total_tokens"] // total if total else 0,
        }

    # ------------------------------------------------------------------
    # TD(λ) Update
    # ------------------------------------------------------------------

    def td_lambda_update(
        self,
        skill_id: str,
        reward: float,
        alpha: float = 0.1,
        gamma: float = 0.9,
        lambda_: float = 0.8,
    ) -> float:
        """Apply a TD(λ) update to the Q-value of *skill_id*.

        This uses an eligibility-trace style update over the history of
        recent outcomes.  For a single-step (no history) this reduces to a
        standard Q-learning update: ``Q ← Q + α * (reward − Q)``.

        Parameters
        ----------
        skill_id : str
            Skill to update.
        reward : float
            Observed scalar reward (typically 0-1).
        alpha : float
            Learning rate.
        gamma : float
            Discount factor.
        lambda_ : float
            Trace decay (0 = TD(0), 1 = Monte-Carlo-like).

        Returns
        -------
        float
            The new Q-value.
        """
        current_q = self.get_q_value(skill_id)
        now = datetime.now(timezone.utc).isoformat()

        # Gather recent outcomes as a trace
        rows = self._conn.execute(
            "SELECT success, latency_ms, tokens_used, user_feedback "
            "FROM outcomes WHERE skill_id = ? ORDER BY id DESC LIMIT 10",
            (skill_id,),
        ).fetchall()

        # Build the TD error with eligibility trace
        td_error = reward - current_q
        trace_weight = 1.0
        cumulative_update = 0.0

        for i, row in enumerate(rows):
            # Weight each historical step by the trace decay
            cumulative_update += trace_weight * td_error
            trace_weight *= gamma * lambda_
            if trace_weight < 1e-6:
                break

        new_q = current_q + alpha * cumulative_update
        new_q = max(0.0, min(1.0, new_q))  # clamp to [0, 1]

        self._conn.execute(
            "UPDATE q_values SET q_value = ?, updated_at = ? WHERE skill_id = ?",
            (new_q, now, skill_id),
        )
        self._conn.commit()
        return new_q

    # ------------------------------------------------------------------
    # Failure retrieval
    # ------------------------------------------------------------------

    def get_failures(self, skill_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve recent failed outcomes for *skill_id*.

        Parameters
        ----------
        skill_id : str
            Skill to query.
        limit : int
            Maximum number of failures to return (default 10).

        Returns
        -------
        list[dict[str, Any]]
            List of failure dicts with keys: skill_id, success, latency_ms,
            tokens_used, user_feedback, recorded_at.
        """
        rows = self._conn.execute(
            "SELECT skill_id, success, latency_ms, tokens_used, "
            "user_feedback, recorded_at "
            "FROM outcomes WHERE skill_id = ? AND success = 0 "
            "ORDER BY id DESC LIMIT ?",
            (skill_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_usage_count(self, skill_id: str) -> int:
        """Return total number of recorded outcomes for *skill_id*."""
        row = self._conn.execute(
            "SELECT success_count, failure_count FROM q_values WHERE skill_id = ?",
            (skill_id,),
        ).fetchone()
        if row is None:
            return 0
        return row["success_count"] + row["failure_count"]

    def get_last_used(self, skill_id: str):
        """Return the most recent outcome datetime for *skill_id*, or None."""
        row = self._conn.execute(
            "SELECT recorded_at FROM outcomes WHERE skill_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (skill_id,),
        ).fetchone()
        if row is None:
            return None
        ts_str = row["recorded_at"]
        try:
            return datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            return None

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


# Backwards-compatible alias expected by the top-level __init__
EffectivenessTracker = QValueTracker
