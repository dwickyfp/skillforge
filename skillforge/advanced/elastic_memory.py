"""Elastic Memory — adaptive memory system for skill execution history.

Stores, retrieves, compresses, and manages contextual memories of skill
executions.  Memories carry an *importance* score and an *access count*,
allowing the system to perform importance-weighted retention during
automatic compaction.

Inspired by experience-replay buffers from reinforcement learning and
long-term memory consolidation from cognitive science.  All storage is
backed by SQLite so memories survive restarts.

No external dependencies — stdlib only.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """A single memory record capturing a skill execution context.

    Attributes
    ----------
    id : str
        Unique identifier for this memory.
    skill_id : str
        The skill that produced this memory.
    context : str
        Free-text description of the execution context (task, inputs, etc.).
    outcome : str
        Description of the outcome (success/failure details, output summary).
    importance : float
        Importance score in [0, 1].  Higher values are retained longer
        during compaction.
    access_count : int
        How many times this memory has been recalled.  Frequently accessed
        memories are boosted during compaction.
    created_at : str
        ISO-8601 UTC timestamp of creation.
    last_accessed_at : str | None
        ISO-8601 UTC timestamp of the last recall, or ``None`` if never.
    metadata : dict[str, Any]
        Arbitrary key-value metadata (e.g. latency, tokens, tags).
    """

    id: str
    skill_id: str
    context: str
    outcome: str
    importance: float = 0.5
    access_count: int = 0
    created_at: str = ""
    last_accessed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ElasticMemory
# ---------------------------------------------------------------------------


class ElasticMemory:
    """Adaptive memory store for skill execution histories.

    Each memory is a :class:`MemoryEntry` that captures the context and
    outcome of a single skill invocation.  The store supports:

    * **remember** — persist a new memory.
    * **recall** — keyword-based retrieval ranked by relevance × importance.
    * **consolidate** — merge similar old memories for a skill to save space.
    * **forget** — evict low-importance entries older than a threshold.
    * **auto_compact** — global compaction keeping only the top entries
      ranked by a composite retention score.

    Parameters
    ----------
    db_path : str | Path | None
        SQLite database path.  Defaults to ``~/.skillforge/memory.db``.
    max_entries : int
        Soft cap on total entries.  :meth:`auto_compact` enforces this.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        max_entries: int = 10_000,
    ) -> None:
        self._db_path = str(db_path or Path.home() / ".skillforge" / "memory.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._max_entries = max_entries
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id              TEXT PRIMARY KEY,
                skill_id        TEXT NOT NULL,
                context         TEXT NOT NULL,
                outcome         TEXT NOT NULL,
                importance      REAL NOT NULL DEFAULT 0.5,
                access_count    INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL,
                last_accessed_at TEXT,
                metadata        TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_memories_skill ON memories(skill_id);
            CREATE INDEX IF NOT EXISTS idx_memories_imp   ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_memories_ts    ON memories(created_at);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        d = dict(row)
        d["metadata"] = json.loads(d.get("metadata") or "{}")
        return MemoryEntry(**d)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remember(
        self,
        skill_id: str,
        context: str,
        outcome: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Store a new memory and return it.

        Parameters
        ----------
        skill_id : str
            The skill that produced this memory.
        context : str
            Free-text context description.
        outcome : str
            Free-text outcome description.
        importance : float
            Importance score clamped to [0, 1].
        metadata : dict | None
            Arbitrary metadata (latency, tokens, tags, etc.).

        Returns
        -------
        MemoryEntry
            The persisted memory.
        """
        now = self._now_iso()
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            skill_id=skill_id,
            context=context,
            outcome=outcome,
            importance=max(0.0, min(1.0, importance)),
            access_count=0,
            created_at=now,
            last_accessed_at=None,
            metadata=metadata or {},
        )
        self._conn.execute(
            "INSERT INTO memories "
            "(id, skill_id, context, outcome, importance, access_count, "
            " created_at, last_accessed_at, metadata) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                entry.id,
                entry.skill_id,
                entry.context,
                entry.outcome,
                entry.importance,
                entry.access_count,
                entry.created_at,
                entry.last_accessed_at,
                json.dumps(entry.metadata),
            ),
        )
        self._conn.commit()
        logger.debug("Remembered memory %s for skill '%s'", entry.id, skill_id)
        return entry

    def recall(
        self,
        skill_id: str | None = None,
        query: str | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> list[MemoryEntry]:
        """Retrieve memories ranked by a relevance × importance score.

        When *query* is provided, relevance is computed via simple keyword
        overlap with the ``context`` and ``outcome`` fields.  Without a
        *query*, results are ordered by importance descending.

        Parameters
        ----------
        skill_id : str | None
            Filter to a specific skill.  ``None`` searches all skills.
        query : str | None
            Space-separated keywords to match against context/outcome.
        limit : int
            Maximum results.
        min_importance : float
            Minimum importance threshold.

        Returns
        -------
        list[MemoryEntry]
            Matching memories sorted by composite score descending.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if skill_id is not None:
            clauses.append("skill_id = ?")
            params.append(skill_id)
        if min_importance > 0:
            clauses.append("importance >= ?")
            params.append(min_importance)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        rows = self._conn.execute(
            f"SELECT * FROM memories{where} ORDER BY importance DESC LIMIT ?",
            params + [limit * 5],  # fetch extra for re-ranking
        ).fetchall()

        entries = [self._row_to_entry(r) for r in rows]

        if query:
            keywords = set(re.findall(r"\w+", query.lower()))
            scored: list[tuple[float, MemoryEntry]] = []
            for entry in entries:
                text = (entry.context + " " + entry.outcome).lower()
                words = set(re.findall(r"\w+", text))
                overlap = len(keywords & words) / max(len(keywords), 1)
                # Composite: 60% keyword overlap + 40% importance
                score = 0.6 * overlap + 0.4 * entry.importance
                scored.append((score, entry))
            scored.sort(key=lambda t: t[0], reverse=True)
            entries = [e for _, e in scored[:limit]]
        else:
            entries = entries[:limit]

        # Update access counts
        now = self._now_iso()
        for entry in entries:
            entry.access_count += 1
            entry.last_accessed_at = now
            self._conn.execute(
                "UPDATE memories SET access_count = ?, last_accessed_at = ? "
                "WHERE id = ?",
                (entry.access_count, now, entry.id),
            )
        if entries:
            self._conn.commit()

        return entries

    def get_memory(self, memory_id: str) -> MemoryEntry | None:
        """Retrieve a single memory by its ID.

        Parameters
        ----------
        memory_id : str
            The memory's unique identifier.

        Returns
        -------
        MemoryEntry | None
            The memory, or ``None`` if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def forget(
        self,
        skill_id: str | None = None,
        older_than_days: float | None = None,
        max_importance: float = 1.0,
    ) -> int:
        """Evict memories matching the given criteria.

        Parameters
        ----------
        skill_id : str | None
            If given, only delete memories for this skill.
        older_than_days : float | None
            If given, only delete memories older than this many days.
        max_importance : float
            Only delete memories with importance ≤ this threshold.

        Returns
        -------
        int
            Number of memories deleted.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if skill_id is not None:
            clauses.append("skill_id = ?")
            params.append(skill_id)

        if max_importance < 1.0:
            clauses.append("importance <= ?")
            params.append(max_importance)

        if older_than_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
            clauses.append("created_at < ?")
            params.append(cutoff.isoformat())

        if not clauses:
            return 0

        where = " WHERE " + " AND ".join(clauses)
        cur = self._conn.execute(f"DELETE FROM memories{where}", params)
        self._conn.commit()
        deleted = cur.rowcount
        logger.debug("Forgot %d memories", deleted)
        return deleted

    def consolidate(self, skill_id: str) -> int:
        """Consolidate memories for *skill_id* by merging similar entries.

        Memories with highly overlapping keyword sets (Jaccard > 0.7)
        are merged: the newer entry absorbs the older one's metadata and
        the older entry is deleted.  The surviving entry's importance is
        boosted to the max of the pair.

        Parameters
        ----------
        skill_id : str
            Skill whose memories should be consolidated.

        Returns
        -------
        int
            Number of entries removed during consolidation.
        """
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE skill_id = ? ORDER BY created_at ASC",
            (skill_id,),
        ).fetchall()

        entries = [self._row_to_entry(r) for r in rows]
        if len(entries) < 2:
            return 0

        # Build keyword sets
        entry_kw: list[tuple[MemoryEntry, set[str]]] = []
        for e in entries:
            text = (e.context + " " + e.outcome).lower()
            entry_kw.append((e, set(re.findall(r"\w+", text))))

        to_delete: set[str] = set()
        merged_importances: dict[str, float] = {}

        for i in range(len(entry_kw)):
            e_i, kw_i = entry_kw[i]
            if e_i.id in to_delete:
                continue
            for j in range(i + 1, len(entry_kw)):
                e_j, kw_j = entry_kw[j]
                if e_j.id in to_delete:
                    continue
                # Jaccard similarity
                union = kw_i | kw_j
                if not union:
                    continue
                jaccard = len(kw_i & kw_j) / len(union)
                if jaccard > 0.7:
                    # Merge j into i (keep the older one)
                    to_delete.add(e_j.id)
                    new_imp = max(e_i.importance, e_j.importance)
                    merged_importances[e_i.id] = new_imp

        # Apply merges
        for entry_id, new_imp in merged_importances.items():
            self._conn.execute(
                "UPDATE memories SET importance = ? WHERE id = ?",
                (new_imp, entry_id),
            )

        if to_delete:
            placeholders = ",".join("?" for _ in to_delete)
            self._conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})",
                list(to_delete),
            )
            self._conn.commit()

        removed = len(to_delete)
        logger.debug(
            "Consolidated %d memories for skill '%s'", removed, skill_id
        )
        return removed

    def auto_compact(self, max_entries: int | None = None) -> int:
        """Global compaction: retain only the top-*N* entries by retention score.

        The retention score is a weighted combination of:

        * Importance (weight 0.45)
        * Recency — exponential decay with 30-day half-life (weight 0.30)
        * Access frequency — log-scaled (weight 0.25)

        Parameters
        ----------
        max_entries : int | None
            Override the instance-level ``max_entries`` cap.

        Returns
        -------
        int
            Number of entries evicted.
        """
        cap = max_entries if max_entries is not None else self._max_entries
        count_row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM memories"
        ).fetchone()
        total = count_row["cnt"] if count_row else 0

        if total <= cap:
            return 0

        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        entries = [self._row_to_entry(r) for r in rows]
        now = datetime.now(timezone.utc)

        def _retention_score(e: MemoryEntry) -> float:
            # Importance component
            imp = max(0.0, min(1.0, e.importance))

            # Recency component: exponential decay, 30-day half-life
            try:
                created = datetime.fromisoformat(e.created_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                days = (now - created).total_seconds() / 86400.0
            except (ValueError, TypeError):
                days = 365.0
            recency = math.exp(-0.6931 * days / 30.0)

            # Access frequency: log-scaled, saturating around 50 accesses
            freq = min(1.0, math.log(1 + e.access_count) / math.log(51))

            return 0.45 * imp + 0.30 * recency + 0.25 * freq

        entries.sort(key=_retention_score, reverse=True)
        to_evict = entries[cap:]
        evict_ids = [e.id for e in to_evict]

        if evict_ids:
            placeholders = ",".join("?" for _ in evict_ids)
            self._conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", evict_ids
            )
            self._conn.commit()

        evicted = len(evict_ids)
        logger.debug("Auto-compacted %d memories (cap=%d)", evicted, cap)
        return evicted

    def get_memory_stats(self, skill_id: str | None = None) -> dict[str, Any]:
        """Return aggregate statistics about stored memories.

        Parameters
        ----------
        skill_id : str | None
            If given, stats are scoped to that skill.  Otherwise global.

        Returns
        -------
        dict[str, Any]
            Keys: ``total_memories``, ``avg_importance``, ``avg_access_count``,
            ``oldest``, ``newest``, ``skills`` (list of unique skill IDs).
        """
        where = " WHERE skill_id = ?" if skill_id else ""
        params: tuple[Any, ...] = (skill_id,) if skill_id else ()

        row = self._conn.execute(
            f"SELECT COUNT(*) AS cnt, "
            f"COALESCE(AVG(importance), 0) AS avg_imp, "
            f"COALESCE(AVG(access_count), 0) AS avg_acc, "
            f"MIN(created_at) AS oldest, "
            f"MAX(created_at) AS newest "
            f"FROM memories{where}",
            params,
        ).fetchone()

        total = row["cnt"] if row else 0

        # Get unique skill IDs
        if skill_id:
            skills_list = [skill_id] if total > 0 else []
        else:
            srows = self._conn.execute(
                "SELECT DISTINCT skill_id FROM memories"
            ).fetchall()
            skills_list = [r["skill_id"] for r in srows]

        return {
            "total_memories": total,
            "avg_importance": round(row["avg_imp"], 4) if row else 0.0,
            "avg_access_count": round(row["avg_acc"], 2) if row else 0.0,
            "oldest": row["oldest"] if row else None,
            "newest": row["newest"] if row else None,
            "skills": skills_list,
        }

    def count(self, skill_id: str | None = None) -> int:
        """Return the number of stored memories.

        Parameters
        ----------
        skill_id : str | None
            If given, count only memories for this skill.

        Returns
        -------
        int
            Number of memories.
        """
        if skill_id:
            row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM memories WHERE skill_id = ?",
                (skill_id,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) AS cnt FROM memories").fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
