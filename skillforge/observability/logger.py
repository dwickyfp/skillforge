"""Structured logger — JSON-structured log entries with SQLite persistence.

Provides structured, machine-parseable log entries that can be queried,
filtered, and analysed.  Designed to complement standard ``logging``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class LogLevel(Enum):
    """Structured log severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """A single structured log entry.

    Attributes:
        id: Unique entry identifier.
        level: Severity level.
        message: Human-readable log message.
        component: The component that generated the log.
        skill_id: Related skill (may be empty).
        trace_id: Optional trace ID for correlation.
        span_id: Optional span ID for correlation.
        metadata: Arbitrary key-value metadata.
        timestamp: When the log entry was created.
    """

    id: str
    level: LogLevel
    message: str
    component: str = ""
    skill_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entry to a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "id": self.id,
            "level": self.level.value,
            "message": self.message,
            "component": self.component,
            "skill_id": self.skill_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_json(self) -> str:
        """Serialize the entry to a JSON string.

        Returns
        -------
        str
            JSON string.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)


class StructuredLogger:
    """Structured logger with SQLite persistence.

    Provides structured, filterable log entries that are stored in a
    database for later analysis and correlation with traces.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the SQLite database.
        Defaults to ``~/.skillforge/logs.db``.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or Path.home() / ".skillforge" / "logs.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create log tables if they do not exist."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS log_entries (
                id          TEXT PRIMARY KEY,
                level       TEXT NOT NULL,
                message     TEXT NOT NULL,
                component   TEXT NOT NULL DEFAULT '',
                skill_id    TEXT NOT NULL DEFAULT '',
                trace_id    TEXT NOT NULL DEFAULT '',
                span_id     TEXT NOT NULL DEFAULT '',
                metadata    TEXT NOT NULL DEFAULT '{}',
                timestamp   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_log_level
                ON log_entries(level);
            CREATE INDEX IF NOT EXISTS idx_log_component
                ON log_entries(component);
            CREATE INDEX IF NOT EXISTS idx_log_skill
                ON log_entries(skill_id);
            CREATE INDEX IF NOT EXISTS idx_log_trace
                ON log_entries(trace_id);
            CREATE INDEX IF NOT EXISTS idx_log_ts
                ON log_entries(timestamp);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _write(self, entry: LogEntry) -> LogEntry:
        """Write a log entry to the database.

        Parameters
        ----------
        entry : LogEntry
            Entry to persist.

        Returns
        -------
        LogEntry
            The persisted entry.
        """
        self._conn.execute(
            "INSERT INTO log_entries VALUES "
            "(:id,:level,:message,:component,:skill_id,:trace_id,"
            ":span_id,:metadata,:timestamp)",
            {
                "id": entry.id,
                "level": entry.level.value,
                "message": entry.message,
                "component": entry.component,
                "skill_id": entry.skill_id,
                "trace_id": entry.trace_id,
                "span_id": entry.span_id,
                "metadata": json.dumps(entry.metadata),
                "timestamp": entry.timestamp.isoformat(),
            },
        )
        self._conn.commit()
        return entry

    def log(
        self,
        level: LogLevel,
        message: str,
        component: str = "",
        skill_id: str = "",
        trace_id: str = "",
        span_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> LogEntry:
        """Record a log entry at the specified level.

        Parameters
        ----------
        level : LogLevel
            Severity level.
        message : str
            Log message.
        component : str
            Source component.
        skill_id : str
            Related skill ID.
        trace_id : str
            Trace ID for correlation.
        span_id : str
            Span ID for correlation.
        metadata : dict[str, Any] | None
            Additional metadata.

        Returns
        -------
        LogEntry
            The persisted log entry.
        """
        entry = LogEntry(
            id=str(uuid.uuid4()),
            level=level,
            message=message,
            component=component,
            skill_id=skill_id,
            trace_id=trace_id,
            span_id=span_id,
            metadata=metadata or {},
        )
        return self._write(entry)

    def debug(
        self, message: str, component: str = "", skill_id: str = "",
        metadata: dict[str, Any] | None = None, **kwargs: Any,
    ) -> LogEntry:
        """Log a debug message."""
        return self.log(LogLevel.DEBUG, message, component, skill_id, metadata=metadata, **kwargs)

    def info(
        self, message: str, component: str = "", skill_id: str = "",
        metadata: dict[str, Any] | None = None, **kwargs: Any,
    ) -> LogEntry:
        """Log an info message."""
        return self.log(LogLevel.INFO, message, component, skill_id, metadata=metadata, **kwargs)

    def warning(
        self, message: str, component: str = "", skill_id: str = "",
        metadata: dict[str, Any] | None = None, **kwargs: Any,
    ) -> LogEntry:
        """Log a warning message."""
        return self.log(LogLevel.WARNING, message, component, skill_id, metadata=metadata, **kwargs)

    def error(
        self, message: str, component: str = "", skill_id: str = "",
        metadata: dict[str, Any] | None = None, **kwargs: Any,
    ) -> LogEntry:
        """Log an error message."""
        return self.log(LogLevel.ERROR, message, component, skill_id, metadata=metadata, **kwargs)

    def critical(
        self, message: str, component: str = "", skill_id: str = "",
        metadata: dict[str, Any] | None = None, **kwargs: Any,
    ) -> LogEntry:
        """Log a critical message."""
        return self.log(LogLevel.CRITICAL, message, component, skill_id, metadata=metadata, **kwargs)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_entries(
        self,
        level: LogLevel | None = None,
        component: str | None = None,
        skill_id: str | None = None,
        trace_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[LogEntry]:
        """Query log entries with optional filters.

        Parameters
        ----------
        level : LogLevel | None
            Filter by severity level.
        component : str | None
            Filter by component.
        skill_id : str | None
            Filter by skill ID.
        trace_id : str | None
            Filter by trace ID.
        since : datetime | None
            Only return entries after this time.
        limit : int
            Maximum entries to return.

        Returns
        -------
        list[LogEntry]
            Matching entries, most recent first.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if level is not None:
            clauses.append("level = ?")
            params.append(level.value)
        if component is not None:
            clauses.append("component = ?")
            params.append(component)
        if skill_id is not None:
            clauses.append("skill_id = ?")
            params.append(skill_id)
        if trace_id is not None:
            clauses.append("trace_id = ?")
            params.append(trace_id)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT * FROM log_entries{where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()

        entries: list[LogEntry] = []
        for r in rows:
            d = dict(r)
            d["level"] = LogLevel(d["level"])
            d["metadata"] = json.loads(d["metadata"])
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
            entries.append(LogEntry(**d))
        return entries

    def get_level_counts(self) -> dict[str, int]:
        """Return the count of entries at each log level.

        Returns
        -------
        dict[str, int]
            Level name → count.
        """
        rows = self._conn.execute(
            "SELECT level, COUNT(*) as cnt FROM log_entries GROUP BY level"
        ).fetchall()
        return {r["level"]: r["cnt"] for r in rows}

    def search(
        self, query: str, limit: int = 50
    ) -> list[LogEntry]:
        """Full-text search across log messages.

        Parameters
        ----------
        query : str
            Search string.
        limit : int
            Maximum results.

        Returns
        -------
        list[LogEntry]
            Matching entries, most recent first.
        """
        pattern = f"%{query}%"
        rows = self._conn.execute(
            "SELECT * FROM log_entries WHERE message LIKE ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (pattern, limit),
        ).fetchall()

        entries: list[LogEntry] = []
        for r in rows:
            d = dict(r)
            d["level"] = LogLevel(d["level"])
            d["metadata"] = json.loads(d["metadata"])
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
            entries.append(LogEntry(**d))
        return entries

    def count(
        self,
        level: LogLevel | None = None,
        component: str | None = None,
        skill_id: str | None = None,
    ) -> int:
        """Count log entries matching filters.

        Parameters
        ----------
        level : LogLevel | None
            Filter by level.
        component : str | None
            Filter by component.
        skill_id : str | None
            Filter by skill ID.

        Returns
        -------
        int
            Number of matching entries.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if level is not None:
            clauses.append("level = ?")
            params.append(level.value)
        if component is not None:
            clauses.append("component = ?")
            params.append(component)
        if skill_id is not None:
            clauses.append("skill_id = ?")
            params.append(skill_id)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        row = self._conn.execute(
            f"SELECT COUNT(*) as cnt FROM log_entries{where}",
            params,
        ).fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
