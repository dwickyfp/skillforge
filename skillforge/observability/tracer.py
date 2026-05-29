"""Distributed tracing for skill executions.

Provides OpenTelemetry-inspired span-based tracing that tracks skill
invocations, their latencies, and parent-child relationships — all
persisted in SQLite for post-mortem analysis.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generator


class SpanStatus(Enum):
    """Status of a tracing span."""

    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


@dataclass
class TraceContext:
    """Lightweight context propagated across spans in a single trace.

    Attributes:
        trace_id: Globally unique trace identifier.
        parent_span_id: The span that initiated the current span.
    """

    trace_id: str
    parent_span_id: str | None = None


@dataclass
class Span:
    """A single unit of work in a distributed trace.

    Attributes:
        span_id: Unique span identifier.
        trace_id: Parent trace ID.
        parent_span_id: Optional parent span ID.
        operation: Human-readable operation name.
        skill_id: The skill involved (may be empty for non-skill ops).
        status: Span status.
        start_time: When the span started.
        end_time: When the span finished.
        duration_ms: Duration in milliseconds.
        attributes: Arbitrary key-value metadata.
        events: Timestamped events within the span.
    """

    span_id: str
    trace_id: str
    parent_span_id: str | None = None
    operation: str = ""
    skill_id: str = ""
    status: SpanStatus = SpanStatus.UNSET
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def finish(self, status: SpanStatus = SpanStatus.OK) -> None:
        """Mark the span as finished.

        Parameters
        ----------
        status : SpanStatus
            Final status of the span.
        """
        self.end_time = datetime.now(timezone.utc)
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status

    def add_event(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> None:
        """Record a timestamped event within this span.

        Parameters
        ----------
        name : str
            Event name.
        attributes : dict[str, Any] | None
            Event metadata.
        """
        self.events.append({
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attributes": attributes or {},
        })

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span.

        Parameters
        ----------
        key : str
            Attribute key.
        value : Any
            Attribute value.
        """
        self.attributes[key] = value


class SkillTracer:
    """Persistent distributed tracer for skill executions.

    Stores spans in SQLite for analysis.  Provides a context-manager
    API for clean span lifecycle management.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the SQLite database.
        Defaults to ``~/.skillforge/traces.db``.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or Path.home() / ".skillforge" / "traces.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create trace tables if they do not exist."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS spans (
                span_id         TEXT PRIMARY KEY,
                trace_id        TEXT NOT NULL,
                parent_span_id  TEXT,
                operation       TEXT NOT NULL DEFAULT '',
                skill_id        TEXT NOT NULL DEFAULT '',
                status          TEXT NOT NULL DEFAULT 'unset',
                start_time      TEXT NOT NULL,
                end_time        TEXT,
                duration_ms     REAL NOT NULL DEFAULT 0.0,
                attributes      TEXT NOT NULL DEFAULT '{}',
                events          TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
            CREATE INDEX IF NOT EXISTS idx_spans_skill ON spans(skill_id);
            CREATE INDEX IF NOT EXISTS idx_spans_start ON spans(start_time);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Trace creation
    # ------------------------------------------------------------------

    def new_trace(self) -> TraceContext:
        """Create a new trace context.

        Returns
        -------
        TraceContext
            Fresh trace context with a unique trace ID.
        """
        return TraceContext(trace_id=str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Span lifecycle
    # ------------------------------------------------------------------

    def start_span(
        self,
        operation: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        skill_id: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span.

        Parameters
        ----------
        operation : str
            Human-readable operation name.
        trace_id : str | None
            Trace to attach to.  A new trace is created when *None*.
        parent_span_id : str | None
            Parent span ID for nesting.
        skill_id : str
            Skill involved in this operation.
        attributes : dict[str, Any] | None
            Initial span attributes.

        Returns
        -------
        Span
            The newly created span.
        """
        tid = trace_id or str(uuid.uuid4())
        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=tid,
            parent_span_id=parent_span_id,
            operation=operation,
            skill_id=skill_id,
            attributes=attributes or {},
        )
        return span

    def end_span(self, span: Span, status: SpanStatus = SpanStatus.OK) -> None:
        """Finish a span and persist it.

        Parameters
        ----------
        span : Span
            The span to finish.
        status : SpanStatus
            Final status.
        """
        span.finish(status)
        self._persist_span(span)

    def _persist_span(self, span: Span) -> None:
        """Write a span to the database."""
        import json

        self._conn.execute(
            "INSERT OR REPLACE INTO spans VALUES "
            "(:span_id,:trace_id,:parent_span_id,:operation,:skill_id,:status,"
            ":start_time,:end_time,:duration_ms,:attributes,:events)",
            {
                "span_id": span.span_id,
                "trace_id": span.trace_id,
                "parent_span_id": span.parent_span_id,
                "operation": span.operation,
                "skill_id": span.skill_id,
                "status": span.status.value,
                "start_time": span.start_time.isoformat(),
                "end_time": span.end_time.isoformat() if span.end_time else None,
                "duration_ms": span.duration_ms,
                "attributes": json.dumps(span.attributes),
                "events": json.dumps(span.events),
            },
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Context-manager API
    # ------------------------------------------------------------------

    @contextmanager
    def trace_span(
        self,
        operation: str,
        skill_id: str = "",
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Context manager that starts and automatically finishes a span.

        On normal exit the span status is ``OK``.
        On exception the status is ``ERROR``.

        Parameters
        ----------
        operation : str
            Operation name.
        skill_id : str
            Skill ID.
        trace_id : str | None
            Trace to attach to.
        parent_span_id : str | None
            Parent span for nesting.
        attributes : dict[str, Any] | None
            Initial attributes.

        Yields
        ------
        Span
            The active span.
        """
        span = self.start_span(
            operation=operation,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            skill_id=skill_id,
            attributes=attributes,
        )
        try:
            yield span
            self.end_span(span, SpanStatus.OK)
        except Exception:
            self.end_span(span, SpanStatus.ERROR)
            raise

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_trace(self, trace_id: str) -> list[Span]:
        """Retrieve all spans in a trace.

        Parameters
        ----------
        trace_id : str
            Trace ID to retrieve.

        Returns
        -------
        list[Span]
            All spans in the trace, ordered by start time.
        """
        import json as _json

        rows = self._conn.execute(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
            (trace_id,),
        ).fetchall()

        spans: list[Span] = []
        for r in rows:
            d = dict(r)
            d["status"] = SpanStatus(d["status"])
            d["attributes"] = _json.loads(d["attributes"])
            d["events"] = _json.loads(d["events"])
            d["start_time"] = datetime.fromisoformat(d["start_time"])
            if d["end_time"]:
                d["end_time"] = datetime.fromisoformat(d["end_time"])
            spans.append(Span(**d))
        return spans

    def get_skill_spans(
        self, skill_id: str, limit: int = 100
    ) -> list[Span]:
        """Retrieve recent spans for a specific skill.

        Parameters
        ----------
        skill_id : str
            Skill to query.
        limit : int
            Maximum spans to return.

        Returns
        -------
        list[Span]
            Recent spans for the skill, most recent first.
        """
        import json as _json

        rows = self._conn.execute(
            "SELECT * FROM spans WHERE skill_id = ? "
            "ORDER BY start_time DESC LIMIT ?",
            (skill_id, limit),
        ).fetchall()

        spans: list[Span] = []
        for r in rows:
            d = dict(r)
            d["status"] = SpanStatus(d["status"])
            d["attributes"] = _json.loads(d["attributes"])
            d["events"] = _json.loads(d["events"])
            d["start_time"] = datetime.fromisoformat(d["start_time"])
            if d["end_time"]:
                d["end_time"] = datetime.fromisoformat(d["end_time"])
            spans.append(Span(**d))
        return spans

    def get_slow_spans(
        self, threshold_ms: float = 1000.0, limit: int = 50
    ) -> list[Span]:
        """Retrieve spans that exceeded a duration threshold.

        Parameters
        ----------
        threshold_ms : float
            Minimum duration in milliseconds.
        limit : int
            Maximum spans to return.

        Returns
        -------
        list[Span]
            Slow spans, sorted by duration descending.
        """
        import json as _json

        rows = self._conn.execute(
            "SELECT * FROM spans WHERE duration_ms >= ? "
            "ORDER BY duration_ms DESC LIMIT ?",
            (threshold_ms, limit),
        ).fetchall()

        spans: list[Span] = []
        for r in rows:
            d = dict(r)
            d["status"] = SpanStatus(d["status"])
            d["attributes"] = _json.loads(d["attributes"])
            d["events"] = _json.loads(d["events"])
            d["start_time"] = datetime.fromisoformat(d["start_time"])
            if d["end_time"]:
                d["end_time"] = datetime.fromisoformat(d["end_time"])
            spans.append(Span(**d))
        return spans

    def get_error_spans(self, limit: int = 50) -> list[Span]:
        """Retrieve spans that ended with an error.

        Parameters
        ----------
        limit : int
            Maximum spans to return.

        Returns
        -------
        list[Span]
            Error spans, most recent first.
        """
        import json as _json

        rows = self._conn.execute(
            "SELECT * FROM spans WHERE status = 'error' "
            "ORDER BY start_time DESC LIMIT ?",
            (limit,),
        ).fetchall()

        spans: list[Span] = []
        for r in rows:
            d = dict(r)
            d["status"] = SpanStatus(d["status"])
            d["attributes"] = _json.loads(d["attributes"])
            d["events"] = _json.loads(d["events"])
            d["start_time"] = datetime.fromisoformat(d["start_time"])
            if d["end_time"]:
                d["end_time"] = datetime.fromisoformat(d["end_time"])
            spans.append(Span(**d))
        return spans

    def get_trace_stats(self) -> dict[str, Any]:
        """Return aggregate trace statistics.

        Returns
        -------
        dict[str, Any]
            Statistics including total traces, total spans, average
            duration, error rate, and slowest operations.
        """
        total_spans = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM spans"
        ).fetchone()["cnt"]

        unique_traces = self._conn.execute(
            "SELECT COUNT(DISTINCT trace_id) as cnt FROM spans"
        ).fetchone()["cnt"]

        error_count = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM spans WHERE status = 'error'"
        ).fetchone()["cnt"]

        avg_duration = self._conn.execute(
            "SELECT AVG(duration_ms) as avg_ms FROM spans"
        ).fetchone()["avg_ms"] or 0.0

        p99_row = self._conn.execute(
            "SELECT duration_ms FROM spans ORDER BY duration_ms DESC LIMIT 1 "
            "OFFSET (SELECT MAX(0, COUNT(*) / 100 - 1) FROM spans)"
        ).fetchone()
        p99_ms = p99_row["duration_ms"] if p99_row else 0.0

        # Top 5 slowest operations
        top_ops = self._conn.execute(
            "SELECT operation, AVG(duration_ms) as avg_ms, COUNT(*) as cnt "
            "FROM spans GROUP BY operation ORDER BY avg_ms DESC LIMIT 5"
        ).fetchall()

        return {
            "total_spans": total_spans,
            "unique_traces": unique_traces,
            "error_count": error_count,
            "error_rate": error_count / total_spans if total_spans else 0.0,
            "avg_duration_ms": round(avg_duration, 2),
            "p99_duration_ms": round(p99_ms, 2),
            "top_operations": [
                {"operation": r["operation"], "avg_ms": round(r["avg_ms"], 2), "count": r["cnt"]}
                for r in top_ops
            ],
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
