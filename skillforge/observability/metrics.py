"""Metrics collector — aggregates and persists numerical skill metrics.

Supports counters, gauges, and histograms (bucketed) with SQLite
persistence and aggregation queries.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class MetricPoint:
    """A single metric data point.

    Attributes:
        name: Metric name (e.g. ``skill.latency_ms``).
        value: Numerical value.
        labels: Dimensional labels (e.g. ``{"skill_id": "…"}``).
        timestamp: When the metric was recorded.
    """

    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MetricSummary:
    """Aggregated summary for a metric.

    Attributes:
        name: Metric name.
        count: Number of data points.
        sum: Sum of all values.
        min: Minimum value.
        max: Maximum value.
        mean: Average value.
        p50: Median (50th percentile) value.
        p95: 95th percentile value.
        p99: 99th percentile value.
    """

    name: str
    count: int = 0
    sum: float = 0.0
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


class MetricsCollector:
    """Persistent metrics collector backed by SQLite.

    Provides a simple interface for recording and querying metrics
    across skill executions.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the SQLite database.
        Defaults to ``~/.skillforge/metrics.db``.
    """

    # Default histogram bucket boundaries
    DEFAULT_BUCKETS: list[float] = [
        1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0,
        1000.0, 2500.0, 5000.0, 10000.0, float("inf"),
    ]

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or Path.home() / ".skillforge" / "metrics.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create metric tables if they do not exist."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metric_points (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                value       REAL NOT NULL,
                labels      TEXT NOT NULL DEFAULT '{}',
                recorded_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mp_name
                ON metric_points(name);
            CREATE INDEX IF NOT EXISTS idx_mp_recorded
                ON metric_points(recorded_at);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        timestamp: datetime | None = None,
    ) -> MetricPoint:
        """Record a single metric data point.

        Parameters
        ----------
        name : str
            Metric name (e.g. ``"skill.latency_ms"``).
        value : float
            Numerical value.
        labels : dict[str, str] | None
            Dimensional labels.
        timestamp : datetime | None
            When the metric occurred.  Defaults to now.

        Returns
        -------
        MetricPoint
            The recorded data point.
        """
        ts = timestamp or datetime.now(timezone.utc)
        point = MetricPoint(name=name, value=value, labels=labels or {}, timestamp=ts)
        self._conn.execute(
            "INSERT INTO metric_points (id, name, value, labels, recorded_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                point.name,
                point.value,
                json.dumps(point.labels),
                point.timestamp.isoformat(),
            ),
        )
        self._conn.commit()
        return point

    def record_counter(
        self, name: str, increment: float = 1.0, labels: dict[str, str] | None = None
    ) -> MetricPoint:
        """Record a counter increment.

        Parameters
        ----------
        name : str
            Counter name.
        increment : float
            Amount to increment.
        labels : dict[str, str] | None
            Dimensional labels.

        Returns
        -------
        MetricPoint
            The recorded data point.
        """
        return self.record(name, increment, labels)

    def record_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> MetricPoint:
        """Record a gauge value (point-in-time snapshot).

        Parameters
        ----------
        name : str
            Gauge name.
        value : float
            Current value.
        labels : dict[str, str] | None
            Dimensional labels.

        Returns
        -------
        MetricPoint
            The recorded data point.
        """
        return self.record(name, value, labels)

    def record_timing(
        self, name: str, duration_ms: float, labels: dict[str, str] | None = None
    ) -> MetricPoint:
        """Record a timing measurement.

        Parameters
        ----------
        name : str
            Timing metric name (e.g. ``"skill.load.duration_ms"``).
        duration_ms : float
            Duration in milliseconds.
        labels : dict[str, str] | None
            Dimensional labels.

        Returns
        -------
        MetricPoint
            The recorded data point.
        """
        return self.record(name, duration_ms, labels)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_metric_points(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[MetricPoint]:
        """Retrieve metric points for a given name.

        Parameters
        ----------
        name : str
            Metric name.
        labels : dict[str, str] | None
            Filter to points whose labels contain all specified pairs.
        since : datetime | None
            Only return points after this time.
        limit : int
            Maximum points to return.

        Returns
        -------
        list[MetricPoint]
            Matching data points, ordered by time.
        """
        clauses = ["name = ?"]
        params: list[Any] = [name]

        if since is not None:
            clauses.append("recorded_at >= ?")
            params.append(since.isoformat())

        where = " AND ".join(clauses)
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT * FROM metric_points WHERE {where} "
            "ORDER BY recorded_at LIMIT ?",
            params,
        ).fetchall()

        points: list[MetricPoint] = []
        for r in rows:
            d = dict(r)
            decoded_labels = json.loads(d["labels"])
            # Filter by labels if requested
            if labels:
                if not all(decoded_labels.get(k) == v for k, v in labels.items()):
                    continue
            points.append(MetricPoint(
                name=d["name"],
                value=d["value"],
                labels=decoded_labels,
                timestamp=datetime.fromisoformat(d["recorded_at"]),
            ))
        return points

    def summarize(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        since: datetime | None = None,
    ) -> MetricSummary:
        """Compute an aggregate summary for a metric.

        Parameters
        ----------
        name : str
            Metric name.
        labels : dict[str, str] | None
            Filter to matching labels.
        since : datetime | None
            Only consider points after this time.

        Returns
        -------
        MetricSummary
            Aggregated statistics.
        """
        points = self.get_metric_points(
            name, labels=labels, since=since, limit=100_000
        )
        if not points:
            return MetricSummary(name=name)

        values = sorted(p.value for p in points)
        n = len(values)
        total = sum(values)

        def _percentile(sorted_vals: list[float], pct: float) -> float:
            """Calculate percentile from a sorted list."""
            if not sorted_vals:
                return 0.0
            k = (len(sorted_vals) - 1) * pct
            f = int(k)
            c = min(f + 1, len(sorted_vals) - 1)
            d = k - f
            return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])

        return MetricSummary(
            name=name,
            count=n,
            sum=total,
            min=values[0],
            max=values[-1],
            mean=total / n,
            p50=_percentile(values, 0.50),
            p95=_percentile(values, 0.95),
            p99=_percentile(values, 0.99),
        )

    def get_histogram(
        self,
        name: str,
        buckets: list[float] | None = None,
        labels: dict[str, str] | None = None,
        since: datetime | None = None,
    ) -> dict[str, int]:
        """Compute a histogram distribution for a metric.

        Parameters
        ----------
        name : str
            Metric name.
        buckets : list[float] | None
            Bucket boundaries.  Defaults to :attr:`DEFAULT_BUCKETS`.
        labels : dict[str, str] | None
            Filter by labels.
        since : datetime | None
            Only consider points after this time.

        Returns
        -------
        dict[str, int]
            Bucket label → count mapping.
        """
        points = self.get_metric_points(
            name, labels=labels, since=since, limit=100_000
        )
        if not points:
            return {}

        bucket_edges = buckets or self.DEFAULT_BUCKETS
        counts: dict[str, int] = {}
        for edge in bucket_edges:
            label = f"<={edge}" if edge != float("inf") else "+Inf"
            counts[label] = 0

        for p in points:
            for edge in bucket_edges:
                label = f"<={edge}" if edge != float("inf") else "+Inf"
                if p.value <= edge:
                    counts[label] += 1
                    break

        return counts

    def get_metric_names(self) -> list[str]:
        """List all unique metric names.

        Returns
        -------
        list[str]
            All metric names, sorted alphabetically.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT name FROM metric_points ORDER BY name"
        ).fetchall()
        return [r["name"] for r in rows]

    def count(
        self,
        name: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> int:
        """Count data points for a metric.

        Parameters
        ----------
        name : str | None
            Metric name.  If *None*, counts all data points.
        labels : dict[str, str] | None
            Filter by labels.

        Returns
        -------
        int
            Number of matching data points.
        """
        if name is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM metric_points WHERE name = ?",
                (name,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM metric_points"
            ).fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
