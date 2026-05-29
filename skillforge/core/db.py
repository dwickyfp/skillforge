"""Database abstraction layer for SkillForge.

Provides a backend-agnostic storage interface so that the rest of the
SkillForge codebase does not depend on any single database engine.  Two
concrete backends ship with this module:

- :class:`SQLiteBackend` — persistent storage via ``sqlite3`` (stdlib).
- :class:`MemoryBackend` — ephemeral in-memory storage for tests and
  transient usage.

All backends implement the :class:`DatabaseBackend` protocol, making it
straightforward to add new storage engines (e.g. PostgreSQL, TinyDB) by
implementing the same interface.

Design goals:

- **Stdlib only** — no third-party dependencies.
- **Full type hints** — every public method is annotated.
- **Context manager support** — ``with DatabaseConnection(...) as conn:``.
- **Thread-safe** — SQLite backend uses WAL mode and connection pooling
  is left to the caller.

No external dependencies — stdlib only.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Container for query results, analogous to a cursor's fetchall().

    Attributes
    ----------
    rows : list[dict[str, Any]]
        List of row dicts (column name → value).
    rowcount : int
        Number of affected rows for write operations.
    lastrowid : int | None
        The last inserted row ID, if applicable.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    rowcount: int = 0
    lastrowid: int | None = None

    @property
    def is_empty(self) -> bool:
        """Whether the result set is empty."""
        return len(self.rows) == 0

    def first(self) -> dict[str, Any] | None:
        """Return the first row or ``None``."""
        return self.rows[0] if self.rows else None

    def scalar(self, column: str, default: Any = None) -> Any:
        """Return the value of *column* from the first row, or *default*."""
        row = self.first()
        if row is None:
            return default
        return row.get(column, default)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __bool__(self) -> bool:
        return not self.is_empty


@dataclass
class ColumnDef:
    """Schema column definition.

    Attributes
    ----------
    name : str
        Column name.
    col_type : str
        SQL type string (e.g. ``TEXT``, ``INTEGER``, ``REAL``).
    nullable : bool
        Whether NULL is allowed.
    primary_key : bool
        Whether this column is the primary key.
    default : Any
        Default value (as a Python literal, serialised to SQL).
    """

    name: str
    col_type: str = "TEXT"
    nullable: bool = True
    primary_key: bool = False
    default: Any = None


@dataclass
class TableSchema:
    """Schema definition for a table.

    Attributes
    ----------
    name : str
        Table name.
    columns : list[ColumnDef]
        Ordered column definitions.
    """

    name: str
    columns: list[ColumnDef] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocol (abstract interface)
# ---------------------------------------------------------------------------


@runtime_checkable
class DatabaseBackend(Protocol):
    """Abstract database backend protocol.

    Any object that satisfies this protocol can be used as a SkillForge
    storage backend.  Methods follow a simplified SQL-ish API.
    """

    def create_table(self, schema: TableSchema) -> None:
        """Create a table if it does not exist."""
        ...

    def drop_table(self, name: str) -> None:
        """Drop a table if it exists."""
        ...

    def table_exists(self, name: str) -> bool:
        """Check whether a table exists."""
        ...

    def insert(self, table: str, data: dict[str, Any]) -> QueryResult:
        """Insert a single row.  Returns the result with ``lastrowid``."""
        ...

    def update(
        self,
        table: str,
        data: dict[str, Any],
        where: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Update rows matching *where*.  If *where* is ``None``, updates all."""
        ...

    def delete(
        self, table: str, where: dict[str, Any] | None = None
    ) -> QueryResult:
        """Delete rows matching *where*.  If ``None``, deletes all."""
        ...

    def select(
        self,
        table: str,
        columns: list[str] | None = None,
        where: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> QueryResult:
        """Select rows from a table."""
        ...

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> QueryResult:
        """Execute raw SQL (backend-specific)."""
        ...

    def begin(self) -> None:
        """Begin a transaction."""
        ...

    def commit(self) -> None:
        """Commit the current transaction."""
        ...

    def rollback(self) -> None:
        """Roll back the current transaction."""
        ...

    def close(self) -> None:
        """Close the backend / release resources."""
        ...


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


class SQLiteBackend:
    """Persistent SQLite backend.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file.
    wal_mode : bool
        Enable WAL journal mode (recommended for concurrent reads).
    foreign_keys : bool
        Enforce foreign key constraints.
    """

    def __init__(
        self,
        db_path: str | Path,
        wal_mode: bool = True,
        foreign_keys: bool = True,
    ) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if wal_mode:
            self._conn.execute("PRAGMA journal_mode=WAL")
        if foreign_keys:
            self._conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.Lock()

    @property
    def path(self) -> str:
        """Return the database file path."""
        return self._db_path

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_table(self, schema: TableSchema) -> None:
        """Create a table from a :class:`TableSchema` definition."""
        col_defs: list[str] = []
        for col in schema.columns:
            parts = [col.name, col.col_type]
            if col.primary_key:
                parts.append("PRIMARY KEY")
            if not col.nullable and not col.primary_key:
                parts.append("NOT NULL")
            if col.default is not None:
                if isinstance(col.default, str):
                    parts.append(f"DEFAULT '{col.default}'")
                else:
                    parts.append(f"DEFAULT {col.default}")
            col_defs.append(" ".join(parts))

        sql = (
            f"CREATE TABLE IF NOT EXISTS {schema.name} "
            f"({', '.join(col_defs)})"
        )
        with self._lock:
            self._conn.execute(sql)
            self._conn.commit()

    def drop_table(self, name: str) -> None:
        """Drop a table if it exists."""
        with self._lock:
            self._conn.execute(f"DROP TABLE IF EXISTS {name}")
            self._conn.commit()

    def table_exists(self, name: str) -> bool:
        """Check whether a table exists in the database."""
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def insert(self, table: str, data: dict[str, Any]) -> QueryResult:
        """Insert a single row and return the result."""
        columns = list(data.keys())
        placeholders = [f":{c}" for c in columns]
        sql = (
            f"INSERT INTO {table} ({', '.join(columns)}) "
            f"VALUES ({', '.join(placeholders)})"
        )
        with self._lock:
            cur = self._conn.execute(sql, data)
            self._conn.commit()
            return QueryResult(
                rowcount=cur.rowcount,
                lastrowid=cur.lastrowid,
            )

    def update(
        self,
        table: str,
        data: dict[str, Any],
        where: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Update rows matching the *where* clause."""
        set_parts = [f"{k} = :{k}" for k in data]
        sql = f"UPDATE {table} SET {', '.join(set_parts)}"
        params: dict[str, Any] = dict(data)

        if where:
            where_parts: list[str] = []
            for i, (k, v) in enumerate(where.items()):
                param_name = f"_w{i}"
                where_parts.append(f"{k} = :{param_name}")
                params[param_name] = v
            sql += " WHERE " + " AND ".join(where_parts)

        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return QueryResult(rowcount=cur.rowcount)

    def delete(
        self, table: str, where: dict[str, Any] | None = None
    ) -> QueryResult:
        """Delete rows matching the *where* clause."""
        sql = f"DELETE FROM {table}"
        params: dict[str, Any] = {}

        if where:
            where_parts: list[str] = []
            for i, (k, v) in enumerate(where.items()):
                param_name = f"_w{i}"
                where_parts.append(f"{k} = :{param_name}")
                params[param_name] = v
            sql += " WHERE " + " AND ".join(where_parts)

        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return QueryResult(rowcount=cur.rowcount)

    def select(
        self,
        table: str,
        columns: list[str] | None = None,
        where: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> QueryResult:
        """Select rows with optional filtering, ordering, and pagination."""
        cols = ", ".join(columns) if columns else "*"
        sql = f"SELECT {cols} FROM {table}"
        params: dict[str, Any] = {}

        if where:
            where_parts: list[str] = []
            for i, (k, v) in enumerate(where.items()):
                param_name = f"_w{i}"
                where_parts.append(f"{k} = :{param_name}")
                params[param_name] = v
            sql += " WHERE " + " AND ".join(where_parts)

        if order_by:
            sql += f" ORDER BY {order_by}"

        if limit is not None:
            sql += " LIMIT :_limit OFFSET :_offset"
            params["_limit"] = limit
            params["_offset"] = offset

        rows = self._conn.execute(sql, params).fetchall()
        return QueryResult(
            rows=[dict(r) for r in rows],
            rowcount=len(rows),
        )

    def execute(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> QueryResult:
        """Execute raw SQL and return the result."""
        with self._lock:
            cur = self._conn.execute(sql, params or {})
            try:
                rows = cur.fetchall()
                return QueryResult(
                    rows=[dict(r) for r in rows],
                    rowcount=cur.rowcount,
                    lastrowid=cur.lastrowid,
                )
            except sqlite3.ProgrammingError:
                # No result set (e.g. DDL, DML without returning)
                return QueryResult(
                    rowcount=cur.rowcount,
                    lastrowid=cur.lastrowid,
                )

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def begin(self) -> None:
        """Begin an explicit transaction."""
        self._conn.execute("BEGIN")

    def commit(self) -> None:
        """Commit the current transaction."""
        self._conn.commit()

    def rollback(self) -> None:
        """Roll back the current transaction."""
        self._conn.rollback()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __repr__(self) -> str:
        return f"SQLiteBackend(path={self._db_path!r})"


# ---------------------------------------------------------------------------
# Memory backend (in-memory, for testing)
# ---------------------------------------------------------------------------


class MemoryBackend:
    """Ephemeral in-memory backend backed by plain Python dicts.

    Useful for unit tests and transient usage where persistence is not
    needed.  Thread-safe via a reentrant lock.

    Parameters
    ----------
    name : str
        Human-readable name for this backend (used in logging).
    """

    def __init__(self, name: str = "memory") -> None:
        self._name = name
        self._tables: dict[str, list[dict[str, Any]]] = {}
        self._schemas: dict[str, TableSchema] = {}
        self._lock = threading.RLock()
        self._auto_id: int = 0

    @property
    def table_names(self) -> list[str]:
        """Return a sorted list of table names."""
        return sorted(self._tables.keys())

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_table(self, schema: TableSchema) -> None:
        """Create a table if it does not exist."""
        with self._lock:
            if schema.name not in self._tables:
                self._tables[schema.name] = []
                self._schemas[schema.name] = schema

    def drop_table(self, name: str) -> None:
        """Drop a table if it exists."""
        with self._lock:
            self._tables.pop(name, None)
            self._schemas.pop(name, None)

    def table_exists(self, name: str) -> bool:
        """Check whether a table exists."""
        return name in self._tables

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def insert(self, table: str, data: dict[str, Any]) -> QueryResult:
        """Insert a row into the in-memory table."""
        if table not in self._tables:
            raise ValueError(f"Table '{table}' does not exist")
        with self._lock:
            row = dict(data)
            # Auto-assign _id if the schema has a primary key not provided
            schema = self._schemas.get(table)
            if schema:
                for col in schema.columns:
                    if col.primary_key and col.name not in row:
                        self._auto_id += 1
                        row[col.name] = self._auto_id
            self._tables[table].append(row)
            self._auto_id += 1
            return QueryResult(rowcount=1, lastrowid=self._auto_id)

    def update(
        self,
        table: str,
        data: dict[str, Any],
        where: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Update matching rows in the in-memory table."""
        if table not in self._tables:
            raise ValueError(f"Table '{table}' does not exist")
        with self._lock:
            count = 0
            for row in self._tables[table]:
                if self._matches(row, where):
                    row.update(data)
                    count += 1
            return QueryResult(rowcount=count)

    def delete(
        self, table: str, where: dict[str, Any] | None = None
    ) -> QueryResult:
        """Delete matching rows from the in-memory table."""
        if table not in self._tables:
            raise ValueError(f"Table '{table}' does not exist")
        with self._lock:
            original_len = len(self._tables[table])
            self._tables[table] = [
                row for row in self._tables[table]
                if not self._matches(row, where)
            ]
            deleted = original_len - len(self._tables[table])
            return QueryResult(rowcount=deleted)

    def select(
        self,
        table: str,
        columns: list[str] | None = None,
        where: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> QueryResult:
        """Select rows with optional filtering and pagination."""
        if table not in self._tables:
            raise ValueError(f"Table '{table}' does not exist")
        with self._lock:
            rows = [r for r in self._tables[table] if self._matches(r, where)]

            # Column projection
            if columns:
                rows = [{k: r.get(k) for k in columns} for r in rows]
            else:
                rows = [dict(r) for r in rows]

            # Ordering
            if order_by:
                desc = order_by.endswith(" DESC")
                col = order_by.replace(" DESC", "").replace(" ASC", "").strip()
                rows.sort(
                    key=lambda r: (r.get(col) is None, r.get(col, "")),
                    reverse=desc,
                )

            # Pagination
            rows = rows[offset:]
            if limit is not None:
                rows = rows[:limit]

            return QueryResult(rows=rows, rowcount=len(rows))

    def execute(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> QueryResult:
        """Execute raw SQL — not supported for MemoryBackend.

        Raises
        ------
        NotImplementedError
            Always, since raw SQL is not parsed by this backend.
        """
        raise NotImplementedError(
            "MemoryBackend does not support raw SQL. Use the CRUD methods."
        )

    # ------------------------------------------------------------------
    # Transactions (no-ops for in-memory)
    # ------------------------------------------------------------------

    def begin(self) -> None:
        """No-op — MemoryBackend has no transaction support."""

    def commit(self) -> None:
        """No-op — MemoryBackend has no transaction support."""

    def rollback(self) -> None:
        """No-op — MemoryBackend has no transaction support."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clear all in-memory data."""
        with self._lock:
            self._tables.clear()
            self._schemas.clear()

    def __repr__(self) -> str:
        return f"MemoryBackend(name={self._name!r}, tables={self.table_names})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matches(
        row: dict[str, Any], where: dict[str, Any] | None
    ) -> bool:
        """Check whether *row* matches all key-value pairs in *where*."""
        if where is None:
            return True
        return all(row.get(k) == v for k, v in where.items())


# ---------------------------------------------------------------------------
# Connection wrapper (context manager)
# ---------------------------------------------------------------------------


class DatabaseConnection:
    """Context manager wrapping any :class:`DatabaseBackend`.

    Provides automatic commit on clean exit and rollback on exception.

    Parameters
    ----------
    backend : DatabaseBackend
        The backend instance to wrap.

    Examples
    --------
    >>> backend = MemoryBackend()
    >>> backend.create_table(TableSchema("t", [ColumnDef("id", "INTEGER", primary_key=True)]))
    >>> with DatabaseConnection(backend) as conn:
    ...     conn.insert("t", {"id": 1})
    ...     result = conn.select("t")
    """

    def __init__(self, backend: DatabaseBackend) -> None:
        self._backend = backend

    @property
    def backend(self) -> DatabaseBackend:
        """Access the underlying backend."""
        return self._backend

    def __enter__(self) -> DatabaseBackend:
        self._backend.begin()
        return self._backend

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        if exc_type is not None:
            self._backend.rollback()
        else:
            self._backend.commit()
        return False  # do not suppress exceptions

    def insert(self, table: str, data: dict[str, Any]) -> QueryResult:
        """Convenience proxy to backend.insert()."""
        return self._backend.insert(table, data)

    def update(
        self,
        table: str,
        data: dict[str, Any],
        where: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Convenience proxy to backend.update()."""
        return self._backend.update(table, data, where)

    def delete(
        self, table: str, where: dict[str, Any] | None = None
    ) -> QueryResult:
        """Convenience proxy to backend.delete()."""
        return self._backend.delete(table, where)

    def select(
        self,
        table: str,
        columns: list[str] | None = None,
        where: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> QueryResult:
        """Convenience proxy to backend.select()."""
        return self._backend.select(table, columns, where, order_by, limit, offset)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_backend(
    backend_type: str = "sqlite",
    **kwargs: Any,
) -> DatabaseBackend:
    """Factory function to create a database backend.

    Parameters
    ----------
    backend_type : str
        One of ``"sqlite"`` or ``"memory"``.
    **kwargs
        Passed to the backend constructor.

    Returns
    -------
    DatabaseBackend
        The created backend instance.

    Raises
    ------
    ValueError
        If *backend_type* is not recognised.

    Examples
    --------
    >>> db = create_backend("sqlite", db_path="/tmp/test.db")
    >>> db = create_backend("memory", name="test")
    """
    if backend_type == "sqlite":
        return SQLiteBackend(**kwargs)
    if backend_type == "memory":
        return MemoryBackend(**kwargs)
    raise ValueError(
        f"Unknown backend type '{backend_type}'. "
        f"Supported: 'sqlite', 'memory'"
    )
