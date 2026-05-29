"""Marketplace registry — persistent store for published, discoverable skills.

Provides a SQLite-backed catalogue where skills can be published, searched,
rated, and tracked for install counts.  Mirrors the structure of
:class:`skillforge.core.registry.SkillRegistry` for consistency.
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


class MarketplaceStatus(Enum):
    """Publication status of a marketplace entry."""

    PENDING = "pending"
    PUBLISHED = "published"
    UNLISTED = "unlisted"
    DEPRECATED = "deprecated"


@dataclass
class MarketplaceEntry:
    """A skill listing in the marketplace.

    Attributes:
        id: Unique listing identifier.
        skill_name: Human-readable skill name.
        publisher: Publisher / author name.
        description: Short summary (~100 words).
        version: Semantic version string (e.g. ``"1.2.0"``).
        tags: List of searchable tags.
        status: Current publication status.
        tier1_metadata: Short metadata for routing.
        tier2_core: Full core prompt / instructions.
        tier3_resources: Auxiliary resource paths or URLs.
        install_count: Number of times this skill has been installed.
        rating: Average user rating (0.0–5.0).
        rating_count: Number of ratings submitted.
        license: License identifier (e.g. ``"MIT"``).
        created_at: When the listing was first created.
        updated_at: When the listing was last updated.
    """

    id: str
    skill_name: str
    publisher: str
    description: str = ""
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    status: MarketplaceStatus = MarketplaceStatus.PENDING
    tier1_metadata: str = ""
    tier2_core: str = ""
    tier3_resources: list[str] = field(default_factory=list)
    install_count: int = 0
    rating: float = 0.0
    rating_count: int = 0
    license: str = "MIT"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class InstallRecord:
    """Record of a skill installation from the marketplace.

    Attributes:
        id: Unique record identifier.
        listing_id: Marketplace listing ID.
        installed_at: When the install happened.
        installed_version: Version that was installed.
        target_registry_id: ID assigned in the local registry.
    """

    id: str
    listing_id: str
    installed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    installed_version: str = "1.0.0"
    target_registry_id: str = ""


class MarketplaceRegistry:
    """Persistent marketplace catalogue backed by SQLite.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the SQLite database file.
        Defaults to ``~/.skillforge/marketplace.db``.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or Path.home() / ".skillforge" / "marketplace.db")
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
        """Create marketplace tables if they do not exist."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS marketplace_listings (
                id              TEXT PRIMARY KEY,
                skill_name      TEXT NOT NULL,
                publisher       TEXT NOT NULL,
                description     TEXT NOT NULL DEFAULT '',
                version         TEXT NOT NULL DEFAULT '1.0.0',
                tags            TEXT NOT NULL DEFAULT '[]',
                status          TEXT NOT NULL DEFAULT 'pending',
                tier1_metadata  TEXT NOT NULL DEFAULT '',
                tier2_core      TEXT NOT NULL DEFAULT '',
                tier3_resources TEXT NOT NULL DEFAULT '[]',
                install_count   INTEGER NOT NULL DEFAULT 0,
                rating          REAL NOT NULL DEFAULT 0.0,
                rating_count    INTEGER NOT NULL DEFAULT 0,
                license         TEXT NOT NULL DEFAULT 'MIT',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mp_status
                ON marketplace_listings(status);
            CREATE INDEX IF NOT EXISTS idx_mp_rating
                ON marketplace_listings(rating DESC);

            CREATE TABLE IF NOT EXISTS marketplace_installs (
                id                  TEXT PRIMARY KEY,
                listing_id          TEXT NOT NULL,
                installed_at        TEXT NOT NULL,
                installed_version   TEXT NOT NULL DEFAULT '1.0.0',
                target_registry_id  TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (listing_id) REFERENCES marketplace_listings(id)
            );
            CREATE INDEX IF NOT EXISTS idx_mpi_listing
                ON marketplace_installs(listing_id);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MarketplaceEntry:
        """Convert a database row to a MarketplaceEntry."""
        d = dict(row)
        d["status"] = MarketplaceStatus(d["status"])
        d["tags"] = json.loads(d["tags"])
        d["tier3_resources"] = json.loads(d["tier3_resources"])
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        d["updated_at"] = datetime.fromisoformat(d["updated_at"])
        return MarketplaceEntry(**d)

    @staticmethod
    def _entry_to_params(e: MarketplaceEntry) -> dict[str, Any]:
        """Convert a MarketplaceEntry to database parameters."""
        return {
            "id": e.id,
            "skill_name": e.skill_name,
            "publisher": e.publisher,
            "description": e.description,
            "version": e.version,
            "tags": json.dumps(e.tags),
            "status": e.status.value,
            "tier1_metadata": e.tier1_metadata,
            "tier2_core": e.tier2_core,
            "tier3_resources": json.dumps(e.tier3_resources),
            "install_count": e.install_count,
            "rating": e.rating,
            "rating_count": e.rating_count,
            "license": e.license,
            "created_at": e.created_at.isoformat(),
            "updated_at": e.updated_at.isoformat(),
        }

    # ------------------------------------------------------------------
    # CRUD — Listings
    # ------------------------------------------------------------------

    def publish(
        self,
        skill_name: str,
        publisher: str,
        description: str = "",
        version: str = "1.0.0",
        tier1_metadata: str = "",
        tier2_core: str = "",
        tier3_resources: list[str] | None = None,
        tags: list[str] | None = None,
        license: str = "MIT",
        listing_id: str | None = None,
        status: MarketplaceStatus = MarketplaceStatus.PUBLISHED,
    ) -> MarketplaceEntry:
        """Publish a new skill to the marketplace.

        Parameters
        ----------
        skill_name : str
            Human-readable skill name.
        publisher : str
            Author / publisher name.
        description : str
            Short description (~100 words).
        version : str
            Semantic version string.
        tier1_metadata : str
            Routing summary (~30 tokens).
        tier2_core : str
            Full core prompt / instructions.
        tier3_resources : list[str] | None
            Auxiliary resource list.
        tags : list[str] | None
            Searchable tags.
        license : str
            License identifier.
        listing_id : str | None
            Explicit listing ID.  A UUID is generated when *None*.
        status : MarketplaceStatus
            Initial publication status.

        Returns
        -------
        MarketplaceEntry
            The newly created listing.

        Raises
        ------
        ValueError
            If a listing with the given *listing_id* already exists.
        """
        lid = listing_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        entry = MarketplaceEntry(
            id=lid,
            skill_name=skill_name,
            publisher=publisher,
            description=description,
            version=version,
            tags=tags or [],
            status=status,
            tier1_metadata=tier1_metadata,
            tier2_core=tier2_core,
            tier3_resources=tier3_resources or [],
            license=license,
            created_at=now,
            updated_at=now,
        )
        try:
            self._conn.execute(
                "INSERT INTO marketplace_listings VALUES "
                "(:id,:skill_name,:publisher,:description,:version,:tags,:status,"
                ":tier1_metadata,:tier2_core,:tier3_resources,:install_count,"
                ":rating,:rating_count,:license,:created_at,:updated_at)",
                self._entry_to_params(entry),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Listing '{lid}' already exists") from exc
        return entry

    def get_listing(self, listing_id: str) -> MarketplaceEntry | None:
        """Retrieve a marketplace listing by ID.

        Parameters
        ----------
        listing_id : str
            The listing identifier.

        Returns
        -------
        MarketplaceEntry | None
            The listing, or *None* if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM marketplace_listings WHERE id = ?", (listing_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def update_listing(
        self, listing_id: str, updates: dict[str, Any]
    ) -> MarketplaceEntry | None:
        """Apply partial updates to a listing.

        Parameters
        ----------
        listing_id : str
            Listing to update.
        updates : dict
            Key-value pairs of fields to change.

        Returns
        -------
        MarketplaceEntry | None
            Updated listing, or *None* if not found.
        """
        allowed = {
            "skill_name", "publisher", "description", "version",
            "tags", "status", "tier1_metadata", "tier2_core",
            "tier3_resources", "rating", "rating_count", "license",
        }
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            return self.get_listing(listing_id)

        if "tags" in filtered:
            filtered["tags"] = json.dumps(filtered["tags"])
        if "tier3_resources" in filtered:
            filtered["tier3_resources"] = json.dumps(filtered["tier3_resources"])
        if "status" in filtered and isinstance(filtered["status"], MarketplaceStatus):
            filtered["status"] = filtered["status"].value

        filtered["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{k} = :{k}" for k in filtered)
        filtered["id"] = listing_id
        cur = self._conn.execute(
            f"UPDATE marketplace_listings SET {set_clause} WHERE id = :id",
            filtered,
        )
        self._conn.commit()
        if cur.rowcount == 0:
            return None
        return self.get_listing(listing_id)

    def list_listings(
        self,
        status: MarketplaceStatus | None = None,
        tags: list[str] | None = None,
        publisher: str | None = None,
        sort_by: str = "rating",
        limit: int = 50,
        offset: int = 0,
    ) -> list[MarketplaceEntry]:
        """List marketplace listings with optional filters.

        Parameters
        ----------
        status : MarketplaceStatus | None
            Filter by publication status.
        tags : list[str] | None
            Filter listings that have at least one of these tags.
        publisher : str | None
            Filter by publisher name.
        sort_by : str
            Sort field: ``"rating"``, ``"install_count"``, or ``"created_at"``.
        limit : int
            Maximum results.
        offset : int
            Pagination offset.

        Returns
        -------
        list[MarketplaceEntry]
            Matching listings sorted as requested.
        """
        clauses: list[str] = []
        params: dict[str, Any] = {}

        if status is not None:
            clauses.append("status = :status")
            params["status"] = status.value
        if publisher is not None:
            clauses.append("publisher = :publisher")
            params["publisher"] = publisher

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        valid_sorts = {"rating", "install_count", "created_at"}
        order_col = sort_by if sort_by in valid_sorts else "rating"
        order_dir = "DESC" if order_col in ("rating", "install_count") else "ASC"

        params["limit"] = limit
        params["offset"] = offset

        rows = self._conn.execute(
            f"SELECT * FROM marketplace_listings{where} "
            f"ORDER BY {order_col} {order_dir} LIMIT :limit OFFSET :offset",
            params,
        ).fetchall()

        entries = [self._row_to_entry(r) for r in rows]

        # Post-filter by tags (JSON in SQLite is cumbersome)
        if tags:
            tag_set = set(tags)
            entries = [e for e in entries if tag_set.intersection(e.tags)]

        return entries

    def search(
        self, query: str, limit: int = 20
    ) -> list[MarketplaceEntry]:
        """Keyword search across skill name, description, and tags.

        Parameters
        ----------
        query : str
            Search string.
        limit : int
            Maximum results.

        Returns
        -------
        list[MarketplaceEntry]
            Matching listings ranked by rating descending.
        """
        pattern = f"%{query}%"
        rows = self._conn.execute(
            "SELECT * FROM marketplace_listings "
            "WHERE skill_name LIKE :q OR description LIKE :q "
            "OR tier1_metadata LIKE :q "
            "ORDER BY rating DESC LIMIT :limit",
            {"q": pattern, "limit": limit},
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    # ------------------------------------------------------------------
    # Ratings
    # ------------------------------------------------------------------

    def rate_listing(
        self, listing_id: str, user_rating: float
    ) -> MarketplaceEntry | None:
        """Submit a rating for a listing (0.0–5.0).

        Updates the rolling average and count.

        Parameters
        ----------
        listing_id : str
            Listing to rate.
        user_rating : float
            Rating value, clamped to [0.0, 5.0].

        Returns
        -------
        MarketplaceEntry | None
            Updated listing, or *None* if not found.
        """
        user_rating = max(0.0, min(5.0, user_rating))
        entry = self.get_listing(listing_id)
        if entry is None:
            return None

        new_count = entry.rating_count + 1
        new_avg = (
            (entry.rating * entry.rating_count + user_rating) / new_count
        )
        return self.update_listing(
            listing_id,
            {"rating": round(new_avg, 2), "rating_count": new_count},
        )

    # ------------------------------------------------------------------
    # Install tracking
    # ------------------------------------------------------------------

    def record_install(
        self,
        listing_id: str,
        installed_version: str = "1.0.0",
        target_registry_id: str = "",
    ) -> InstallRecord:
        """Record that a listing was installed and bump the install count.

        Parameters
        ----------
        listing_id : str
            The marketplace listing that was installed.
        installed_version : str
            Version string that was installed.
        target_registry_id : str
            ID assigned to the skill in the local registry.

        Returns
        -------
        InstallRecord
            The persisted install record.
        """
        record_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        record = InstallRecord(
            id=record_id,
            listing_id=listing_id,
            installed_at=now,
            installed_version=installed_version,
            target_registry_id=target_registry_id,
        )

        self._conn.execute(
            "INSERT INTO marketplace_installs VALUES (:id,:listing_id,"
            ":installed_at,:installed_version,:target_registry_id)",
            {
                "id": record.id,
                "listing_id": record.listing_id,
                "installed_at": record.installed_at.isoformat(),
                "installed_version": record.installed_version,
                "target_registry_id": record.target_registry_id,
            },
        )

        # Bump install count on the listing
        self._conn.execute(
            "UPDATE marketplace_listings SET install_count = install_count + 1, "
            "updated_at = :now WHERE id = :lid",
            {"now": now.isoformat(), "lid": listing_id},
        )
        self._conn.commit()
        return record

    def get_install_history(
        self, listing_id: str, limit: int = 50
    ) -> list[InstallRecord]:
        """Get installation history for a listing.

        Parameters
        ----------
        listing_id : str
            Listing to query.
        limit : int
            Maximum records to return.

        Returns
        -------
        list[InstallRecord]
            Install records, most recent first.
        """
        rows = self._conn.execute(
            "SELECT * FROM marketplace_installs WHERE listing_id = ? "
            "ORDER BY installed_at DESC LIMIT ?",
            (listing_id, limit),
        ).fetchall()

        results: list[InstallRecord] = []
        for r in rows:
            d = dict(r)
            d["installed_at"] = datetime.fromisoformat(d["installed_at"])
            results.append(InstallRecord(**d))
        return results

    def get_install_count(self, listing_id: str) -> int:
        """Return the current install count for a listing.

        Parameters
        ----------
        listing_id : str
            Listing to query.

        Returns
        -------
        int
            Install count, or 0 if listing not found.
        """
        row = self._conn.execute(
            "SELECT install_count FROM marketplace_listings WHERE id = ?",
            (listing_id,),
        ).fetchone()
        return row["install_count"] if row else 0

    # ------------------------------------------------------------------
    # Listing count
    # ------------------------------------------------------------------

    def count(self, status: MarketplaceStatus | None = None) -> int:
        """Return the number of listings, optionally filtered by status.

        Parameters
        ----------
        status : MarketplaceStatus | None
            If given, only count listings with this status.

        Returns
        -------
        int
            Number of matching listings.
        """
        if status is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM marketplace_listings WHERE status = ?",
                (status.value,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM marketplace_listings"
            ).fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Deprecation
    # ------------------------------------------------------------------

    def deprecate_listing(self, listing_id: str) -> MarketplaceEntry | None:
        """Mark a listing as deprecated.

        Parameters
        ----------
        listing_id : str
            Listing to deprecate.

        Returns
        -------
        MarketplaceEntry | None
            Updated listing, or *None* if not found.
        """
        return self.update_listing(
            listing_id, {"status": MarketplaceStatus.DEPRECATED}
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
