"""Detect and resolve conflicts between skills.

Conflicts arise when two or more skills overlap in purpose, contain
contradictory instructions, or form circular dependency chains.  The
:class:`ConflictDetector` scans all active skills, identifies conflicts,
and offers automated resolution strategies (merge, deprecate, manual).

A lightweight SQLite table (``skill_conflicts``) persists detected
conflicts so that resolution state survives across sessions.
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from skillforge.core.registry import Skill, SkillLifecycle, SkillRegistry
from skillforge.core.graph import SkillDependencyGraph


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SkillConflict:
    """Represents a detected conflict between two skills.

    Attributes:
        conflict_id: Unique identifier for this conflict record.
        skill_a_id: Registry ID of the first skill involved.
        skill_b_id: Registry ID of the second skill involved.
        conflict_type: One of ``overlap``, ``contradiction``, or
            ``circular_dependency``.
        severity: One of ``low``, ``medium``, ``high``, or ``critical``.
        description: Human-readable explanation of the conflict.
        detected_at: UTC timestamp when the conflict was first detected.
        resolved: Whether the conflict has been resolved.
    """

    conflict_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    skill_a_id: str = ""
    skill_b_id: str = ""
    conflict_type: Literal["overlap", "contradiction", "circular_dependency"] = "overlap"
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    description: str = ""
    detected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved: bool = False


# ---------------------------------------------------------------------------
# ConflictDetector
# ---------------------------------------------------------------------------

class ConflictDetector:
    """Detects and resolves conflicts between registered skills.

    Parameters:
        registry: The :class:`~skillforge.core.registry.SkillRegistry` to
            scan for conflicts.
        graph: The :class:`~skillforge.core.graph.SkillDependencyGraph` used
            to detect circular dependency chains.
        db_path: Path to the SQLite database that persists conflict records.
            Defaults to ``skill_conflicts.db`` in the current directory.
    """

    # Opposite-word pairs used for contradiction detection.
    _OPPOSITES: list[tuple[set[str], set[str]]] = [
        ({"always"}, {"never"}),
        ({"must"}, {"must not", "mustn't"}),
        ({"do"}, {"don't", "do not"}),
        ({"include"}, {"exclude"}),
        ({"enable"}, {"disable"}),
        ({"start"}, {"stop"}),
    ]

    # Minimum keyword overlap ratio to flag an *overlap* conflict.
    _OVERLAP_THRESHOLD: float = 0.60

    def __init__(
        self,
        registry: SkillRegistry,
        graph: SkillDependencyGraph,
        db_path: str = "skill_conflicts.db",
    ) -> None:
        self._registry = registry
        self._graph = graph
        self._db_path = db_path
        self._conn = sqlite3.connect(self._db_path)
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create the ``skill_conflicts`` table if it does not exist."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skill_conflicts (
                conflict_id   TEXT PRIMARY KEY,
                skill_a_id    TEXT NOT NULL,
                skill_b_id    TEXT NOT NULL,
                conflict_type TEXT NOT NULL,
                severity      TEXT NOT NULL,
                description   TEXT NOT NULL,
                detected_at   TEXT NOT NULL,
                resolved      INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_conflicts(self) -> list[SkillConflict]:
        """Scan all active skills and return a list of newly detected conflicts.

        Detection strategies:
        1. **Overlap** — keyword overlap between two skill descriptions
           exceeds the configured threshold (default 60 %).
        2. **Contradiction** — tier-2 core instructions contain opposite
           directives (e.g. "always X" vs "never X").
        3. **Circular dependency** — a cycle exists in the dependency graph.

        Returns:
            A list of :class:`SkillConflict` instances for every newly
            detected conflict.  Previously persisted conflicts are not
            duplicated.
        """
        all_skills = self._registry.list_skills(lifecycle=SkillLifecycle.ACTIVE)
        conflicts: list[SkillConflict] = []

        # Pair-wise checks for overlap and contradiction.
        for i, skill_a in enumerate(all_skills):
            for skill_b in all_skills[i + 1 :]:
                overlap = self._check_keyword_overlap(skill_a, skill_b)
                if overlap is not None:
                    conflicts.append(overlap)

                contradiction = self._check_contradiction(skill_a, skill_b)
                if contradiction is not None:
                    conflicts.append(contradiction)

        # Circular dependency detection via DFS on the graph.
        conflicts.extend(self._check_circular_dependencies())

        # Persist only genuinely new conflicts.
        persisted: list[SkillConflict] = []
        for conflict in conflicts:
            if not self._conflict_exists(conflict):
                self._persist_conflict(conflict)
                persisted.append(conflict)

        return persisted

    def resolve_conflict(
        self,
        conflict_id: str,
        strategy: Literal["merge", "deprecate", "manual"] = "deprecate",
    ) -> Optional[SkillConflict]:
        """Resolve a conflict using the specified strategy.

        Strategies:
        - ``merge``: Combine the ``tier2_core`` of both skills into the
          higher-Q skill and archive the other.
        - ``deprecate``: Mark the skill with the lower Q-value as
          deprecated.
        - ``manual``: Flag the conflict for human review (no auto-action).

        Parameters:
            conflict_id: The unique ID of the conflict to resolve.
            strategy: Resolution strategy to apply.

        Returns:
            The updated :class:`SkillConflict` if found, else ``None``.
        """
        conflict = self._load_conflict(conflict_id)
        if conflict is None:
            return None

        if strategy == "merge":
            self._resolve_merge(conflict)
        elif strategy == "deprecate":
            self._resolve_deprecate(conflict)
        # "manual" — no automatic action taken.

        conflict.resolved = True
        self._update_conflict(conflict)
        return conflict

    def get_conflicts(
        self,
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
    ) -> list[SkillConflict]:
        """Retrieve persisted conflicts with optional filters.

        Parameters:
            severity: Filter by severity level (``low``, ``medium``,
                ``high``, ``critical``).  ``None`` means no filter.
            resolved: Filter by resolved status.  ``None`` means no filter.

        Returns:
            A list of matching :class:`SkillConflict` records.
        """
        query = "SELECT * FROM skill_conflicts WHERE 1=1"
        params: list[Any] = []

        if severity is not None:
            query += " AND severity = ?"
            params.append(severity)
        if resolved is not None:
            query += " AND resolved = ?"
            params.append(int(resolved))

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_conflict(row) for row in rows]

    def auto_resolve_all(
        self,
        strategy: Literal["merge", "deprecate", "manual"] = "deprecate",
    ) -> list[SkillConflict]:
        """Auto-resolve every unresolved conflict.

        Parameters:
            strategy: The resolution strategy to apply to all unresolved
                conflicts.

        Returns:
            A list of conflicts that were successfully resolved.
        """
        unresolved = self.get_conflicts(resolved=False)
        resolved_conflicts: list[SkillConflict] = []
        for conflict in unresolved:
            result = self.resolve_conflict(conflict.conflict_id, strategy)
            if result is not None:
                resolved_conflicts.append(result)
        return resolved_conflicts

    # ------------------------------------------------------------------
    # Overlap detection
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Lowercase tokenise *text* into a set of words."""
        return set(re.findall(r"[a-z]+", text.lower()))

    def _check_keyword_overlap(
        self, skill_a: Skill, skill_b: Skill
    ) -> Optional[SkillConflict]:
        """Return a :class:`SkillConflict` if keyword overlap exceeds threshold."""
        desc_a = self._tokenize(skill_a.tier1_metadata)
        desc_b = self._tokenize(skill_b.tier1_metadata)

        if not desc_a or not desc_b:
            return None

        intersection = desc_a & desc_b
        union = desc_a | desc_b
        jaccard = len(intersection) / len(union) if union else 0.0

        if jaccard < self._OVERLAP_THRESHOLD:
            return None

        severity: Literal["low", "medium", "high", "critical"]
        if jaccard > 0.90:
            severity = "critical"
        elif jaccard > 0.80:
            severity = "high"
        elif jaccard > 0.70:
            severity = "medium"
        else:
            severity = "low"

        return SkillConflict(
            skill_a_id=skill_a.id,
            skill_b_id=skill_b.id,
            conflict_type="overlap",
            severity=severity,
            description=(
                f"Keyword overlap ({jaccard:.0%}) between "
                f"'{skill_a.id}' and '{skill_b.id}'."
            ),
        )

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    def _check_contradiction(
        self, skill_a: Skill, skill_b: Skill
    ) -> Optional[SkillConflict]:
        """Return a :class:`SkillConflict` if tier-2 cores contradict each other."""
        core_a = (skill_a.tier2_core or "").lower()
        core_b = (skill_b.tier2_core or "").lower()

        if not core_a or not core_b:
            return None

        for positive_set, negative_set in self._OPPOSITES:
            has_positive_a = any(w in core_a for w in positive_set)
            has_negative_a = any(w in core_a for w in negative_set)
            has_positive_b = any(w in core_b for w in positive_set)
            has_negative_b = any(w in core_b for w in negative_set)

            # Cross-skill contradiction: one says "always", other says "never".
            if (has_positive_a and has_negative_b) or (
                has_negative_a and has_positive_b
            ):
                return SkillConflict(
                    skill_a_id=skill_a.id,
                    skill_b_id=skill_b.id,
                    conflict_type="contradiction",
                    severity="high",
                    description=(
                        f"Contradictory instructions between "
                        f"'{skill_a.id}' and '{skill_b.id}': "
                        f"positive={positive_set} vs negative={negative_set}."
                    ),
                )

        return None

    # ------------------------------------------------------------------
    # Circular dependency detection
    # ------------------------------------------------------------------

    def _check_circular_dependencies(self) -> list[SkillConflict]:
        """Detect cycles in the dependency graph via iterative DFS."""
        all_nodes = self._graph.get_skills()

        # Build forward adjacency from the graph.
        adj: dict[str, list[str]] = {}
        for node_id in all_nodes:
            deps = self._graph.get_dependencies(node_id)
            adj[node_id] = [dep_id for dep_id, _ in deps]

        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def _dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbour in adj.get(node, []):
                if neighbour not in visited:
                    _dfs(neighbour, path)
                elif neighbour in rec_stack:
                    # Found a cycle — extract it.
                    cycle_start = path.index(neighbour)
                    cycle = path[cycle_start:] + [neighbour]
                    cycles.append(cycle)

            path.pop()
            rec_stack.discard(node)

        for node in all_nodes:
            if node not in visited:
                _dfs(node, [])

        conflicts: list[SkillConflict] = []
        seen_pairs: set[frozenset[str]] = set()

        for cycle in cycles:
            for idx in range(len(cycle) - 1):
                pair = frozenset({cycle[idx], cycle[idx + 1]})
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                a, b = sorted(pair)
                conflicts.append(
                    SkillConflict(
                        skill_a_id=a,
                        skill_b_id=b,
                        conflict_type="circular_dependency",
                        severity="critical",
                        description=(
                            f"Circular dependency detected involving "
                            f"'{a}' and '{b}' "
                            f"(cycle: {' -> '.join(cycle)})."
                        ),
                    )
                )

        return conflicts

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def _resolve_merge(self, conflict: SkillConflict) -> None:
        """Merge tier-2 cores into the higher-Q skill and deprecate the other."""
        skill_a = self._registry.get_skill(conflict.skill_a_id, tier=3)
        skill_b = self._registry.get_skill(conflict.skill_b_id, tier=3)

        if skill_a is None or skill_b is None:
            return

        # Keep the higher-Q skill, deprecate the lower-Q one.
        if skill_a.q_value >= skill_b.q_value:
            keeper, loser = skill_a, skill_b
        else:
            keeper, loser = skill_b, skill_a

        merged_core = self._merge_cores(keeper.tier2_core, loser.tier2_core)
        self._registry.update_skill(keeper.id, {"tier2_core": merged_core})
        self._registry.update_skill(
            loser.id,
            {
                "lifecycle": SkillLifecycle.DEPRECATED,
                "tier1_metadata": f"{loser.tier1_metadata} [merged into {keeper.id}]",
            },
        )

    def _resolve_deprecate(self, conflict: SkillConflict) -> None:
        """Deprecate the lower-Q skill."""
        skill_a = self._registry.get_skill(conflict.skill_a_id, tier=2)
        skill_b = self._registry.get_skill(conflict.skill_b_id, tier=2)

        if skill_a is None or skill_b is None:
            return

        if skill_a.q_value >= skill_b.q_value:
            loser = skill_b
        else:
            loser = skill_a

        self._registry.update_skill(
            loser.id,
            {
                "lifecycle": SkillLifecycle.DEPRECATED,
                "tier1_metadata": (
                    f"{loser.tier1_metadata} [deprecated: conflict {conflict.conflict_id}]"
                ),
            },
        )

    @staticmethod
    def _merge_cores(core_a: Optional[str], core_b: Optional[str]) -> str:
        """Merge two tier-2 core instruction blocks by concatenation."""
        parts: list[str] = []
        if core_a:
            parts.append(core_a.strip())
        if core_b:
            parts.append(core_b.strip())
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _conflict_exists(self, conflict: SkillConflict) -> bool:
        """Check whether a conflict with the same skill pair and type exists."""
        row = self._conn.execute(
            """
            SELECT 1 FROM skill_conflicts
            WHERE skill_a_id = ? AND skill_b_id = ? AND conflict_type = ?
            """,
            (conflict.skill_a_id, conflict.skill_b_id, conflict.conflict_type),
        ).fetchone()
        return row is not None

    def _persist_conflict(self, conflict: SkillConflict) -> None:
        """Insert a new conflict record into the database."""
        self._conn.execute(
            """
            INSERT INTO skill_conflicts
                (conflict_id, skill_a_id, skill_b_id, conflict_type,
                 severity, description, detected_at, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conflict.conflict_id,
                conflict.skill_a_id,
                conflict.skill_b_id,
                conflict.conflict_type,
                conflict.severity,
                conflict.description,
                conflict.detected_at,
                int(conflict.resolved),
            ),
        )
        self._conn.commit()

    def _load_conflict(self, conflict_id: str) -> Optional[SkillConflict]:
        """Load a single conflict by its ID."""
        row = self._conn.execute(
            "SELECT * FROM skill_conflicts WHERE conflict_id = ?",
            (conflict_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_conflict(row)

    def _update_conflict(self, conflict: SkillConflict) -> None:
        """Update an existing conflict record."""
        self._conn.execute(
            """
            UPDATE skill_conflicts
            SET resolved = ?, description = ?, severity = ?
            WHERE conflict_id = ?
            """,
            (
                int(conflict.resolved),
                conflict.description,
                conflict.severity,
                conflict.conflict_id,
            ),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_conflict(row: tuple[Any, ...]) -> SkillConflict:
        """Convert a raw SQLite row into a :class:`SkillConflict`."""
        return SkillConflict(
            conflict_id=row[0],
            skill_a_id=row[1],
            skill_b_id=row[2],
            conflict_type=row[3],  # type: ignore[arg-type]
            severity=row[4],  # type: ignore[arg-type]
            description=row[5],
            detected_at=row[6],
            resolved=bool(row[7]),
        )
