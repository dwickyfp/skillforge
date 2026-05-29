"""SkillForge main orchestrator — ties all core components together.

Provides a single unified API for skill management, progressive loading,
outcome tracking, evolution, and self-diagnosis.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .core.diagnosis import SelfDiagnosisEngine
from .core.evolution import EvolutionLoop, EvolutionReport
from .core.graph import SkillDependencyGraph
from .core.loader import ProgressiveLoader
from .core.registry import Skill, SkillLifecycle, SkillRegistry
from .core.tracker import EffectivenessTracker, Outcome, QValueTracker

logger = logging.getLogger(__name__)

_DEFAULT_DB = "~/.skillforge/skillforge.db"


class SkillForge:
    """Main SkillForge orchestrator.

    Initialises and wires together every core component — registry,
    effectiveness tracker, progressive loader, dependency graph, evolution
    loop, and self-diagnosis engine — behind a single, cohesive API.

    Parameters
    ----------
    db_path : str | Path
        Path to the shared SQLite database.  All database-backed
        components will use this file.  Parent directories are created
        automatically.  Defaults to ``~/.skillforge/skillforge.db``.
    llm_fn : Callable[[str], str] | None
        Optional callable that accepts a prompt string and returns an
        LLM response string.  Passed to the :class:`SelfDiagnosisEngine`
        for advanced failure analysis.  When *None* the engine falls
        back to rule-based heuristics.

    Examples
    --------
    >>> forge = SkillForge()
    >>> forge.register_skill("greeting", "Respond to greetings")
    >>> skills = forge.load_skill("hello world")
    >>> forge.record_outcome(skills[0].id, success=True, latency_ms=120)
    >>> forge.close()
    """

    def __init__(
        self,
        db_path: str | Path = _DEFAULT_DB,
        llm_fn: Callable[[str], str] | None = None,
    ) -> None:
        resolved = Path(db_path).expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._db_path: Path = resolved
        self._llm_fn = llm_fn

        # --- Core persistent components (share the same DB) ---
        self._registry = SkillRegistry(db_path=self._db_path)
        self._tracker = QValueTracker(db_path=self._db_path)

        # --- In-memory components ---
        self._graph = SkillDependencyGraph()

        # --- Composed helpers ---
        self._loader = ProgressiveLoader(self._registry, self._tracker)
        self._diagnosis = SelfDiagnosisEngine(
            registry=self._registry,  # type: ignore[arg-type]
            tracker=self._tracker,  # type: ignore[arg-type]
            llm_fn=llm_fn,
        )
        self._evolution = EvolutionLoop(
            registry=self._registry,  # type: ignore[arg-type]
            tracker=self._tracker,  # type: ignore[arg-type]
            graph=self._graph,  # type: ignore[arg-type]
            diagnosis=self._diagnosis,  # type: ignore[arg-type]
        )

        self._closed = False
        logger.debug("SkillForge initialised (db=%s)", self._db_path)

    # ------------------------------------------------------------------
    # Skill loading
    # ------------------------------------------------------------------

    def load_skill(
        self,
        query: str,
        tier: int = 1,
        routing: str = "q_value",
        limit: int = 5,
    ) -> list[Skill]:
        """Load skills matching *query* at the requested detail *tier*.

        Delegates to :class:`ProgressiveLoader` which performs tier-by-tier
        retrieval and ranks results by the chosen *routing* strategy.

        Parameters
        ----------
        query : str
            Natural-language query or keywords.
        tier : int
            Maximum detail level (1 = metadata only, 2 = core prompt,
            3 = full with resources).
        routing : str
            Ranking strategy — ``"q_value"``, ``"success_rate"``,
            ``"usage_count"``, or ``"relevance"``.
        limit : int
            Maximum number of skills to return.

        Returns
        -------
        list[Skill]
            Ranked list of matching skills at the requested tier.
        """
        self._ensure_open()
        return self._loader.load_skill(query, tier=tier, routing=routing, limit=limit)

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        skill_id: str,
        success: bool,
        latency_ms: float | None = None,
        tokens_used: int | None = None,
        user_feedback: float | None = None,
    ) -> None:
        """Record an execution outcome and update the skill's rolling stats.

        The outcome is persisted to the :class:`QValueTracker` **and** the
        skill's ``usage_count`` / ``success_rate`` fields in the registry
        are updated to stay in sync.

        Parameters
        ----------
        skill_id : str
            The skill that produced the outcome.
        success : bool
            Whether the execution was successful.
        latency_ms : float | None
            Wall-clock latency in milliseconds (defaults to 0.0).
        tokens_used : int | None
            Number of LLM tokens consumed (defaults to 0).
        user_feedback : float | None
            Optional user feedback score on a 0–5 scale.
        """
        self._ensure_open()

        outcome = Outcome(
            skill_id=skill_id,
            success=success,
            latency_ms=latency_ms if latency_ms is not None else 0.0,
            tokens_used=tokens_used if tokens_used is not None else 0,
            user_feedback=user_feedback,
        )
        self._tracker.record_outcome(outcome)

        # Keep registry-level counters in sync
        stats = self._tracker.get_stats(skill_id)
        self._registry.update_skill(
            skill_id,
            {
                "usage_count": stats["total_outcomes"],
                "success_rate": stats["success_rate"],
                "q_value": stats["q_value"],
            },
        )

        logger.debug(
            "Recorded outcome for '%s': success=%s latency=%.1fms",
            skill_id,
            success,
            outcome.latency_ms,
        )

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def run_evolution_loop(
        self,
        thresholds: dict[str, float] | None = None,
    ) -> EvolutionReport:
        """Run a full evolution cycle across all skills.

        Evaluates every skill's health, evolves under-performers, prunes
        dead skills, and returns a detailed report.

        Parameters
        ----------
        thresholds : dict[str, float] | None
            Optional overrides for the default evolution thresholds
            (e.g. ``{"q_critical": 0.25}``).  Unspecified keys fall back
            to :attr:`EvolutionLoop.DEFAULT_THRESHOLDS`.

        Returns
        -------
        EvolutionReport
            Comprehensive summary of the cycle.
        """
        self._ensure_open()
        report = self._evolution.run_evolution_loop(thresholds=thresholds)
        logger.info(
            "Evolution cycle complete: %d evaluated, %d evolved, %d pruned",
            report.total_skills_evaluated,
            report.skills_evolved,
            report.skills_pruned,
        )
        return report

    # ------------------------------------------------------------------
    # Stats / queries
    # ------------------------------------------------------------------

    def get_skill_stats(
        self,
        skill_id: str | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Return performance statistics for one or all skills.

        Parameters
        ----------
        skill_id : str | None
            If given, return a stats dict for that single skill.
            If *None*, return a list of stats dicts for every registered
            skill.

        Returns
        -------
        dict | list[dict]
            Statistics merged from the tracker (Q-value, success rate,
            latency, tokens) and the registry (name, version, lifecycle,
            tags, usage count).
        """
        self._ensure_open()

        if skill_id is not None:
            return self._build_stats(skill_id)

        skills = self._registry.list_skills(limit=10_000)
        return [self._build_stats(s.id, skill=s) for s in skills]

    def _build_stats(
        self,
        skill_id: str,
        skill: Skill | None = None,
    ) -> dict[str, Any]:
        """Merge tracker stats with registry metadata for one skill."""
        tracker_stats = self._tracker.get_stats(skill_id)

        if skill is None:
            skill = self._registry.get_skill(skill_id, tier=1)

        registry_info: dict[str, Any] = {}
        if skill is not None:
            registry_info = {
                "id": skill.id,
                "name": skill.name,
                "version": skill.version,
                "lifecycle": skill.lifecycle.value,
                "tags": skill.tags,
                "created_at": skill.created_at.isoformat(),
                "updated_at": skill.updated_at.isoformat(),
            }

        return {**registry_info, **tracker_stats}

    # ------------------------------------------------------------------
    # Skill registration / import
    # ------------------------------------------------------------------

    def register_skill(
        self,
        name: str,
        tier1_metadata: str,
        tier2_core: str = "",
        tier3_resources: list[str] | None = None,
        tags: list[str] | None = None,
        skill_id: str | None = None,
    ) -> Skill:
        """Register a brand-new skill in the forge.

        Parameters
        ----------
        name : str
            Human-readable skill name.
        tier1_metadata : str
            Short summary (~30 tokens) used for routing.
        tier2_core : str
            Full core prompt / instructions.
        tier3_resources : list[str] | None
            Auxiliary file paths or URLs.
        tags : list[str] | None
            Descriptive tags for search and filtering.
        skill_id : str | None
            Explicit ID.  A UUID is generated when *None*.

        Returns
        -------
        Skill
            The newly registered skill.
        """
        self._ensure_open()
        skill = self._registry.register_skill(
            name=name,
            tier1_metadata=tier1_metadata,
            tier2_core=tier2_core,
            tier3_resources=tier3_resources,
            tags=tags,
            skill_id=skill_id,
        )
        self._graph.add_skill(skill.id, metadata={"name": name})
        logger.info("Registered skill '%s' (%s)", skill.name, skill.id)
        return skill

    def import_skill(self, skill_dict: dict[str, Any]) -> Skill:
        """Import a skill from a dictionary (e.g. agentskills.io format).

        Expected keys (all optional except *name*):

        * ``name`` — skill name (required).
        * ``id`` / ``skill_id`` — explicit identifier.
        * ``description`` / ``tier1_metadata`` — short summary.
        * ``instructions`` / ``tier2_core`` — full instructions.
        * ``resources`` / ``tier3_resources`` — auxiliary file list.
        * ``tags`` — list of tag strings.
        * ``version`` — ignored on import (always starts at 1).

        Parameters
        ----------
        skill_dict : dict
            Skill specification dictionary.

        Returns
        -------
        Skill
            The imported (registered) skill.

        Raises
        ------
        ValueError
            If *name* is missing from the dictionary.
        """
        self._ensure_open()

        name = skill_dict.get("name")
        if not name:
            raise ValueError("skill_dict must contain a 'name' key")

        tier1 = (
            skill_dict.get("tier1_metadata")
            or skill_dict.get("description")
            or ""
        )
        tier2 = (
            skill_dict.get("tier2_core")
            or skill_dict.get("instructions")
            or ""
        )
        tier3 = (
            skill_dict.get("tier3_resources")
            or skill_dict.get("resources")
            or []
        )
        tags = skill_dict.get("tags") or []
        skill_id = skill_dict.get("id") or skill_dict.get("skill_id")

        skill = self._registry.register_skill(
            name=name,
            tier1_metadata=tier1,
            tier2_core=tier2,
            tier3_resources=tier3,
            tags=tags,
            skill_id=skill_id,
        )
        self._graph.add_skill(skill.id, metadata={"name": name})
        logger.info("Imported skill '%s' (%s)", skill.name, skill.id)
        return skill

    # ------------------------------------------------------------------
    # Listing / searching
    # ------------------------------------------------------------------

    def list_skills(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[Skill]:
        """List registered skills with optional filters.

        Parameters
        ----------
        filters : dict | None
            Optional filter criteria.  Accepted keys:

            * ``lifecycle`` — a :class:`SkillLifecycle` value or its
              string name (e.g. ``"active"``).
            * ``tags`` — list of tag strings (OR logic).
            * ``limit`` — max results (default 100).
            * ``offset`` — pagination offset (default 0).

        Returns
        -------
        list[Skill]
            Matching skills ordered by Q-value descending.
        """
        self._ensure_open()

        lifecycle: SkillLifecycle | None = None
        tags: list[str] | None = None
        limit = 100
        offset = 0

        if filters:
            raw_lifecycle = filters.get("lifecycle")
            if raw_lifecycle is not None:
                if isinstance(raw_lifecycle, SkillLifecycle):
                    lifecycle = raw_lifecycle
                else:
                    lifecycle = SkillLifecycle(str(raw_lifecycle))
            tags = filters.get("tags")
            limit = int(filters.get("limit", 100))
            offset = int(filters.get("offset", 0))

        return self._registry.list_skills(
            lifecycle=lifecycle,
            tags=tags,
            limit=limit,
            offset=offset,
        )

    def search_skills(self, query: str, limit: int = 20) -> list[Skill]:
        """Keyword search across skill names and metadata.

        Parameters
        ----------
        query : str
            Search string.
        limit : int
            Maximum results.

        Returns
        -------
        list[Skill]
            Matching skills ordered by Q-value descending.
        """
        self._ensure_open()
        return self._registry.search_skills(query, limit=limit)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release all resources (database connections, etc.).

        After calling ``close()`` the forge instance must not be used.
        """
        if self._closed:
            return
        self._registry.close()
        self._tracker.close()
        self._closed = True
        logger.debug("SkillForge closed (db=%s)", self._db_path)

    def _ensure_open(self) -> None:
        """Raise if the forge has been closed."""
        if self._closed:
            raise RuntimeError("SkillForge instance has been closed")

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "SkillForge":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        try:
            skill_count = len(self._registry.list_skills(limit=10_000))
        except Exception:
            skill_count = -1
        return (
            f"SkillForge(db={str(self._db_path)!r}, "
            f"status={status!r}, skills={skill_count})"
        )

    def __del__(self) -> None:
        # Best-effort cleanup; never raises.
        try:
            self.close()
        except Exception:
            pass
