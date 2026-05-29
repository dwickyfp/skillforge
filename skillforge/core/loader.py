"""Progressive skill loader with tier-by-tier retrieval and routing strategies."""

from __future__ import annotations

from typing import Any

from .registry import Skill, SkillLifecycle, SkillRegistry
from .tracker import QValueTracker


class ProgressiveLoader:
    """Loads skills progressively by tier, with multiple routing strategies.

    Parameters
    ----------
    registry : SkillRegistry
        Shared skill registry instance.
    tracker : QValueTracker
        Shared outcome tracker instance.
    """

    ROUTING_STRATEGIES = ("q_value", "success_rate", "usage_count", "relevance")

    def __init__(self, registry: SkillRegistry, tracker: QValueTracker) -> None:
        self._registry = registry
        self._tracker = tracker

    # ------------------------------------------------------------------
    # Relevance
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_relevance(query: str, skill: Skill) -> float:
        """Simple keyword-overlap relevance score (no embeddings needed).

        Returns a float in [0, 1] based on the fraction of query tokens
        found in the skill's name, tags, or tier1_metadata.
        """
        if not query:
            return 0.0

        query_tokens = set(query.lower().split())
        if not query_tokens:
            return 0.0

        corpus = " ".join(
            [skill.name, skill.tier1_metadata] + skill.tags
        ).lower()

        corpus_tokens = set(corpus.split())
        if not corpus_tokens:
            return 0.0

        matches = query_tokens & corpus_tokens
        return len(matches) / len(query_tokens)

    # ------------------------------------------------------------------
    # Main loading
    # ------------------------------------------------------------------

    def load_skill(
        self,
        query: str,
        tier: int = 1,
        routing: str = "q_value",
        limit: int = 5,
    ) -> list[Skill]:
        """Load skills that match *query*, returned at the requested detail *tier*.

        Skills are loaded tier-by-tier: tier-1 metadata is always evaluated
        first, then tier-2 core prompts, then tier-3 resources — only for
        the top-ranked results.

        Parameters
        ----------
        query : str
            Natural-language query or keywords.
        tier : int
            Maximum detail level (1, 2, or 3).
        routing : str
            Ranking strategy — one of ``q_value``, ``success_rate``,
            ``usage_count``, or ``relevance``.
        limit : int
            Maximum number of skills to return.

        Returns
        -------
        list[Skill]
            Ranked and trimmed list of skills at the requested tier level.
        """
        if routing not in self.ROUTING_STRATEGIES:
            raise ValueError(
                f"Unknown routing '{routing}'. Choose from {self.ROUTING_STRATEGIES}"
            )

        # Tier 1: broad candidate retrieval
        candidates = self._registry.search_skills(query, limit=limit * 4)

        # Filter out non-active skills unless nothing else matches
        active = [s for s in candidates if s.lifecycle == SkillLifecycle.ACTIVE]
        if not active:
            active = candidates  # fallback to all matches

        # Enrich with tracker data and compute relevance
        enriched: list[dict[str, Any]] = []
        for skill in active:
            stats = self._tracker.get_stats(skill.id)
            skill.q_value = stats["q_value"]
            skill.success_rate = stats["success_rate"]
            relevance = self._compute_relevance(query, skill)
            enriched.append({"skill": skill, "relevance": relevance, "stats": stats})

        # Sort by chosen routing strategy
        if routing == "q_value":
            enriched.sort(key=lambda e: e["skill"].q_value, reverse=True)
        elif routing == "success_rate":
            enriched.sort(key=lambda e: e["skill"].success_rate, reverse=True)
        elif routing == "usage_count":
            enriched.sort(key=lambda e: e["skill"].usage_count, reverse=True)
        elif routing == "relevance":
            enriched.sort(key=lambda e: e["relevance"], reverse=True)

        top = enriched[:limit]

        # Progressive tier loading: fetch full detail at requested tier
        results: list[Skill] = []
        for entry in top:
            full = self._registry.get_skill(entry["skill"].id, tier=tier)
            if full is not None:
                results.append(full)

        return results

    # ------------------------------------------------------------------
    # Sticky skills
    # ------------------------------------------------------------------

    def get_sticky_skills(self, limit: int = 5) -> list[Skill]:
        """Return the top skills to keep pre-loaded in context.

        Ranked by a composite of Q-value and usage count so that both
        high-quality and frequently-used skills surface.
        """
        all_skills = self._registry.list_skills(
            lifecycle=SkillLifecycle.ACTIVE, limit=200
        )
        for skill in all_skills:
            stats = self._tracker.get_stats(skill.id)
            skill.q_value = stats["q_value"]
            skill.success_rate = stats["success_rate"]

        # Composite score: weighted sum of q_value and normalised usage
        max_usage = max((s.usage_count for s in all_skills), default=1) or 1
        all_skills.sort(
            key=lambda s: 0.6 * s.q_value + 0.4 * (s.usage_count / max_usage),
            reverse=True,
        )
        return all_skills[:limit]
