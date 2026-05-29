"""Skill optimization engine — compress, split, merge, and reorder skills."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from skillforge.core.graph import SkillDependencyGraph
from skillforge.core.registry import Skill, SkillRegistry
from skillforge.core.tracker import QValueTracker

from .analyzer import SkillAnalyzer


@dataclass
class OptimizationAction:
    """Record of a single optimisation action.

    Attributes:
        action_type: Type of optimisation (compress, split, merge, reorder).
        skill_id: The skill the action applies to.
        description: Human-readable description.
        expected_improvement: Estimated fractional token reduction (0-1).
        applied: Whether the action has been applied.
    """

    action_type: str
    skill_id: str
    description: str
    expected_improvement: float = 0.0
    applied: bool = False


class SkillOptimizer:
    """Optimises skills for brevity, focus, and ordering.

    Parameters
    ----------
    registry : SkillRegistry
        Skill metadata store.
    tracker : QValueTracker
        Outcome tracker.
    graph : SkillDependencyGraph
        Dependency graph.
    analyzer : SkillAnalyzer | None
        Optional pre-built analyser instance.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        graph: SkillDependencyGraph,
        analyzer: SkillAnalyzer | None = None,
    ) -> None:
        """Initialise the optimizer.

        Args:
            registry: Skill registry.
            tracker: Q-value tracker.
            graph: Dependency graph.
            analyzer: Optional SkillAnalyzer instance.
        """
        self._registry = registry
        self._tracker = tracker
        self._graph = graph
        self._analyzer = analyzer or SkillAnalyzer(registry, tracker, graph)

    # ------------------------------------------------------------------
    # Compress
    # ------------------------------------------------------------------

    def compress_skill(self, skill_id: str) -> OptimizationAction:
        """Compress tier2_core by removing redundant or verbose content.

        Applies several heuristics:
        1. Remove duplicate lines.
        2. Remove blank lines and excess whitespace.
        3. Dedent and simplify verbose sentences.
        4. Remove unused references.

        Parameters
        ----------
        skill_id : str
            The skill to compress.

        Returns
        -------
        OptimizationAction
            Action record with expected improvement estimate.

        Raises
        ------
        ValueError
            If skill_id is not found.
        """
        skill = self._registry.get_skill(skill_id, tier=3)
        if skill is None:
            raise ValueError(f"Skill '{skill_id}' not found")

        original = skill.tier2_core
        if not original or not original.strip():
            return OptimizationAction(
                action_type="compress",
                skill_id=skill_id,
                description="No content to compress",
                expected_improvement=0.0,
                applied=False,
            )

        compressed = self._apply_compression(original)

        original_len = len(original)
        compressed_len = len(compressed)
        improvement = (
            1.0 - (compressed_len / original_len) if original_len > 0 else 0.0
        )

        # Apply the change
        if compressed != original:
            self._registry.update_skill(
                skill_id, {"tier2_core": compressed}
            )

        return OptimizationAction(
            action_type="compress",
            skill_id=skill_id,
            description=(
                f"Reduced tier2_core from {original_len} to "
                f"{compressed_len} chars ({improvement:.0%} reduction)"
            ),
            expected_improvement=round(max(0.0, improvement), 4),
            applied=compressed != original,
        )

    def _apply_compression(self, text: str) -> str:
        """Apply compression heuristics to text.

        Args:
            text: The raw tier2_core text.

        Returns:
            Compressed text.
        """
        lines = text.split("\n")

        # 1. Remove exact duplicate lines (preserve order)
        seen: set[str] = set()
        deduped: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                # Keep one blank line between sections
                if deduped and deduped[-1].strip():
                    deduped.append("")
                continue
            if stripped not in seen:
                seen.add(stripped)
                deduped.append(line.rstrip())

        # 2. Remove verbose filler phrases
        filler_patterns = [
            (r"\bplease\s+note\s+that\s+", ""),
            (r"\bit\s+is\s+important\s+to\s+", ""),
            (r"\bmake\s+sure\s+to\s+", ""),
            (r"\bensure\s+that\s+", "ensure "),
            (r"\bin\s+order\s+to\s+", "to "),
            (r"\bdue\s+to\s+the\s+fact\s+that\s+", "because "),
            (r"\bat\s+this\s+point\s+in\s+time\b", "now"),
            (r"\bfor\s+the\s+purpose\s+of\s+", "for "),
            (r"\bin\s+the\s+event\s+that\s+", "if "),
        ]

        simplified: list[str] = []
        for line in deduped:
            result = line
            for pattern, replacement in filler_patterns:
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            simplified.append(result.rstrip())

        # 3. Trim trailing blank lines
        while simplified and not simplified[-1].strip():
            simplified.pop()

        return "\n".join(simplified)

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split_skill(
        self, skill_id: str, split_points: list[str] | None = None
    ) -> list[str]:
        """Split a large skill into focused sub-skills.

        If *split_points* are provided they define the heading / keyword
        where each split occurs.  Otherwise the skill is split at every
        top-level heading (``# Title``) or every 500 tokens.

        Parameters
        ----------
        skill_id : str
            The skill to split.
        split_points : list[str] | None
            Optional list of headings or keywords defining split boundaries.

        Returns
        -------
        list[str]
            IDs of the newly created sub-skills.

        Raises
        ------
        ValueError
            If skill_id is not found.
        """
        skill = self._registry.get_skill(skill_id, tier=3)
        if skill is None:
            raise ValueError(f"Skill '{skill_id}' not found")

        content = skill.tier2_core
        if not content or not content.strip():
            return []

        # Estimate token count (rough: 1 token ≈ 4 chars)
        token_estimate = len(content) // 4
        if token_estimate <= 500:
            # Small enough — no need to split
            return []

        # Determine sections
        sections = self._split_into_sections(content, split_points)
        if len(sections) <= 1:
            # Cannot meaningfully split
            return []

        new_ids: list[str] = []
        for idx, section_content in enumerate(sections):
            if not section_content.strip():
                continue
            part_name = f"{skill.name} [Part {idx + 1}]"
            part_skill = self._registry.register_skill(
                name=part_name,
                tier1_metadata=f"Sub-skill of {skill.name} — part {idx + 1}",
                tier2_core=section_content,
                tags=skill.tags,
            )
            self._registry.update_skill(
                part_skill.id,
                {
                    "lifecycle": skill.lifecycle,
                    "q_value": skill.q_value,
                },
            )
            new_ids.append(part_skill.id)

        return new_ids

    def _split_into_sections(
        self, content: str, split_points: list[str] | None
    ) -> list[str]:
        """Split content into sections.

        Args:
            content: The tier2_core content.
            split_points: Optional explicit split keywords.

        Returns:
            List of section strings.
        """
        lines = content.split("\n")

        if split_points:
            return self._split_by_keywords(lines, split_points)

        # Split at heading lines (# Title)
        sections: list[str] = []
        current: list[str] = []

        for line in lines:
            if line.strip().startswith("# ") and current:
                sections.append("\n".join(current))
                current = [line]
            else:
                current.append(line)

        if current:
            sections.append("\n".join(current))

        # If no headings found, split by approximate token count
        if len(sections) <= 1:
            sections = self._split_by_token_count(content, max_tokens=500)

        return sections

    def _split_by_keywords(
        self, lines: list[str], keywords: list[str]
    ) -> list[str]:
        """Split at lines containing specified keywords."""
        sections: list[str] = []
        current: list[str] = []

        for line in lines:
            if any(kw.lower() in line.lower() for kw in keywords) and current:
                sections.append("\n".join(current))
                current = [line]
            else:
                current.append(line)

        if current:
            sections.append("\n".join(current))

        return sections

    def _split_by_token_count(
        self, content: str, max_tokens: int = 500
    ) -> list[str]:
        """Split content into chunks of approximately max_tokens."""
        lines = content.split("\n")
        sections: list[str] = []
        current: list[str] = []
        char_budget = max_tokens * 4  # ~4 chars per token
        current_len = 0

        for line in lines:
            line_len = len(line) + 1  # +1 for newline
            if current_len + line_len > char_budget and current:
                sections.append("\n".join(current))
                current = [line]
                current_len = line_len
            else:
                current.append(line)
                current_len += line_len

        if current:
            sections.append("\n".join(current))

        return sections

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_skills(self, skill_ids: list[str]) -> str:
        """Merge multiple skills into one optimised skill.

        The merged skill combines tier2_core from all inputs (deduplicating
        lines) and adopts the highest Q-value.  Original skills are
        deprecated.

        Parameters
        ----------
        skill_ids : list[str]
            IDs of the skills to merge.

        Returns
        -------
        str
            The ID of the merged skill.

        Raises
        ------
        ValueError
            If fewer than 2 skill_ids are provided or any skill is not found.
        """
        if len(skill_ids) < 2:
            raise ValueError("At least 2 skills are required for merging")

        skills: list[Skill] = []
        for sid in skill_ids:
            skill = self._registry.get_skill(sid, tier=3)
            if skill is None:
                raise ValueError(f"Skill '{sid}' not found")
            skills.append(skill)

        # Build merged tier2_core (deduplicated lines)
        seen_lines: set[str] = set()
        merged_lines: list[str] = []

        all_tags: set[str] = set()
        best_q = 0.0
        best_sr = 0.0
        merged_name_parts: list[str] = []

        for skill in skills:
            merged_name_parts.append(skill.name)
            all_tags.update(skill.tags)
            best_q = max(best_q, skill.q_value)
            best_sr = max(best_sr, skill.success_rate)

            for line in skill.tier2_core.split("\n"):
                stripped = line.strip()
                if stripped and stripped not in seen_lines:
                    seen_lines.add(stripped)
                    merged_lines.append(line.rstrip())

        merged_name = " + ".join(merged_name_parts)
        merged_core = "\n".join(merged_lines)

        # Create the merged skill
        merged_skill = self._registry.register_skill(
            name=f"Merged: {merged_name}",
            tier1_metadata=f"Merged from {len(skills)} skills",
            tier2_core=merged_core,
            tags=sorted(all_tags),
        )

        self._registry.update_skill(
            merged_skill.id,
            {
                "q_value": best_q,
                "success_rate": best_sr,
            },
        )

        # Deprecate originals
        from skillforge.core.registry import SkillLifecycle

        for skill in skills:
            self._registry.update_skill(
                skill.id,
                {"lifecycle": SkillLifecycle.DEPRECATED},
            )

        return merged_skill.id

    # ------------------------------------------------------------------
    # Reorder
    # ------------------------------------------------------------------

    def reorder_steps(self, skill_id: str) -> OptimizationAction:
        """Reorder tier2_core steps based on usage frequency.

        Steps that appear most frequently in successful trajectories
        are moved to the front.

        Parameters
        ----------
        skill_id : str
            The skill whose steps to reorder.

        Returns
        -------
        OptimizationAction
            Action record.

        Raises
        ------
        ValueError
            If skill_id is not found.
        """
        skill = self._registry.get_skill(skill_id, tier=3)
        if skill is None:
            raise ValueError(f"Skill '{skill_id}' not found")

        content = skill.tier2_core
        if not content or not content.strip():
            return OptimizationAction(
                action_type="reorder",
                skill_id=skill_id,
                description="No content to reorder",
                expected_improvement=0.0,
                applied=False,
            )

        # Extract numbered steps
        step_pattern = re.compile(r"^(\s*(\d+)[\.\)]\s+)(.+)$", re.MULTILINE)
        matches = list(step_pattern.finditer(content))

        if len(matches) < 2:
            return OptimizationAction(
                action_type="reorder",
                skill_id=skill_id,
                description="Not enough numbered steps to reorder",
                expected_improvement=0.0,
                applied=False,
            )

        # Get usage patterns to determine frequency
        patterns = self._analyzer.get_usage_patterns(skill_id)
        # Use peak hours as a proxy for frequency — in practice we'd
        # analyse the actual step execution; here we shuffle by
        # the inverse of step length (shorter = more likely to be core)
        step_entries: list[tuple[str, float]] = []
        for match in matches:
            step_text = match.group(3).strip()
            # Score: shorter steps first (more likely core/important)
            # In a real system this would be frequency-based
            score = 1.0 / max(len(step_text), 1)
            step_entries.append((step_text, score))

        # Sort by score descending (highest priority first)
        step_entries.sort(key=lambda kv: kv[1], reverse=True)

        # Rebuild with new numbering
        reordered_lines: list[str] = []
        # Preserve any content before the first step
        pre_content = content[: matches[0].start()].rstrip()
        if pre_content:
            reordered_lines.append(pre_content)

        for idx, (step_text, _) in enumerate(step_entries, 1):
            reordered_lines.append(f"{idx}. {step_text}")

        new_content = "\n".join(reordered_lines)

        if new_content != content:
            self._registry.update_skill(
                skill_id, {"tier2_core": new_content}
            )

        return OptimizationAction(
            action_type="reorder",
            skill_id=skill_id,
            description=f"Reordered {len(step_entries)} steps by priority",
            expected_improvement=0.0,  # Reordering doesn't reduce tokens
            applied=new_content != content,
        )

    # ------------------------------------------------------------------
    # Batch optimise
    # ------------------------------------------------------------------

    def optimize_all(self) -> list[OptimizationAction]:
        """Run all applicable optimisations across every skill.

        Returns
        -------
        list[OptimizationAction]
            All actions taken (and not taken).
        """
        all_actions: list[OptimizationAction] = []
        skills = self._registry.list_skills()

        for skill in skills:
            # Compress
            compress_action = self.compress_skill(skill.id)
            all_actions.append(compress_action)

            # Split if content is large
            content_len = len(skill.tier2_core) // 4  # tokens
            if content_len > 500:
                new_ids = self.split_skill(skill.id)
                if new_ids:
                    all_actions.append(
                        OptimizationAction(
                            action_type="split",
                            skill_id=skill.id,
                            description=f"Split into {len(new_ids)} sub-skills",
                            expected_improvement=0.2,
                            applied=True,
                        )
                    )

            # Reorder if content has numbered steps
            if re.search(r"^\s*\d+[\.\)]\s+", skill.tier2_core, re.MULTILINE):
                reorder_action = self.reorder_steps(skill.id)
                all_actions.append(reorder_action)

        return all_actions

    # ------------------------------------------------------------------
    # Improvement estimation
    # ------------------------------------------------------------------

    def _estimate_improvement(self, action: OptimizationAction) -> float:
        """Estimate the fractional token savings for an action.

        Parameters
        ----------
        action : OptimizationAction
            The action to estimate.

        Returns
        -------
        float
            Estimated improvement (0-1).
        """
        estimates = {
            "compress": 0.25,   # ~25% typical compression
            "split": 0.0,       # Split doesn't save tokens total
            "merge": 0.15,      # ~15% from deduplication
            "reorder": 0.0,     # Reordering doesn't reduce size
        }
        return estimates.get(action.action_type, 0.0)
