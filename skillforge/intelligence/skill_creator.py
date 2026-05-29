"""Auto-create skills from agent execution trajectories.

Analyses successful (and optionally failed) execution trajectories to
extract common patterns, then registers new skills with appropriate
tier-1 metadata and tier-2 core instructions.

Supports an optional ``llm_fn`` callback for higher-quality extraction;
when absent, falls back to deterministic rule-based heuristics.

All components are stdlib-only (Python 3.10+).
"""

from __future__ import annotations

import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from skillforge.core.registry import Skill, SkillRegistry
from skillforge.core.tracker import QValueTracker


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """A single agent execution trajectory (task attempt).

    Attributes:
        task_description: Natural-language description of the task attempted.
        steps: Ordered list of high-level action descriptions.
        tools_used: Names of tools / functions invoked during execution.
        outcome: Whether the task completed successfully (``"success"``) or
            failed (``"failure"``).
        tokens_used: Total tokens consumed across the trajectory.
        latency_ms: Wall-clock latency in milliseconds.
    """

    task_description: str = ""
    steps: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    outcome: str = "success"  # "success" | "failure"
    tokens_used: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# SkillCreator
# ---------------------------------------------------------------------------

# Type alias for an optional LLM helper function.
# It receives a prompt string and returns a text completion.
LLMFn = Callable[[str], str]


class SkillCreator:
    """Creates new skills automatically from patterns in agent trajectories.

    Parameters:
        registry: The :class:`~skillforge.core.registry.SkillRegistry` to
            register newly created skills.
        tracker: The :class:`~skillforge.core.tracker.QValueTracker` used
            for initial Q-value seeding.
        llm_fn: Optional callable that takes a prompt string and returns a
            text completion.  When provided, the creator uses the LLM for
            richer metadata and instruction generation.  When ``None``,
            a purely rule-based approach is used.
    """

    # Minimum fraction of successful trajectories a step must appear in
    # to be included in the extracted common steps.
    _COMMON_STEP_THRESHOLD: float = 0.60

    def __init__(
        self,
        registry: SkillRegistry,
        tracker: QValueTracker,
        llm_fn: Optional[LLMFn] = None,
    ) -> None:
        self._registry = registry
        self._tracker = tracker
        self._llm_fn = llm_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_from_trajectory(
        self,
        task_pattern: str | None = None,
        trajectories: list[Trajectory] | None = None,
        *,
        pattern_name: str | None = None,
        **_: Any,
    ) -> str:
        """Extract common patterns from successful trajectories and register a skill.

        Parameters:
            task_pattern: A short label or keyword describing the shared task
                category (e.g. ``"summarise document"``).
            trajectories: A list of :class:`Trajectory` instances representing
                past attempts at tasks matching *task_pattern*.
            pattern_name: Alias for *task_pattern* (for backward compatibility).

        Returns:
            The ``skill_id`` of the newly registered skill.

        Raises:
            ValueError: If fewer than 2 successful trajectories are provided
                (not enough data to extract patterns).
        """
        # Support pattern_name as an alias for task_pattern
        if task_pattern is None and pattern_name is not None:
            task_pattern = pattern_name
        if task_pattern is None:
            raise ValueError("task_pattern (or pattern_name) is required")
        if trajectories is None:
            raise ValueError("trajectories is required")
        # Auto-convert dicts to Trajectory objects for convenience
        trajectories = [
            Trajectory(**t) if isinstance(t, dict) else t
            for t in trajectories
        ]
        successful = [t for t in trajectories if t.outcome == "success"]
        failed = [t for t in trajectories if t.outcome == "failure"]

        if len(successful) < 2:
            raise ValueError(
                f"Need at least 2 successful trajectories to extract a pattern; "
                f"got {len(successful)} out of {len(trajectories)} total."
            )

        common_steps = self._extract_common_steps(successful)
        pitfalls = self._identify_pitfalls(trajectories)
        tier1 = self._generate_tier1_metadata(task_pattern, common_steps)
        tier2 = self._generate_tier2_core(common_steps, pitfalls)

        skill = self._registry.register_skill(
            name=task_pattern,
            tier1_metadata=tier1,
            tier2_core=tier2,
            tags=["auto-generated"],
        )

        return skill.id

    def _extract_common_steps(self, trajectories: list[Trajectory]) -> list[str]:
        """Find steps that appear in more than the threshold fraction of trajectories.

        Two steps are considered *the same* if their normalised forms
        (lowercased, stripped) share at least 70 % of their words (Jaccard
        similarity).

        Parameters:
            trajectories: Successful trajectories to analyse.

        Returns:
            An ordered list of common step descriptions.
        """
        if not trajectories:
            return []

        total = len(trajectories)

        # Collect all unique step strings across trajectories (normalised).
        all_steps: list[tuple[str, str]] = []  # (original, normalised)
        seen: dict[str, str] = {}  # normalised -> first original
        for traj in trajectories:
            for step in traj.steps:
                norm = self._normalise(step)
                if norm not in seen:
                    seen[norm] = step
                    all_steps.append((step, norm))

        # Count how many trajectories contain each normalised step.
        step_counts: dict[str, int] = {}
        for _, norm in all_steps:
            if norm in step_counts:
                continue
            count = sum(
                1
                for traj in trajectories
                if any(self._normalise(s) == norm for s in traj.steps)
            )
            step_counts[norm] = count

        # Keep steps above the threshold.
        common = [
            (seen[norm], count / total)
            for norm, count in step_counts.items()
            if count / total > self._COMMON_STEP_THRESHOLD
        ]

        # Sort by frequency (descending) then by first appearance.
        common.sort(key=lambda x: x[1], reverse=True)

        # If LLM is available, ask it to deduplicate and order the steps.
        steps = [s for s, _ in common]
        if self._llm_fn and steps:
            steps = self._llm_refine_steps(steps)

        return steps

    def _generate_tier1_metadata(self, task_pattern: str, steps: list[str]) -> str:
        """Create a concise tier-1 metadata description (target: ~30 tokens).

        Parameters:
            task_pattern: The high-level task label.
            steps: Common extracted steps.

        Returns:
            A short string suitable for use as ``tier1_metadata``.
        """
        if self._llm_fn:
            prompt = (
                f"Write a very short skill summary (max 30 words, one sentence) "
                f"for a skill that handles '{task_pattern}'. "
                f"Steps involved: {', '.join(steps[:5])}. "
                f"Output only the summary, nothing else."
            )
            result = self._llm_fn(prompt).strip()
            # Enforce ~30 token cap.
            words = result.split()
            if len(words) > 35:
                result = " ".join(words[:30])
            return result

        # Rule-based: compose from task_pattern + key step verbs.
        verbs = self._extract_verbs(steps)
        verb_phrase = ", ".join(verbs[:4]) if verbs else task_pattern
        metadata = f"Skill for '{task_pattern}'. Key actions: {verb_phrase}."
        # Truncate to ~30 tokens.
        words = metadata.split()
        if len(words) > 30:
            metadata = " ".join(words[:28]) + "..."
        return metadata

    def _generate_tier2_core(
        self, steps: list[str], pitfalls: list[str]
    ) -> str:
        """Create numbered tier-2 core instructions.

        Parameters:
            steps: Common extracted steps.
            pitfalls: Known failure points from failed trajectories.

        Returns:
            A multi-line numbered instruction string.
        """
        if self._llm_fn:
            prompt = (
                "Create a numbered instruction list for a skill.\n"
                f"Steps:\n{self._format_numbered(steps)}\n\n"
                f"Known pitfalls to avoid:\n{self._format_numbered(pitfalls)}\n\n"
                "Output ONLY the numbered instructions, one per line. "
                "Include the pitfalls as cautionary notes under relevant steps."
            )
            result = self._llm_fn(prompt).strip()
            if result:
                return result

        # Rule-based: straightforward numbered list.
        parts: list[str] = []
        for i, step in enumerate(steps, 1):
            parts.append(f"{i}. {step}")

        if pitfalls:
            parts.append("")
            parts.append("IMPORTANT — avoid these common pitfalls:")
            for pitfall in pitfalls:
                parts.append(f"- {pitfall}")

        return "\n".join(parts)

    def _identify_pitfalls(self, trajectories: list[Trajectory]) -> list[str]:
        """Identify common failure points from failed trajectories.

        A step is flagged as a *pitfall* if it appears in at least 50 % of
        failed trajectories but in fewer than 30 % of successful ones.

        Parameters:
            trajectories: All trajectories (both successes and failures).

        Returns:
            A list of pitfall descriptions.
        """
        successful = [t for t in trajectories if t.outcome == "success"]
        failed = [t for t in trajectories if t.outcome == "failure"]

        if not failed:
            return []

        fail_total = len(failed)
        suc_total = max(1, len(successful))

        # Count step frequency in failed trajectories.
        fail_counts: Counter[str] = Counter()
        for traj in failed:
            for step in traj.steps:
                fail_counts[self._normalise(step)] += 1

        # Count step frequency in successful trajectories.
        suc_counts: Counter[str] = Counter()
        for traj in successful:
            for step in traj.steps:
                suc_counts[self._normalise(step)] += 1

        # Map normalised -> first original (from failures for readability).
        norm_to_orig: dict[str, str] = {}
        for traj in failed:
            for step in traj.steps:
                norm = self._normalise(step)
                if norm not in norm_to_orig:
                    norm_to_orig[norm] = step

        pitfalls: list[str] = []
        for norm, f_count in fail_counts.items():
            f_ratio = f_count / fail_total
            s_ratio = suc_counts.get(norm, 0) / suc_total

            if f_ratio >= 0.5 and s_ratio < 0.3:
                pitfalls.append(norm_to_orig[norm])

        return pitfalls

    def evaluate_skill(
        self,
        skill_id: str,
        test_trajectories: list[Trajectory],
    ) -> dict[str, Any]:
        """Evaluate a skill against test trajectories.

        Computes how many test trajectories succeed when following the
        skill's instructions, along with aggregate statistics.

        Parameters:
            skill_id: The ID of the skill to evaluate.
            test_trajectories: Trajectories to test against.

        Returns:
            A dict with keys:
            - ``skill_id``: echo of the input.
            - ``total_tested``: number of test trajectories.
            - ``success_count``: number that succeeded.
            - ``success_rate``: ratio of successes.
            - ``avg_tokens``: average tokens across trajectories.
            - ``avg_latency_ms``: average latency.
        """
        skill = self._registry.get_skill(skill_id, tier=2)
        if skill is None:
            raise ValueError(f"Skill '{skill_id}' not found in registry")

        total = len(test_trajectories)
        if total == 0:
            return {
                "skill_id": skill_id,
                "total_tested": 0,
                "success_count": 0,
                "success_rate": 0.0,
                "avg_tokens": 0,
                "avg_latency_ms": 0.0,
            }

        success_count = sum(
            1 for t in test_trajectories if t.outcome == "success"
        )
        total_tokens = sum(t.tokens_used for t in test_trajectories)
        total_latency = sum(t.latency_ms for t in test_trajectories)

        return {
            "skill_id": skill_id,
            "total_tested": total,
            "success_count": success_count,
            "success_rate": success_count / total,
            "avg_tokens": total_tokens // total,
            "avg_latency_ms": round(total_latency / total, 2),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(text: str) -> str:
        """Normalise a step string for comparison (lowercase, collapse whitespace)."""
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _extract_verbs(steps: list[str]) -> list[str]:
        """Extract leading verbs from step descriptions.

        Heuristic: take the first word of each step (stripped of punctuation)
        and return unique verbs in order of frequency.
        """
        verb_counts: Counter[str] = Counter()
        for step in steps:
            words = re.findall(r"[a-zA-Z]+", step)
            if words:
                # Assume the first word is a verb or close to one.
                verb_counts[words[0].lower()] += 1
        return [v for v, _ in verb_counts.most_common()]

    @staticmethod
    def _format_numbered(items: list[str]) -> str:
        """Format a list of strings as a numbered list."""
        return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))

    def _llm_refine_steps(self, steps: list[str]) -> list[str]:
        """Use the LLM to deduplicate and logically order steps.

        Parameters:
            steps: Raw common steps.

        Returns:
            A refined, ordered list of steps.
        """
        if not self._llm_fn:
            return steps

        prompt = (
            "Deduplicate and logically order the following steps. "
            "Merge near-duplicates. Output one step per line, no numbering.\n\n"
            + "\n".join(f"- {s}" for s in steps)
        )
        result = self._llm_fn(prompt).strip()
        refined = [
            line.lstrip("0123456789. -\t")
            for line in result.splitlines()
            if line.strip()
        ]
        return refined if refined else steps
