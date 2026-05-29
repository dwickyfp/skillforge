"""
Reinforcement-learning skill optimizer inspired by Evolving-RL.

Applies iterative optimisation actions (compress, split, reorder) to
individual skills and evaluates each action with a composite reward signal
derived from Q-value improvement, success-rate delta, and token savings.

The optimiser co-evolves skill representations alongside a learned reward
model — mirroring the co-evolution concept from Evolving-RL where both
the policy and the environment representation improve in tandem.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrainingStep:
    """A single optimisation step recorded during RL training.

    Attributes
    ----------
    skill_id : str
        The skill that was optimised.
    action : str
        The optimisation action applied (``optimize``, ``compress``,
        ``split``, ``merge``, ``reorder``).
    reward : float
        Scalar reward computed by :meth:`RLOptimizer._evaluate_reward`.
    before_q : float
        Q-value of the skill *before* the action.
    after_q : float
        Q-value of the skill *after* the action.
    iteration : int
        Which iteration of the optimisation loop this step belongs to.
    timestamp : str
        ISO-8601 UTC timestamp of when the step was recorded.
    """

    skill_id: str
    action: str
    reward: float
    before_q: float
    after_q: float
    iteration: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Protocols (structural typing for loose coupling)
# ---------------------------------------------------------------------------


class SkillRegistryProto(Protocol):
    """Minimal registry interface required by :class:`RLOptimizer`."""

    def get_skill(self, skill_id: str, tier: int = 1) -> Any: ...  # pragma: no cover
    def update_skill(self, skill_id: str, updates: dict[str, Any]) -> Any: ...  # pragma: no cover
    def list_skills(self, **kwargs: Any) -> list[Any]: ...  # pragma: no cover
    def register_skill(self, name: str, tier1_metadata: str, **kwargs: Any) -> Any: ...  # pragma: no cover


class TrackerProto(Protocol):
    """Minimal Q-value tracker interface required by :class:`RLOptimizer`."""

    def get_q_value(self, skill_id: str) -> float: ...  # pragma: no cover
    def get_success_rate(self, skill_id: str) -> float: ...  # pragma: no cover
    def get_stats(self, skill_id: str) -> dict[str, Any]: ...  # pragma: no cover


class EvolutionProto(Protocol):
    """Minimal evolution-loop interface required by :class:`RLOptimizer`."""

    def evolve_skill(self, skill_id: str) -> bool: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# RLOptimizer
# ---------------------------------------------------------------------------


class RLOptimizer:
    """Reinforcement-learning optimizer for skills.

    Iteratively applies structural transformations (compression, splitting,
    reordering) to a skill's ``tier2_core`` content and evaluates each
    transformation with a composite reward signal.  The best-performing
    transformation is kept, mirroring the co-evolution concept from
    Evolving-RL.

    Parameters
    ----------
    registry : SkillRegistryProto
        Skill registry for reading and updating skill definitions.
    tracker : TrackerProto
        Outcome / Q-value tracker for performance signals.
    evolution : EvolutionProto
        Evolution loop for fallback skill-level evolution.
    db_path : str | Path | None
        SQLite database path.  Defaults to ``~/.skillforge/rl_optimizer.db``.
    """

    # Reward weights
    _W_SUCCESS: float = 0.6
    _W_QVALUE: float = 0.3
    _W_TOKENS: float = 0.1

    def __init__(
        self,
        registry: SkillRegistryProto,
        tracker: TrackerProto,
        evolution: EvolutionProto,
        db_path: str | Path | None = None,
    ) -> None:
        self._registry = registry
        self._tracker = tracker
        self._evolution = evolution
        self._db_path = str(
            db_path or Path.home() / ".skillforge" / "rl_optimizer.db"
        )
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create the ``rl_optimization_log`` table if it does not exist."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS rl_optimization_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_id    TEXT NOT NULL,
                action      TEXT NOT NULL,
                reward      REAL NOT NULL,
                before_q    REAL NOT NULL,
                after_q     REAL NOT NULL,
                iteration   INTEGER NOT NULL DEFAULT 0,
                recorded_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rl_log_skill
                ON rl_optimization_log(skill_id);
            CREATE INDEX IF NOT EXISTS idx_rl_log_action
                ON rl_optimization_log(action);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self, skill_id: str, max_iterations: int = 10
    ) -> list[TrainingStep]:
        """Iteratively optimise a skill using RL-inspired actions.

        For each iteration the optimizer tries compression, split, and
        reorder actions.  The action producing the highest reward is
        retained; the others are rolled back.

        Parameters
        ----------
        skill_id : str
            Identifier of the skill to optimise.
        max_iterations : int
            Maximum number of optimisation iterations.

        Returns
        -------
        list[TrainingStep]
            Ordered list of accepted training steps.
        """
        steps: list[TrainingStep] = []
        skill = self._registry.get_skill(skill_id, tier=3)
        if skill is None:
            logger.warning("Skill '%s' not found — skipping optimisation", skill_id)
            return steps

        for iteration in range(max_iterations):
            before_q = self._tracker.get_q_value(skill_id)
            original_core: str = skill.tier2_core if hasattr(skill, "tier2_core") else ""

            best_reward: float = -1.0
            best_action: str | None = None
            best_core: str | None = None

            # --- Try compression ----------------------------------------
            try:
                compress_reward = self._try_compression(skill_id)
                if compress_reward > best_reward:
                    best_reward = compress_reward
                    best_action = "compress"
                    updated = self._registry.get_skill(skill_id, tier=3)
                    best_core = updated.tier2_core if updated else original_core
            except Exception as exc:  # noqa: BLE001
                logger.debug("Compression failed for '%s': %s", skill_id, exc)
            # Rollback to original if compression was not the best
            if best_action != "compress":
                self._registry.update_skill(skill_id, {"tier2_core": original_core})

            # --- Try split ----------------------------------------------
            try:
                split_ids = self._try_split(skill_id)
                if split_ids:
                    # Reward = avg Q of sub-skills relative to parent
                    sub_qs = [self._tracker.get_q_value(sid) for sid in split_ids]
                    split_reward = sum(sub_qs) / len(sub_qs) - before_q
                    if split_reward > best_reward:
                        best_reward = split_reward
                        best_action = "split"
                        best_core = original_core  # parent unchanged
            except Exception as exc:  # noqa: BLE001
                logger.debug("Split failed for '%s': %s", skill_id, exc)

            # --- Try reorder --------------------------------------------
            try:
                reorder_reward = self._try_reorder(skill_id)
                if reorder_reward > best_reward:
                    best_reward = reorder_reward
                    best_action = "reorder"
                    updated = self._registry.get_skill(skill_id, tier=3)
                    best_core = updated.tier2_core if updated else original_core
            except Exception as exc:  # noqa: BLE001
                logger.debug("Reorder failed for '%s': %s", skill_id, exc)
            if best_action != "reorder":
                self._registry.update_skill(skill_id, {"tier2_core": original_core})

            # No improvement found this iteration — stop early
            if best_action is None or best_reward <= 0:
                logger.info(
                    "No positive reward at iteration %d for '%s' — stopping",
                    iteration,
                    skill_id,
                )
                break

            after_q = self._tracker.get_q_value(skill_id)
            step = TrainingStep(
                skill_id=skill_id,
                action=best_action,
                reward=best_reward,
                before_q=before_q,
                after_q=after_q,
                iteration=iteration,
            )
            steps.append(step)
            self._log_step(step)

            # Refresh skill reference for next iteration
            skill = self._registry.get_skill(skill_id, tier=3)
            if skill is None:
                break

        return steps

    # Backward-compatible alias
    optimize_skill = optimize

    def batch_optimize(self, threshold: float = 0.5) -> list[TrainingStep]:
        """Optimise all skills whose Q-value is below *threshold*.

        Parameters
        ----------
        threshold : float
            Q-value cutoff; skills at or above this value are skipped.

        Returns
        -------
        list[TrainingStep]
            Aggregated training steps across all optimised skills.
        """
        all_steps: list[TrainingStep] = []
        skills = self._registry.list_skills()
        for skill in skills:
            sid: str = skill.id if hasattr(skill, "id") else str(skill)
            q_val = self._tracker.get_q_value(sid)
            if q_val < threshold:
                logger.info(
                    "Batch-optimising skill '%s' (Q=%.3f < %.3f)",
                    sid,
                    q_val,
                    threshold,
                )
                steps = self.optimize(sid)
                all_steps.extend(steps)
        return all_steps

    def get_optimization_history(
        self, skill_id: str | None = None
    ) -> list[TrainingStep]:
        """Return logged optimisation steps.

        Parameters
        ----------
        skill_id : str | None
            If provided, filter to a single skill; otherwise return all.

        Returns
        -------
        list[TrainingStep]
            Chronological list of recorded training steps.
        """
        if skill_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM rl_optimization_log WHERE skill_id = ? "
                "ORDER BY id ASC",
                (skill_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM rl_optimization_log ORDER BY id ASC"
            ).fetchall()
        return [
            TrainingStep(
                skill_id=row["skill_id"],
                action=row["action"],
                reward=row["reward"],
                before_q=row["before_q"],
                after_q=row["after_q"],
                iteration=row["iteration"],
                timestamp=row["recorded_at"],
            )
            for row in rows
        ]

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal optimisation actions
    # ------------------------------------------------------------------

    def _try_compression(self, skill_id: str) -> float:
        """Compress ``tier2_core`` by removing redundancy and measure reward.

        Compression removes duplicate lines, excessive whitespace, and
        boilerplate phrases.  The reward is the improvement in Q-value
        after the compression is applied.

        Parameters
        ----------
        skill_id : str
            The skill to compress.

        Returns
        -------
        float
            Scalar reward (positive if compression improved Q-value).
        """
        skill = self._registry.get_skill(skill_id, tier=3)
        if skill is None:
            return 0.0

        original_core: str = skill.tier2_core
        before_stats = self._tracker.get_stats(skill_id)
        before_outcomes: dict[str, float] = {
            "success_rate": before_stats.get("success_rate", 0.5),
            "q_value": before_stats.get("q_value", 0.5),
            "avg_tokens": before_stats.get("avg_tokens", 0),
        }

        # Compress: deduplicate lines, collapse whitespace
        compressed = self._compress_text(original_core)
        self._registry.update_skill(skill_id, {"tier2_core": compressed})

        after_stats = self._tracker.get_stats(skill_id)
        after_outcomes: dict[str, float] = {
            "success_rate": after_stats.get("success_rate", 0.5),
            "q_value": after_stats.get("q_value", 0.5),
            "avg_tokens": after_stats.get("avg_tokens", 0),
        }

        reward = self._evaluate_reward(
            skill_id, before_outcomes, after_outcomes
        )
        return reward

    def _try_split(self, skill_id: str) -> list[str]:
        """Split a skill into sub-skills if ``tier2_core`` exceeds 500 tokens.

        Token count is approximated as ``len(text.split()) * 4 / 3``
        (a rough word-to-token ratio).

        Parameters
        ----------
        skill_id : str
            The skill to potentially split.

        Returns
        -------
        list[str]
            List of new sub-skill IDs if a split was performed, empty list
            otherwise.
        """
        skill = self._registry.get_skill(skill_id, tier=3)
        if skill is None:
            return []

        core: str = skill.tier2_core
        approx_tokens = int(len(core.split()) * 4 / 3) if core else 0

        if approx_tokens <= 500:
            return []

        # Split by sections (headings or double-newline blocks)
        sections = re.split(r"\n{2,}", core.strip())
        if len(sections) < 2:
            # Cannot meaningfully split a single block
            return []

        sub_ids: list[str] = []
        for idx, section in enumerate(sections):
            sub_name = f"{skill.name}_part{idx + 1}"
            sub_id = f"{skill_id}_sub_{uuid.uuid4().hex[:8]}"
            self._registry.register_skill(
                name=sub_name,
                tier1_metadata=skill.tier1_metadata,
                tier2_core=section.strip(),
                tags=skill.tags if hasattr(skill, "tags") else [],
                skill_id=sub_id,
            )
            sub_ids.append(sub_id)

        logger.info(
            "Split skill '%s' into %d sub-skills", skill_id, len(sub_ids)
        )
        return sub_ids

    def _try_reorder(self, skill_id: str) -> float:
        """Reorder numbered steps by usage-frequency heuristic.

        Extracts numbered items (e.g. ``1. …``, ``2. …``) from
        ``tier2_core`` and reorders them so that shorter (presumed
        higher-frequency) steps appear first.

        Parameters
        ----------
        skill_id : str
            The skill whose steps to reorder.

        Returns
        -------
        float
            Scalar reward (positive if reordering improved Q-value).
        """
        skill = self._registry.get_skill(skill_id, tier=3)
        if skill is None:
            return 0.0

        original_core: str = skill.tier2_core
        before_stats = self._tracker.get_stats(skill_id)
        before_outcomes: dict[str, float] = {
            "success_rate": before_stats.get("success_rate", 0.5),
            "q_value": before_stats.get("q_value", 0.5),
            "avg_tokens": before_stats.get("avg_tokens", 0),
        }

        reordered = self._reorder_steps(original_core)
        self._registry.update_skill(skill_id, {"tier2_core": reordered})

        after_stats = self._tracker.get_stats(skill_id)
        after_outcomes: dict[str, float] = {
            "success_rate": after_stats.get("success_rate", 0.5),
            "q_value": after_stats.get("q_value", 0.5),
            "avg_tokens": after_stats.get("avg_tokens", 0),
        }

        reward = self._evaluate_reward(
            skill_id, before_outcomes, after_outcomes
        )
        return reward

    def _evaluate_reward(
        self,
        skill_id: str,
        before_outcomes: dict[str, float],
        after_outcomes: dict[str, float],
    ) -> float:
        """Compute composite reward from before/after outcome snapshots.

        ``reward = Δ success_rate × 0.6 + Δ q_value × 0.3 + token_savings × 0.1``

        Token savings are normalised as the fractional reduction in
        average tokens (clamped to ``[0, 1]``).

        Parameters
        ----------
        skill_id : str
            The skill being evaluated (used for logging only).
        before_outcomes : dict[str, float]
            Pre-action metrics (``success_rate``, ``q_value``, ``avg_tokens``).
        after_outcomes : dict[str, float]
            Post-action metrics with the same keys.

        Returns
        -------
        float
            Scalar reward, typically in ``[-1, 1]``.
        """
        delta_sr = (
            after_outcomes.get("success_rate", 0.5)
            - before_outcomes.get("success_rate", 0.5)
        )
        delta_q = (
            after_outcomes.get("q_value", 0.5)
            - before_outcomes.get("q_value", 0.5)
        )

        before_tok = before_outcomes.get("avg_tokens", 0)
        after_tok = after_outcomes.get("avg_tokens", 0)
        if before_tok > 0:
            token_savings = max(0.0, (before_tok - after_tok) / before_tok)
        else:
            token_savings = 0.0

        reward = (
            delta_sr * self._W_SUCCESS
            + delta_q * self._W_QVALUE
            + token_savings * self._W_TOKENS
        )
        logger.debug(
            "Reward for '%s': Δsr=%.4f Δq=%.4f tok_sav=%.4f → %.4f",
            skill_id,
            delta_sr,
            delta_q,
            token_savings,
            reward,
        )
        return reward

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compress_text(text: str) -> str:
        """Compress text by removing duplicate lines and collapsing whitespace.

        Parameters
        ----------
        text : str
            Raw text content to compress.

        Returns
        -------
        str
            Compressed text.
        """
        if not text:
            return text

        lines: list[str] = text.splitlines()
        seen: set[str] = set()
        unique_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                unique_lines.append(stripped)
            elif not stripped:
                # Keep single blank separators
                if unique_lines and unique_lines[-1] != "":
                    unique_lines.append("")

        compressed = "\n".join(unique_lines)
        # Collapse runs of 3+ newlines to 2
        compressed = re.sub(r"\n{3,}", "\n\n", compressed)
        return compressed.strip()

    @staticmethod
    def _reorder_steps(text: str) -> str:
        """Reorder numbered steps so shorter ones come first.

        Lines matching ``^\\d+[\\.\\)]`` are treated as numbered steps.
        They are sorted by character length (ascending) and renumbered.

        Parameters
        ----------
        text : str
            Text potentially containing numbered steps.

        Returns
        -------
        str
            Text with reordered and renumbered steps.
        """
        if not text:
            return text

        lines = text.splitlines()
        prefix_lines: list[str] = []
        steps: list[str] = []
        step_pattern = re.compile(r"^\s*\d+[.)]\s*")

        for line in lines:
            if step_pattern.match(line):
                steps.append(line)
            else:
                if not steps:
                    prefix_lines.append(line)
                else:
                    # Non-step line after steps — treat as continuation
                    steps[-1] = steps[-1] + "\n" + line

        if not steps:
            return text

        # Sort by length (shorter first = higher frequency heuristic)
        steps.sort(key=len)
        # Renumber
        renumbered: list[str] = []
        for idx, step in enumerate(steps, start=1):
            new_step = step_pattern.sub(f"{idx}. ", step, count=1)
            renumbered.append(new_step)

        result_lines = prefix_lines + renumbered
        return "\n".join(result_lines)

    def _log_step(self, step: TrainingStep) -> None:
        """Persist a training step to the ``rl_optimization_log`` table.

        Parameters
        ----------
        step : TrainingStep
            The step to persist.
        """
        self._conn.execute(
            "INSERT INTO rl_optimization_log "
            "(skill_id, action, reward, before_q, after_q, iteration, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                step.skill_id,
                step.action,
                step.reward,
                step.before_q,
                step.after_q,
                step.iteration,
                step.timestamp,
            ),
        )
        self._conn.commit()
