"""
Reinforcement-learning skill optimizer inspired by Evolving-RL.

Applies iterative optimisation actions (compress, split, reorder) to
individual skills and evaluates each action with a composite reward signal
derived from Q-value improvement, success-rate delta, and token savings.

The optimiser co-evolves skill representations alongside a learned reward
model — mirroring the co-evolution concept from Evolving-RL where both
the policy and the environment representation improve in tandem.

Phase 5a Enhancements:
- **Contextual bandit** action selection (epsilon-greedy with decay).
- **Experience replay** buffer for sample-efficient learning.
- **Curriculum scheduler** that orders skills by difficulty for batch
  optimisation.
- **Reward model** learned from historical outcomes.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import math
import random
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


@dataclass
class Experience:
    """A single transition stored in the experience replay buffer.

    Attributes
    ----------
    state_hash : str
        Hash of the skill state (context) at the time of action.
    action : str
        The action taken.
    reward : float
        Observed reward.
    next_state_hash : str
        Hash of the skill state after the action.
    timestamp : str
        When the experience was recorded.
    """

    state_hash: str
    action: str
    reward: float
    next_state_hash: str
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
# Experience Replay Buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Fixed-capacity experience replay buffer with uniform sampling.

    Stores :class:`Experience` instances and supports batch sampling
    for the learned reward model.

    Parameters
    ----------
    capacity : int
        Maximum number of experiences to retain.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self._capacity = capacity
        self._buffer: collections.deque[Experience] = collections.deque(
            maxlen=capacity
        )

    @property
    def size(self) -> int:
        """Current number of stored experiences."""
        return len(self._buffer)

    def push(self, experience: Experience) -> None:
        """Add an experience to the buffer.

        Parameters
        ----------
        experience : Experience
            The transition to store.
        """
        self._buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample a random batch of experiences.

        Parameters
        ----------
        batch_size : int
            Number of experiences to sample.

        Returns
        -------
        list[Experience]
            Random sample (may be smaller than *batch_size* if the
            buffer is not full enough).
        """
        batch_size = min(batch_size, len(self._buffer))
        return random.sample(list(self._buffer), batch_size)

    def get_all(self) -> list[Experience]:
        """Return all stored experiences."""
        return list(self._buffer)

    def clear(self) -> None:
        """Remove all stored experiences."""
        self._buffer.clear()


# ---------------------------------------------------------------------------
# Contextual Bandit
# ---------------------------------------------------------------------------


class ContextualBandit:
    """Epsilon-greedy contextual bandit for action selection.

    Maintains per-action reward statistics and selects actions based on
    context features (skill complexity, current Q-value, content length).

    Parameters
    ----------
    actions : list[str]
        Available actions.
    epsilon_start : float
        Initial exploration rate.
    epsilon_min : float
        Minimum exploration rate.
    epsilon_decay : float
        Multiplicative decay applied after each action selection.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        epsilon_start: float = 0.3,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        self._actions = actions or ["compress", "split", "reorder"]
        self._epsilon = epsilon_start
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay

        # Per-action statistics: action → {count, total_reward, avg_reward}
        self._stats: dict[str, dict[str, float]] = {
            a: {"count": 0.0, "total_reward": 0.0, "avg_reward": 0.0}
            for a in self._actions
        }

        # Context-bucket → action → cumulative reward
        # Buckets are simple strings derived from discretised features.
        self._context_values: dict[str, dict[str, float]] = {}

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self._epsilon

    def select_action(self, context: dict[str, float]) -> str:
        """Select an action given a context.

        Parameters
        ----------
        context : dict[str, float]
            Context features (e.g. ``{"q_value": 0.3, "length": 0.7}``).

        Returns
        -------
        str
            The selected action.
        """
        if random.random() < self._epsilon:
            action = random.choice(self._actions)
        else:
            bucket = self._context_to_bucket(context)
            if bucket in self._context_values:
                cv = self._context_values[bucket]
                action = max(cv, key=cv.get)  # type: ignore[arg-type]
            else:
                # No data for this context — pick best global action
                action = max(
                    self._stats,
                    key=lambda a: self._stats[a]["avg_reward"],
                )

        # Decay epsilon
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
        return action

    def update(
        self, context: dict[str, float], action: str, reward: float
    ) -> None:
        """Update bandit statistics after observing a reward.

        Parameters
        ----------
        context : dict[str, float]
            The context in which the action was taken.
        action : str
            The action that was taken.
        reward : float
            The observed reward.
        """
        # Global stats
        stats = self._stats[action]
        stats["count"] += 1
        stats["total_reward"] += reward
        stats["avg_reward"] = stats["total_reward"] / stats["count"]

        # Context-bucket stats
        bucket = self._context_to_bucket(context)
        if bucket not in self._context_values:
            self._context_values[bucket] = {a: 0.0 for a in self._actions}
        self._context_values[bucket][action] += reward

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Return per-action reward statistics."""
        return dict(self._stats)

    @staticmethod
    def _context_to_bucket(context: dict[str, float]) -> str:
        """Discretise continuous context features into a bucket key.

        Each feature is quantised to 5 levels (0-4) and combined into
        a string key.
        """
        parts: list[str] = []
        for key in sorted(context):
            val = context[key]
            level = min(4, max(0, int(val * 5)))
            parts.append(f"{key}={level}")
        return "|".join(parts)


# ---------------------------------------------------------------------------
# Learned Reward Model
# ---------------------------------------------------------------------------


class RewardModel:
    """Lightweight learned reward model trained from experience replay.

    Uses a simple linear model (one weight per feature) to predict
    reward from (context, action) pairs.  Updated via gradient descent
    on replay samples.

    Parameters
    ----------
    feature_dim : int
        Number of context features.
    actions : list[str]
        Available actions.
    learning_rate : float
        SGD learning rate.
    """

    def __init__(
        self,
        feature_dim: int = 4,
        actions: list[str] | None = None,
        learning_rate: float = 0.01,
    ) -> None:
        self._actions = actions or ["compress", "split", "reorder"]
        self._lr = learning_rate
        self._feature_dim = feature_dim

        # Weight matrix: action → feature weights (list of floats)
        self._weights: dict[str, list[float]] = {
            a: [0.0] * feature_dim for a in self._actions
        }
        self._bias: dict[str, float] = {a: 0.0 for a in self._actions}
        self._update_count: int = 0

    @property
    def update_count(self) -> int:
        """Number of training updates performed."""
        return self._update_count

    def predict(self, features: list[float], action: str) -> float:
        """Predict reward for a (features, action) pair.

        Parameters
        ----------
        features : list[float]
            Context feature vector.
        action : str
            The action to evaluate.

        Returns
        -------
        float
            Predicted reward.
        """
        w = self._weights.get(action, [0.0] * self._feature_dim)
        b = self._bias.get(action, 0.0)
        return sum(f * wi for f, wi in zip(features, w)) + b

    def predict_best_action(self, features: list[float]) -> tuple[str, float]:
        """Return the action with the highest predicted reward.

        Parameters
        ----------
        features : list[float]
            Context feature vector.

        Returns
        -------
        tuple[str, float]
            (best_action, predicted_reward).
        """
        best_action = self._actions[0]
        best_reward = float("-inf")
        for action in self._actions:
            pred = self.predict(features, action)
            if pred > best_reward:
                best_reward = pred
                best_action = action
        return best_action, best_reward

    def train_step(
        self, features: list[float], action: str, target_reward: float
    ) -> float:
        """Apply one gradient-descent step for a single sample.

        Uses MSE loss: ``L = (predicted - target)²``.

        Parameters
        ----------
        features : list[float]
            Context feature vector.
        action : str
            The action taken.
        target_reward : float
            The observed (target) reward.

        Returns
        -------
        float
            The squared error for this sample.
        """
        pred = self.predict(features, action)
        error = pred - target_reward
        # Gradient descent on linear model
        w = self._weights[action]
        for i in range(len(w)):
            w[i] -= self._lr * error * features[i]
        self._bias[action] -= self._lr * error
        self._update_count += 1
        return error * error

    def train_batch(
        self,
        replay_buffer: ReplayBuffer,
        context_fn: Any,
        batch_size: int = 64,
    ) -> float:
        """Train on a batch sampled from the replay buffer.

        Parameters
        ----------
        replay_buffer : ReplayBuffer
            Source of experience samples.
        context_fn : callable
            Function ``(state_hash) -> list[float]`` to convert state
            hashes into feature vectors.
        batch_size : int
            Number of samples to train on.

        Returns
        -------
        float
            Mean squared error over the batch.
        """
        if replay_buffer.size == 0:
            return 0.0
        samples = replay_buffer.sample(batch_size)
        total_se = 0.0
        for exp in samples:
            features = context_fn(exp.state_hash)
            total_se += self.train_step(features, exp.action, exp.reward)
        return total_se / len(samples)

    def get_weights(self) -> dict[str, dict[str, Any]]:
        """Return current model weights for inspection."""
        return {
            action: {"weights": list(self._weights[action]), "bias": self._bias[action]}
            for action in self._actions
        }


# ---------------------------------------------------------------------------
# Curriculum Scheduler
# ---------------------------------------------------------------------------


class CurriculumScheduler:
    """Orders skills by estimated difficulty for staged batch optimisation.

    Difficulty is computed from skill properties: low Q-value, long content,
    and high complexity all contribute to higher difficulty.

    Three stages are supported:
    - **warmup**: only easiest skills (difficulty < 0.33)
    - **main**: medium-difficulty skills (0.33 ≤ difficulty < 0.66)
    - **advanced**: hardest skills (difficulty ≥ 0.66)

    Parameters
    ----------
    tracker : TrackerProto
        For reading Q-values and stats.
    """

    def __init__(self, tracker: TrackerProto) -> None:
        self._tracker = tracker
        self._stage = "warmup"
        self._stage_history: list[str] = ["warmup"]

    @property
    def current_stage(self) -> str:
        """Current curriculum stage."""
        return self._stage

    @property
    def stage_history(self) -> list[str]:
        """History of stage transitions."""
        return list(self._stage_history)

    def compute_difficulty(self, skill: Any) -> float:
        """Estimate skill difficulty in ``[0, 1]``.

        Parameters
        ----------
        skill : Any
            A skill object with ``id``, ``tier2_core``, ``q_value`` attrs.

        Returns
        -------
        float
            Difficulty score (higher = harder).
        """
        sid = skill.id if hasattr(skill, "id") else str(skill)
        q_val = self._tracker.get_q_value(sid)
        core: str = skill.tier2_core if hasattr(skill, "tier2_core") else ""
        content_len = len(core.split()) if core else 0

        # Low Q-value → harder, long content → harder
        q_factor = 1.0 - q_val  # invert: low Q = high difficulty
        len_factor = min(1.0, content_len / 500)  # normalise to 500 words

        difficulty = 0.7 * q_factor + 0.3 * len_factor
        return max(0.0, min(1.0, difficulty))

    def order_skills(self, skills: list[Any]) -> list[Any]:
        """Sort skills by ascending difficulty.

        Parameters
        ----------
        skills : list[Any]
            Skills to order.

        Returns
        -------
        list[Any]
            Skills sorted from easiest to hardest.
        """
        scored = [(self.compute_difficulty(s), s) for s in skills]
        scored.sort(key=lambda x: x[0])
        return [s for _, s in scored]

    def get_stage_skills(
        self,
        skills: list[Any],
        stage: str | None = None,
    ) -> list[Any]:
        """Return skills belonging to a curriculum stage.

        Parameters
        ----------
        skills : list[Any]
            All available skills.
        stage : str | None
            Stage to filter by (defaults to current stage).

        Returns
        -------
        list[Any]
            Skills in the specified stage.
        """
        stage = stage or self._stage
        boundaries = {
            "warmup": (0.0, 0.33),
            "main": (0.33, 0.66),
            "advanced": (0.66, 1.01),
        }
        lo, hi = boundaries.get(stage, (0.0, 1.01))
        result: list[Any] = []
        for s in skills:
            d = self.compute_difficulty(s)
            if lo <= d < hi:
                result.append(s)
        return self.order_skills(result)

    def advance_stage(self) -> str:
        """Move to the next curriculum stage.

        Returns
        -------
        str
            The new current stage.
        """
        progression = ["warmup", "main", "advanced"]
        idx = progression.index(self._stage)
        if idx < len(progression) - 1:
            self._stage = progression[idx + 1]
            self._stage_history.append(self._stage)
            logger.info("Curriculum advanced to stage '%s'", self._stage)
        return self._stage

    def should_advance(self, recent_rewards: list[float], threshold: float = 0.0) -> bool:
        """Check if recent performance warrants advancing to the next stage.

        Parameters
        ----------
        recent_rewards : list[float]
            Rewards from recent optimisation steps in the current stage.
        threshold : float
            Mean reward threshold to advance.

        Returns
        -------
        bool
            True if the scheduler should advance.
        """
        if len(recent_rewards) < 3:
            return False
        return sum(recent_rewards) / len(recent_rewards) > threshold


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

    Phase 5a adds contextual bandit action selection, experience replay,
    a learned reward model, and curriculum-based batch scheduling.

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
    replay_capacity : int
        Maximum experiences in the replay buffer.
    bandit_epsilon : float
        Initial epsilon for the contextual bandit.
    reward_lr : float
        Learning rate for the learned reward model.
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
        replay_capacity: int = 10_000,
        bandit_epsilon: float = 0.3,
        reward_lr: float = 0.01,
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

        # Phase 5a components
        self._bandit = ContextualBandit(
            epsilon_start=bandit_epsilon,
        )
        self._replay_buffer = ReplayBuffer(capacity=replay_capacity)
        self._reward_model = RewardModel(
            feature_dim=4,
            learning_rate=reward_lr,
        )
        self._curriculum = CurriculumScheduler(tracker=tracker)

        # Per-optimisation-run reward tracking for curriculum decisions
        self._current_stage_rewards: list[float] = []

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

            CREATE TABLE IF NOT EXISTS rl_experiences (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                state_hash    TEXT NOT NULL,
                action        TEXT NOT NULL,
                reward        REAL NOT NULL,
                next_state    TEXT NOT NULL,
                recorded_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rl_exp_state
                ON rl_experiences(state_hash);
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

        Phase 5a: action selection uses the contextual bandit; experiences
        are stored in the replay buffer and the reward model is updated.

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

            # Build context features for the bandit
            context = self._build_context(skill_id, skill)
            state_hash = self._hash_state(original_core, before_q)

            # --- Select action via bandit ---
            # In Phase 5a, we still try all actions but use the bandit
            # to decide which to evaluate first (and for exploration).
            selected_action = self._bandit.select_action(context)

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
            next_state_hash = self._hash_state(best_core or original_core, after_q)

            # --- Store experience ---
            exp = Experience(
                state_hash=state_hash,
                action=best_action,
                reward=best_reward,
                next_state_hash=next_state_hash,
            )
            self._replay_buffer.push(exp)
            self._persist_experience(exp)

            # --- Update bandit and reward model ---
            self._bandit.update(context, best_action, best_reward)
            self._current_stage_rewards.append(best_reward)

            # Train reward model on a mini-batch
            if self._replay_buffer.size >= 16:
                mse = self._reward_model.train_batch(
                    self._replay_buffer,
                    context_fn=self._state_hash_to_features,
                    batch_size=32,
                )
                logger.debug("Reward model MSE: %.6f", mse)

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

        Uses the curriculum scheduler to order skills by difficulty.

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
        candidates = [
            s for s in skills
            if self._tracker.get_q_value(
                s.id if hasattr(s, "id") else str(s)
            ) < threshold
        ]

        # Order by curriculum difficulty
        ordered = self._curriculum.order_skills(candidates)
        self._current_stage_rewards = []

        for skill in ordered:
            sid: str = skill.id if hasattr(skill, "id") else str(skill)
            q_val = self._tracker.get_q_value(sid)
            logger.info(
                "Batch-optimising skill '%s' (Q=%.3f < %.3f, stage=%s)",
                sid,
                q_val,
                threshold,
                self._curriculum.current_stage,
            )
            steps = self.optimize(sid)
            all_steps.extend(steps)

        # Check if curriculum should advance
        if self._curriculum.should_advance(self._current_stage_rewards):
            self._curriculum.advance_stage()

        return all_steps

    def curriculum_optimize(self) -> list[TrainingStep]:
        """Run full curriculum-based optimisation through all stages.

        Progresses through warmup → main → advanced, optimising
        stage-appropriate skills at each level.

        Returns
        -------
        list[TrainingStep]
            All training steps across all curriculum stages.
        """
        all_steps: list[TrainingStep] = []
        stages = ["warmup", "main", "advanced"]

        for stage in stages:
            self._curriculum._stage = stage
            self._curriculum._stage_history.append(stage)
            logger.info("Starting curriculum stage '%s'", stage)

            skills = self._registry.list_skills()
            stage_skills = self._curriculum.get_stage_skills(skills, stage)
            self._current_stage_rewards = []

            for skill in stage_skills:
                sid: str = skill.id if hasattr(skill, "id") else str(skill)
                steps = self.optimize(sid)
                all_steps.extend(steps)

            avg_reward = (
                sum(self._current_stage_rewards) / len(self._current_stage_rewards)
                if self._current_stage_rewards
                else 0.0
            )
            logger.info(
                "Curriculum stage '%s' complete: %d steps, avg_reward=%.4f",
                stage,
                len(self._current_stage_rewards),
                avg_reward,
            )

        return all_steps

    def predict_reward(self, skill_id: str, action: str) -> float:
        """Predict the reward for a (skill, action) pair using the learned model.

        Parameters
        ----------
        skill_id : str
            Target skill.
        action : str
            The action to evaluate.

        Returns
        -------
        float
            Predicted reward.
        """
        skill = self._registry.get_skill(skill_id, tier=3)
        if skill is None:
            return 0.0
        context = self._build_context(skill_id, skill)
        features = self._context_to_features(context)
        return self._reward_model.predict(features, action)

    def recommend_action(self, skill_id: str) -> tuple[str, float]:
        """Recommend the best action for a skill using the learned model.

        Parameters
        ----------
        skill_id : str
            Target skill.

        Returns
        -------
        tuple[str, float]
            (recommended_action, predicted_reward).
        """
        skill = self._registry.get_skill(skill_id, tier=3)
        if skill is None:
            return "compress", 0.0
        context = self._build_context(skill_id, skill)
        features = self._context_to_features(context)
        return self._reward_model.predict_best_action(features)

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

    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostic information about all RL components.

        Returns
        -------
        dict[str, Any]
            Diagnostics including bandit stats, replay buffer size,
            reward model weights, and curriculum state.
        """
        return {
            "bandit": {
                "epsilon": self._bandit.epsilon,
                "action_stats": self._bandit.get_stats(),
            },
            "replay_buffer": {
                "size": self._replay_buffer.size,
                "capacity": self._replay_buffer._capacity,
            },
            "reward_model": {
                "update_count": self._reward_model.update_count,
                "weights": self._reward_model.get_weights(),
            },
            "curriculum": {
                "current_stage": self._curriculum.current_stage,
                "stage_history": self._curriculum.stage_history,
            },
        }

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
    # Context / state helpers (Phase 5a)
    # ------------------------------------------------------------------

    def _build_context(self, skill_id: str, skill: Any) -> dict[str, float]:
        """Build a context feature dict for the bandit.

        Features (all normalised to [0, 1]):
        - ``q_value``: current Q-value
        - ``success_rate``: current success rate
        - ``content_length``: word count / 500
        - ``complexity``: unique-word ratio

        Parameters
        ----------
        skill_id : str
            Skill identifier.
        skill : Any
            Skill object.

        Returns
        -------
        dict[str, float]
            Context features.
        """
        q_val = self._tracker.get_q_value(skill_id)
        sr = self._tracker.get_success_rate(skill_id)
        core: str = skill.tier2_core if hasattr(skill, "tier2_core") else ""
        words = core.split() if core else []
        content_length = min(1.0, len(words) / 500)
        unique_ratio = len(set(words)) / max(1, len(words))

        return {
            "q_value": max(0.0, min(1.0, q_val)),
            "success_rate": max(0.0, min(1.0, sr)),
            "content_length": content_length,
            "complexity": unique_ratio,
        }

    @staticmethod
    def _context_to_features(context: dict[str, float]) -> list[float]:
        """Convert context dict to a feature vector (sorted by key)."""
        return [context[k] for k in sorted(context)]

    def _build_context_from_skill_id(self, skill_id: str) -> dict[str, float]:
        """Build context for an arbitrary skill ID."""
        skill = self._registry.get_skill(skill_id, tier=3)
        if skill is None:
            return {"q_value": 0.5, "success_rate": 0.5, "content_length": 0.0, "complexity": 0.0}
        return self._build_context(skill_id, skill)

    @staticmethod
    def _hash_state(core: str, q_value: float) -> str:
        """Produce a deterministic hash of skill state."""
        content = f"{core[:200]}|{q_value:.4f}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _state_hash_to_features(self, state_hash: str) -> list[float]:
        """Convert a state hash back to features for the reward model.

        Since we can't perfectly reconstruct the state, we derive
        features from the hash itself (deterministic pseudo-features).
        This is a pragmatic compromise for the stdlib-only constraint.
        """
        # Use hash bytes as pseudo-features
        raw = bytes.fromhex(state_hash)
        features: list[float] = []
        for i in range(4):
            features.append(raw[i % len(raw)] / 255.0)
        return features

    def _persist_experience(self, exp: Experience) -> None:
        """Persist an experience to the ``rl_experiences`` table."""
        self._conn.execute(
            "INSERT INTO rl_experiences "
            "(state_hash, action, reward, next_state, recorded_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (exp.state_hash, exp.action, exp.reward, exp.next_state_hash, exp.timestamp),
        )
        self._conn.commit()

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
