"""A/B Testing framework for comparing skill variants.

Enables controlled experiments that route traffic between two or more skill
variants, record outcomes, and apply statistical tests to determine whether
one variant significantly outperforms the others.

Key concepts:

- **Experiment**: a named test with a set of variants and a target metric.
- **Variant**: a skill (or skill version) participating in the experiment.
- **Assignment**: deterministic or random traffic routing to a variant.
- **Outcome**: a recorded success/failure for a single invocation.
- **Result**: aggregated metrics and statistical significance.

Statistical tests are implemented in pure stdlib (no scipy) and cover
the two most common cases: two-proportion z-test and chi-squared
goodness-of-fit for multi-variant experiments.

No external dependencies — stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ExperimentStatus(Enum):
    """Lifecycle status of an A/B experiment."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class MetricType(Enum):
    """Supported metric types for experiment evaluation."""

    SUCCESS_RATE = "success_rate"
    LATENCY = "latency"
    TOKEN_EFFICIENCY = "token_efficiency"
    USER_RATING = "user_rating"


class AssignmentStrategy(Enum):
    """Traffic assignment strategies."""

    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    WEIGHTED_RANDOM = "weighted_random"
    HASH_BASED = "hash_based"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    """A skill variant participating in an A/B experiment.

    Attributes
    ----------
    skill_id : str
        The SkillForge skill ID this variant routes to.
    weight : float
        Relative traffic weight (used by WEIGHTED_RANDOM).  Must be > 0.
    description : str
        Optional human-readable label for this variant.
    """

    skill_id: str
    weight: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"Variant weight must be > 0, got {self.weight}")


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment.

    Attributes
    ----------
    name : str
        Human-readable experiment name.
    variants : list[Variant]
        Two or more variants to compare.
    metric : MetricType
        The primary metric to evaluate.
    assignment_strategy : AssignmentStrategy
        How incoming requests are routed to variants.
    alpha : float
        Significance level for statistical tests (default 0.05).
    min_sample_size : int
        Minimum total outcomes before evaluation is attempted.
    max_duration_hours : int | None
        Optional auto-stop after this many hours.
    """

    name: str
    variants: list[Variant]
    metric: MetricType = MetricType.SUCCESS_RATE
    assignment_strategy: AssignmentStrategy = AssignmentStrategy.RANDOM
    alpha: float = 0.05
    min_sample_size: int = 30
    max_duration_hours: int | None = None

    def __post_init__(self) -> None:
        if len(self.variants) < 2:
            raise ValueError(
                f"An experiment requires at least 2 variants, got {len(self.variants)}"
            )
        if not 0 < self.alpha < 1:
            raise ValueError(f"Alpha must be in (0, 1), got {self.alpha}")
        if self.min_sample_size < 2:
            raise ValueError(
                f"min_sample_size must be >= 2, got {self.min_sample_size}"
            )


@dataclass
class Outcome:
    """A single recorded outcome for an experiment invocation.

    Attributes
    ----------
    id : str
        Unique outcome identifier.
    experiment_id : str
        The experiment this outcome belongs to.
    variant_index : int
        Index of the variant that was assigned.
    success : bool
        Whether the invocation was considered a success.
    metric_value : float
        The observed metric value (e.g. latency in ms, rating 0-1).
    context : dict[str, Any]
        Free-form context (user_id, input_hash, etc.).
    created_at : str
        ISO-8601 UTC timestamp.
    """

    id: str
    experiment_id: str
    variant_index: int
    success: bool
    metric_value: float = 1.0
    context: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class VariantStats:
    """Aggregated statistics for a single variant.

    Attributes
    ----------
    variant_index : int
        Position in the experiment's variant list.
    skill_id : str
        The skill backing this variant.
    sample_size : int
        Total number of outcomes recorded.
    successes : int
        Number of successful outcomes.
    success_rate : float
        successes / sample_size (0.0 if no samples).
    mean_metric : float
        Average metric_value across all outcomes.
    variance_metric : float
        Variance of metric_value across outcomes.
    """

    variant_index: int
    skill_id: str
    sample_size: int = 0
    successes: int = 0
    success_rate: float = 0.0
    mean_metric: float = 0.0
    variance_metric: float = 0.0


@dataclass
class ExperimentResult:
    """Result of evaluating a completed (or in-progress) experiment.

    Attributes
    ----------
    experiment_id : str
        The experiment identifier.
    status : ExperimentStatus
        Current experiment status.
    variant_stats : list[VariantStats]
        Per-variant aggregated statistics.
    best_variant_index : int | None
        Index of the winning variant, or ``None`` if inconclusive.
    is_significant : bool
        Whether the statistical test reached the configured alpha.
    p_value : float | None
        The computed p-value, or ``None`` if insufficient data.
    test_name : str
        Name of the statistical test applied.
    summary : str
        Human-readable summary of the result.
    evaluated_at : str
        ISO-8601 UTC timestamp of evaluation.
    """

    experiment_id: str
    status: ExperimentStatus
    variant_stats: list[VariantStats] = field(default_factory=list)
    best_variant_index: int | None = None
    is_significant: bool = False
    p_value: float | None = None
    test_name: str = ""
    summary: str = ""
    evaluated_at: str = ""

    def __post_init__(self) -> None:
        if not self.evaluated_at:
            self.evaluated_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Statistical helpers (stdlib-only)
# ---------------------------------------------------------------------------


def _normal_cdf(x: float) -> float:
    """Approximate the standard normal CDF using Abramowitz & Stegun 7.1.26.

    Accurate to ~1.5e-7 for all x.
    """
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    erf_approx = 1.0 - poly * math.exp(-(x * x))
    return 0.5 * (1.0 + sign * erf_approx)


def _two_proportion_z_test(
    n1: int, s1: int, n2: int, s2: int
) -> tuple[float, float]:
    """Two-proportion z-test returning (z_statistic, two_tailed_p_value).

    Parameters
    ----------
    n1, s1 : int
        Sample size and successes for group 1.
    n2, s2 : int
        Sample size and successes for group 2.

    Returns
    -------
    tuple[float, float]
        ``(z, p)`` where *z* is the z-statistic and *p* the two-tailed
        p-value.  Returns ``(0.0, 1.0)`` when the test is undefined
        (e.g. zero variance).
    """
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1 = s1 / n1
    p2 = s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    variance = p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)
    if variance <= 0:
        return 0.0, 1.0
    z = (p1 - p2) / math.sqrt(variance)
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return z, p


def _chi_squared_test(observed: list[int], expected: list[float]) -> tuple[float, float]:
    """Chi-squared goodness-of-fit test.

    Parameters
    ----------
    observed : list[int]
        Observed counts per category.
    expected : list[float]
        Expected counts per category (must be > 0).

    Returns
    -------
    tuple[float, float]
        ``(chi2, p_value)``.  Returns ``(0.0, 1.0)`` for degenerate inputs.
    """
    if len(observed) != len(expected) or len(observed) < 2:
        return 0.0, 1.0
    if any(e <= 0 for e in expected):
        return 0.0, 1.0

    chi2 = sum(
        (o - e) ** 2 / e for o, e in zip(observed, expected)
    )
    df = len(observed) - 1

    # Approximate p-value from chi-squared using Wilson-Hilferty normal
    # approximation: Z ≈ ((X/k)^(1/3) - 1 + 2/(9k)) / sqrt(2/(9k))
    if df <= 0:
        return chi2, 1.0
    ratio = (chi2 / df) ** (1.0 / 3.0)
    correction = 2.0 / (9.0 * df)
    z_wh = (ratio - 1.0 + correction) / math.sqrt(correction)
    p = 1.0 - _normal_cdf(z_wh)
    return chi2, max(0.0, min(1.0, p))


# ---------------------------------------------------------------------------
# ABTestRunner
# ---------------------------------------------------------------------------


class ABTestRunner:
    """Orchestrates A/B experiments: creation, assignment, recording, and
    evaluation.

    All state is persisted in a SQLite database so experiments survive
    process restarts.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the SQLite database.  Defaults to
        ``~/.skillforge/ab_testing.db``.
    seed : int | None
        Optional random seed for reproducible assignment in tests.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        seed: int | None = None,
    ) -> None:
        self._db_path = str(db_path or Path.home() / ".skillforge" / "ab_testing.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_tables()

        self._rng = random.Random(seed)
        self._round_robin_counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create tables if they do not already exist."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id              TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'created',
                config_json     TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                started_at      TEXT,
                completed_at    TEXT
            );
            CREATE TABLE IF NOT EXISTS outcomes (
                id              TEXT PRIMARY KEY,
                experiment_id   TEXT NOT NULL,
                variant_index   INTEGER NOT NULL,
                success         INTEGER NOT NULL,
                metric_value    REAL NOT NULL DEFAULT 1.0,
                context_json    TEXT NOT NULL DEFAULT '{}',
                created_at      TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );
            CREATE INDEX IF NOT EXISTS idx_outcomes_experiment
                ON outcomes(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_outcomes_variant
                ON outcomes(experiment_id, variant_index);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Experiment lifecycle
    # ------------------------------------------------------------------

    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment and return its ID.

        Parameters
        ----------
        config : ExperimentConfig
            Full experiment configuration.

        Returns
        -------
        str
            The unique experiment ID.

        Raises
        ------
        ValueError
            If an experiment with the same name already exists and is
            still running.
        """
        experiment_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        config_dict: dict[str, Any] = {
            "name": config.name,
            "variants": [
                {
                    "skill_id": v.skill_id,
                    "weight": v.weight,
                    "description": v.description,
                }
                for v in config.variants
            ],
            "metric": config.metric.value,
            "assignment_strategy": config.assignment_strategy.value,
            "alpha": config.alpha,
            "min_sample_size": config.min_sample_size,
            "max_duration_hours": config.max_duration_hours,
        }
        try:
            self._conn.execute(
                "INSERT INTO experiments (id, name, status, config_json, created_at) "
                "VALUES (:id, :name, :status, :config, :now)",
                {
                    "id": experiment_id,
                    "name": config.name,
                    "status": ExperimentStatus.CREATED.value,
                    "config": json.dumps(config_dict),
                    "now": now,
                },
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Experiment '{config.name}' conflict") from exc

        logger.info(
            "Created experiment '%s' (%s) with %d variants",
            config.name,
            experiment_id,
            len(config.variants),
        )
        return experiment_id

    def start_experiment(self, experiment_id: str) -> None:
        """Transition an experiment to RUNNING status."""
        cur = self._conn.execute(
            "UPDATE experiments SET status = :s, started_at = :now "
            "WHERE id = :id AND status = 'created'",
            {
                "s": ExperimentStatus.RUNNING.value,
                "now": datetime.now(timezone.utc).isoformat(),
                "id": experiment_id,
            },
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise ValueError(
                f"Experiment '{experiment_id}' not found or not in 'created' status"
            )
        logger.info("Started experiment %s", experiment_id)

    def pause_experiment(self, experiment_id: str) -> None:
        """Transition a running experiment to PAUSED."""
        cur = self._conn.execute(
            "UPDATE experiments SET status = :s WHERE id = :id AND status = 'running'",
            {"s": ExperimentStatus.PAUSED.value, "id": experiment_id},
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise ValueError(
                f"Experiment '{experiment_id}' not found or not in 'running' status"
            )

    def resume_experiment(self, experiment_id: str) -> None:
        """Transition a paused experiment back to RUNNING."""
        cur = self._conn.execute(
            "UPDATE experiments SET status = :s WHERE id = :id AND status = 'paused'",
            {"s": ExperimentStatus.RUNNING.value, "id": experiment_id},
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise ValueError(
                f"Experiment '{experiment_id}' not found or not in 'paused' status"
            )

    def complete_experiment(self, experiment_id: str) -> ExperimentResult:
        """Finalise an experiment, evaluate it, and return the result."""
        cur = self._conn.execute(
            "UPDATE experiments SET status = :s, completed_at = :now "
            "WHERE id = :id AND status IN ('running', 'paused')",
            {
                "s": ExperimentStatus.COMPLETED.value,
                "now": datetime.now(timezone.utc).isoformat(),
                "id": experiment_id,
            },
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise ValueError(
                f"Experiment '{experiment_id}' not found or cannot be completed"
            )
        logger.info("Completed experiment %s", experiment_id)
        return self.evaluate(experiment_id)

    def abandon_experiment(self, experiment_id: str) -> None:
        """Mark an experiment as abandoned (discarding results)."""
        cur = self._conn.execute(
            "UPDATE experiments SET status = :s WHERE id = :id AND status != 'abandoned'",
            {"s": ExperimentStatus.ABANDONED.value, "id": experiment_id},
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise ValueError(f"Experiment '{experiment_id}' not found")

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def assign_variant(
        self, experiment_id: str, context_key: str = ""
    ) -> int:
        """Assign an incoming request to a variant index.

        Parameters
        ----------
        experiment_id : str
            The running experiment.
        context_key : str
            An identifier for the user/session (used by HASH_BASED
            strategy for sticky assignments).  Ignored by RANDOM.

        Returns
        -------
        int
            The zero-based index of the assigned variant.

        Raises
        ------
        ValueError
            If the experiment is not running or not found.
        """
        row = self._conn.execute(
            "SELECT config_json, status FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        if row["status"] != ExperimentStatus.RUNNING.value:
            raise ValueError(
                f"Experiment '{experiment_id}' is not running "
                f"(status={row['status']})"
            )

        config_dict = json.loads(row["config_json"])
        variants = config_dict["variants"]
        strategy = AssignmentStrategy(config_dict["assignment_strategy"])
        n = len(variants)

        if strategy == AssignmentStrategy.RANDOM:
            return self._rng.randint(0, n - 1)

        if strategy == AssignmentStrategy.ROUND_ROBIN:
            counter = self._round_robin_counters.get(experiment_id, 0)
            self._round_robin_counters[experiment_id] = counter + 1
            return counter % n

        if strategy == AssignmentStrategy.WEIGHTED_RANDOM:
            weights = [v["weight"] for v in variants]
            total = sum(weights)
            r = self._rng.random() * total
            cumulative = 0.0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    return i
            return n - 1  # fallback for floating-point edge case

        if strategy == AssignmentStrategy.HASH_BASED:
            digest = hashlib.sha256(
                f"{experiment_id}:{context_key}".encode()
            ).hexdigest()
            return int(digest[:8], 16) % n

        raise ValueError(f"Unknown assignment strategy: {strategy}")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        experiment_id: str,
        variant_index: int,
        success: bool,
        metric_value: float = 1.0,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Record a single outcome for an experiment.

        Parameters
        ----------
        experiment_id : str
            The experiment to record for.
        variant_index : int
            The variant that was assigned.
        success : bool
            Whether the invocation was successful.
        metric_value : float
            The observed metric value.
        context : dict[str, Any] | None
            Optional context metadata.

        Returns
        -------
        str
            The outcome ID.
        """
        outcome_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO outcomes "
            "(id, experiment_id, variant_index, success, metric_value, "
            "context_json, created_at) "
            "VALUES (:id, :eid, :vi, :s, :mv, :ctx, :now)",
            {
                "id": outcome_id,
                "eid": experiment_id,
                "vi": variant_index,
                "s": 1 if success else 0,
                "mv": metric_value,
                "ctx": json.dumps(context or {}),
                "now": now,
            },
        )
        self._conn.commit()
        return outcome_id

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _load_config(self, experiment_id: str) -> dict[str, Any]:
        """Load and return the experiment config dict."""
        row = self._conn.execute(
            "SELECT config_json, status FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        return json.loads(row["config_json"])

    def _compute_variant_stats(
        self, experiment_id: str, config: dict[str, Any]
    ) -> list[VariantStats]:
        """Compute per-variant statistics from recorded outcomes."""
        variants = config["variants"]
        stats: list[VariantStats] = []

        for i, variant in enumerate(variants):
            rows = self._conn.execute(
                "SELECT success, metric_value FROM outcomes "
                "WHERE experiment_id = ? AND variant_index = ?",
                (experiment_id, i),
            ).fetchall()

            n = len(rows)
            if n == 0:
                stats.append(
                    VariantStats(
                        variant_index=i,
                        skill_id=variant["skill_id"],
                    )
                )
                continue

            successes = sum(1 for r in rows if r["success"])
            values = [r["metric_value"] for r in rows]
            mean = sum(values) / n
            if n > 1:
                variance = sum((v - mean) ** 2 for v in values) / (n - 1)
            else:
                variance = 0.0

            stats.append(
                VariantStats(
                    variant_index=i,
                    skill_id=variant["skill_id"],
                    sample_size=n,
                    successes=successes,
                    success_rate=successes / n,
                    mean_metric=mean,
                    variance_metric=variance,
                )
            )

        return stats

    def evaluate(self, experiment_id: str) -> ExperimentResult:
        """Evaluate an experiment's current outcomes.

        For two variants, applies a **two-proportion z-test**.
        For three or more variants, applies a **chi-squared test**.

        Returns
        -------
        ExperimentResult
            Aggregated statistics and significance verdict.
        """
        row = self._conn.execute(
            "SELECT config_json, status FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        config = json.loads(row["config_json"])
        status = ExperimentStatus(row["status"])
        stats = self._compute_variant_stats(experiment_id, config)
        total_samples = sum(s.sample_size for s in stats)
        min_sample = config.get("min_sample_size", 30)

        result = ExperimentResult(
            experiment_id=experiment_id,
            status=status,
            variant_stats=stats,
        )

        if total_samples < min_sample:
            result.summary = (
                f"Insufficient data: {total_samples}/{min_sample} samples collected"
            )
            return result

        n_variants = len(config["variants"])

        if n_variants == 2:
            # Two-proportion z-test
            s1, s2 = stats[0], stats[1]
            z, p = _two_proportion_z_test(
                s1.sample_size, s1.successes,
                s2.sample_size, s2.successes,
            )
            result.test_name = "two_proportion_z_test"
            result.p_value = p
            alpha = config.get("alpha", 0.05)
            result.is_significant = p < alpha

            if result.is_significant:
                winner = 0 if s1.success_rate > s2.success_rate else 1
                result.best_variant_index = winner
                loser = 1 - winner
                result.summary = (
                    f"Significant result (p={p:.4f}): variant {winner} "
                    f"({config['variants'][winner]['skill_id']}, "
                    f"SR={stats[winner].success_rate:.2%}) outperforms "
                    f"variant {loser} "
                    f"(SR={stats[loser].success_rate:.2%})"
                )
            else:
                result.summary = (
                    f"No significant difference (p={p:.4f}, α={alpha})"
                )

        else:
            # Chi-squared test across multiple variants
            observed = [s.successes for s in stats]
            total_successes = sum(observed)
            # Expected: equal distribution of successes
            if total_successes == 0 or n_variants == 0:
                result.summary = "No successes recorded across any variant"
                return result
            expected = [total_successes / n_variants] * n_variants
            chi2, p = _chi_squared_test(observed, expected)

            result.test_name = "chi_squared"
            result.p_value = p
            alpha = config.get("alpha", 0.05)
            result.is_significant = p < alpha

            if result.is_significant:
                winner = max(range(n_variants), key=lambda i: stats[i].success_rate)
                result.best_variant_index = winner
                result.summary = (
                    f"Significant result (χ²={chi2:.4f}, p={p:.4f}): "
                    f"variant {winner} "
                    f"({config['variants'][winner]['skill_id']}, "
                    f"SR={stats[winner].success_rate:.2%}) is best"
                )
            else:
                result.summary = (
                    f"No significant difference (χ²={chi2:.4f}, p={p:.4f}, "
                    f"α={alpha})"
                )

        return result

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_experiment_status(self, experiment_id: str) -> ExperimentStatus:
        """Return the current status of an experiment."""
        row = self._conn.execute(
            "SELECT status FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        return ExperimentStatus(row["status"])

    def list_experiments(
        self, status: ExperimentStatus | None = None
    ) -> list[dict[str, Any]]:
        """List experiments, optionally filtered by status.

        Returns
        -------
        list[dict[str, Any]]
            Each dict has keys ``id``, ``name``, ``status``, ``created_at``.
        """
        if status is not None:
            rows = self._conn.execute(
                "SELECT id, name, status, created_at FROM experiments "
                "WHERE status = ? ORDER BY created_at DESC",
                (status.value,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, name, status, created_at FROM experiments "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_outcomes(
        self,
        experiment_id: str,
        variant_index: int | None = None,
        limit: int = 1000,
    ) -> list[Outcome]:
        """Retrieve recorded outcomes for an experiment.

        Parameters
        ----------
        experiment_id : str
            The experiment to query.
        variant_index : int | None
            Filter to a specific variant.
        limit : int
            Maximum outcomes to return.

        Returns
        -------
        list[Outcome]
        """
        if variant_index is not None:
            rows = self._conn.execute(
                "SELECT * FROM outcomes "
                "WHERE experiment_id = ? AND variant_index = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (experiment_id, variant_index, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM outcomes WHERE experiment_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (experiment_id, limit),
            ).fetchall()

        return [
            Outcome(
                id=r["id"],
                experiment_id=r["experiment_id"],
                variant_index=r["variant_index"],
                success=bool(r["success"]),
                metric_value=r["metric_value"],
                context=json.loads(r["context_json"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
