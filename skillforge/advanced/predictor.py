"""
Skill performance predictor using lightweight OLS regression.

Predicts future skill performance from historical outcome data using
simple ordinary-least-squares linear regression.  No external
dependencies — all mathematics is implemented in pure Python.

Provides decline detection, performance forecasting, and intelligent
skill recommendation based on predicted (not just historical) metrics.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SkillPrediction:
    """Predicted performance metrics for a skill.

    Attributes
    ----------
    skill_id : str
        Identifier of the skill.
    predicted_success_rate : float
        Predicted success rate (0-1) for the next invocation window.
    predicted_tokens : float
        Predicted token usage for the next invocation.
    predicted_latency : float
        Predicted latency in milliseconds for the next invocation.
    confidence : float
        Confidence in the prediction (0-1), computed as ``1 / (1 + var)``.
    recommendation : str
        Human-readable recommendation (e.g. ``"use"``, ``"monitor"``,
        ``"avoid"``).
    """

    skill_id: str
    predicted_success_rate: float
    predicted_tokens: float
    predicted_latency: float
    confidence: float
    recommendation: str


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class SkillRegistryProto(Protocol):
    """Minimal registry interface required by :class:`SkillPredictor`."""

    def get_skill(self, skill_id: str, tier: int = 1) -> Any: ...  # pragma: no cover
    def list_skills(self, **kwargs: Any) -> list[Any]: ...  # pragma: no cover
    def search_skills(self, query: str, limit: int = 20) -> list[Any]: ...  # pragma: no cover


class TrackerProto(Protocol):
    """Minimal Q-value tracker interface required by :class:`SkillPredictor`."""

    def get_q_value(self, skill_id: str) -> float: ...  # pragma: no cover
    def get_success_rate(self, skill_id: str) -> float: ...  # pragma: no cover
    def get_stats(self, skill_id: str) -> dict[str, Any]: ...  # pragma: no cover


class GraphProto(Protocol):
    """Minimal dependency-graph interface required by :class:`SkillPredictor`."""

    def get_skills(self) -> list[str]: ...  # pragma: no cover
    def downstream_impact(self, skill_id: str) -> list[str]: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# SkillPredictor
# ---------------------------------------------------------------------------


class SkillPredictor:
    """Predicts skill performance from historical outcome data.

    Uses OLS linear regression on the last *N* outcomes to forecast
    success rate, token usage, and latency.  Confidence is derived from
    outcome variance (lower variance → higher confidence).

    Parameters
    ----------
    registry : SkillRegistryProto
        Skill registry for metadata queries.
    tracker : TrackerProto
        Outcome / Q-value tracker with a ``_conn`` attribute exposing the
        underlying SQLite connection for outcome queries.
    graph : GraphProto
        Skill dependency graph for context-aware recommendations.
    """

    def __init__(
        self,
        registry: SkillRegistryProto,
        tracker: TrackerProto,
        graph: GraphProto,
    ) -> None:
        self._registry = registry
        self._tracker = tracker
        self._graph = graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_performance(self, skill_id: str) -> SkillPrediction:
        """Predict future performance for a single skill.

        Fits a linear regression on the last 20 outcomes for ``success``,
        ``tokens_used``, and ``latency_ms``.  The predicted value for the
        next step (x = 21) is used as the forecast.

        Parameters
        ----------
        skill_id : str
            Identifier of the skill to predict.

        Returns
        -------
        SkillPrediction
            Forecasted metrics with confidence and recommendation.
        """
        outcomes = self._get_recent_outcomes(skill_id, limit=20)

        if len(outcomes) < 2:
            # Not enough data — fall back to current stats
            stats = self._tracker.get_stats(skill_id)
            return SkillPrediction(
                skill_id=skill_id,
                predicted_success_rate=stats.get("success_rate", 0.5),
                predicted_tokens=float(stats.get("avg_tokens", 0)),
                predicted_latency=float(stats.get("avg_latency_ms", 0.0)),
                confidence=0.1,
                recommendation="insufficient_data",
            )

        xs = list(range(len(outcomes)))
        successes = [1.0 if o["success"] else 0.0 for o in outcomes]
        tokens = [float(o["tokens_used"]) for o in outcomes]
        latencies = [float(o["latency_ms"]) for o in outcomes]

        # Linear regression for each metric
        sr_slope, sr_intercept = self._linear_regression(xs, successes)
        tk_slope, tk_intercept = self._linear_regression(xs, tokens)
        lat_slope, lat_intercept = self._linear_regression(xs, latencies)

        next_x = float(len(outcomes))
        predicted_sr = max(0.0, min(1.0, sr_slope * next_x + sr_intercept))
        predicted_tokens = max(0.0, tk_slope * next_x + tk_intercept)
        predicted_latency = max(0.0, lat_slope * next_x + lat_intercept)

        # Confidence = 1 / (1 + variance of success outcomes)
        variance = self._variance(successes)
        confidence = 1.0 / (1.0 + variance)

        recommendation = self._make_recommendation(predicted_sr, confidence)

        return SkillPrediction(
            skill_id=skill_id,
            predicted_success_rate=predicted_sr,
            predicted_tokens=predicted_tokens,
            predicted_latency=predicted_latency,
            confidence=confidence,
            recommendation=recommendation,
        )

    def predict_for_task(self, query: str) -> list[SkillPrediction]:
        """Predict performance of the top-5 matching skills for a query.

        Parameters
        ----------
        query : str
            Natural-language query describing the task.

        Returns
        -------
        list[SkillPrediction]
            Predictions for up to 5 matching skills, sorted by predicted
            success rate descending.
        """
        matching_skills = self._registry.search_skills(query, limit=5)
        predictions: list[SkillPrediction] = []
        for skill in matching_skills:
            sid: str = skill.id if hasattr(skill, "id") else str(skill)
            pred = self.predict_performance(sid)
            predictions.append(pred)

        predictions.sort(key=lambda p: p.predicted_success_rate, reverse=True)
        return predictions

    def detect_decline(self, skill_id: str, window: int = 20) -> bool:
        """Detect if a skill's performance is declining.

        Fits a linear regression on the Q-values from the last *window*
        outcomes.  A negative slope indicates decline.

        Parameters
        ----------
        skill_id : str
            Identifier of the skill.
        window : int
            Number of recent outcomes to consider.

        Returns
        -------
        bool
            ``True`` if a statistically meaningful negative trend is detected.
        """
        outcomes = self._get_recent_outcomes(skill_id, limit=window)
        if len(outcomes) < 3:
            return False

        xs = list(range(len(outcomes)))
        # Use success as a proxy for Q-value trajectory
        successes = [1.0 if o["success"] else 0.0 for o in outcomes]
        slope, _ = self._linear_regression(xs, successes)

        # Consider declining if slope is meaningfully negative
        # (less than -0.01 per step, i.e. > 1% decline per invocation)
        return slope < -0.01

    def recommend_skills(
        self, query: str, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Recommend skills based on predicted performance.

        Unlike simple Q-value ranking, this considers predicted future
        performance, confidence, and decline status.

        Parameters
        ----------
        query : str
            Natural-language task description.
        top_k : int
            Maximum number of recommendations to return.

        Returns
        -------
        list[dict[str, Any]]
            Each dict contains ``skill_id``, ``prediction``
            (:class:`SkillPrediction`), ``score`` (composite ranking
            score), ``declining`` (bool), and ``reason`` (str).
        """
        matching_skills = self._registry.search_skills(query, limit=10)
        candidates: list[dict[str, Any]] = []

        for skill in matching_skills:
            sid: str = skill.id if hasattr(skill, "id") else str(skill)
            pred = self.predict_performance(sid)
            declining = self.detect_decline(sid)

            # Composite score: predicted success * confidence, penalised
            # if declining
            score = pred.predicted_success_rate * pred.confidence
            if declining:
                score *= 0.7  # 30% penalty for declining skills

            reason = self._recommendation_reason(pred, declining)

            candidates.append(
                {
                    "skill_id": sid,
                    "prediction": pred,
                    "score": score,
                    "declining": declining,
                    "reason": reason,
                }
            )

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _linear_regression(
        xs: Sequence[int | float], ys: Sequence[float]
    ) -> tuple[float, float]:
        """Simple OLS linear regression.

        Fits ``y = slope * x + intercept`` using the standard closed-form
        solution.

        Parameters
        ----------
        xs : list[int | float]
            Independent variable values.
        ys : list[float]
            Dependent variable values.

        Returns
        -------
        tuple[float, float]
            ``(slope, intercept)`` of the best-fit line.  Returns
            ``(0.0, mean(ys))`` if the regression is degenerate (e.g. all
            x values identical).
        """
        n = len(xs)
        if n == 0:
            return 0.0, 0.0
        if n == 1:
            return 0.0, ys[0]

        sum_x = sum(float(x) for x in xs)
        sum_y = sum(ys)
        sum_xy = sum(float(x) * y for x, y in zip(xs, ys))
        sum_x2 = sum(float(x) ** 2 for x in xs)

        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-12:
            # All x values are the same — no slope
            return 0.0, sum_y / n

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        return slope, intercept

    @staticmethod
    def _variance(values: list[float]) -> float:
        """Compute population variance of a list of floats.

        Parameters
        ----------
        values : list[float]
            The data points.

        Returns
        -------
        float
            Population variance.  Returns ``0.0`` for empty lists.
        """
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def _get_recent_outcomes(
        self, skill_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Fetch recent outcomes from the tracker's SQLite database.

        Accesses the tracker's internal ``_conn`` attribute to query the
        ``outcomes`` table directly.

        Parameters
        ----------
        skill_id : str
            The skill to query outcomes for.
        limit : int
            Maximum number of outcomes to return.

        Returns
        -------
        list[dict[str, Any]]
            Chronological list of outcome dicts with keys ``success``,
            ``tokens_used``, ``latency_ms``.
        """
        conn: sqlite3.Connection | None = getattr(self._tracker, "_conn", None)
        if conn is None:
            logger.warning(
                "Tracker has no '_conn' attribute — cannot query outcomes"
            )
            return []

        try:
            rows = conn.execute(
                "SELECT success, tokens_used, latency_ms "
                "FROM outcomes WHERE skill_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (skill_id, limit),
            ).fetchall()
            # Reverse to chronological order (oldest first)
            results = [
                {
                    "success": bool(row["success"]),
                    "tokens_used": row["tokens_used"],
                    "latency_ms": row["latency_ms"],
                }
                for row in reversed(rows)
            ]
            return results
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to fetch outcomes for '%s': %s", skill_id, exc
            )
            return []

    @staticmethod
    def _make_recommendation(
        predicted_sr: float, confidence: float
    ) -> str:
        """Generate a human-readable recommendation string.

        Parameters
        ----------
        predicted_sr : float
            Predicted success rate (0-1).
        confidence : float
            Prediction confidence (0-1).

        Returns
        -------
        str
            One of ``"use"``, ``"use_with_caution"``, ``"monitor"``,
            ``"avoid"``, or ``"insufficient_confidence"``.
        """
        if confidence < 0.3:
            return "insufficient_confidence"
        if predicted_sr >= 0.8:
            return "use"
        if predicted_sr >= 0.6:
            return "use_with_caution"
        if predicted_sr >= 0.4:
            return "monitor"
        return "avoid"

    @staticmethod
    def _recommendation_reason(
        pred: SkillPrediction, declining: bool
    ) -> str:
        """Generate a human-readable reason for a recommendation.

        Parameters
        ----------
        pred : SkillPrediction
            The prediction result.
        declining : bool
            Whether the skill is in decline.

        Returns
        -------
        str
            Explanatory reason string.
        """
        parts: list[str] = []
        if declining:
            parts.append("performance declining")
        if pred.predicted_success_rate >= 0.8:
            parts.append("high predicted success")
        elif pred.predicted_success_rate < 0.4:
            parts.append("low predicted success")
        if pred.confidence >= 0.7:
            parts.append("high confidence")
        elif pred.confidence < 0.3:
            parts.append("low confidence")
        return ", ".join(parts) if parts else "no strong signal"
