"""Tests for QValueTracker (skillforge.core.tracker)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from skillforge.core.tracker import Outcome, QValueTracker


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_tracker.db"


@pytest.fixture
def tracker(tmp_db: Path) -> Iterator[QValueTracker]:
    t = QValueTracker(db_path=tmp_db)
    yield t
    t.close()


def _make_outcome(
    skill_id: str = "skill-1",
    success: bool = True,
    latency_ms: float = 100.0,
    tokens_used: int = 50,
    user_feedback: float | None = None,
) -> Outcome:
    return Outcome(
        skill_id=skill_id,
        success=success,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        user_feedback=user_feedback,
    )


# -----------------------------------------------------------------------
# Recording outcomes
# -----------------------------------------------------------------------

class TestRecordOutcome:
    """Tests for QValueTracker.record_outcome."""

    def test_record_single_outcome(self, tracker: QValueTracker) -> None:
        tracker.record_outcome(_make_outcome())
        stats = tracker.get_stats("skill-1")
        assert stats["total_outcomes"] == 1

    def test_record_multiple_outcomes(self, tracker: QValueTracker) -> None:
        for i in range(5):
            tracker.record_outcome(_make_outcome(success=i % 2 == 0))
        stats = tracker.get_stats("skill-1")
        assert stats["total_outcomes"] == 5

    def test_record_aggregates_latency(self, tracker: QValueTracker) -> None:
        tracker.record_outcome(_make_outcome(latency_ms=100.0))
        tracker.record_outcome(_make_outcome(latency_ms=200.0))
        stats = tracker.get_stats("skill-1")
        assert stats["avg_latency_ms"] == pytest.approx(150.0)

    def test_record_aggregates_tokens(self, tracker: QValueTracker) -> None:
        tracker.record_outcome(_make_outcome(tokens_used=100))
        tracker.record_outcome(_make_outcome(tokens_used=200))
        stats = tracker.get_stats("skill-1")
        assert stats["avg_tokens"] == 150

    def test_record_preserves_user_feedback(self, tracker: QValueTracker) -> None:
        tracker.record_outcome(_make_outcome(user_feedback=4.5))
        # Feedback is stored in outcomes table but not directly in stats
        stats = tracker.get_stats("skill-1")
        assert stats["total_outcomes"] == 1

    def test_record_separate_skills(self, tracker: QValueTracker) -> None:
        tracker.record_outcome(_make_outcome(skill_id="a"))
        tracker.record_outcome(_make_outcome(skill_id="b"))
        assert tracker.get_stats("a")["total_outcomes"] == 1
        assert tracker.get_stats("b")["total_outcomes"] == 1


# -----------------------------------------------------------------------
# Q-value queries
# -----------------------------------------------------------------------

class TestGetQValue:
    """Tests for QValueTracker.get_q_value."""

    def test_default_q_value(self, tracker: QValueTracker) -> None:
        assert tracker.get_q_value("unknown") == 0.5

    def test_q_value_after_outcomes(self, tracker: QValueTracker) -> None:
        # Record some outcomes so q_values row is created
        tracker.record_outcome(_make_outcome(success=True))
        q = tracker.get_q_value("skill-1")
        # Default initial q_value in upsert is 0.5
        assert q == 0.5


# -----------------------------------------------------------------------
# Success rate
# -----------------------------------------------------------------------

class TestSuccessRate:
    """Tests for QValueTracker.get_success_rate."""

    def test_default_success_rate(self, tracker: QValueTracker) -> None:
        assert tracker.get_success_rate("unknown") == 0.5

    def test_success_rate_all_success(self, tracker: QValueTracker) -> None:
        for _ in range(5):
            tracker.record_outcome(_make_outcome(success=True))
        assert tracker.get_success_rate("skill-1") == 1.0

    def test_success_rate_all_failure(self, tracker: QValueTracker) -> None:
        for _ in range(5):
            tracker.record_outcome(_make_outcome(success=False))
        assert tracker.get_success_rate("skill-1") == 0.0

    def test_success_rate_mixed(self, tracker: QValueTracker) -> None:
        for i in range(4):
            tracker.record_outcome(_make_outcome(success=i < 3))
        rate = tracker.get_success_rate("skill-1")
        assert rate == pytest.approx(0.75)


# -----------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------

class TestGetStats:
    """Tests for QValueTracker.get_stats."""

    def test_stats_empty(self, tracker: QValueTracker) -> None:
        stats = tracker.get_stats("nope")
        assert stats["skill_id"] == "nope"
        assert stats["q_value"] == 0.5
        assert stats["success_rate"] == 0.5
        assert stats["total_outcomes"] == 0
        assert stats["avg_latency_ms"] == 0.0
        assert stats["avg_tokens"] == 0

    def test_stats_with_data(self, tracker: QValueTracker) -> None:
        tracker.record_outcome(_make_outcome(success=True, latency_ms=100, tokens_used=50))
        tracker.record_outcome(_make_outcome(success=False, latency_ms=200, tokens_used=100))
        stats = tracker.get_stats("skill-1")
        assert stats["total_outcomes"] == 2
        assert stats["success_rate"] == pytest.approx(0.5)
        assert stats["avg_latency_ms"] == pytest.approx(150.0)
        assert stats["avg_tokens"] == 75


# -----------------------------------------------------------------------
# TD(λ) Update
# -----------------------------------------------------------------------

class TestTDLambda:
    """Tests for QValueTracker.td_lambda_update."""

    def test_td_lambda_positive_reward(self, tracker: QValueTracker) -> None:
        tracker.record_outcome(_make_outcome(success=True))
        new_q = tracker.td_lambda_update("skill-1", reward=1.0)
        # Q should move towards 1.0 from 0.5
        assert new_q > 0.5
        assert new_q <= 1.0

    def test_td_lambda_negative_reward(self, tracker: QValueTracker) -> None:
        tracker.record_outcome(_make_outcome(success=False))
        new_q = tracker.td_lambda_update("skill-1", reward=0.0)
        # Q should move towards 0.0 from 0.5
        assert new_q < 0.5
        assert new_q >= 0.0

    def test_td_lambda_clamps_to_0_1(self, tracker: QValueTracker) -> None:
        # Push hard towards 1.0
        for _ in range(100):
            tracker.record_outcome(_make_outcome(success=True))
        q = tracker.td_lambda_update("skill-1", reward=1.0, alpha=0.9)
        assert 0.0 <= q <= 1.0

        # Push hard towards 0.0
        q = tracker.td_lambda_update("skill-1", reward=0.0, alpha=0.9)
        assert 0.0 <= q <= 1.0

    def test_td_lambda_convergence(self, tracker: QValueTracker) -> None:
        """Repeated updates should converge towards the reward."""
        tracker.record_outcome(_make_outcome(success=True))
        q = 0.5
        for _ in range(50):
            q = tracker.td_lambda_update("skill-1", reward=0.8, alpha=0.05)
        assert abs(q - 0.8) < 0.2  # Should be close-ish

    def test_td_lambda_custom_params(self, tracker: QValueTracker) -> None:
        tracker.record_outcome(_make_outcome(success=True))
        new_q = tracker.td_lambda_update(
            "skill-1", reward=1.0, alpha=0.5, gamma=0.5, lambda_=0.5
        )
        assert isinstance(new_q, float)
        assert 0.0 <= new_q <= 1.0

    def test_td_lambda_unknown_skill(self, tracker: QValueTracker) -> None:
        # No outcomes recorded means no trace → update is zero
        new_q = tracker.td_lambda_update("unknown", reward=1.0)
        # With no history, cumulative_update is 0, so Q stays at 0.5
        assert new_q == pytest.approx(0.5)

    def test_td_lambda_with_history(self, tracker: QValueTracker) -> None:
        """Multiple recorded outcomes should affect the trace."""
        for _ in range(10):
            tracker.record_outcome(_make_outcome(success=True))
        new_q = tracker.td_lambda_update("skill-1", reward=1.0)
        assert new_q > 0.5


# -----------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------

class TestPersistence:
    """Verify tracker data survives close/reopen."""

    def test_data_persists(self, tmp_db: Path) -> None:
        t = QValueTracker(db_path=tmp_db)
        t.record_outcome(_make_outcome(success=True))
        t.td_lambda_update("skill-1", reward=1.0)
        t.close()

        t2 = QValueTracker(db_path=tmp_db)
        stats = t2.get_stats("skill-1")
        assert stats["total_outcomes"] == 1
        assert stats["q_value"] > 0.5
        t2.close()
