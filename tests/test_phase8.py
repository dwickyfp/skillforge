"""Tests for Phase 8: Production — Resilience + Caching."""
import sys
import time
import tempfile
import os
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from skillforge.core.resilience import (
    CircuitBreaker, CircuitState, CircuitStats, CircuitOpenError,
    RetryPolicy, Bulkhead, BulkheadStats, BulkheadFullError,
    GracefulDegradation, ResilientExecutor, ExecutorStats,
)
from skillforge.core.cache import (
    TTLCache, LRUCache, CacheStats, CachedStore, cached, cache_key,
)


# =====================================================================
# Circuit Breaker Tests
# =====================================================================

class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker("test")
        stats = cb.get_stats()
        assert stats.state == CircuitState.CLOSED
        assert stats.failure_count == 0

    def test_success_stays_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        for _ in range(10):
            cb.call(lambda: True)
        assert cb.get_stats().state == CircuitState.CLOSED

    def test_failures_open_circuit(self):
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=999)
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(self._fail)
        assert cb.get_stats().state == CircuitState.OPEN

    def test_open_rejects_calls(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=999)
        with pytest.raises(ValueError):
            cb.call(self._fail)
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: True)

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1, half_open_max=1)
        with pytest.raises(ValueError):
            cb.call(self._fail)
        time.sleep(0.15)
        result = cb.call(lambda: "ok")
        assert result == "ok"

    def test_excluded_exceptions_dont_count(self):
        cb = CircuitBreaker("test", failure_threshold=3, excluded_exceptions=(ValueError,))
        for _ in range(5):
            with pytest.raises(ValueError):
                cb.call(self._fail)
        assert cb.get_stats().state == CircuitState.CLOSED

    def test_manual_reset(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=999)
        with pytest.raises(ValueError):
            cb.call(self._fail)
        assert cb.get_stats().state == CircuitState.OPEN
        cb.reset()
        assert cb.get_stats().state == CircuitState.CLOSED

    def test_success_resets_consecutive_failures(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(self._fail)
        cb.call(lambda: "ok")
        assert cb.get_stats().consecutive_failures == 0

    @staticmethod
    def _fail():
        raise ValueError("fail")


# =====================================================================
# Retry Policy Tests
# =====================================================================

class TestRetryPolicy:
    def test_no_retry_on_success(self):
        rp = RetryPolicy(max_attempts=3, base_delay=0.01)
        result = rp.execute(lambda: "ok")
        assert result == "ok"

    def test_retry_on_failure(self):
        attempts = 0
        def flaky():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("transient")
            return "ok"

        rp = RetryPolicy(max_attempts=5, base_delay=0.01, jitter=False)
        result = rp.execute(flaky)
        assert result == "ok"
        assert attempts == 3

    def test_exhausted_retries_raises(self):
        from skillforge.core.resilience import RetryExhaustedError
        rp = RetryPolicy(max_attempts=2, base_delay=0.01, jitter=False)
        with pytest.raises(RetryExhaustedError):
            rp.execute(self._always_fail)

    def test_custom_retriable_exceptions(self):
        rp = RetryPolicy(max_attempts=3, base_delay=0.01, retriable_exceptions=(TypeError,))
        with pytest.raises(ValueError):
            rp.execute(lambda: (_ for _ in ()).throw(ValueError("no")))

    @staticmethod
    def _always_fail():
        raise RuntimeError("always")


# =====================================================================
# Bulkhead Tests
# =====================================================================

class TestBulkhead:
    def test_admits_within_capacity(self):
        bh = Bulkhead("test", max_concurrent=2)
        bh.execute(lambda: "ok")
        assert bh.get_stats().current_executions == 0

    def test_rejects_at_capacity(self):
        bh = Bulkhead("test", max_concurrent=1, max_wait=0.0)
        barrier = threading.Event()

        def blocking():
            bh.execute(lambda: barrier.wait(2.0))
            return "ok"

        bg = threading.Thread(target=blocking)
        bg.start()
        time.sleep(0.05)

        with pytest.raises(BulkheadFullError):
            bh.execute(lambda: "fail")

        barrier.set()
        bg.join()

    def test_stats(self):
        bh = Bulkhead("test", max_concurrent=5)
        for _ in range(3):
            bh.execute(lambda: "ok")
        stats = bh.get_stats()
        assert stats.total_calls == 3


# =====================================================================
# Graceful Degradation Tests
# =====================================================================

class TestGracefulDegradation:
    def test_primary_success(self):
        gd = GracefulDegradation("test", fallbacks=[lambda: "backup"])
        assert gd.execute(lambda: "primary") == "primary"

    def test_fallback_to_backup(self):
        gd = GracefulDegradation("test", fallbacks=[lambda: "backup"])
        result = gd.execute(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert result == "backup"

    def test_all_fail_raises(self):
        from skillforge.core.resilience import FallbackChainExhaustedError
        gd = GracefulDegradation("test", fallbacks=[lambda: (_ for _ in ()).throw(RuntimeError("b"))])
        with pytest.raises(FallbackChainExhaustedError):
            gd.execute(lambda: (_ for _ in ()).throw(RuntimeError("a")))

    def test_add_fallback(self):
        gd = GracefulDegradation("test")
        gd.add_fallback(lambda: "fallback1")
        assert gd.execute(lambda: (_ for _ in ()).throw(RuntimeError("fail"))) == "fallback1"


# =====================================================================
# Resilient Executor Tests
# =====================================================================

class TestResilientExecutor:
    def test_basic_execution(self):
        re = ResilientExecutor(name="test")
        assert re.execute(lambda: "ok") == "ok"

    def test_with_retry_policy(self):
        attempts = 0
        def flaky():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise RuntimeError("transient")
            return "ok"

        re = ResilientExecutor(name="test", retry_policy=RetryPolicy(max_attempts=3, base_delay=0.01, jitter=False))
        assert re.execute(flaky) == "ok"

    def test_with_circuit_breaker(self):
        cb = CircuitBreaker("db", failure_threshold=3)
        re = ResilientExecutor(name="test", circuit_breaker=cb)
        assert re.execute(lambda: "ok") == "ok"

    def test_stats(self):
        re = ResilientExecutor(name="test")
        re.execute(lambda: "ok")
        stats = re.get_stats()
        assert stats.total_attempts == 1
        assert stats.total_successes == 1


# =====================================================================
# TTL Cache Tests
# =====================================================================

class TestTTLCache:
    def test_put_and_get(self):
        cache = TTLCache(ttl_seconds=10.0)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_miss_returns_none(self):
        cache = TTLCache(ttl_seconds=10.0)
        assert cache.get("missing") is None

    def test_ttl_expiry(self):
        cache = TTLCache(ttl_seconds=0.05)
        cache.put("key1", "value1")
        time.sleep(0.1)
        assert cache.get("key1") is None

    def test_invalidate(self):
        cache = TTLCache(ttl_seconds=10.0)
        cache.put("key1", "value1")
        assert cache.invalidate("key1") is True
        assert cache.get("key1") is None

    def test_max_size_eviction(self):
        cache = TTLCache(ttl_seconds=999, max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # should evict oldest
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_stats(self):
        cache = TTLCache(ttl_seconds=10.0)
        cache.put("a", 1)
        cache.get("a")  # hit
        cache.get("b")  # miss
        stats = cache.stats
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5

    def test_clear(self):
        cache = TTLCache(ttl_seconds=10.0)
        cache.put("a", 1)
        cache.clear()
        assert cache.get("a") is None


# =====================================================================
# LRU Cache Tests
# =====================================================================

class TestLRUCache:
    def test_put_and_get(self):
        cache = LRUCache(max_size=10)
        cache.put("a", 1)
        assert cache.get("a") == 1

    def test_lru_eviction(self):
        cache = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_access_moves_to_end(self):
        cache = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # move "a" to end
        cache.put("c", 3)  # evicts "b" (oldest)
        assert cache.get("a") == 1
        assert cache.get("b") is None


# =====================================================================
# Cached Store Tests
# =====================================================================

class TestCachedStore:
    def test_fetches_on_miss(self):
        calls = []
        def fetch(key):
            calls.append(key)
            return f"value_{key}"

        store = CachedStore(fetch, ttl_seconds=10.0)
        assert store.get("x") == "value_x"
        assert len(calls) == 1

    def test_caches_on_hit(self):
        calls = []
        def fetch(key):
            calls.append(key)
            return f"value_{key}"

        store = CachedStore(fetch, ttl_seconds=10.0)
        store.get("x")
        store.get("x")  # should be cached
        assert len(calls) == 1


# =====================================================================
# Cached Decorator Tests
# =====================================================================

class TestCachedDecorator:
    def test_caches_result(self):
        calls = []
        @cached(ttl_seconds=10.0)
        def expensive(x):
            calls.append(x)
            return x * 2

        assert expensive(5) == 10
        assert expensive(5) == 10  # cached
        assert len(calls) == 1

    def test_different_args_different_cache(self):
        calls = []
        @cached(ttl_seconds=10.0)
        def expensive(x):
            calls.append(x)
            return x * 2

        assert expensive(5) == 10
        assert expensive(3) == 6
        assert len(calls) == 2


# =====================================================================
# Cache Key Tests
# =====================================================================

class TestCacheKey:
    def test_deterministic(self):
        k1 = cache_key("a", "b")
        k2 = cache_key("a", "b")
        assert k1 == k2

    def test_different_args_different_key(self):
        k1 = cache_key("a")
        k2 = cache_key("b")
        assert k1 != k2


# =====================================================================
# Integration Tests
# =====================================================================

class TestPhase8Integration:
    def test_resilient_executor_with_all_layers(self):
        cb = CircuitBreaker("db", failure_threshold=3)
        rp = RetryPolicy(max_attempts=2, base_delay=0.01, jitter=False)
        re = ResilientExecutor(
            name="db-call",
            circuit_breaker=cb,
            retry_policy=rp,
        )
        assert re.execute(lambda: "ok") == "ok"

    def test_cached_skill_stats(self):
        from skillforge import SkillForge
        import tempfile
        db = os.path.join(tempfile.mkdtemp(), "test.db")
        forge = SkillForge(db_path=db)
        forge.register_skill("test", "Test Skill", tier2_core="Step 1\nStep 2")
        forge.record_outcome("test", success=True, latency_ms=1000, tokens_used=1000)

        cache = TTLCache(ttl_seconds=10.0)

        def get_stats_cached(skill_id):
            cached_val = cache.get(skill_id)
            if cached_val:
                return cached_val
            stats = forge.get_skill_stats(skill_id)
            cache.put(skill_id, stats)
            return stats

        stats1 = get_stats_cached("test")
        stats2 = get_stats_cached("test")
        assert stats1 == stats2
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1
        forge.close()

    def test_resilient_executor_with_degradation(self):
        gd = GracefulDegradation("test", fallbacks=[lambda: "degraded"])
        re = ResilientExecutor(name="test", degradation=gd)
        result = re.execute(lambda: "ok")
        assert result == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
