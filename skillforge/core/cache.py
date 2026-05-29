"""Performance Caching Module.

Provides production-grade caching for SkillForge operations:

- :class:`TTLCache` — time-to-live cache with automatic expiry.
- :class:`LRUCache` — Least Recently Used eviction cache.
- :class:`CacheStats` — hit/miss tracking for cache monitoring.
- :class:`CachedStore` — composable cache layer over any data source.
- :func:`cached` — function decorator for automatic result caching.

All implementations are stdlib-only, thread-safe, and zero-dependency.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Generic, Optional, TypeVar

T = TypeVar("T")

logger = __import__("logging").getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache statistics
# ---------------------------------------------------------------------------

@dataclass
class CacheStats:
    """Cache performance counters."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    inserts: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses


# ---------------------------------------------------------------------------
# TTL Cache
# ---------------------------------------------------------------------------

class TTLCache(Generic[T]):
    """Thread-safe Time-To-Live cache.

    Entries expire after ``ttl_seconds`` and are lazily evicted on access.

    Parameters
    ----------
    ttl_seconds : float
        Default time-to-live in seconds.
    max_size : int
        Maximum number of entries (0 = unlimited).
    """

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 1024) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: dict[str, tuple[float, T]] = {}
        self._stats = CacheStats()
        self._lock = threading.Lock()

    def get(self, key: str) -> T | None:
        """Return cached value or ``None`` if expired/missing."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._stats.misses += 1
                return None
            ts, value = entry
            if time.monotonic() - ts > self._ttl:
                del self._store[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None
            self._stats.hits += 1
            return value

    def put(self, key: str, value: T) -> None:
        """Store a value with the default TTL."""
        with self._lock:
            if self._max_size > 0 and len(self._store) >= self._max_size and key not in self._store:
                self._evict_expired()
            if self._max_size > 0 and len(self._store) >= self._max_size:
                self._evict_oldest()
            self._store[key] = (time.monotonic(), value)
            self._stats.inserts += 1
            self._stats.size = len(self._store)

    def invalidate(self, key: str) -> bool:
        """Remove a specific key. Returns ``True`` if key existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                self._stats.size = len(self._store)
                return True
            return False

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()
            self._stats.size = 0

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            self._stats.size = len(self._store)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                inserts=self._stats.inserts,
                size=self._stats.size,
            )

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (ts, _) in self._store.items() if now - ts > self._ttl]
        for k in expired:
            del self._store[k]
            self._stats.evictions += 1

    def _evict_oldest(self) -> None:
        if self._store:
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest_key]
            self._stats.evictions += 1


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------

class LRUCache(Generic[T]):
    """Thread-safe Least Recently Used cache.

    Parameters
    ----------
    max_size : int
        Maximum entries before eviction.
    """

    def __init__(self, max_size: int = 256) -> None:
        self._max_size = max_size
        self._store: OrderedDict[str, T] = OrderedDict()
        self._stats = CacheStats()
        self._lock = threading.Lock()

    def get(self, key: str) -> T | None:
        with self._lock:
            if key not in self._store:
                self._stats.misses += 1
                return None
            self._store.move_to_end(key)
            self._stats.hits += 1
            return self._store[key]

    def put(self, key: str, value: T) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            elif len(self._store) >= self._max_size:
                self._store.popitem(last=False)
                self._stats.evictions += 1
            self._store[key] = value
            self._stats.inserts += 1
            self._stats.size = len(self._store)

    def invalidate(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                self._stats.size = len(self._store)
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._stats.size = 0

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            self._stats.size = len(self._store)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                inserts=self._stats.inserts,
                size=self._stats.size,
            )


# ---------------------------------------------------------------------------
# Cached Store — composable cache layer
# ---------------------------------------------------------------------------

class CachedStore(Generic[T]):
    """Wraps a data source with a TTL cache.

    Parameters
    ----------
    fetch_fn : callable
        Function that fetches data given a key.
    ttl_seconds : float
        Cache TTL in seconds.
    max_size : int
        Maximum cache entries.
    """

    def __init__(
        self,
        fetch_fn: Callable[[str], T],
        ttl_seconds: float = 300.0,
        max_size: int = 1024,
    ) -> None:
        self._fetch = fetch_fn
        self._cache: TTLCache[T] = TTLCache(ttl_seconds, max_size)

    def get(self, key: str) -> T:
        """Fetch from cache, falling back to source on miss."""
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value = self._fetch(key)
        self._cache.put(key, value)
        return value

    def invalidate(self, key: str) -> bool:
        return self._cache.invalidate(key)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def stats(self) -> CacheStats:
        return self._cache.stats


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------

def cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate a deterministic cache key from arguments."""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(key_data.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Function decorator
# ---------------------------------------------------------------------------

def cached(
    ttl_seconds: float = 300.0,
    max_size: int = 256,
    key_fn: Callable[..., str] | None = None,
) -> Callable:
    """Decorator that caches function results with TTL.

    Parameters
    ----------
    ttl_seconds : float
        Cache TTL in seconds.
    max_size : int
        Maximum entries.
    key_fn : callable, optional
        Custom key generation function. Defaults to ``cache_key``.

    Example::

        @cached(ttl_seconds=60)
        def get_skill_stats(skill_id: str) -> dict:
            # expensive query
            return result
    """
    _cache: TTLCache = TTLCache(ttl_seconds, max_size)
    _key_fn = key_fn or cache_key

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = _key_fn(*args, **kwargs)
            result = _cache.get(key)
            if result is not None:
                return result
            value = fn(*args, **kwargs)
            _cache.put(key, value)
            return value

        wrapper.cache = _cache  # type: ignore[attr-defined]
        wrapper.cache_clear = _cache.clear  # type: ignore[attr-defined]
        wrapper.cache_stats = lambda: _cache.stats  # type: ignore[attr-defined]
        return wrapper

    return decorator
