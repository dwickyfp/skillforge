"""Resilience & Error Handling Module.

Provides production-grade fault tolerance primitives:

- :class:`CircuitBreaker` — prevents cascading failures by short-circuiting
  calls to a failing dependency after a configurable error threshold.
- :class:`RetryPolicy` — configurable retry with exponential backoff and
  jitter for transient failures.
- :class:`Bulkhead` — concurrency limiter that isolates resource pools.
- :class:`GracefulDegradation` — cascading fallback chains.
- :class:`ResilientExecutor` — composable executor that chains a circuit
  breaker, retry policy, and bulkhead around any callable.

All classes are stdlib-only with full type hints.  Thread-safety is achieved
via :class:`threading.Lock` where shared mutable state exists.
"""

from __future__ import annotations

import enum
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ======================================================================
# Circuit Breaker
# ======================================================================


class CircuitState(enum.Enum):
    """Circuit breaker states.

    * **CLOSED** — normal operation; calls pass through.
    * **OPEN** — calls are rejected immediately with :class:`CircuitOpenError`.
    * **HALF_OPEN** — a limited number of probe calls are allowed through
      to test whether the downstream has recovered.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitStats:
    """Snapshot of circuit breaker statistics."""

    state: CircuitState
    failure_count: int
    success_count: int
    total_calls: int
    consecutive_failures: int
    last_failure_time: datetime | None
    last_state_change: datetime


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is in the OPEN state."""

    def __init__(self, name: str, remaining_seconds: float) -> None:
        self.name = name
        self.remaining_seconds = remaining_seconds
        super().__init__(
            f"Circuit '{name}' is OPEN. "
            f"Retry in {remaining_seconds:.1f}s."
        )


class CircuitBreaker:
    """State-machine circuit breaker.

    Parameters
    ----------
    name : str
        Human-readable identifier for logging / diagnostics.
    failure_threshold : int
        Number of consecutive failures before the circuit opens.
    recovery_timeout : float
        Seconds to wait in OPEN before transitioning to HALF_OPEN.
    half_open_max : int
        Number of probe calls allowed in HALF_OPEN before deciding.
    excluded_exceptions : tuple[type[BaseException], ...]
        Exception types that **should not** count as failures (e.g.
        validation errors that are not the downstream's fault).
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max: int = 1,
        excluded_exceptions: tuple[type[BaseException], ...] = (),
    ) -> None:
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max = half_open_max
        self._excluded_exceptions = excluded_exceptions

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._total_calls = 0
        self._consecutive_failures = 0
        self._half_open_calls = 0
        self._last_failure_time: datetime | None = None
        self._last_state_change = datetime.now(timezone.utc)
        self._opened_at: float = 0.0
        self._lock = threading.Lock()

    # -- public API ---------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current state, auto-transitioning from OPEN when timeout elapses."""
        with self._lock:
            self._maybe_half_open()
            return self._state

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute *fn* through the circuit breaker.

        Parameters
        ----------
        fn : Callable[..., T]
            The protected callable.
        *args, **kwargs
            Forwarded to *fn*.

        Returns
        -------
        T
            The return value of *fn*.

        Raises
        ------
        CircuitOpenError
            If the circuit is OPEN and the recovery timeout has not elapsed.
        Exception
            Any exception raised by *fn* (after recording it).
        """
        with self._lock:
            self._maybe_half_open()
            state = self._state

            if state == CircuitState.OPEN:
                remaining = self._recovery_timeout - (time.monotonic() - self._opened_at)
                raise CircuitOpenError(self._name, max(0.0, remaining))

            if state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max:
                    remaining = self._recovery_timeout - (
                        time.monotonic() - self._opened_at
                    )
                    raise CircuitOpenError(self._name, max(0.0, remaining))
                self._half_open_calls += 1

            self._total_calls += 1

        # Execute outside the lock so the callable can be long-running.
        try:
            result = fn(*args, **kwargs)
        except self._excluded_exceptions:
            raise  # don't count as failure
        except Exception:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def record_success(self) -> None:
        """Manually record a success (useful for external integrations)."""
        self._record_success()

    def record_failure(self) -> None:
        """Manually record a failure."""
        self._record_failure()

    def reset(self) -> None:
        """Reset the breaker to the CLOSED state with zero counters."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._consecutive_failures = 0
            self._half_open_calls = 0
            self._last_failure_time = None
            self._last_state_change = datetime.now(timezone.utc)

    def get_stats(self) -> CircuitStats:
        """Return a snapshot of current statistics."""
        with self._lock:
            self._maybe_half_open()
            return CircuitStats(
                state=self._state,
                failure_count=self._failure_count,
                success_count=self._success_count,
                total_calls=self._total_calls,
                consecutive_failures=self._consecutive_failures,
                last_failure_time=self._last_failure_time,
                last_state_change=self._last_state_change,
            )

    # -- internals ----------------------------------------------------

    def _record_success(self) -> None:
        with self._lock:
            self._success_count += 1
            self._consecutive_failures = 0
            if self._state == CircuitState.HALF_OPEN:
                # Probe succeeded → close
                self._transition(CircuitState.CLOSED)
                self._half_open_calls = 0

    def _record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._consecutive_failures += 1
            self._last_failure_time = datetime.now(timezone.utc)

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed → re-open
                self._transition(CircuitState.OPEN)
            elif (
                self._state == CircuitState.CLOSED
                and self._consecutive_failures >= self._failure_threshold
            ):
                self._transition(CircuitState.OPEN)

    def _transition(self, new_state: CircuitState) -> None:
        if self._state == new_state:
            return
        logger.info(
            "Circuit '%s': %s → %s",
            self._name,
            self._state.value,
            new_state.value,
        )
        self._state = new_state
        self._last_state_change = datetime.now(timezone.utc)
        if new_state == CircuitState.OPEN:
            self._opened_at = time.monotonic()
            self._half_open_calls = 0

    def _maybe_half_open(self) -> None:
        if self._state != CircuitState.OPEN:
            return
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= self._recovery_timeout:
            self._transition(CircuitState.HALF_OPEN)

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self._name!r}, "
            f"state={self._state.value}, "
            f"failures={self._consecutive_failures}/{self._failure_threshold})"
        )


# ======================================================================
# Retry Policy
# ======================================================================


@dataclass
class RetryEvent:
    """Record of a single retry attempt."""

    attempt: int
    delay: float
    exception: Exception
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, attempts: int, last_exception: BaseException) -> None:
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"All {attempts} retry attempts exhausted. "
            f"Last error: {last_exception!r}"
        )


class RetryPolicy:
    """Configurable retry with exponential back-off and jitter.

    Parameters
    ----------
    max_attempts : int
        Total number of attempts (including the first call).  ``1`` means
        no retries.
    base_delay : float
        Initial delay in seconds between retries.
    max_delay : float
        Cap on the delay (prevents unbounded exponential growth).
    backoff_factor : float
        Multiplier applied to the delay after each attempt.
    jitter : bool
        If *True*, uniform random jitter of ±50 % is added to each delay.
    retriable_exceptions : tuple[type[BaseException], ...]
        Exception types that trigger a retry.  Non-retriable exceptions
        propagate immediately.
    on_retry : Callable[[RetryEvent], None] | None
        Optional callback invoked before each retry sleep.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retriable_exceptions: tuple[type[BaseException], ...] = (Exception,),
        on_retry: Callable[[RetryEvent], None] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if backoff_factor < 1.0:
            raise ValueError("backoff_factor must be >= 1.0")

        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._backoff_factor = backoff_factor
        self._jitter = jitter
        self._retriable_exceptions = retriable_exceptions
        self._on_retry = on_retry

    def execute(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call *fn* with retries according to this policy.

        Parameters
        ----------
        fn : Callable[..., T]
            The callable to protect.
        *args, **kwargs
            Forwarded to *fn*.

        Returns
        -------
        T
            The return value of *fn* on the first successful attempt.

        Raises
        ------
        RetryExhaustedError
            If all attempts fail.
        Exception
            Any non-retriable exception raised by *fn*.
        """
        last_exc: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except self._retriable_exceptions as exc:
                last_exc = exc
                if attempt == self._max_attempts:
                    break

                delay = self._compute_delay(attempt)
                event = RetryEvent(
                    attempt=attempt,
                    delay=delay,
                    exception=exc if isinstance(exc, Exception) else Exception(str(exc)),
                )
                if self._on_retry:
                    try:
                        self._on_retry(event)
                    except Exception:
                        pass  # callback errors must not break retry
                logger.debug(
                    "Retry attempt %d/%d after %.2fs: %s",
                    attempt,
                    self._max_attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)

        raise RetryExhaustedError(
            self._max_attempts, last_exc  # type: ignore[arg-type]
        )

    def _compute_delay(self, attempt: int) -> float:
        delay = self._base_delay * (self._backoff_factor ** (attempt - 1))
        delay = min(delay, self._max_delay)
        if self._jitter:
            delay = delay * random.uniform(0.5, 1.5)
        return max(0.0, delay)


# ======================================================================
# Bulkhead
# ======================================================================


class BulkheadFullError(Exception):
    """Raised when the bulkhead concurrency limit is reached."""

    def __init__(self, name: str, max_concurrent: int) -> None:
        self.name = name
        self.max_concurrent = max_concurrent
        super().__init__(
            f"Bulkhead '{name}' is full ({max_concurrent} concurrent callers). "
            f"Try again later."
        )


@dataclass
class BulkheadStats:
    """Snapshot of bulkhead statistics."""

    name: str
    max_concurrent: int
    current_executions: int
    available_permits: int
    total_calls: int
    rejected_calls: int


class Bulkhead:
    """Concurrency limiter / resource pool isolator.

    Parameters
    ----------
    name : str
        Identifier for logging.
    max_concurrent : int
        Maximum number of concurrent executions allowed.
    max_wait : float
        Maximum seconds to wait for a permit.  ``0`` means fail immediately.
    """

    def __init__(
        self,
        name: str = "default",
        max_concurrent: int = 10,
        max_wait: float = 0.0,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._name = name
        self._max_concurrent = max_concurrent
        self._max_wait = max_wait
        self._semaphore = threading.Semaphore(max_concurrent)
        self._current = 0
        self._total_calls = 0
        self._rejected_calls = 0
        self._lock = threading.Lock()

    def execute(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute *fn* within the bulkhead's concurrency limit.

        Parameters
        ----------
        fn : Callable[..., T]
            The callable to protect.
        *args, **kwargs
            Forwarded to *fn*.

        Returns
        -------
        T
            Return value of *fn*.

        Raises
        ------
        BulkheadFullError
            If no permit is available within *max_wait* seconds.
        """
        acquired = self._semaphore.acquire(timeout=self._max_wait)
        if not acquired:
            with self._lock:
                self._rejected_calls += 1
            raise BulkheadFullError(self._name, self._max_concurrent)

        with self._lock:
            self._current += 1
            self._total_calls += 1

        try:
            return fn(*args, **kwargs)
        finally:
            with self._lock:
                self._current -= 1
            self._semaphore.release()

    def get_stats(self) -> BulkheadStats:
        """Return a snapshot of bulkhead statistics."""
        with self._lock:
            return BulkheadStats(
                name=self._name,
                max_concurrent=self._max_concurrent,
                current_executions=self._current,
                available_permits=self._max_concurrent - self._current,
                total_calls=self._total_calls,
                rejected_calls=self._rejected_calls,
            )


# ======================================================================
# Graceful Degradation
# ======================================================================


class FallbackChainExhaustedError(Exception):
    """Raised when every fallback in the chain has failed."""

    def __init__(self, errors: list[Exception]) -> None:
        self.errors = errors
        summary = "; ".join(repr(e) for e in errors[-3:])
        super().__init__(
            f"All {len(errors)} fallback(s) exhausted. Latest: {summary}"
        )


class GracefulDegradation:
    """Execute a primary function with cascading fallbacks.

    Parameters
    ----------
    name : str
        Identifier for logging.
    fallbacks : list[Callable[..., Any]]
        Ordered list of fallback callables.  They share the same signature
        as the primary function.
    on_fallback : Callable[[int, Exception], None] | None
        Callback invoked when a fallback is triggered, receiving the
        fallback index (0-based) and the triggering exception.
    """

    def __init__(
        self,
        name: str = "default",
        fallbacks: list[Callable[..., Any]] | None = None,
        on_fallback: Callable[[int, Exception], None] | None = None,
    ) -> None:
        self._name = name
        self._fallbacks: list[Callable[..., Any]] = fallbacks or []
        self._on_fallback = on_fallback

    def execute(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call *fn*, falling back through the chain on failure.

        Parameters
        ----------
        fn : Callable[..., T]
            Primary callable.
        *args, **kwargs
            Forwarded to the callable being attempted.

        Returns
        -------
        T
            Return value of the first successful callable.

        Raises
        ------
        FallbackChainExhaustedError
            If every callable (primary + fallbacks) raises.
        """
        errors: list[Exception] = []
        chain: list[Callable[..., Any]] = [fn] + self._fallbacks

        for idx, callable_ in enumerate(chain):
            try:
                return callable_(*args, **kwargs)
            except Exception as exc:
                errors.append(exc)
                logger.debug(
                    "Fallback '%s' attempt %d/%d failed: %s",
                    self._name,
                    idx + 1,
                    len(chain),
                    exc,
                )
                if self._on_fallback and idx < len(chain) - 1:
                    try:
                        self._on_fallback(idx, exc)
                    except Exception:
                        pass

        raise FallbackChainExhaustedError(errors)

    def add_fallback(self, fn: Callable[..., Any]) -> None:
        """Append a fallback callable to the chain."""
        self._fallbacks.append(fn)


# ======================================================================
# Resilient Executor (composable)
# ======================================================================


@dataclass
class ExecutorStats:
    """Aggregated statistics from the resilient executor's components."""

    circuit: CircuitStats | None = None
    bulkhead: BulkheadStats | None = None
    total_attempts: int = 0
    total_retries: int = 0
    total_successes: int = 0
    total_failures: int = 0


class ResilientExecutor:
    """Composable resilience wrapper combining circuit breaker, retry, and
    bulkhead around a callable.

    Execution order:
    1. Bulkhead (concurrency check)
    2. Circuit breaker (availability check)
    3. Retry policy (transient failure handling)
    4. Actual callable

    Parameters
    ----------
    name : str
        Identifier for logging.
    circuit_breaker : CircuitBreaker | None
        Optional circuit breaker.
    retry_policy : RetryPolicy | None
        Optional retry policy.
    bulkhead : Bulkhead | None
        Optional bulkhead.
    degradation : GracefulDegradation | None
        Optional fallback chain wrapping the entire execution.
    """

    def __init__(
        self,
        name: str = "default",
        circuit_breaker: CircuitBreaker | None = None,
        retry_policy: RetryPolicy | None = None,
        bulkhead: Bulkhead | None = None,
        degradation: GracefulDegradation | None = None,
    ) -> None:
        self._name = name
        self._circuit = circuit_breaker
        self._retry = retry_policy
        self._bulkhead = bulkhead
        self._degradation = degradation
        self._total_attempts = 0
        self._total_retries = 0
        self._total_successes = 0
        self._total_failures = 0
        self._lock = threading.Lock()

    def execute(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute *fn* through all configured resilience layers.

        Parameters
        ----------
        fn : Callable[..., T]
            The callable to protect.
        *args, **kwargs
            Forwarded to *fn*.

        Returns
        -------
        T
            Return value on success.

        Raises
        ------
        Exception
            Propagates from the innermost failing layer.
        """
        with self._lock:
            self._total_attempts += 1

        def _inner() -> T:
            if self._retry:
                return self._retry.execute(self._invoke_through_breaker, fn, *args, **kwargs)
            return self._invoke_through_breaker(fn, *args, **kwargs)

        try:
            if self._bulkhead:
                result = self._bulkhead.execute(_inner)
            elif self._degradation:
                result = self._degradation.execute(_inner)
            else:
                result = _inner()

            with self._lock:
                self._total_successes += 1
            return result

        except Exception:
            with self._lock:
                self._total_failures += 1
            raise

    def _invoke_through_breaker(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        if self._circuit:
            return self._circuit.call(fn, *args, **kwargs)
        return fn(*args, **kwargs)

    def get_stats(self) -> ExecutorStats:
        """Return aggregated statistics from all layers."""
        with self._lock:
            return ExecutorStats(
                circuit=self._circuit.get_stats() if self._circuit else None,
                bulkhead=self._bulkhead.get_stats() if self._bulkhead else None,
                total_attempts=self._total_attempts,
                total_retries=self._total_retries,
                total_successes=self._total_successes,
                total_failures=self._total_failures,
            )

    def __repr__(self) -> str:
        parts = [f"name={self._name!r}"]
        if self._circuit:
            parts.append(f"circuit={self._circuit._name!r}")
        if self._retry:
            parts.append(f"retry(max={self._retry._max_attempts})")
        if self._bulkhead:
            parts.append(f"bulkhead={self._bulkhead._name!r}")
        return f"ResilientExecutor({', '.join(parts)})"
