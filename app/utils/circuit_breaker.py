# app/utils/circuit_breaker.py
# Reusable async-safe circuit breaker with exponential backoff retries.

import time
import asyncio
from typing import Callable, Awaitable, Optional

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_time_sec: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_time_sec = recovery_time_sec
        self._failures = 0
        self._state = "closed"  # closed|open|half_open
        self._opened_at: Optional[float] = None

    def can_attempt(self) -> bool:
        if self._state == "closed":
            return True
        if self._state == "open":
            if (time.time() - (self._opened_at or 0)) >= self.recovery_time_sec:
                self._state = "half_open"
                return True
            return False
        if self._state == "half_open":
            return True
        return True

    def on_success(self):
        self._failures = 0
        self._state = "closed"
        self._opened_at = None

    def on_failure(self):
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._state = "open"
            self._opened_at = time.time()

async def with_retries_async(
        op: Callable[[], Awaitable],
        is_retryable: Callable[[Exception], bool],
        breaker: CircuitBreaker,
        max_attempts: int = 3,
        base_backoff: float = 0.5,
        jitter: float = 0.1
):
    attempt = 0
    last_err = None
    while attempt < max_attempts:
        if not breaker.can_attempt():
            raise RuntimeError("Circuit open; skipping attempt")
        try:
            result = await op()
            breaker.on_success()
            return result
        except Exception as e:
            last_err = e
            if not is_retryable(e):
                breaker.on_failure()
                raise
            breaker.on_failure()
            sleep_s = base_backoff * (2 ** attempt) + min(jitter, 0.05)
            await asyncio.sleep(sleep_s)
            attempt += 1
    raise last_err if last_err else RuntimeError("Operation failed with no exception")
