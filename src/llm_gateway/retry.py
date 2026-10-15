"""Retry provider calls on 429 and 5xx.

- If the provider says how long to wait (retry-after), wait that long plus a little jitter.
- Otherwise exponential backoff with full jitter: sleep a random time in [0, base * 2^n].
  The randomness spreads out clients that failed together, so they don't all retry at the
  same instant and fail together again.
- Other 4xx errors are the caller's fault; retrying cannot fix them.
- A retry-after longer than `max_retry_after_s` (e.g. a daily quota) fails fast instead of
  holding an HTTP request open for an hour.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from llm_gateway.errors import ProviderError

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base_delay_s: float = 0.5
    max_delay_s: float = 20.0
    max_retry_after_s: float = 60.0
    sleep: Callable[[float], Awaitable[None]] = field(default=asyncio.sleep, repr=False)
    rand: Callable[[], float] = field(default=random.random, repr=False)

    def backoff(self, attempt: int) -> float:
        cap = min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1))
        return cap * self.rand()

    def delay_for(self, error: ProviderError, attempt: int) -> float | None:
        """Seconds to wait before the next attempt, or None to give up."""
        if not error.retryable or attempt >= self.max_attempts:
            return None
        if error.retry_after is None:
            return self.backoff(attempt)
        if error.retry_after > self.max_retry_after_s:
            return None
        return error.retry_after * (1 + 0.1 * self.rand())

    async def run(self, call: Callable[[int], Awaitable[T]]) -> T:
        """Run `call(attempt)` with attempt = 1, 2, ... until it succeeds or we give up."""
        attempt = 1
        while True:
            try:
                return await call(attempt)
            except ProviderError as error:
                delay = self.delay_for(error, attempt)
                if delay is None:
                    raise
                log.warning(
                    "retrying provider call",
                    extra={
                        "attempt": attempt,
                        "status": error.status_code,
                        "delay_s": round(delay, 3),
                        "error": error.message,
                    },
                )
                await self.sleep(delay)
                attempt += 1
