"""Client-side rate limiting, so the gateway slows down before Groq has to say 429.

Per model (limits are per model on Groq, so each model gets its own budget):
- requests/min, tokens/min and output tokens/min: token buckets. When a bucket is empty,
  the call waits. (Groq enforces output tokens/min on some models, e.g. 1000 on
  qwen3.8-27b, without listing it on its rate-limits page.)
- requests/day and tokens/day: 24-hour sliding windows. When a window is full, the call is
  refused right away with a clear error: waiting hours inside an HTTP request helps nobody.

Token costs are not known until the reply comes back, so each call reserves an estimate
(prompt estimate + completion budget) and `reconcile` corrects the books with the real
usage afterwards.

Limits live in this process's memory: two gateway processes would each think they have the
whole budget. One process is the deployment model here; daily windows are seeded from the
metrics table on first use so a restart doesn't forget today's usage.
"""

import asyncio
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from llm_gateway.catalog import load_yaml, lookup
from llm_gateway.errors import BudgetExhausted

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]

DAY_S = 24 * 3600.0
_EPSILON = 1e-6


@dataclass(frozen=True)
class ModelLimits:
    rpm: int | None = None
    tpm: int | None = None
    otpm: int | None = None  # output (completion) tokens per minute
    rpd: int | None = None
    tpd: int | None = None


@dataclass(frozen=True)
class Usage:
    """Tokens of one call: reserved up front (estimates) or reported afterwards (actual)."""

    prompt: int
    completion: int

    @property
    def total(self) -> int:
        return self.prompt + self.completion


def load_limits(path: Path) -> dict[str, ModelLimits]:
    return {model: ModelLimits(**(values or {})) for model, values in load_yaml(path).items()}


class TokenBucket:
    """Holds up to `capacity` units and refills at `capacity / period_s` units per second.

    The capacity sets the largest burst; the refill rate sets the long-run average. A request
    bigger than the whole bucket waits for a full bucket, takes it all and leaves the bucket
    in debt, so oversized requests are slowed down instead of blocked forever.
    """

    def __init__(
        self,
        capacity: float,
        period_s: float = 60.0,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.capacity = float(capacity)
        self.rate = self.capacity / period_s
        self._level = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._updated = clock()
        # Waiters queue on the lock, so they are served in arrival order.
        self._lock = asyncio.Lock()

    @property
    def level(self) -> float:
        self._refill()
        return self._level

    def _refill(self) -> None:
        now = self._clock()
        self._level = min(self.capacity, self._level + (now - self._updated) * self.rate)
        self._updated = now

    async def acquire(self, amount: float) -> float:
        """Take `amount` units, waiting until they are available. Returns seconds waited."""
        need = min(amount, self.capacity)
        waited = 0.0
        async with self._lock:
            while True:
                self._refill()
                # The tolerance stops float rounding (level = 0.9999999 for need = 1) from
                # turning into an endless series of ever-tinier sleeps.
                if self._level >= need - _EPSILON:
                    self._level -= amount
                    return waited
                delay = (need - self._level) / self.rate
                await self._sleep(delay)
                waited += delay

    def adjust(self, delta: float) -> None:
        """Charge `delta` more units (negative refunds), e.g. once real usage is known."""
        self._refill()
        self._level = min(self.capacity, self._level - delta)


class SlidingWindow:
    """Sum of amounts added in the last `window_s` seconds, against a fixed limit."""

    def __init__(self, limit: float, window_s: float = DAY_S, *, clock: Clock = time.time):
        self.limit = float(limit)
        self.window_s = window_s
        self._clock = clock
        self._entries: deque[tuple[float, float]] = deque()
        self._total = 0.0

    def _prune(self) -> None:
        cutoff = self._clock() - self.window_s
        while self._entries and self._entries[0][0] <= cutoff:
            self._total -= self._entries.popleft()[1]

    @property
    def used(self) -> float:
        self._prune()
        return self._total

    def add(self, amount: float) -> None:
        self._prune()
        self._entries.append((self._clock(), amount))
        self._total += amount

    def wait_for(self, amount: float) -> float:
        """0 if `amount` fits now, else seconds until enough old usage expires (inf: never)."""
        self._prune()
        excess = self._total + amount - self.limit
        if excess <= 0:
            return 0.0
        now = self._clock()
        freed = 0.0
        for ts, value in self._entries:
            freed += value
            if freed >= excess:
                return ts + self.window_s - now
        return math.inf


class ModelLimiter:
    def __init__(
        self,
        model: str,
        limits: ModelLimits,
        *,
        clock: Clock = time.monotonic,
        wall_clock: Clock = time.time,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.model = model

        def bucket(capacity: int | None) -> TokenBucket | None:
            return TokenBucket(capacity, clock=clock, sleep=sleep) if capacity else None

        def window(limit: int | None) -> SlidingWindow | None:
            return SlidingWindow(limit, clock=wall_clock) if limit else None

        self.rpm, self.tpm, self.otpm = bucket(limits.rpm), bucket(limits.tpm), bucket(limits.otpm)
        self.rpd, self.tpd = window(limits.rpd), window(limits.tpd)

    def seed(self, requests: int, tokens: int) -> None:
        """Count usage from before this process started (read from the metrics table)."""
        if self.rpd and requests:
            self.rpd.add(requests)
        if self.tpd and tokens:
            self.tpd.add(tokens)

    def _check_daily(self, window: SlidingWindow | None, amount: float, what: str) -> None:
        if window is None:
            return
        wait = window.wait_for(amount)
        if wait <= 0:
            return
        if math.isinf(wait):
            raise BudgetExhausted(
                f"{self.model}: this request ({amount:.0f} {what}) is larger than the whole "
                f"daily {what} budget ({window.limit:.0f})"
            )
        raise BudgetExhausted(
            f"{self.model}: daily {what} budget ({window.limit:.0f}) is used up; "
            f"room again in ~{math.ceil(wait / 60)} min",
            retry_after=wait,
        )

    async def acquire(self, reserved: Usage) -> float:
        # Daily budgets first: refuse before queueing behind the per-minute buckets.
        self._check_daily(self.rpd, 1, "requests")
        self._check_daily(self.tpd, reserved.total, "tokens")
        waited = 0.0
        for bucket, amount in (
            (self.rpm, 1),
            (self.tpm, reserved.total),
            (self.otpm, reserved.completion),
        ):
            if bucket:
                waited += await bucket.acquire(amount)
        if self.rpd:
            self.rpd.add(1)
        if self.tpd:
            self.tpd.add(reserved.total)
        return waited

    def reconcile(self, reserved: Usage, actual: Usage) -> None:
        if self.tpm:
            self.tpm.adjust(actual.total - reserved.total)
        if self.otpm:
            self.otpm.adjust(actual.completion - reserved.completion)
        if self.tpd:
            self.tpd.add(actual.total - reserved.total)


class RateLimiter:
    """Routes each call to its model's limiter; models with no limits are not limited."""

    def __init__(
        self,
        limits: dict[str, ModelLimits],
        *,
        usage_lookup: Callable[[str], tuple[int, int]] | None = None,
        clock: Clock = time.monotonic,
        wall_clock: Clock = time.time,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._limits = limits
        self._usage_lookup = usage_lookup
        self._clock, self._wall_clock, self._sleep = clock, wall_clock, sleep
        self._limiters: dict[str, ModelLimiter | None] = {}

    def _for(self, model: str) -> ModelLimiter | None:
        if model not in self._limiters:
            limits = lookup(self._limits, model)
            limiter = None
            if limits is not None:
                limiter = ModelLimiter(
                    model,
                    limits,
                    clock=self._clock,
                    wall_clock=self._wall_clock,
                    sleep=self._sleep,
                )
                if self._usage_lookup is not None:
                    limiter.seed(*self._usage_lookup(model))
            self._limiters[model] = limiter
        return self._limiters[model]

    async def acquire(self, model: str, reserved: Usage) -> float:
        limiter = self._for(model)
        return await limiter.acquire(reserved) if limiter else 0.0

    def reconcile(self, model: str, reserved: Usage, actual: Usage) -> None:
        limiter = self._for(model)
        if limiter:
            limiter.reconcile(reserved, actual)
