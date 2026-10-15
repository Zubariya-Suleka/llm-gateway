import asyncio

import pytest

from llm_gateway.errors import BudgetExhausted, ProviderError
from llm_gateway.retry import RetryPolicy


class Calls:
    """call(attempt) that fails with the given errors first, then returns "ok"."""

    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.attempts: list[int] = []

    async def __call__(self, attempt: int) -> str:
        self.attempts.append(attempt)
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def policy(sleeps: list[float], rand: float = 0.0, **kwargs) -> RetryPolicy:
    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return RetryPolicy(sleep=sleep, rand=lambda: rand, **kwargs)


def test_success_needs_no_retry():
    sleeps: list[float] = []
    calls = Calls()
    assert asyncio.run(policy(sleeps).run(calls)) == "ok"
    assert calls.attempts == [1]
    assert sleeps == []


def test_429_honours_retry_after():
    sleeps: list[float] = []
    calls = Calls(ProviderError("slow down", status_code=429, retry_after=3.0))

    assert asyncio.run(policy(sleeps).run(calls)) == "ok"
    assert calls.attempts == [1, 2]
    assert sleeps == [3.0]


def test_retry_after_gets_at_most_10_percent_jitter():
    sleeps: list[float] = []
    calls = Calls(ProviderError("slow down", status_code=429, retry_after=2.0))
    asyncio.run(policy(sleeps, rand=1.0).run(calls))
    assert sleeps == [pytest.approx(2.2)]


def test_5xx_without_retry_after_backs_off_exponentially():
    sleeps: list[float] = []
    errors = [ProviderError("boom", status_code=503) for _ in range(3)]
    # rand=1.0 picks the top of each jitter range, exposing the exponential caps.
    asyncio.run(policy(sleeps, rand=1.0, base_delay_s=0.5).run(Calls(*errors)))
    assert sleeps == [0.5, 1.0, 2.0]


def test_backoff_is_capped():
    p = RetryPolicy(base_delay_s=1.0, max_delay_s=5.0, rand=lambda: 1.0)
    assert p.backoff(10) == 5.0


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_other_4xx_is_never_retried(status):
    sleeps: list[float] = []
    calls = Calls(ProviderError("bad", status_code=status))
    with pytest.raises(ProviderError):
        asyncio.run(policy(sleeps).run(calls))
    assert calls.attempts == [1]


def test_gives_up_after_max_attempts():
    sleeps: list[float] = []
    calls = Calls(*[ProviderError("boom", status_code=500) for _ in range(10)])
    with pytest.raises(ProviderError):
        asyncio.run(policy(sleeps, max_attempts=5).run(calls))
    assert calls.attempts == [1, 2, 3, 4, 5]
    assert len(sleeps) == 4


def test_huge_retry_after_fails_fast():
    sleeps: list[float] = []
    calls = Calls(ProviderError("daily quota", status_code=429, retry_after=3600))
    with pytest.raises(ProviderError):
        asyncio.run(policy(sleeps, max_retry_after_s=60).run(calls))
    assert sleeps == []


def test_local_budget_refusal_is_not_retried():
    sleeps: list[float] = []
    calls = Calls(BudgetExhausted("daily budget used up"))
    with pytest.raises(BudgetExhausted):
        asyncio.run(policy(sleeps).run(calls))
    assert calls.attempts == [1]
