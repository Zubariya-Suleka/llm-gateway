import asyncio

import pytest
from conftest import FakeTime

from llm_gateway.errors import BudgetExhausted
from llm_gateway.ratelimit import (
    ModelLimiter,
    ModelLimits,
    RateLimiter,
    SlidingWindow,
    TokenBucket,
    load_limits,
)


def bucket(t: FakeTime, capacity: float) -> TokenBucket:
    return TokenBucket(capacity, 60.0, clock=t.clock, sleep=t.sleep)


def test_bucket_allows_a_full_burst_then_waits_for_refill():
    t = FakeTime()
    b = bucket(t, 30)  # 30/min = one every 2 s

    async def run():
        for _ in range(30):
            assert await b.acquire(1) == 0
        return await b.acquire(1)

    assert asyncio.run(run()) == pytest.approx(2.0)


def test_bucket_refills_over_time():
    t = FakeTime()
    b = bucket(t, 60)
    asyncio.run(b.acquire(60))
    t.now += 30
    assert b.level == pytest.approx(30)
    t.now += 1000
    assert b.level == pytest.approx(60)  # never above capacity


def test_oversized_request_takes_a_full_bucket_and_leaves_debt():
    t = FakeTime()
    b = bucket(t, 100)

    async def run():
        first = await b.acquire(250)  # bigger than the bucket: allowed once it is full
        second = await b.acquire(1)  # must wait for the debt (-150) to refill to +1
        return first, second

    first, second = asyncio.run(run())
    assert first == 0
    assert second == pytest.approx(151 / (100 / 60))


def test_adjust_refunds_overestimates():
    t = FakeTime()
    b = bucket(t, 1000)
    asyncio.run(b.acquire(800))
    b.adjust(-500)  # reserved 800, really used 300
    assert b.level == pytest.approx(700)


def test_sliding_window_expires_old_usage():
    t = FakeTime()
    w = SlidingWindow(10, window_s=100, clock=t.clock)
    w.add(6)
    t.now += 50
    w.add(4)
    assert w.wait_for(1) == pytest.approx(50)  # the 6 expires 50 s from now
    t.now += 51
    assert w.used == 4
    assert w.wait_for(6) == 0
    assert w.wait_for(11) == float("inf")


def model_limiter(t: FakeTime, limits: ModelLimits) -> ModelLimiter:
    return ModelLimiter("groq/m", limits, clock=t.clock, wall_clock=t.clock, sleep=t.sleep)


def test_daily_request_budget_refuses_immediately_with_retry_after():
    t = FakeTime()
    limiter = model_limiter(t, ModelLimits(rpd=2))

    async def run():
        await limiter.acquire(10)
        await limiter.acquire(10)
        await limiter.acquire(10)

    with pytest.raises(BudgetExhausted, match="daily requests budget") as info:
        asyncio.run(run())
    assert info.value.retry_after == pytest.approx(24 * 3600)
    assert t.sleeps == []  # refused, not queued


def test_daily_token_budget_uses_real_usage_after_reconcile():
    t = FakeTime()
    limiter = model_limiter(t, ModelLimits(tpd=1000))

    async def run():
        await limiter.acquire(900)  # reserve 900 ...
        limiter.reconcile(900, 100)  # ... really used 100
        await limiter.acquire(800)  # fits: 100 + 800 <= 1000

    asyncio.run(run())


def test_tokens_per_minute_waits():
    t = FakeTime()
    limiter = model_limiter(t, ModelLimits(tpm=6000))

    async def run():
        await limiter.acquire(6000)
        return await limiter.acquire(3000)

    assert asyncio.run(run()) == pytest.approx(30.0)  # 3000 tokens at 100 tokens/s


def test_seeding_from_metrics_counts_earlier_usage():
    t = FakeTime()
    limiter = RateLimiter(
        {"groq/m": ModelLimits(rpd=5)},
        usage_lookup=lambda model: (5, 0),
        clock=t.clock,
        wall_clock=t.clock,
        sleep=t.sleep,
    )
    with pytest.raises(BudgetExhausted):
        asyncio.run(limiter.acquire("groq/m", 1))


def test_wildcard_limits_apply_and_unlisted_providers_are_unlimited():
    t = FakeTime()
    limiter = RateLimiter(
        {"groq/*": ModelLimits(rpm=1)}, clock=t.clock, wall_clock=t.clock, sleep=t.sleep
    )

    async def run():
        await limiter.acquire("groq/any-model", 1)
        waited = await limiter.acquire("groq/any-model", 1)
        unlimited = [await limiter.acquire("mock/x", 10**9) for _ in range(5)]
        return waited, unlimited

    waited, unlimited = asyncio.run(run())
    assert waited == pytest.approx(60.0)
    assert unlimited == [0.0] * 5


def test_load_limits_from_yaml(tmp_path):
    path = tmp_path / "limits.yaml"
    path.write_text("groq/a: {rpm: 30, tpm: 8000}\ngroq/*: {rpm: 10}\n", encoding="utf-8")
    assert load_limits(path) == {
        "groq/a": ModelLimits(rpm=30, tpm=8000),
        "groq/*": ModelLimits(rpm=10),
    }
    assert load_limits(tmp_path / "missing.yaml") == {}
