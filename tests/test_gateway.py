import asyncio

import pytest
from conftest import ScriptedProvider, completion

from llm_gateway.errors import BadRequest, BudgetExhausted, ProviderError
from llm_gateway.ratelimit import ModelLimits
from llm_gateway.schemas import ChatCompletionRequest


def req(model="fake/m", temperature=0.0, **kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        temperature=temperature,
        **kwargs,
    )


def test_success_writes_a_full_metrics_row(make_gateway):
    fake = ScriptedProvider(completion(prompt_tokens=1000, completion_tokens=500))
    gateway, metrics = make_gateway({"fake": fake})

    result = asyncio.run(gateway.complete(req(), project="sql-agent", trace_id="t-1"))

    assert result.cache == "MISS"
    assert result.body["model"] == "fake/m"  # gateway name, not the provider's
    assert fake.calls[0][0] == "m"
    assert fake.calls[0][1]["model"] == "m"
    [row] = metrics.query()
    assert row["status"] == 200
    assert row["project"] == "sql-agent"
    assert row["trace_id"] == "t-1"
    assert row["provider"] == "fake"
    assert row["attempts"] == 1
    assert row["cache_hit"] == 0
    assert (row["prompt_tokens"], row["completion_tokens"]) == (1000, 500)
    assert row["tokens_per_sec"] == pytest.approx(500 / 0.1)  # from usage.completion_time
    assert row["est_cost_usd"] == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1e6)
    assert row["latency_ms"] is not None
    assert row["total_ms"] >= row["latency_ms"]


def test_prompt_cache_hits_are_recorded_and_priced_at_the_cached_rate(make_gateway):
    fake = ScriptedProvider(
        completion(prompt_tokens=1000, completion_tokens=500, cached_tokens=600)
    )
    gateway, metrics = make_gateway({"fake": fake})

    asyncio.run(gateway.complete(req()))

    [row] = metrics.query()
    assert row["cached_tokens"] == 600
    # 400 uncached at $1/M + 600 cached at $0.5/M + 500 output at $2/M
    assert row["est_cost_usd"] == pytest.approx((400 * 1.0 + 600 * 0.5 + 500 * 2.0) / 1e6)


def test_prompt_cache_hits_do_not_use_up_the_token_budget(make_gateway):
    # Each call: 1500 prompt tokens, 1400 of them from Groq's prompt cache, 50 output.
    fake = ScriptedProvider(
        completion(prompt_tokens=1500, completion_tokens=50, cached_tokens=1400)
    )
    gateway, _ = make_gateway({"fake": fake}, limits={"fake/*": ModelLimits(tpd=2000)})

    # 150 counted tokens per call fit three times; 1550 per call would fail on the 2nd.
    for _ in range(3):
        asyncio.run(gateway.complete(req(temperature=0.5)))
    assert len(fake.calls) == 3


def test_failure_writes_a_row_and_reraises(make_gateway):
    fake = ScriptedProvider(ProviderError("bad tool schema", status_code=400))
    gateway, metrics = make_gateway({"fake": fake})

    with pytest.raises(ProviderError):
        asyncio.run(gateway.complete(req()))

    [row] = metrics.query()
    assert row["status"] == 400
    assert row["error"] == "bad tool schema"
    assert row["prompt_tokens"] is None


def test_unknown_provider_is_recorded_too(make_gateway):
    gateway, metrics = make_gateway()
    with pytest.raises(BadRequest):
        asyncio.run(gateway.complete(req(model="nope/x")))
    assert metrics.query()[0]["status"] == 400


def test_temperature_zero_is_served_from_cache_the_second_time(make_gateway):
    fake = ScriptedProvider(completion())
    gateway, metrics = make_gateway({"fake": fake})

    first = asyncio.run(gateway.complete(req()))
    second = asyncio.run(gateway.complete(req()))

    assert (first.cache, second.cache) == ("MISS", "HIT")
    assert second.body == first.body
    assert len(fake.calls) == 1
    hit = metrics.query()[1]
    assert hit["cache_hit"] == 1
    assert hit["attempts"] == 0
    assert hit["est_cost_usd"] == 0.0


def test_nonzero_temperature_bypasses_the_cache(make_gateway):
    fake = ScriptedProvider(completion())
    gateway, _ = make_gateway({"fake": fake})

    results = [asyncio.run(gateway.complete(req(temperature=0.7))) for _ in range(2)]

    assert [r.cache for r in results] == ["BYPASS", "BYPASS"]
    assert len(fake.calls) == 2


def test_failed_calls_are_not_cached(make_gateway):
    fake = ScriptedProvider(ProviderError("bad", status_code=400), completion())
    gateway, _ = make_gateway({"fake": fake})

    with pytest.raises(ProviderError):
        asyncio.run(gateway.complete(req()))
    assert asyncio.run(gateway.complete(req())).cache == "MISS"


def test_retry_on_429_then_success_is_one_row_with_two_attempts(make_gateway):
    fake = ScriptedProvider(
        ProviderError("rate limited", status_code=429, retry_after=1.5), completion()
    )
    gateway, metrics = make_gateway({"fake": fake})

    result = asyncio.run(gateway.complete(req()))

    assert result.attempts == 2
    [row] = metrics.query()
    assert row["status"] == 200
    assert row["attempts"] == 2


def test_daily_budget_refusal_reaches_the_caller_unretried(make_gateway):
    fake = ScriptedProvider(completion())
    gateway, metrics = make_gateway({"fake": fake}, limits={"fake/*": ModelLimits(rpd=1)})

    asyncio.run(gateway.complete(req(temperature=0.5)))
    with pytest.raises(BudgetExhausted):
        asyncio.run(gateway.complete(req(temperature=0.5)))

    assert len(fake.calls) == 1
    refused = metrics.query()[1]
    assert refused["status"] == 429
    assert refused["attempts"] == 0  # never reached the provider
