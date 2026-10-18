"""Small pure helpers: token estimates, pricing, mock provider, metrics queries."""

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from llm_gateway.errors import ProviderError
from llm_gateway.metrics import MetricsStore, RequestRecord, utc_iso
from llm_gateway.pricing import Price, Pricing
from llm_gateway.providers.mock import MockProvider
from llm_gateway.tokens import (
    COMPLETION_RESERVE,
    completion_budget,
    count_tokens,
    estimate_prompt_tokens,
)


def test_token_estimate_grows_with_messages_and_tools():
    one = [{"role": "user", "content": "How many orders were delivered late in 2018?"}]
    two = one + [{"role": "assistant", "content": "Let me check the orders table first."}]
    tools = [{"type": "function", "function": {"name": "run_sql", "description": "Run SQL"}}]

    assert count_tokens("") == 0
    assert 0 < estimate_prompt_tokens(one) < estimate_prompt_tokens(two)
    assert estimate_prompt_tokens(one, tools) > estimate_prompt_tokens(one)


def test_token_estimate_handles_content_parts_and_tool_calls():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hello there"}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {}}]},
    ]
    assert estimate_prompt_tokens(messages) > 8


def test_completion_budget():
    assert completion_budget(None) == COMPLETION_RESERVE
    assert completion_budget(50) == 50
    assert completion_budget(100_000) == COMPLETION_RESERVE


def test_pricing_exact_wildcard_and_unknown():
    pricing = Pricing({"groq/a": Price(0.1, 0.5), "mock/*": Price(0, 0)})
    assert pricing.cost("groq/a", 1_000_000, 2_000_000) == pytest.approx(1.1)
    assert pricing.cost("mock/anything", 10, 10) == 0
    assert pricing.cost("groq/unknown", 10, 10) is None


def test_cached_prompt_tokens_use_the_cached_price_when_known():
    pricing = Pricing({"groq/a": Price(0.1, 0.5, 0.05), "groq/b": Price(0.1, 0.5)})
    # 600K uncached at $0.10/M + 400K cached at $0.05/M
    assert pricing.cost("groq/a", 1_000_000, 0, cached_tokens=400_000) == pytest.approx(0.08)
    # No cached price: charged as normal input, never guessed cheaper.
    assert pricing.cost("groq/b", 1_000_000, 0, cached_tokens=400_000) == pytest.approx(0.1)


def test_old_metrics_db_gains_new_columns(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE requests (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,"
        " project TEXT, trace_id TEXT, provider TEXT, model TEXT NOT NULL,"
        " status INTEGER NOT NULL, cache_hit INTEGER NOT NULL, attempts INTEGER NOT NULL,"
        " prompt_tokens INTEGER, completion_tokens INTEGER, latency_ms REAL, total_ms REAL,"
        " completion_time_s REAL, tokens_per_sec REAL, est_cost_usd REAL, error TEXT)"
    )
    old.execute(
        "INSERT INTO requests (ts, model, status, cache_hit, attempts)"
        " VALUES ('2026-10-18T00:00:00.000+00:00', 'groq/a', 200, 0, 1)"
    )
    old.commit()
    old.close()

    store = MetricsStore(path)
    store.record(RequestRecord(model="groq/a", status=200, prompt_tokens=10, cached_tokens=4))

    assert [r["cached_tokens"] for r in store.query()] == [None, 4]
    store.close()


def test_mock_fail_models_raise_with_that_status():
    with pytest.raises(ProviderError) as info:
        asyncio.run(MockProvider().complete("fail-503", {"messages": []}))
    assert info.value.status_code == 503


def test_mock_sleeps_for_sampled_latency():
    sleeps: list[float] = []

    async def sleep(s: float) -> None:
        sleeps.append(s)

    mock = MockProvider(latency_ms=200, jitter_ms=0, sleep=sleep)
    body = asyncio.run(mock.complete("x", {"messages": [{"role": "user", "content": "hey"}]}))
    assert sleeps == [pytest.approx(0.2)]
    assert body["usage"]["completion_time"] == pytest.approx(0.2)
    assert body["choices"][0]["message"]["content"] == "[mock:x] hey"


def test_metrics_query_filters_and_usage_since(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    old = utc_iso(datetime.now(UTC) - timedelta(days=2))
    store.record(RequestRecord(model="groq/a", ts=old, status=200, attempts=1, prompt_tokens=5))
    store.record(
        RequestRecord(
            model="groq/a",
            project="p",
            status=200,
            attempts=2,
            prompt_tokens=10,
            completion_tokens=5,
        )
    )
    store.record(RequestRecord(model="groq/a", status=200, cache_hit=True, prompt_tokens=10))
    store.record(RequestRecord(model="groq/b", project="p", status=500))

    day_ago = datetime.now(UTC) - timedelta(days=1)
    assert len(store.query()) == 4
    assert len(store.query(since=day_ago)) == 3
    assert len(store.query(project="p")) == 2
    assert len(store.query(model="groq/b")) == 1
    # Old row and cache hit excluded: 2 attempts, 15 tokens.
    assert store.usage_since("groq/a", day_ago) == (2, 15)
