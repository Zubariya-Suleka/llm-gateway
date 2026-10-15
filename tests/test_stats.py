from datetime import timedelta

import pytest

from llm_gateway.stats import format_table, parse_since, percentile, summarize


@pytest.mark.parametrize(
    ("text", "expected"),
    [("90s", timedelta(seconds=90)), ("30m", timedelta(minutes=30)), ("7d", timedelta(days=7))],
)
def test_parse_since(text, expected):
    assert parse_since(text) == expected


@pytest.mark.parametrize("text", ["7", "d7", "1w", ""])
def test_parse_since_rejects_garbage(text):
    with pytest.raises(ValueError):
        parse_since(text)


def test_percentile_interpolates():
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(values, 50) == pytest.approx(55)
    assert percentile(values, 95) == pytest.approx(95.5)
    assert percentile([42], 95) == 42
    assert percentile([], 50) is None


def row(model="groq/a", status=200, cache_hit=0, latency_ms=100.0, tps=500.0, cost=0.001):
    return {
        "model": model,
        "status": status,
        "cache_hit": cache_hit,
        "latency_ms": latency_ms,
        "tokens_per_sec": tps,
        "est_cost_usd": cost,
    }


def test_summarize_per_model():
    rows = [
        row(latency_ms=100),
        row(latency_ms=300),
        row(cache_hit=1, latency_ms=1, tps=None, cost=0.0),
        row(status=429, latency_ms=None, tps=None, cost=None),
        row(model="groq/b", cost=None),
    ]

    a, b = summarize(rows)

    assert a.model == "groq/a"
    assert a.requests == 4
    assert a.cache_hit_pct == 25
    assert a.error_pct == 25
    assert a.p50_ms == 100  # of [1, 100, 300]
    assert a.median_tokens_per_sec == 500  # cache hits excluded
    assert a.cost_per_1k_usd == pytest.approx(1000 * (0.001 + 0.001 + 0.0) / 3)
    assert b.cost_per_1k_usd is None


def test_markdown_table_has_header_and_one_line_per_model():
    table = format_table(summarize([row(), row(model="groq/b")]), markdown=True)
    lines = table.splitlines()
    assert lines[0].startswith("| model | requests")
    assert len(lines) == 4
    assert "| groq/b |" in lines[3]


def test_text_table_renders_missing_values_as_dash():
    line = format_table(summarize([row(cost=None)])).splitlines()[2]
    assert line.split()[6] == "-"  # the $/1K column
