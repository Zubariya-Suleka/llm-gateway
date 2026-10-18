"""Per-model summary of the metrics table, for `gateway stats` and the benchmark."""

import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


def parse_since(text: str) -> timedelta:
    """'90s', '30m', '12h', '7d' -> timedelta."""
    match = re.fullmatch(r"(\d+)([smhd])", text.strip())
    if not match:
        raise ValueError(f"expected a duration like 30m, 12h or 7d, got {text!r}")
    return timedelta(**{_UNITS[match[2]]: int(match[1])})


def percentile(values: Sequence[float], q: float) -> float | None:
    """q-th percentile (0-100) with linear interpolation between ranks."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@dataclass
class ModelStats:
    model: str
    requests: int
    cache_hit_pct: float
    p50_ms: float | None
    p95_ms: float | None
    median_tokens_per_sec: float | None
    # Share of prompt tokens the provider served from its own prompt cache (not our cache).
    prompt_cached_pct: float | None
    cost_per_1k_usd: float | None
    error_pct: float


def summarize(rows: Iterable[Mapping[str, Any]]) -> list[ModelStats]:
    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    result = []
    for model in sorted(by_model):
        group = by_model[model]
        ok = [r for r in group if r["status"] == 200]
        latencies = [r["latency_ms"] for r in ok if r["latency_ms"] is not None]
        # Speed only means something for calls that actually generated tokens.
        speeds = [r["tokens_per_sec"] for r in ok if not r["cache_hit"] and r["tokens_per_sec"]]
        costs = [r["est_cost_usd"] for r in ok if r["est_cost_usd"] is not None]
        called = [r for r in ok if not r["cache_hit"]]
        prompt_total = sum(r.get("prompt_tokens") or 0 for r in called)
        cached_total = sum(r.get("cached_tokens") or 0 for r in called)
        result.append(
            ModelStats(
                model=model,
                requests=len(group),
                cache_hit_pct=100 * sum(bool(r["cache_hit"]) for r in group) / len(group),
                p50_ms=percentile(latencies, 50),
                p95_ms=percentile(latencies, 95),
                median_tokens_per_sec=statistics.median(speeds) if speeds else None,
                prompt_cached_pct=100 * cached_total / prompt_total if prompt_total else None,
                cost_per_1k_usd=1000 * statistics.fmean(costs) if costs else None,
                error_pct=100 * (len(group) - len(ok)) / len(group),
            )
        )
    return result


_HEADERS = [
    "model",
    "requests",
    "cache hit %",
    "p50 ms",
    "p95 ms",
    "tokens/s",
    "prompt cached %",
    "$/1K req",
    "error %",
]


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def format_table(stats: Sequence[ModelStats], *, markdown: bool = False) -> str:
    rows = [
        [
            s.model,
            str(s.requests),
            _fmt(s.cache_hit_pct, ".0f"),
            _fmt(s.p50_ms, ".0f"),
            _fmt(s.p95_ms, ".0f"),
            _fmt(s.median_tokens_per_sec, ".0f"),
            _fmt(s.prompt_cached_pct, ".0f"),
            _fmt(s.cost_per_1k_usd, ".4f"),
            _fmt(s.error_pct, ".1f"),
        ]
        for s in stats
    ]
    if markdown:
        lines = ["| " + " | ".join(_HEADERS) + " |", "|" + "---|" * len(_HEADERS)]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines)

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(_HEADERS)]

    def line(cells: list[str]) -> str:
        # Model name left-aligned, numbers right-aligned.
        return "  ".join(
            c.ljust(w) if i == 0 else c.rjust(w)
            for i, (c, w) in enumerate(zip(cells, widths, strict=True))
        )

    return "\n".join([line(_HEADERS), line(["-" * w for w in widths]), *map(line, rows)])
