from pathlib import Path
from typing import Any

import pytest

from llm_gateway.cache import ResponseCache
from llm_gateway.gateway import Gateway
from llm_gateway.metrics import MetricsStore
from llm_gateway.pricing import Price, Pricing
from llm_gateway.providers.mock import MockProvider
from llm_gateway.ratelimit import ModelLimits, RateLimiter
from llm_gateway.retry import RetryPolicy


async def no_sleep(_: float) -> None:
    pass


class FakeTime:
    """A clock that only moves when something sleeps on it."""

    def __init__(self) -> None:
        self.now = 1_000_000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class ScriptedProvider:
    """Plays back outcomes in order (an exception is raised, a dict returned); the last repeats."""

    name = "fake"

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def complete(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((model, payload))
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        pass


def completion(
    content: str = "hi",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    cached_tokens: int | None = None,
) -> dict:
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "completion_time": 0.1,
    }
    if cached_tokens is not None:  # Groq adds this when its prompt cache was used
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return {
        "id": "x",
        "object": "chat.completion",
        "model": "provider-name",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": usage,
    }


@pytest.fixture
def make_gateway(tmp_path: Path):
    """Build a Gateway on temp SQLite files. Returns (gateway, metrics_store)."""

    def build(
        providers: dict | None = None,
        limits: dict[str, ModelLimits] | None = None,
        retry: RetryPolicy | None = None,
    ) -> tuple[Gateway, MetricsStore]:
        metrics = MetricsStore(tmp_path / "metrics.db")
        gateway = Gateway(
            providers=providers or {"mock": MockProvider()},
            cache=ResponseCache(tmp_path / "cache.db"),
            limiter=RateLimiter(limits or {}),
            retry=retry or RetryPolicy(sleep=no_sleep, rand=lambda: 0.0),
            metrics=metrics,
            pricing=Pricing({"fake/*": Price(1.0, 2.0, 0.5), "mock/*": Price(0, 0)}),
        )
        return gateway, metrics

    return build
