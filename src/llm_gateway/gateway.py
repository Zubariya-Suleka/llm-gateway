"""The request pipeline: cache -> retry -> rate limiter -> provider, one metrics row per request.

Why this order:
- Cache first: a hit costs no quota and never waits behind the limiter.
- Retry wraps the limiter: every attempt waits for budget like a fresh request, because
  Groq counts every attempt against the per-minute limits.
- The metrics row is written in `finally`, so failures are recorded as well as successes.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from llm_gateway.cache import ResponseCache, cache_key, is_cacheable
from llm_gateway.config import Settings
from llm_gateway.errors import GatewayError
from llm_gateway.metrics import MetricsStore, RequestRecord
from llm_gateway.pricing import Pricing
from llm_gateway.providers.base import Provider
from llm_gateway.providers.groq import GroqProvider
from llm_gateway.providers.local import LocalProvider
from llm_gateway.providers.mock import MockProvider
from llm_gateway.ratelimit import RateLimiter, Usage, load_limits
from llm_gateway.retry import RetryPolicy
from llm_gateway.router import resolve
from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.tokens import completion_budget, estimate_prompt_tokens

log = logging.getLogger(__name__)

CacheStatus = Literal["HIT", "MISS", "BYPASS"]


@dataclass
class Result:
    body: dict[str, Any]
    cache: CacheStatus  # BYPASS: not cacheable (temperature != 0)
    attempts: int


class Gateway:
    def __init__(
        self,
        *,
        providers: Mapping[str, Provider],
        cache: ResponseCache,
        limiter: RateLimiter,
        retry: RetryPolicy,
        metrics: MetricsStore,
        pricing: Pricing,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._providers = providers
        self._cache = cache
        self._limiter = limiter
        self._retry = retry
        self._metrics = metrics
        self._pricing = pricing
        self._clock = clock

    @classmethod
    def from_settings(cls, settings: Settings) -> "Gateway":
        metrics = MetricsStore(settings.metrics_path)

        def usage_last_24h(model: str) -> tuple[int, int]:
            return metrics.usage_since(model, datetime.now(UTC) - timedelta(days=1))

        api_key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
        providers: dict[str, Provider] = {
            "groq": GroqProvider(
                api_key, base_url=settings.groq_base_url, timeout_s=settings.groq_timeout_s
            ),
            "mock": MockProvider(
                latency_ms=settings.mock_latency_ms, jitter_ms=settings.mock_latency_jitter_ms
            ),
            "local": LocalProvider(),
        }
        return cls(
            providers=providers,
            cache=ResponseCache(settings.cache_path),
            limiter=RateLimiter(
                load_limits(settings.config_dir / "limits.yaml"), usage_lookup=usage_last_24h
            ),
            retry=RetryPolicy(max_attempts=settings.retry_max_attempts),
            metrics=metrics,
            pricing=Pricing.from_yaml(settings.config_dir / "pricing.yaml"),
        )

    async def complete(
        self,
        request: ChatCompletionRequest,
        *,
        project: str | None = None,
        trace_id: str | None = None,
    ) -> Result:
        rec = RequestRecord(model=request.model, project=project, trace_id=trace_id)
        started = self._clock()
        try:
            return await self._complete(request, rec)
        except GatewayError as exc:
            rec.status, rec.error = exc.status_code, exc.message
            raise
        except Exception as exc:
            rec.status, rec.error = 500, f"{type(exc).__name__}: {exc}"
            raise
        finally:
            rec.total_ms = (self._clock() - started) * 1000
            await self._record(rec)

    async def _complete(self, request: ChatCompletionRequest, rec: RequestRecord) -> Result:
        provider, model_id = resolve(request.model, self._providers)
        rec.provider = provider.name

        key = None
        if is_cacheable(request.temperature):
            key = cache_key(request.cache_material())
            lookup_started = self._clock()
            cached = await asyncio.to_thread(self._cache.get, key)
            if cached is not None:
                usage = cached.get("usage") or {}
                rec.cache_hit, rec.status = True, 200
                rec.latency_ms = (self._clock() - lookup_started) * 1000
                rec.prompt_tokens = usage.get("prompt_tokens")
                rec.completion_tokens = usage.get("completion_tokens")
                rec.est_cost_usd = 0.0  # nothing was sent to the provider
                return Result(cached, "HIT", 0)

        payload = request.provider_payload(model_id)
        reserved = Usage(
            prompt=estimate_prompt_tokens(request.messages, request.tools),
            completion=completion_budget(request.completion_limit()),
        )

        async def attempt(n: int) -> dict[str, Any]:
            await self._limiter.acquire(request.model, reserved)
            rec.attempts = n
            call_started = self._clock()
            try:
                return await provider.complete(model_id, payload)
            finally:
                rec.latency_ms = (self._clock() - call_started) * 1000

        body = {**await self._retry.run(attempt), "model": request.model}
        self._account_usage(rec, body, reserved)
        rec.status = 200
        if key is not None:
            await asyncio.to_thread(self._cache.put, key, request.model, body)
        return Result(body, "MISS" if key else "BYPASS", rec.attempts)

    def _account_usage(self, rec: RequestRecord, body: dict[str, Any], reserved: Usage) -> None:
        usage = body.get("usage") or {}
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        rec.prompt_tokens, rec.completion_tokens, rec.cached_tokens = prompt, completion, cached

        if prompt is not None and completion is not None:
            # Groq doesn't count prompt-cache hits towards its limits, so neither do we.
            self._limiter.reconcile(rec.model, reserved, Usage(prompt - cached, completion))
            rec.est_cost_usd = self._pricing.cost(rec.model, prompt, completion, cached)

        # Groq reports generation time in usage; otherwise fall back to wall-clock latency.
        completion_time = usage.get("completion_time")
        rec.completion_time_s = completion_time
        seconds = completion_time if completion_time else (rec.latency_ms or 0) / 1000
        if completion and seconds > 0:
            rec.tokens_per_sec = completion / seconds

    async def _record(self, rec: RequestRecord) -> None:
        try:
            await asyncio.to_thread(self._metrics.record, rec)
        except Exception:
            # Losing a metrics row must never fail the caller's request.
            log.exception("failed to write metrics row")
        log.info(
            "chat completion",
            extra={
                "model": rec.model,
                "project": rec.project,
                "trace_id": rec.trace_id,
                "status": rec.status,
                "cache_hit": rec.cache_hit,
                "attempts": rec.attempts,
                "latency_ms": round(rec.latency_ms or 0, 1),
                "total_ms": round(rec.total_ms or 0, 1),
                "error": rec.error,
            },
        )

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
        self._cache.close()
        self._metrics.close()
