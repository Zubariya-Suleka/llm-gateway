"""Fake provider: an OpenAI-shaped reply after a sampled latency. No network, no quota.

Used for tests, for trying the gateway without a key, and for load tests (Week 6), where
it measures the gateway itself instead of Groq's rate limits.

Special model names:
    mock/fail-<status>   always fails with that status, e.g. mock/fail-429, mock/fail-500
"""

import asyncio
import random
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from llm_gateway.errors import ProviderError
from llm_gateway.tokens import count_tokens, estimate_prompt_tokens


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


class MockProvider:
    name = "mock"

    def __init__(
        self,
        *,
        latency_ms: float = 0.0,
        jitter_ms: float = 0.0,
        seed: int | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._latency_ms = latency_ms
        self._jitter_ms = jitter_ms
        self._rng = random.Random(seed)
        self._sleep = sleep

    async def complete(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        if model.startswith("fail-"):
            status = int(model.removeprefix("fail-"))
            raise ProviderError(
                f"mock failure {status}",
                status_code=status,
                retry_after=0.0 if status == 429 else None,
            )

        latency_s = max(0.0, self._rng.gauss(self._latency_ms, self._jitter_ms)) / 1000
        if latency_s:
            await self._sleep(latency_s)

        response_format = (payload.get("response_format") or {}).get("type")
        if response_format in ("json_object", "json_schema"):
            content = "{}"
        else:
            content = f"[mock:{model}] {_last_user_text(payload['messages'])[:200]}"

        prompt_tokens = estimate_prompt_tokens(payload["messages"], payload.get("tools"))
        completion_tokens = count_tokens(content)
        return {
            "id": f"mock-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "completion_time": latency_s,
                "total_time": latency_s,
            },
        }

    async def aclose(self) -> None:
        pass
