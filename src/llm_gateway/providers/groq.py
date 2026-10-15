"""Groq via its OpenAI-compatible HTTP API, called with plain httpx.

No SDK: the gateway forwards OpenAI-shaped JSON as-is, and needs the raw status code and
`retry-after` header for its own retry layer (an SDK would retry on its own, hidden from
the metrics). Docs: https://console.groq.com/docs/api-reference
"""

from typing import Any

import httpx

from llm_gateway.errors import ProviderError


def _retry_after(response: httpx.Response) -> float | None:
    # Groq sends seconds ("2"). The HTTP-date form is legal but not used by Groq.
    value = response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error") or {}
        message = error.get("message") if isinstance(error, dict) else str(error)
    except ValueError:
        message = None
    return f"groq {response.status_code}: {message or response.text[:500] or 'no body'}"


class GroqProvider:
    name = "groq"

    def __init__(
        self,
        api_key: str | None,
        *,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_s: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._has_key = bool(api_key)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=timeout_s, transport=transport
        )

    async def complete(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._has_key:
            raise ProviderError(
                "GROQ_API_KEY is not set; add it to .env", status_code=503, retryable=False
            )
        try:
            response = await self._client.post("chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"groq timed out: {exc!r}", status_code=504) from exc
        except httpx.TransportError as exc:
            raise ProviderError(f"cannot reach groq: {exc!r}", status_code=502) from exc

        if response.status_code >= 400:
            raise ProviderError(
                _error_message(response),
                status_code=response.status_code,
                retry_after=_retry_after(response),
            )
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()
