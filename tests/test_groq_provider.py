import asyncio
import json

import httpx
import pytest
from conftest import completion

from llm_gateway.errors import ProviderError
from llm_gateway.providers.groq import GroqProvider

PAYLOAD = {"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "hi"}]}


def provider(handler, api_key="test-key") -> GroqProvider:
    return GroqProvider(api_key, transport=httpx.MockTransport(handler))


def call(p: GroqProvider):
    return asyncio.run(p.complete("openai/gpt-oss-20b", PAYLOAD))


def test_success_posts_to_openai_path_with_bearer_token():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion())

    body = call(provider(handler))

    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"] == PAYLOAD
    assert body["usage"]["completion_tokens"] == 20


def test_429_carries_retry_after_and_message():
    def handler(request):
        return httpx.Response(
            429,
            headers={"retry-after": "2.5"},
            json={"error": {"message": "Rate limit reached for model", "type": "tokens"}},
        )

    with pytest.raises(ProviderError) as info:
        call(provider(handler))
    assert info.value.status_code == 429
    assert info.value.retry_after == 2.5
    assert info.value.retryable
    assert "Rate limit reached" in info.value.message


def test_400_is_not_retryable():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "bad tool schema"}})

    with pytest.raises(ProviderError) as info:
        call(provider(handler))
    assert not info.value.retryable
    assert info.value.retry_after is None


def test_unparseable_tool_call_keeps_what_the_model_wrote():
    def handler(request):
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Parsing failed. The model generated output that could not be"
                    " parsed.",
                    "type": "invalid_request_error",
                    "code": "tool_use_failed",
                    "failed_generation": '{"name": "submit_answer", "arguments": {"sql": "SEL',
                }
            },
        )

    with pytest.raises(ProviderError) as info:
        call(provider(handler))
    assert not info.value.retryable  # same input, same broken output: retrying won't help
    assert info.value.details == {
        "provider_code": "tool_use_failed",
        "failed_generation": '{"name": "submit_answer", "arguments": {"sql": "SEL',
    }


def test_network_failure_is_a_retryable_502():
    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(ProviderError) as info:
        call(provider(handler))
    assert info.value.status_code == 502
    assert info.value.retryable


def test_missing_key_fails_without_calling_groq():
    def handler(request):
        raise AssertionError("must not be called")

    with pytest.raises(ProviderError, match="GROQ_API_KEY") as info:
        call(provider(handler, api_key=None))
    assert not info.value.retryable
