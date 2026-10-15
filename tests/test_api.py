import pytest
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.config import Settings


@pytest.fixture
def client(make_gateway):
    gateway, metrics = make_gateway()
    settings = Settings(_env_file=None)
    with TestClient(create_app(settings, gateway=gateway)) as client:
        client.metrics = metrics
        yield client


def body(**overrides):
    return {
        "model": "mock/test",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
        **overrides,
    }


def test_returns_openai_shape_and_cache_header(client):
    first = client.post("/v1/chat/completions", json=body(), headers={"X-Project": "tests"})
    second = client.post("/v1/chat/completions", json=body())

    assert first.status_code == 200
    data = first.json()
    assert data["model"] == "mock/test"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["usage"]["total_tokens"] > 0
    assert (first.headers["x-cache"], second.headers["x-cache"]) == ("MISS", "HIT")
    assert client.metrics.query()[0]["project"] == "tests"


def test_nonzero_temperature_says_bypass(client):
    response = client.post("/v1/chat/completions", json=body(temperature=0.7))
    assert response.headers["x-cache"] == "BYPASS"


def test_tools_and_response_format_pass_validation(client):
    response = client.post(
        "/v1/chat/completions",
        json=body(
            tools=[{"type": "function", "function": {"name": "run_sql", "parameters": {}}}],
            tool_choice="auto",
            response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {}}},
            max_tokens=50,
            reasoning_effort="low",  # unknown to the gateway: passed through
        ),
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "{}"


@pytest.mark.parametrize(
    ("payload", "status", "error_type"),
    [
        (body(model="openai/gpt-4o"), 400, "invalid_request_error"),
        (body(model="no-prefix"), 400, "invalid_request_error"),
        (body(stream=True), 400, "invalid_request_error"),
        (body(model="mock/fail-500"), 500, "provider_error"),
        (body(model="mock/fail-401"), 401, "provider_error"),
    ],
)
def test_errors_use_the_openai_error_shape(client, payload, status, error_type):
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == status
    assert response.json()["error"]["type"] == error_type


def test_persistent_429_is_retried_then_returned_with_retry_after(client):
    response = client.post("/v1/chat/completions", json=body(model="mock/fail-429"))
    assert response.status_code == 429
    assert response.headers["retry-after"] == "0"
    assert client.metrics.query()[-1]["attempts"] == 5


def test_missing_messages_is_a_validation_error(client):
    response = client.post("/v1/chat/completions", json={"model": "mock/x", "messages": []})
    assert response.status_code == 422
