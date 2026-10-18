from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.config import Settings


def test_healthz_returns_ok():
    client = TestClient(create_app(Settings(_env_file=None)))

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["groq_key_configured"] is False


def test_empty_key_counts_as_not_configured():
    client = TestClient(create_app(Settings(_env_file=None, groq_api_key="")))
    assert client.get("/healthz").json()["groq_key_configured"] is False
