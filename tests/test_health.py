from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.config import Settings


def test_healthz_returns_ok():
    client = TestClient(create_app(Settings(_env_file=None)))

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["groq_key_configured"] is False
