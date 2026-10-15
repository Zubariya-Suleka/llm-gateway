"""Settings read from the environment and `.env`. Secrets live only there.

Relative paths are resolved from the directory the gateway is started in (the repo root).
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_timeout_s: float = 120.0

    # models.yaml, limits.yaml, pricing.yaml
    config_dir: Path = Path("config")

    cache_path: Path = Path(".cache/cache.db")
    metrics_path: Path = Path("data/metrics.db")
    # tiktoken downloads its vocabulary on first use; keep it in the repo, not on C:.
    tiktoken_cache_dir: Path = Path(".cache/tiktoken")

    retry_max_attempts: int = 5

    # mock/ provider: latency is sampled from a normal distribution (Week 6 replaces this
    # with the real Groq latency distribution from the metrics table).
    mock_latency_ms: float = 200.0
    mock_latency_jitter_ms: float = 50.0

    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
