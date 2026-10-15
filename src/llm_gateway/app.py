"""FastAPI app. Routes are added here as each layer lands."""

from fastapi import FastAPI

from llm_gateway import __version__
from llm_gateway.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="llm-gateway", version=__version__)
    app.state.settings = settings

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app
