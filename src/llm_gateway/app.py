"""FastAPI app: POST /v1/chat/completions (OpenAI-compatible) and GET /healthz."""

import asyncio
import math
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse

from llm_gateway import __version__
from llm_gateway.config import Settings, get_settings
from llm_gateway.errors import BadRequest, GatewayError
from llm_gateway.gateway import Gateway
from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.tokens import count_tokens


def create_app(settings: Settings | None = None, gateway: Gateway | None = None) -> FastAPI:
    """Pass `gateway` to inject one (tests); otherwise it is built from settings on startup."""
    settings = settings or get_settings()
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(settings.tiktoken_cache_dir.resolve()))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.gateway = gateway or Gateway.from_settings(settings)
        # Load (and on the first run, download) the tokenizer now, not inside a request.
        await asyncio.to_thread(count_tokens, "warm up")
        yield
        await app.state.gateway.aclose()

    app = FastAPI(title="llm-gateway", version=__version__, lifespan=lifespan)
    app.state.settings = settings

    @app.exception_handler(GatewayError)
    async def gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        headers = {}
        if exc.retry_after is not None:
            headers["Retry-After"] = str(math.ceil(exc.retry_after))
        return JSONResponse(
            {"error": {"message": exc.message, "type": exc.error_type, "code": exc.status_code}},
            status_code=exc.status_code,
            headers=headers,
        )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "groq_key_configured": settings.groq_api_key is not None,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest,
        request: Request,
        response: Response,
        x_project: Annotated[str | None, Header()] = None,
        x_trace_id: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        if body.stream:
            raise BadRequest("stream=true is not supported; send stream=false")
        result = await request.app.state.gateway.complete(
            body, project=x_project, trace_id=x_trace_id
        )
        response.headers["X-Cache"] = result.cache
        response.headers["X-Attempts"] = str(result.attempts)
        return result.body

    return app
