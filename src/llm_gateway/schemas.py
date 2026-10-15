"""OpenAI chat-completions request shape.

Only the fields the gateway itself reads are declared; anything else (top_p, seed, stop,
reasoning_effort, ...) is kept and passed through to the provider untouched. Responses are
passed through as plain dicts in the OpenAI shape.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    # "<provider>/<model-id>", e.g. "groq/openai/gpt-oss-20b"
    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False

    def completion_limit(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens

    def cache_material(self) -> dict[str, Any]:
        """Every field the caller set, with the gateway model name: the input to the cache key."""
        return self.model_dump(exclude_none=True)

    def provider_payload(self, model_id: str) -> dict[str, Any]:
        """The body sent to the provider: same request, provider's own model id, no streaming."""
        payload = self.model_dump(exclude_none=True)
        payload.pop("stream", None)
        payload["model"] = model_id
        return payload
