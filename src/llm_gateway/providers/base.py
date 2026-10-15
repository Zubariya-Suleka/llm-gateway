from typing import Any, Protocol


class Provider(Protocol):
    name: str

    async def complete(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one chat completion. Raises ProviderError on failure.

        `model` is the provider's own model id (prefix stripped); `payload` is the OpenAI
        request body with that id already in `payload["model"]`.
        """
        ...

    async def aclose(self) -> None: ...
