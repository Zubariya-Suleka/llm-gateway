"""Pick a provider from the model name prefix: "groq/openai/gpt-oss-20b" -> groq."""

from collections.abc import Mapping

from llm_gateway.errors import BadRequest
from llm_gateway.providers.base import Provider


def split_model(model: str) -> tuple[str, str]:
    """Split on the first "/" only: Groq model ids contain slashes themselves."""
    provider, sep, model_id = model.partition("/")
    if not sep or not provider or not model_id:
        raise BadRequest(f"model must look like '<provider>/<model-id>', got {model!r}")
    return provider, model_id


def resolve(model: str, providers: Mapping[str, Provider]) -> tuple[Provider, str]:
    name, model_id = split_model(model)
    provider = providers.get(name)
    if provider is None:
        known = ", ".join(sorted(providers))
        raise BadRequest(f"unknown provider {name!r} in {model!r}; known: {known}")
    return provider, model_id
