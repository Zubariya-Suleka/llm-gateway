import pytest

from llm_gateway.errors import BadRequest
from llm_gateway.providers.mock import MockProvider
from llm_gateway.router import resolve, split_model


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("groq/openai/gpt-oss-20b", ("groq", "openai/gpt-oss-20b")),
        ("groq/llama-3.1-8b-instant", ("groq", "llama-3.1-8b-instant")),
        (
            "local/HuggingFaceTB/SmolLM2-360M-Instruct",
            ("local", "HuggingFaceTB/SmolLM2-360M-Instruct"),
        ),
        ("mock/bench", ("mock", "bench")),
    ],
)
def test_split_on_first_slash_only(model, expected):
    assert split_model(model) == expected


@pytest.mark.parametrize("model", ["gpt-4o", "/gpt", "groq/", ""])
def test_malformed_model_is_a_bad_request(model):
    with pytest.raises(BadRequest):
        split_model(model)


def test_unknown_provider_lists_known_ones():
    with pytest.raises(BadRequest, match="known: mock"):
        resolve("openai/gpt-4o", {"mock": MockProvider()})
