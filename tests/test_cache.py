import pytest

from llm_gateway.cache import ResponseCache, cache_key, is_cacheable

BASE = {
    "model": "groq/openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "hi"}],
    "temperature": 0.0,
    "max_tokens": 100,
}


def test_key_is_stable_across_dict_order():
    reordered = {
        "max_tokens": 100,
        "temperature": 0.0,
        "messages": [{"content": "hi", "role": "user"}],
        "model": "groq/openai/gpt-oss-20b",
    }
    assert cache_key(BASE) == cache_key(reordered)
    assert len(cache_key(BASE)) == 64  # sha256 hex


@pytest.mark.parametrize(
    "change",
    [
        {"model": "groq/openai/gpt-oss-120b"},
        {"messages": [{"role": "user", "content": "hello"}]},
        {"max_tokens": 101},
        {"tools": [{"type": "function", "function": {"name": "f"}}]},
        {"response_format": {"type": "json_object"}},
        {"tool_choice": "required"},
        {"seed": 7},
    ],
)
def test_any_input_that_changes_the_answer_changes_the_key(change):
    assert cache_key({**BASE, **change}) != cache_key(BASE)


def test_fields_that_cannot_change_the_answer_are_ignored():
    assert cache_key({**BASE, "stream": False, "user": "zubi", "seed": None}) == cache_key(BASE)


@pytest.mark.parametrize(
    ("temperature", "cacheable"), [(0, True), (0.0, True), (None, False), (0.7, False)]
)
def test_only_explicit_temperature_zero_is_cacheable(temperature, cacheable):
    assert is_cacheable(temperature) is cacheable


def test_put_get_round_trip_survives_reopen(tmp_path):
    path = tmp_path / "cache.db"
    cache = ResponseCache(path)
    assert cache.get("k") is None
    cache.put("k", "mock/x", {"choices": [], "usage": {"prompt_tokens": 1}})
    cache.close()

    reopened = ResponseCache(path)
    assert reopened.get("k") == {"choices": [], "usage": {"prompt_tokens": 1}}
    reopened.close()
