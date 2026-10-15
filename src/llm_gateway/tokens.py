"""Rough token counts, used only to reserve rate-limit budget before a call.

cl100k is not the tokenizer of any Groq model, but it is close enough for budgeting, and
the limiter corrects itself with the real usage the provider reports after each call.
"""

import json
import logging
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)

# Tokens reserved for the reply when the caller sets no (or a large) max_tokens.
COMPLETION_RESERVE = 512
# Role markers and separators the chat template adds around each message.
_PER_MESSAGE_OVERHEAD = 4


@lru_cache(maxsize=1)
def _encoding():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # e.g. no network on the very first run
        log.warning("tiktoken unavailable, estimating tokens as chars/4", extra={"error": str(exc)})
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    encoding = _encoding()
    if encoding is None:
        return max(1, len(text) // 4)
    return len(encoding.encode(text, disallowed_special=()))


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):  # content parts: [{"type": "text", "text": "..."}]
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    parts = [content or ""]
    if message.get("tool_calls"):
        parts.append(json.dumps(message["tool_calls"]))
    return "\n".join(parts)


def estimate_prompt_tokens(
    messages: Iterable[dict[str, Any]], tools: list[dict[str, Any]] | None = None
) -> int:
    total = sum(count_tokens(_message_text(m)) + _PER_MESSAGE_OVERHEAD for m in messages)
    if tools:
        total += count_tokens(json.dumps(tools))
    return total


def completion_budget(max_tokens: int | None) -> int:
    return min(max_tokens, COMPLETION_RESERVE) if max_tokens else COMPLETION_RESERVE
