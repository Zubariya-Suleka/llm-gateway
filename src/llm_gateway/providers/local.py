"""Local Hugging Face model on CPU, e.g. local/HuggingFaceTB/SmolLM2-360M-Instruct.

Optional: needs the `local` extra (`uv sync --extra local`), so torch never lands in the
default install or CI. Models load on first use and stay in memory; weights are cached
wherever HF_HOME points (keep it off C:).

Plain text only: tools are passed to the chat template, but tool calls in the output are
not parsed, and response_format is not enforced. A small local model is a baseline here,
not a worker.
"""

import asyncio
import threading
import time
import uuid
from typing import Any

from llm_gateway.errors import ProviderError

DEFAULT_MAX_NEW_TOKENS = 256


def _text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


class LocalProvider:
    name = "local"

    def __init__(self) -> None:
        self._models: dict[str, tuple[Any, Any]] = {}
        # One generation at a time: they compete for the same CPU cores anyway.
        self._lock = threading.Lock()

    async def complete(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        # generate() blocks for seconds; a worker thread keeps the event loop serving others.
        return await asyncio.to_thread(self._complete_blocking, model, payload)

    def _load(self, model: str) -> tuple[Any, Any]:
        if model in self._models:
            return self._models[model]
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ProviderError(
                "local provider needs the 'local' extra: uv sync --extra local",
                status_code=501,
                retryable=False,
            ) from exc
        try:
            tokenizer = AutoTokenizer.from_pretrained(model)
            # bf16 halves RAM; on this class of CPU it measured as fast as fp32.
            lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
        except OSError as exc:
            raise ProviderError(
                f"cannot load local model {model!r}: {exc}", status_code=400, retryable=False
            ) from exc
        lm.eval()
        self._models[model] = (tokenizer, lm)
        return tokenizer, lm

    def _complete_blocking(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        import torch

        with self._lock:
            tokenizer, lm = self._load(model)
            messages = [
                {"role": m.get("role", "user"), "content": _text(m.get("content"))}
                for m in payload["messages"]
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                tools=payload.get("tools"),
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            prompt_tokens = inputs["input_ids"].shape[1]
            max_new = (
                payload.get("max_completion_tokens")
                or payload.get("max_tokens")
                or DEFAULT_MAX_NEW_TOKENS
            )
            temperature = payload.get("temperature", 1.0)
            # Temperature 0 means greedy; sampling at 0 would divide the logits by zero.
            sampling = (
                {"do_sample": False}
                if temperature == 0
                else {"do_sample": True, "temperature": temperature}
            )

            # Timed around generate() only, so a first-call model load is not counted. Unlike
            # Groq's completion_time, this includes prompt processing (prefill).
            started = time.perf_counter()
            # inference_mode is thread-local, so it is entered here, on the worker thread.
            with torch.inference_mode():
                output = lm.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    **sampling,
                )
            elapsed = time.perf_counter() - started

        completion = output[0, prompt_tokens:].tolist()
        eos = lm.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish_reason = "stop" if completion and completion[-1] in eos_ids else "length"
        return {
            "id": f"local-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": tokenizer.decode(completion, skip_special_tokens=True),
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(completion),
                "total_tokens": prompt_tokens + len(completion),
                "completion_time": elapsed,
                "total_time": elapsed,
            },
        }

    async def aclose(self) -> None:
        self._models.clear()
