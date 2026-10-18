"""Benchmark models through the gateway: 50 fixed prompts per model at temperature 0.

    uv run python scripts/benchmark.py                        # models from config/models.yaml
    uv run python scripts/benchmark.py --models mock/bench    # no API key, no quota
    uv run python scripts/benchmark.py --url http://127.0.0.1:8000   # a running gateway

By default the gateway runs in-process (same code path as HTTP, via httpx's ASGI
transport), so there is no server to start. Requests are tagged X-Project: benchmark and
the table is built from the metrics rows this run wrote.

Run it twice: the second run should be ~100% cache hits with near-zero latency.
Budget: ~50 requests and roughly 40-60K tokens per Groq model, well inside the free daily
limits; the rate limiter paces it, so expect several minutes per run.
"""

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx

from llm_gateway.app import create_app
from llm_gateway.catalog import load_yaml
from llm_gateway.config import get_settings
from llm_gateway.log import setup_logging
from llm_gateway.metrics import MetricsStore
from llm_gateway.stats import format_table, summarize

PROMPTS_FILE = Path(__file__).with_name("benchmark_prompts.json")


def build_prompts() -> list[dict[str, str]]:
    data = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    prompts = [
        {"id": f"short-{i:02d}", "bucket": "short", "content": text}
        for i, text in enumerate(data["short"], 1)
    ]
    passages = data["passages"]
    for i, passage in enumerate(passages, 1):
        for j, task in enumerate(data["medium_tasks"], 1):
            prompts.append(
                {
                    "id": f"medium-{i:02d}{'ab'[j - 1]}",
                    "bucket": "medium",
                    "content": f"{task}\n\n{passage['text']}",
                }
            )
    for i, passage in enumerate(passages):
        # Rotate the passage order so each long prompt is a different text.
        rotated = passages[i:] + passages[:i]
        context = "\n\n".join(p["text"] for p in rotated)
        prompts.append(
            {
                "id": f"long-{i + 1:02d}",
                "bucket": "long",
                "content": f"{context}\n\nUsing only the text above, answer in two or three "
                f"sentences: {passage['question']}",
            }
        )
    return prompts


async def run_model(
    client: httpx.AsyncClient,
    model: str,
    prompts: list[dict[str, str]],
    *,
    concurrency: int,
    max_tokens: int,
    run_id: str,
) -> Counter:
    outcomes: Counter = Counter()
    semaphore = asyncio.Semaphore(concurrency)
    started = time.perf_counter()

    async def one(prompt: dict[str, str]) -> None:
        async with semaphore:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt["content"]}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                },
                headers={"X-Project": "benchmark", "X-Trace-Id": f"{run_id}/{prompt['id']}"},
            )
        if response.status_code == 200:
            outcomes[response.headers.get("x-cache", "?")] += 1
        else:
            outcomes[f"HTTP {response.status_code}"] += 1
            message = response.json().get("error", {}).get("message", response.text[:200])
            print(f"  {model} {prompt['id']}: {response.status_code} {message}", file=sys.stderr)
        done = sum(outcomes.values())
        if done % 10 == 0:
            print(f"  {model}: {done}/{len(prompts)}", file=sys.stderr)

    await asyncio.gather(*(one(p) for p in prompts))
    elapsed = time.perf_counter() - started
    print(f"  {model}: done in {elapsed:.0f}s {dict(outcomes)}", file=sys.stderr)
    return outcomes


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    models = args.models or load_yaml(settings.config_dir / "models.yaml").get("benchmark", [])
    if not models:
        sys.exit("no models: pass --models or list them under 'benchmark' in config/models.yaml")
    prompts = build_prompts()[: args.limit]
    run_id = datetime.now(UTC).strftime("bench-%Y%m%dT%H%M%S")
    started = datetime.now(UTC)
    print(f"{run_id}: {len(prompts)} prompts x {len(models)} models", file=sys.stderr)

    async def all_models(client: httpx.AsyncClient) -> None:
        # Models in parallel (Groq limits are per model); each one paced by its own limiter.
        await asyncio.gather(
            *(
                run_model(
                    client,
                    m,
                    prompts,
                    concurrency=args.concurrency,
                    max_tokens=args.max_tokens,
                    run_id=run_id,
                )
                for m in models
            )
        )

    if args.url:
        async with httpx.AsyncClient(base_url=args.url, timeout=600) as client:
            await all_models(client)
    else:
        app = create_app(settings)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://gateway", timeout=None
            ) as client,
        ):
            await all_models(client)

    store = MetricsStore(settings.metrics_path)
    rows = [r for r in store.query(since=started, project="benchmark") if r["model"] in models]
    store.close()
    if not rows:
        print("no metrics rows written (is --url pointing at a gateway on this machine?)")
        return 1
    print(f"\n{run_id} (latency = model call only; cache hits ~0 ms)\n")
    print(format_table(summarize(rows), markdown=True))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", help="gateway model names, e.g. mock/bench")
    parser.add_argument("--url", help="benchmark a running gateway instead of in-process")
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts")
    parser.add_argument("--concurrency", type=int, default=4, help="in-flight requests per model")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    setup_logging(args.log_level)
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
