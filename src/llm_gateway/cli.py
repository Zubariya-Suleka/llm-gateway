"""`gateway` command-line entry point."""

import argparse
import sys
from datetime import UTC, datetime

import uvicorn

from llm_gateway.app import create_app
from llm_gateway.config import get_settings
from llm_gateway.log import setup_logging
from llm_gateway.metrics import MetricsStore
from llm_gateway.stats import format_table, parse_since, summarize


def _serve(args: argparse.Namespace) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    uvicorn.run(
        create_app(settings),
        host=args.host or settings.host,
        port=args.port or settings.port,
        log_config=None,  # keep our JSON logging; uvicorn's loggers propagate to root
    )


def _stats(args: argparse.Namespace) -> None:
    try:
        since = datetime.now(UTC) - parse_since(args.since)
    except ValueError as exc:
        sys.exit(str(exc))
    store = MetricsStore(get_settings().metrics_path)
    rows = store.query(since=since, project=args.project, model=args.model)
    store.close()
    if args.exclude_cache_hits:
        rows = [r for r in rows if not r["cache_hit"]]
    if not rows:
        print(f"no requests in the last {args.since}")
        return
    print(format_table(summarize(rows), markdown=args.markdown))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gateway")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP gateway")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(func=_serve)

    stats = sub.add_parser("stats", help="per-model latency, speed, cost and error summary")
    stats.add_argument("--since", default="7d", help="window, e.g. 30m, 12h, 7d (default 7d)")
    stats.add_argument("--project", help="only requests with this X-Project")
    stats.add_argument("--model", help="only this model, e.g. groq/openai/gpt-oss-20b")
    stats.add_argument(
        "--exclude-cache-hits", action="store_true", help="only calls that reached a provider"
    )
    stats.add_argument("--markdown", action="store_true", help="print a markdown table")
    stats.set_defaults(func=_stats)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
