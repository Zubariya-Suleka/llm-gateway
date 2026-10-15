"""`gateway` command-line entry point."""

import argparse

import uvicorn

from llm_gateway.app import create_app
from llm_gateway.config import get_settings
from llm_gateway.log import setup_logging


def _serve(args: argparse.Namespace) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    uvicorn.run(
        create_app(settings),
        host=args.host or settings.host,
        port=args.port or settings.port,
        log_config=None,  # keep our JSON logging; uvicorn's loggers propagate to root
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gateway")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP gateway")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
