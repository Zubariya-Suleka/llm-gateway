set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

default:
    @just --list

# install deps and git hooks
setup:
    uv sync
    uv run pre-commit install

test:
    uv run pytest

lint:
    uv run ruff check .
    uv run ruff format --check .

# auto-fix lint and formatting
fmt:
    uv run ruff check --fix .
    uv run ruff format .

run:
    uv run gateway serve

# also install torch + transformers for local/ models (~1 GB, CPU-only)
setup-local:
    uv sync --extra local

# per-model latency / speed / cost summary, e.g. `just stats 1d`
stats since="7d":
    uv run gateway stats --since {{since}}

# 50 fixed prompts x the models in config/models.yaml (spends Groq free quota)
bench *args:
    uv run python scripts/benchmark.py {{args}}
