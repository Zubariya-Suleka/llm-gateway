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
