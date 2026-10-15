"""Per-model YAML tables (limits.yaml, pricing.yaml, models.yaml).

Tables are keyed by the full gateway model name. A "<provider>/*" key is the fallback for
any model of that provider that has no entry of its own.
"""

from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def lookup(table: dict[str, T], model: str) -> T | None:
    if model in table:
        return table[model]
    provider = model.partition("/")[0]
    return table.get(f"{provider}/*")
