"""Estimated cost per request from config/pricing.yaml (USD per 1M tokens).

The free tier costs $0. The estimate shows what the same traffic would cost on the paid
tier, which is the number that matters when deciding whether a design can scale.
"""

from dataclasses import dataclass
from pathlib import Path

from llm_gateway.catalog import load_yaml, lookup


@dataclass(frozen=True)
class Price:
    input_per_m: float
    output_per_m: float


class Pricing:
    def __init__(self, table: dict[str, Price]) -> None:
        self._table = table

    @classmethod
    def from_yaml(cls, path: Path) -> "Pricing":
        raw = load_yaml(path)
        return cls({m: Price(float(p["input"]), float(p["output"])) for m, p in raw.items()})

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
        """None when the model has no price, so unknown costs never show up as $0."""
        price = lookup(self._table, model)
        if price is None:
            return None
        return (prompt_tokens * price.input_per_m + completion_tokens * price.output_per_m) / 1e6
