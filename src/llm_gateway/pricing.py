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
    # Prompt tokens served from the provider's prompt cache. None: no discount known.
    cached_input_per_m: float | None = None


class Pricing:
    def __init__(self, table: dict[str, Price]) -> None:
        self._table = table

    @classmethod
    def from_yaml(cls, path: Path) -> "Pricing":
        raw = load_yaml(path)
        return cls(
            {
                model: Price(
                    float(p["input"]),
                    float(p["output"]),
                    float(p["cached_input"]) if "cached_input" in p else None,
                )
                for model, p in raw.items()
            }
        )

    def cost(
        self, model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0
    ) -> float | None:
        """None when the model has no price, so unknown costs never show up as $0."""
        price = lookup(self._table, model)
        if price is None:
            return None
        cached_price = (
            price.cached_input_per_m if price.cached_input_per_m is not None else price.input_per_m
        )
        return (
            (prompt_tokens - cached_tokens) * price.input_per_m
            + cached_tokens * cached_price
            + completion_tokens * price.output_per_m
        ) / 1e6
