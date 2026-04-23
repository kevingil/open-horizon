"""Per-model pricing for cost accounting. USD per 1M tokens."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None


# Conservative defaults; override via Settings if prices change.
MODEL_PRICES: dict[str, ModelPrice] = {
    "claude-opus-4-7": ModelPrice(15.0, 75.0, 18.75, 1.5),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0, 3.75, 0.3),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0, 1.25, 0.1),
}


def estimate_cost_usd(
    model_id: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    price = MODEL_PRICES.get(model_id)
    if price is None:
        return 0.0
    total = (input_tokens / 1_000_000) * price.input_per_mtok
    total += (output_tokens / 1_000_000) * price.output_per_mtok
    if price.cache_write_per_mtok:
        total += (cache_write_tokens / 1_000_000) * price.cache_write_per_mtok
    if price.cache_read_per_mtok:
        total += (cache_read_tokens / 1_000_000) * price.cache_read_per_mtok
    return round(total, 6)
