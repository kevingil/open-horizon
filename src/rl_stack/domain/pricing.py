"""Per-model pricing for cost accounting. USD per 1M tokens.

Public OpenAI/Anthropic models carry list prices; self-hosted models
(`vllm:*`, `ollama:*`, `local:*`) are treated as zero-cost. The lookup is
prefix-aware so any `vllm:Qwen/Qwen2.5-7B` style id matches without needing
an entry per model.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None


# Conservative list prices. Update as providers change theirs; users with
# private pricing should override via their own ModelPrice registration.
MODEL_PRICES: dict[str, ModelPrice] = {
    # OpenAI
    "gpt-4o": ModelPrice(2.5, 10.0, cache_read_per_mtok=1.25),
    "gpt-4o-mini": ModelPrice(0.15, 0.6, cache_read_per_mtok=0.075),
    "gpt-4.1": ModelPrice(2.0, 8.0, cache_read_per_mtok=0.5),
    "gpt-4.1-mini": ModelPrice(0.4, 1.6, cache_read_per_mtok=0.1),
    "gpt-4.1-nano": ModelPrice(0.1, 0.4, cache_read_per_mtok=0.025),
    "o3-mini": ModelPrice(1.1, 4.4, cache_read_per_mtok=0.55),
    # Anthropic (also reachable via the OpenAI-compat endpoint)
    "claude-opus-4-7": ModelPrice(15.0, 75.0, 18.75, 1.5),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0, 3.75, 0.3),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0, 1.25, 0.1),
}

# Model id prefixes that always cost $0 (self-hosted via vLLM, SGLang,
# Ollama, etc.).
ZERO_COST_PREFIXES = ("vllm:", "sglang:", "ollama:", "local:")


def estimate_cost_usd(
    model_id: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    if model_id.startswith(ZERO_COST_PREFIXES):
        return 0.0
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
