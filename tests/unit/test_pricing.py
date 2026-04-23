from __future__ import annotations

from rl_stack.domain.pricing import estimate_cost_usd


def test_haiku_cost_matches_published_rate() -> None:
    # 1M input + 1M output against Haiku 4.5: $1 + $5.
    cost = estimate_cost_usd(
        "claude-haiku-4-5", input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert cost == 6.0


def test_cache_read_is_much_cheaper() -> None:
    plain = estimate_cost_usd("claude-sonnet-4-6", input_tokens=100_000)
    cached = estimate_cost_usd("claude-sonnet-4-6", cache_read_tokens=100_000)
    assert cached < plain / 5


def test_unknown_model_costs_zero_rather_than_raising() -> None:
    assert estimate_cost_usd("unknown", input_tokens=1_000_000) == 0.0
