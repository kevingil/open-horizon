from __future__ import annotations

from domain.pricing import estimate_cost_usd


def test_haiku_cost_matches_published_rate() -> None:
    # 1M input + 1M output against Haiku 4.5: $1 + $5.
    cost = estimate_cost_usd(
        "claude-haiku-4-5", input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert cost == 6.0


def test_opus_4_7_uses_2026_pricing() -> None:
    # Opus 4.7 dropped to $5 / $25 in early 2026; the long-context
    # surcharge tier was retired at the same time.
    cost = estimate_cost_usd(
        "claude-opus-4-7", input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert cost == 30.0


def test_gpt_5_4_mini_cost_matches_published_rate() -> None:
    # 1M input + 1M output against gpt-5.4-mini: $0.75 + $4.50.
    cost = estimate_cost_usd(
        "gpt-5.4-mini", input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert cost == 5.25


def test_gpt_4_1_nano_is_cheapest_openai_tier() -> None:
    nano = estimate_cost_usd("gpt-4.1-nano", input_tokens=1_000_000)
    mini = estimate_cost_usd("gpt-5.4-mini", input_tokens=1_000_000)
    assert nano < mini


def test_gpt_4o_mini_legacy_pricing_still_resolves() -> None:
    # Legacy entry retained so re-scoring historical runs stays accurate.
    cost = estimate_cost_usd(
        "gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert cost == 0.75


def test_cache_read_is_much_cheaper() -> None:
    plain = estimate_cost_usd("claude-sonnet-4-6", input_tokens=100_000)
    cached = estimate_cost_usd("claude-sonnet-4-6", cache_read_tokens=100_000)
    assert cached < plain / 5


def test_self_hosted_prefixes_cost_zero() -> None:
    for model in (
        "vllm:Qwen/Qwen2.5-7B",
        "sglang:Qwen/Qwen2.5-7B-Instruct",
        "ollama:qwen2.5:7b",
        "local:my-adapter",
    ):
        assert estimate_cost_usd(model, input_tokens=10_000_000) == 0.0


def test_unknown_model_costs_zero_rather_than_raising() -> None:
    assert estimate_cost_usd("unknown", input_tokens=1_000_000) == 0.0
