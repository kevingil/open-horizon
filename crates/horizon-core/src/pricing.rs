//! Per-model pricing for cost accounting. USD per 1M tokens.
//!
//! Self-hosted prefixes (`vllm:`, `sglang:`, `ollama:`, `local:`) cost $0.
//! Last refreshed: 2026-05.

use crate::round_to;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPrice {
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
    pub cache_write_per_mtok: Option<f64>,
    pub cache_read_per_mtok: Option<f64>,
}

const fn price(
    input: f64,
    output: f64,
    cache_write: Option<f64>,
    cache_read: Option<f64>,
) -> ModelPrice {
    ModelPrice {
        input_per_mtok: input,
        output_per_mtok: output,
        cache_write_per_mtok: cache_write,
        cache_read_per_mtok: cache_read,
    }
}

pub const MODEL_PRICES: &[(&str, ModelPrice)] = &[
    ("gpt-5", price(10.0, 30.0, None, Some(2.5))),
    ("gpt-5.4", price(10.0, 30.0, None, Some(2.5))),
    ("gpt-5.4-mini", price(0.75, 4.5, None, Some(0.15))),
    ("gpt-5.4-nano", price(0.15, 1.2, None, Some(0.04))),
    ("gpt-4.1", price(2.0, 8.0, None, Some(0.5))),
    ("gpt-4.1-mini", price(0.4, 1.6, None, Some(0.1))),
    ("gpt-4.1-nano", price(0.1, 0.4, None, Some(0.025))),
    ("gpt-4o", price(2.5, 10.0, None, Some(1.25))),
    ("gpt-4o-mini", price(0.15, 0.6, None, Some(0.075))),
    ("o3-mini", price(1.1, 4.4, None, Some(0.55))),
    ("claude-opus-4-7", price(5.0, 25.0, Some(6.25), Some(0.5))),
    ("claude-opus-4-6", price(5.0, 25.0, Some(6.25), Some(0.5))),
    ("claude-sonnet-4-6", price(3.0, 15.0, Some(3.75), Some(0.3))),
    ("claude-haiku-4-5", price(1.0, 5.0, Some(1.25), Some(0.1))),
];

pub const ZERO_COST_PREFIXES: &[&str] = &["vllm:", "sglang:", "ollama:", "local:"];

pub fn lookup(model_id: &str) -> Option<ModelPrice> {
    MODEL_PRICES
        .iter()
        .find(|(id, _)| *id == model_id)
        .map(|(_, p)| *p)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_write_tokens: u64,
    pub cache_read_tokens: u64,
}

impl TokenUsage {
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens + self.cache_write_tokens + self.cache_read_tokens
    }
}

pub fn estimate_cost_usd(model_id: &str, usage: TokenUsage) -> f64 {
    if ZERO_COST_PREFIXES.iter().any(|p| model_id.starts_with(p)) {
        return 0.0;
    }
    let Some(price) = lookup(model_id) else {
        return 0.0;
    };
    let mtok = 1_000_000f64;
    let mut total = (usage.input_tokens as f64 / mtok) * price.input_per_mtok;
    total += (usage.output_tokens as f64 / mtok) * price.output_per_mtok;
    if let Some(w) = price.cache_write_per_mtok {
        total += (usage.cache_write_tokens as f64 / mtok) * w;
    }
    if let Some(r) = price.cache_read_per_mtok {
        total += (usage.cache_read_tokens as f64 / mtok) * r;
    }
    round_to(total, 6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_hosted_is_free() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        assert_eq!(estimate_cost_usd("sglang:Qwen/Qwen3-8B", usage), 0.0);
        assert_eq!(estimate_cost_usd("unknown-model", usage), 0.0);
    }

    #[test]
    fn list_price_applies() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        assert_eq!(estimate_cost_usd("gpt-4.1-mini", usage), 2.0);
        let cached = TokenUsage {
            cache_read_tokens: 1_000_000,
            ..Default::default()
        };
        assert_eq!(estimate_cost_usd("claude-haiku-4-5", cached), 0.1);
    }
}
