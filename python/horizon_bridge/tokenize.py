"""Tokenizer-backed turn recorder.

Tokenizes prompt and completion separately and returns the concatenated
ids with an attention mask (1 over every token) and a loss mask (1 over
completion tokens only). Padding and truncation are the trainer's call.
"""
from __future__ import annotations

from typing import Any

_TOKENIZERS: dict[str, Any] = {}


def _load(name: str) -> Any:
    if name in _TOKENIZERS:
        return _TOKENIZERS[name]
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tok = AutoTokenizer.from_pretrained(name)
    _TOKENIZERS[name] = tok
    return tok


def encode_pair(tokenizer: Any, prompt: str, completion: str) -> dict[str, Any]:
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    completion_ids = list(tokenizer.encode(completion, add_special_tokens=False))
    all_ids = prompt_ids + completion_ids
    return {
        "prompt_ids": all_ids,
        "completion_ids": [],  # consumers use loss_mask over the full sequence
        "attention_mask": [1] * len(all_ids),
        "loss_mask": [0] * len(prompt_ids) + [1] * len(completion_ids),
        "token_count": len(all_ids),
    }


def tokenize_pair(params: dict[str, Any]) -> dict[str, Any]:
    tokenizer = params.get("_tokenizer") or _load(str(params["tokenizer"]))
    return encode_pair(tokenizer, str(params.get("prompt", "")), str(params.get("completion", "")))
