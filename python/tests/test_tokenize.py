from __future__ import annotations

from horizon_bridge.tokenize import tokenize_pair


class _CharTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


def test_masks_split_prompt_and_completion() -> None:
    out = tokenize_pair({"_tokenizer": _CharTokenizer(), "prompt": "ab", "completion": "cd"})
    assert out["prompt_ids"] == [97, 98, 99, 100]
    assert out["attention_mask"] == [1, 1, 1, 1]
    assert out["loss_mask"] == [0, 0, 1, 1]
    assert out["token_count"] == 4
