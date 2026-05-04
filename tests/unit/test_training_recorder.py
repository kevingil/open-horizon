"""Unit tests for the Phase E TrainingRecorder."""
from __future__ import annotations

from infrastructure.store.memory import InMemoryArtifactStore
from infrastructure.training.recorder import (
    NullTrainingRecorder,
    TokenizedTrainingRecorder,
)


class _FakeTokenizer:
    """Tokenizer stub: maps every word to its length, every char to its
    ord(). Deterministic; lets the test assert exact id sequences."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


def test_null_recorder_returns_none_and_writes_nothing() -> None:
    store = InMemoryArtifactStore()
    recorder = NullTrainingRecorder()
    result = recorder.record(
        run_id="run-1", step_index=0,
        prompt_text="hi", completion_text="ok",
        sampling_args={}, model_name="stub", store=store,
    )
    assert result is None
    assert store.list_turn_training("run-1") == []


def test_tokenized_recorder_writes_record_and_loss_mask() -> None:
    store = InMemoryArtifactStore()
    recorder = TokenizedTrainingRecorder(tokenizer=_FakeTokenizer())
    result = recorder.record(
        run_id="run-1", step_index=2,
        prompt_text="ab", completion_text="cd",
        sampling_args={"max_tokens": 64},
        model_name="gpt-5.4-mini", store=store,
    )
    assert result is not None
    # 2 prompt chars + 2 completion chars
    assert len(result.attention_mask) == 4
    assert all(m == 1 for m in result.attention_mask)
    # loss_mask: 0 over prompt span, 1 over completion span
    assert result.loss_mask == [0, 0, 1, 1]
    # full ids = prompt ids ++ completion ids
    assert result.prompt_ids == [ord("a"), ord("b"), ord("c"), ord("d")]
    assert result.token_count == 4
    assert result.model_name == "gpt-5.4-mini"
    # Persisted in the store under (run_id, step_index)
    fetched = store.get_turn_training("run-1", 2)
    assert fetched is not None
    assert fetched.id == result.id
    assert fetched.loss_mask == result.loss_mask


def test_in_memory_store_lists_turns_in_step_order() -> None:
    store = InMemoryArtifactStore()
    recorder = TokenizedTrainingRecorder(tokenizer=_FakeTokenizer())
    for step in (4, 0, 2):
        recorder.record(
            run_id="run-1", step_index=step,
            prompt_text="x", completion_text="y",
            sampling_args={}, model_name="m", store=store,
        )
    rows = store.list_turn_training("run-1")
    assert [r.step_index for r in rows] == [0, 2, 4]
