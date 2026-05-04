"""Per-turn token-level recorder for trainer-ready trajectories.

Phase E ships an optional `TrainingRecorder` slot on the coordinator.
When wired, every turn produces a `TurnTrainingRecord` (prompt_ids,
completion_ids, attention_mask, loss_mask, sampling_args, model_name)
which gets persisted alongside (but separate from) the trajectory in
the ArtifactStore. The trajectory itself just gains a cheap
`has_training_metadata` flag per step.

Two implementations:
  - `NullTrainingRecorder`: no-op default; what every test gets.
  - `TokenizedTrainingRecorder(tokenizer)`: real tokenizer-backed
    impl that produces ids/masks for the trainer to consume. Lives
    behind the `[train]` extra so the default install stays Mac-runnable
    without HF transformers in scope.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from domain.contracts import ArtifactStore
from domain.models import TurnTrainingRecord


class TrainingRecorder(ABC):
    """Capture trainer-ready data per turn.

    Called from the coordinator once per (action, observation) turn pair
    after the LLM completion has come back. Implementations are free to
    no-op when the run isn't training data, or to tokenize and persist.
    """

    @abstractmethod
    def record(
        self,
        *,
        run_id: str,
        step_index: int,
        prompt_text: str,
        completion_text: str,
        sampling_args: dict[str, Any],
        model_name: str,
        store: ArtifactStore,
    ) -> TurnTrainingRecord | None:
        ...


@dataclass
class NullTrainingRecorder(TrainingRecorder):
    """No-op recorder. Default; never writes to the store."""

    def record(
        self,
        *,
        run_id: str,
        step_index: int,
        prompt_text: str,
        completion_text: str,
        sampling_args: dict[str, Any],
        model_name: str,
        store: ArtifactStore,
    ) -> TurnTrainingRecord | None:
        return None


@dataclass
class TokenizedTrainingRecorder(TrainingRecorder):
    """Tokenizer-backed recorder.

    Tokenizes the prompt and completion separately; emits a single
    concatenated id list for the trainer plus an attention mask (all 1s
    over real tokens) and a loss mask (1 over completion tokens, 0 over
    prompt). The split between prompt and completion is the contract
    GRPO-style trainers need: prompt_ids supervises nothing; completion
    tokens get the gradient.

    Pads / truncates internally are explicitly NOT done here - that's
    the trainer's call, not the recorder's.
    """

    # `tokenizer` is whatever the [train] extra provides; we duck-type
    # the slice we need so the import doesn't leak into the default
    # install (HF transformers stays optional).
    tokenizer: Any

    def record(
        self,
        *,
        run_id: str,
        step_index: int,
        prompt_text: str,
        completion_text: str,
        sampling_args: dict[str, Any],
        model_name: str,
        store: ArtifactStore,
    ) -> TurnTrainingRecord:
        prompt_ids = list(self.tokenizer.encode(prompt_text, add_special_tokens=False))
        completion_ids = list(
            self.tokenizer.encode(completion_text, add_special_tokens=False),
        )
        all_ids = prompt_ids + completion_ids
        attention_mask = [1] * len(all_ids)
        loss_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
        record = TurnTrainingRecord(
            id=f"turn-{uuid4().hex[:8]}",
            run_id=run_id,
            step_index=step_index,
            prompt_ids=all_ids,
            completion_ids=[],  # consumers should use `loss_mask` over the full sequence.
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            sampling_args=dict(sampling_args),
            model_name=model_name,
            token_count=len(all_ids),
        )
        store.save_turn_training(record)
        return record
