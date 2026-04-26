"""GRPO-lite trainer tests with a synthetic torch model.

We avoid downloading any HF checkpoint by passing a tiny hand-rolled
LM through the trainer's `model_loader` hook. The point isn't to verify
that LoRA/transformers integrate (that's what the smoke test is for) -
the point is to prove the advantage-weighted NLL step actually moves
the model toward high-advantage responses and away from low-advantage
ones, and that the metric callback fires once per step.
"""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from rl_stack.domain.models import (
    ArtifactRecord,
    RewardRecord,
    RunDetail,
    RunManifest,
    RunStatus,
    TaskSpec,
    ToolPermission,
    TrainingMetricPoint,
    TrainingRunRecord,
    TrajectoryRecord,
    TrajectoryStep,
)
from rl_stack.infrastructure.adapters.local import LocalAdapterRegistry
from rl_stack.infrastructure.training.grpo import (
    GrpoTrainer,
    TokenizerLike,
    _build_examples,
    _weighted_nll_step,
)

# --- Synthetic LM + tokenizer -------------------------------------------


class CharTokenizer(TokenizerLike):
    """One-character-per-token. Vocab is fixed-ASCII subset; padding happens
    naturally because we never pad - we just truncate."""

    vocab_size: int = 64

    def __init__(self) -> None:
        self.eos = 0  # \x00
        # Map any character to an int in [0, vocab_size); collisions are fine
        # for a synthetic test that just needs gradient flow.

    def _encode(self, text: str) -> list[int]:
        return [(ord(c) % (self.vocab_size - 1)) + 1 for c in text]

    def encode_pair(self, prompt: str, response: str) -> tuple[list[int], int]:
        prompt_ids = [*self._encode(prompt), self.eos]
        response_ids = self._encode(response)
        return [*prompt_ids, *response_ids], len(prompt_ids)


class TinyLM(nn.Module):
    """A 2-layer transformer-ish toy LM. Not realistic; just enough nonzero
    gradients on every parameter so the optimiser actually moves."""

    def __init__(self, vocab_size: int = 64, hidden: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, vocab_size)
        self.act = nn.GELU()

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor):
        x = self.embed(input_ids)
        x = self.lin2(self.act(self.lin1(x)))
        # Standard shifted CE: predict token n+1 given tokens up to n.
        shift_logits = x[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        class _Out:
            pass

        out = _Out()
        out.loss = loss
        out.logits = x
        return out


# --- Fixtures -----------------------------------------------------------


def _detail(task_id: str, action_text: str, reward: float) -> RunDetail:
    task = TaskSpec(
        id=task_id, prompt=f"Prompt {task_id}", repo_snapshot=".",
        tool_permissions=[ToolPermission.read], horizon=2,
        success_criteria=["ok"],
    )
    steps = [TrajectoryStep(index=0, actor="policy", kind="action", content=action_text)]
    traj = TrajectoryRecord(id=f"{task_id}-traj", task_id=task_id, steps=steps)
    rew = RewardRecord(trajectory_id=traj.id, terminal_reward=reward, step_rewards=[], provenance="t")
    manifest = RunManifest(
        id=f"run-{task_id}-{int(reward * 100)}",
        model_id="m", dataset_slice="s", infra_target="mac-local",
        seed=1, status=RunStatus.completed, estimated_cost_usd=0.0,
    )
    return RunDetail(
        manifest=manifest, task=task, trajectory=traj, reward=rew,
        artifacts=[ArtifactRecord(name="a", kind="manifest", path="p")],
    )


# --- Pure-function tests -------------------------------------------------


def test_build_examples_normalises_advantages_within_a_group() -> None:
    samples = [
        _detail("task-a", "high-reward action", reward=0.9),
        _detail("task-a", "low-reward action", reward=0.1),
        _detail("task-a", "mid-reward action", reward=0.5),
    ]
    examples = _build_examples(samples)
    assert len(examples) == 3
    advs = sorted(e.advantage for e in examples)
    # Mean ~ 0, std-normalised: extremes are roughly +1 / -1.
    assert advs[0] < -0.5 and advs[-1] > 0.5
    assert abs(sum(advs)) < 1e-6


def test_build_examples_groups_by_task_independently() -> None:
    samples = [
        _detail("task-a", "x", reward=0.9),
        _detail("task-a", "y", reward=0.1),
        _detail("task-b", "z", reward=0.5),  # group of 1
    ]
    examples = _build_examples(samples)
    by_resp = {e.response: e.advantage for e in examples}
    # Group-relative: task-a's "x" and "y" are normalised against each other.
    assert by_resp["x"] > 0 > by_resp["y"]
    # Single-sample group: weak signal centred at 0.5.
    assert by_resp["z"] == pytest.approx(0.0)


def test_build_examples_emits_one_per_policy_step() -> None:
    """Sample with multiple policy steps should produce multiple examples."""
    base = _detail("task-a", "step-1", reward=0.5)
    base.trajectory.steps.append(
        TrajectoryStep(index=2, actor="policy", kind="action", content="step-2"),
    )
    examples = _build_examples([base])
    assert {e.response for e in examples} == {"step-1", "step-2"}


# --- Inner training loop -------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_weighted_nll_descends_for_positive_advantage(seed: int) -> None:
    """A response with positive advantage should see its NLL drop after a
    gradient step. (Negative-advantage responses do the opposite, by
    construction of the loss.)"""
    torch.manual_seed(seed)
    model = TinyLM()
    tokenizer = CharTokenizer()

    from rl_stack.infrastructure.training.grpo import _Example

    pos = [_Example(prompt="ABC", response="XYZ", advantage=1.0)]
    # Measure NLL before the step, run the step, measure again.
    before = _nll_only(model, tokenizer, pos[0])
    loss, _ = _weighted_nll_step(model, tokenizer, pos, max_seq_len=64)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    opt.zero_grad()
    loss.backward()
    opt.step()
    after = _nll_only(model, tokenizer, pos[0])
    assert after < before


def test_weighted_nll_climbs_for_negative_advantage() -> None:
    torch.manual_seed(0)
    model = TinyLM()
    tokenizer = CharTokenizer()
    from rl_stack.infrastructure.training.grpo import _Example

    neg = [_Example(prompt="ABC", response="XYZ", advantage=-1.0)]
    before = _nll_only(model, tokenizer, neg[0])
    loss, _ = _weighted_nll_step(model, tokenizer, neg, max_seq_len=64)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    opt.zero_grad()
    loss.backward()
    opt.step()
    after = _nll_only(model, tokenizer, neg[0])
    assert after > before


def _nll_only(model, tokenizer, example) -> float:
    """Forward a single example, return the cross-entropy on response tokens."""
    ids, prompt_len = tokenizer.encode_pair(example.prompt, example.response)
    input_ids = torch.tensor([ids], dtype=torch.long)
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100
    out = model(input_ids=input_ids, labels=labels)
    return float(out.loss.item())


# --- Full trainer.train() with a synthetic loader ------------------------


def test_grpo_trainer_runs_full_pipeline_with_synthetic_model(tmp_path: Path) -> None:
    """Exercises every piece: model load (synthetic), optimizer setup,
    metric callback, adapter registration. Catches API drift between the
    abstract Trainer contract and what the orchestration layer expects."""
    registry = LocalAdapterRegistry(root=tmp_path / "adapters")

    def loader(_base_model: str, _opts: dict):
        return TinyLM(), CharTokenizer()

    saver_called: dict[str, Path] = {}

    def saver(_model, target: Path) -> None:
        (target / "adapter_marker.txt").write_text("synthetic-adapter")
        saver_called["target"] = target

    trainer = GrpoTrainer(
        model_loader=loader,
        adapter_saver=saver,
    )
    samples = [
        _detail("task-a", "high", reward=0.9),
        _detail("task-a", "low", reward=0.1),
    ]
    record = TrainingRunRecord(
        id="trun-1",
        hyperparams={"steps": 4, "batch_size": 2, "lr": 0.1},
    )

    metrics: list[TrainingMetricPoint] = []
    adapter = trainer.train(record, samples, parent=None, on_metric=metrics.append, adapters=registry)

    assert len(metrics) == 4
    assert all(p.loss == p.loss for p in metrics)  # not NaN
    assert "mean_advantage" in metrics[0].extra
    assert adapter.training_run_id == "trun-1"
    assert "grpo-lite" in adapter.tags
    assert saver_called["target"] == registry.path_for(adapter.id)
    assert (Path(adapter.path) / "adapter_marker.txt").read_text() == "synthetic-adapter"


def test_grpo_trainer_rejects_empty_sample_set() -> None:
    trainer = GrpoTrainer(model_loader=lambda *_: (TinyLM(), CharTokenizer()))
    record = TrainingRunRecord(id="trun-1")
    registry = LocalAdapterRegistry(root=Path("/tmp/never-created"))
    with pytest.raises(ValueError):
        trainer.train(record, samples=[], parent=None, on_metric=lambda _: None, adapters=registry)
