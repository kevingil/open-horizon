"""Unit tests for the prime-rl trainer's static helpers.

The CLI subprocess + GPU dependency tests live in tests/smoke/; here we
just exercise the parts that turn rollouts + turn_training rows into
the inputs prime-rl will consume.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from domain.models import (
    AdapterRecord,
    ArtifactRecord,
    RewardRecord,
    RunDetail,
    RunManifest,
    RunStatus,
    TaskSpec,
    ToolPermission,
    TrainingMetricPoint,
    TrainingRunRecord,
    TrainingStatus,
    TrajectoryRecord,
    TurnTrainingRecord,
    utc_now,
)
from infrastructure.adapters.local import LocalAdapterRegistry
from infrastructure.store.memory import InMemoryArtifactStore
from infrastructure.training.prime_rl import (
    DEFAULT_METRIC_PATTERN,
    PrimeRLTrainer,
)


def _seed_run_with_turns(store: InMemoryArtifactStore, run_id: str, *, turns: int) -> RunDetail:
    task = TaskSpec(
        id=f"task-{run_id}", prompt="solve",
        repo_snapshot=".", tool_permissions=[ToolPermission.read],
        horizon=turns, success_criteria=["done"],
    )
    traj = TrajectoryRecord(id=f"traj-{run_id}", task_id=task.id, steps=[])
    reward = RewardRecord(
        trajectory_id=traj.id, terminal_reward=0.7, step_rewards=[],
        provenance="test",
    )
    manifest = RunManifest(
        id=run_id, model_id="openai:test", dataset_slice="s",
        infra_target="mac-local", seed=1, status=RunStatus.completed,
        estimated_cost_usd=0.0,
    )
    detail = RunDetail(
        manifest=manifest, task=task, trajectory=traj, reward=reward,
        artifacts=[ArtifactRecord(name="m", kind="manifest", path="p")],
    )
    store.save_run(detail)
    for i in range(turns):
        store.save_turn_training(
            TurnTrainingRecord(
                id=f"tt-{run_id}-{i}", run_id=run_id, step_index=i * 2,
                prompt_ids=[1, 2, 3], completion_ids=[],
                attention_mask=[1, 1, 1, 1, 1],
                loss_mask=[0, 0, 0, 1, 1],
                sampling_args={"max_tokens": 64},
                model_name="gpt-5.4-mini", token_count=5,
            ),
        )
    return detail


def test_dataset_writer_emits_one_jsonl_line_per_turn(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    sample = _seed_run_with_turns(store, "run-A", turns=3)
    trainer = PrimeRLTrainer(artifact_store=store)
    out = tmp_path / "ds.jsonl"
    n = trainer._write_dataset([sample], out)
    assert n == 3
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["run_id"] == "run-A"
    assert first["reward"] == pytest.approx(0.7)
    assert first["loss_mask"] == [0, 0, 0, 1, 1]
    assert first["model_name"] == "gpt-5.4-mini"


def test_dataset_writer_handles_runs_without_turn_training_rows(tmp_path: Path) -> None:
    """Sample runs without recorded turn_training rows just get skipped
    in the dataset rather than blowing up the trainer."""
    store = InMemoryArtifactStore()
    bare = RunDetail(
        manifest=RunManifest(
            id="run-bare", model_id="m", dataset_slice="s",
            infra_target="t", seed=1, status=RunStatus.completed,
        ),
        task=TaskSpec(
            id="t-bare", prompt="x", repo_snapshot=".",
            tool_permissions=[ToolPermission.read], horizon=1,
            success_criteria=["x"],
        ),
        trajectory=TrajectoryRecord(id="tr-bare", task_id="t-bare", steps=[]),
        reward=RewardRecord(
            trajectory_id="tr-bare", terminal_reward=0.0, step_rewards=[],
            provenance="x",
        ),
        artifacts=[],
    )
    trainer = PrimeRLTrainer(artifact_store=store)
    out = tmp_path / "ds.jsonl"
    n = trainer._write_dataset([bare], out)
    assert n == 0
    assert out.read_text() == ""


def test_render_config_falls_back_to_minimal_default(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    trainer = PrimeRLTrainer(
        artifact_store=store, base_model="Qwen/Qwen3-8B",
    )
    run = TrainingRunRecord(
        id="trun-1", status=TrainingStatus.running,
        adapter_in=None, adapter_out=None, sample_run_ids=["run-A"],
        hyperparams={"learning_rate": 1e-5, "batch_size": 4, "use_lora": True},
        metrics=[], created_at=utc_now(), updated_at=utc_now(),
    )
    cfg = trainer._render_config(
        workdir=tmp_path,
        dataset_path=tmp_path / "ds.jsonl",
        output_dir=tmp_path / "out",
        run=run,
        parent=None,
    )
    body = cfg.read_text()
    assert 'base_model = "Qwen/Qwen3-8B"' in body
    assert 'training_run_id = "trun-1"' in body
    assert "learning_rate = 1e-05" in body
    assert "batch_size = 4" in body
    assert "use_lora = true" in body


def test_render_config_uses_template_substitutions(tmp_path: Path) -> None:
    template = tmp_path / "tpl.toml"
    template.write_text(
        "policy.model = '{{base_model}}'\n"
        "data.path = '{{dataset_path}}'\n"
        "out = '{{output_dir}}'\n"
        "parent = '{{parent_adapter_id}}'\n",
    )
    store = InMemoryArtifactStore()
    trainer = PrimeRLTrainer(artifact_store=store, config_template_path=template)
    run = TrainingRunRecord(
        id="trun-2", sample_run_ids=[], hyperparams={},
        created_at=utc_now(), updated_at=utc_now(),
    )
    parent = AdapterRecord(
        id="parent-x", base_model="Qwen/Qwen3-8B", path=str(tmp_path / "parent"),
    )
    cfg = trainer._render_config(
        workdir=tmp_path,
        dataset_path=tmp_path / "ds.jsonl",
        output_dir=tmp_path / "out",
        run=run,
        parent=parent,
    )
    body = cfg.read_text()
    assert "Qwen/Qwen3-8B" in body
    assert str(tmp_path / "ds.jsonl") in body
    assert str(tmp_path / "out") in body
    assert "parent-x" in body


def test_default_metric_pattern_extracts_step_and_loss() -> None:
    # Primary pattern locates step + loss; reward + kl come from
    # secondary regexes inside _stream_metrics so any one of them
    # missing on a given line doesn't drop the whole metric.
    line = "step=42  loss=0.3142  reward=0.71  kl=0.0024"
    match = DEFAULT_METRIC_PATTERN.search(line)
    assert match is not None
    assert match.group("step") == "42"
    assert match.group("loss") == "0.3142"


def test_default_metric_pattern_skips_non_metric_lines() -> None:
    for line in (
        "Loading dataset...",
        "[INFO] checkpoint saved",
        "Traceback (most recent call last):",
    ):
        assert DEFAULT_METRIC_PATTERN.search(line) is None


def test_stream_metrics_emits_only_on_matches() -> None:
    store = InMemoryArtifactStore()
    trainer = PrimeRLTrainer(
        artifact_store=store,
        metric_pattern=re.compile(
            r"step=(?P<step>\d+) loss=(?P<loss>[\d.]+)",
        ),
    )
    captured: list[TrainingMetricPoint] = []

    class _StreamLike:
        def __iter__(self):
            yield from [
                "Loading...\n",
                "step=0 loss=1.0\n",
                "step=1 loss=0.5\n",
                "warning: foo\n",
            ]

    trainer._stream_metrics(_StreamLike(), captured.append)
    assert [p.step for p in captured] == [0, 1]
    assert captured[0].loss == 1.0


def test_train_rejects_empty_sample_set(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    trainer = PrimeRLTrainer(artifact_store=store)
    registry = LocalAdapterRegistry(root=tmp_path / "adapters")
    run = TrainingRunRecord(
        id="trun-empty", sample_run_ids=[], hyperparams={},
        created_at=utc_now(), updated_at=utc_now(),
    )
    with pytest.raises(RuntimeError, match="at least one sample"):
        trainer.train(run, [], None, lambda _p: None, registry)
