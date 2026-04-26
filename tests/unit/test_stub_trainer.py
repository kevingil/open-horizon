from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rl_stack.domain.models import (
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
    TrajectoryRecord,
)
from rl_stack.infrastructure.adapters.local import LocalAdapterRegistry
from rl_stack.infrastructure.training.stub import StubTrainer


def _detail(reward: float) -> RunDetail:
    task = TaskSpec(
        id="t", prompt="p", repo_snapshot=".",
        tool_permissions=[ToolPermission.read], horizon=2, success_criteria=["ok"],
    )
    traj = TrajectoryRecord(id="tr", task_id=task.id, steps=[])
    rew = RewardRecord(trajectory_id=traj.id, terminal_reward=reward, step_rewards=[], provenance="t")
    manifest = RunManifest(
        id="run-x", model_id="m", dataset_slice="s", infra_target="mac-local",
        seed=1, status=RunStatus.completed, estimated_cost_usd=0.01,
    )
    return RunDetail(
        manifest=manifest, task=task, trajectory=traj, reward=rew,
        artifacts=[ArtifactRecord(name="a", kind="manifest", path="p")],
    )


def test_stub_trainer_emits_decreasing_loss(tmp_path: Path) -> None:
    registry = LocalAdapterRegistry(root=tmp_path / "adapters")
    trainer = StubTrainer()
    record = TrainingRunRecord(id="trun-1", hyperparams={"steps": 6})
    samples = [_detail(0.5), _detail(0.3)]

    points: list[TrainingMetricPoint] = []
    adapter = trainer.train(record, samples, parent=None, on_metric=points.append, adapters=registry)

    assert len(points) == 6
    losses = [p.loss for p in points]
    assert losses[0] > losses[-1], "loss should decrease over the synthetic curve"
    assert all(p.kl >= 0 for p in points)
    assert all(0 <= p.mean_reward <= 1 for p in points if p.mean_reward is not None)
    assert adapter.training_run_id == "trun-1"
    assert adapter.parent_id is None
    assert "stub" in adapter.tags


def test_stub_trainer_copies_parent_weights(tmp_path: Path) -> None:
    registry = LocalAdapterRegistry(root=tmp_path / "adapters")
    parent_dir = registry.path_for("adapter-parent")
    parent_dir.mkdir(parents=True)
    (parent_dir / "weights.bin").write_bytes(b"PARENT")
    parent = registry.register(
        AdapterRecord(
            id="adapter-parent",
            base_model="vllm:Qwen/Qwen2.5-0.5B",
            path=str(parent_dir),
            created_at=datetime.now(UTC),
        )
    )

    trainer = StubTrainer()
    record = TrainingRunRecord(id="trun-1", hyperparams={"steps": 2}, adapter_in=parent.id)
    child = trainer.train(record, samples=[], parent=parent, on_metric=lambda _: None, adapters=registry)

    assert child.parent_id == parent.id
    assert child.base_model == parent.base_model
    copied = Path(child.path) / "weights.bin"
    assert copied.exists() and copied.read_bytes() == b"PARENT"


def test_stub_trainer_is_deterministic_for_same_inputs(tmp_path: Path) -> None:
    """Same hyperparams + samples → same metric curve. Adapter id differs (uuid),
    weights are stubs, but loss/mean_reward/kl values are stable."""
    registry = LocalAdapterRegistry(root=tmp_path / "adapters")
    trainer = StubTrainer()
    record = TrainingRunRecord(id="trun-1", hyperparams={"steps": 4})
    samples = [_detail(0.5)]
    pts_a: list[TrainingMetricPoint] = []
    pts_b: list[TrainingMetricPoint] = []
    trainer.train(record, samples, parent=None, on_metric=pts_a.append, adapters=registry)
    trainer.train(record, samples, parent=None, on_metric=pts_b.append, adapters=registry)
    assert [p.loss for p in pts_a] == [p.loss for p in pts_b]
    assert [p.mean_reward for p in pts_a] == [p.mean_reward for p in pts_b]
