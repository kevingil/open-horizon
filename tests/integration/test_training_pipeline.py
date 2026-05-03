from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from application.event_bus import EventBus
from application.training import TrainingRequest, TrainingService
from domain.events import DomainEvent
from domain.models import (
    ArtifactRecord,
    RewardRecord,
    RunDetail,
    RunManifest,
    RunStatus,
    TaskSpec,
    ToolPermission,
    TrainingStatus,
    TrajectoryRecord,
)
from infrastructure.adapters.local import LocalAdapterRegistry
from infrastructure.store.memory import InMemoryArtifactStore
from infrastructure.training.memory_store import InMemoryTrainingStore
from infrastructure.training.stub import StubTrainer


def _run_detail(run_id: str, reward: float = 0.5) -> RunDetail:
    task = TaskSpec(
        id=f"{run_id}-t", prompt="p", repo_snapshot=".",
        tool_permissions=[ToolPermission.read], horizon=2, success_criteria=["ok"],
    )
    traj = TrajectoryRecord(id=f"{run_id}-tr", task_id=task.id, steps=[])
    rew = RewardRecord(trajectory_id=traj.id, terminal_reward=reward, step_rewards=[], provenance="t")
    manifest = RunManifest(
        id=run_id, model_id="m", dataset_slice="s", infra_target="mac-local",
        seed=1, status=RunStatus.completed, estimated_cost_usd=0.01,
    )
    return RunDetail(
        manifest=manifest, task=task, trajectory=traj, reward=rew,
        artifacts=[ArtifactRecord(name="a", kind="manifest", path="p")],
    )


@pytest.fixture
def service(tmp_path: Path) -> tuple[TrainingService, EventBus, InMemoryArtifactStore, InMemoryTrainingStore, LocalAdapterRegistry]:
    bus = EventBus()
    rollouts = InMemoryArtifactStore()
    rollouts.save_run(_run_detail("run-1", reward=0.5))
    rollouts.save_run(_run_detail("run-2", reward=0.7))
    training_store = InMemoryTrainingStore()
    registry = LocalAdapterRegistry(root=tmp_path / "adapters")
    svc = TrainingService(
        trainer=StubTrainer(),
        artifact_store=rollouts,
        training_store=training_store,
        adapter_registry=registry,
        event_bus=bus,
    )
    return svc, bus, rollouts, training_store, registry


@pytest.mark.asyncio
async def test_training_pipeline_emits_lifecycle_and_publishes_adapter(service) -> None:
    svc, bus, _, training_store, registry = service

    captured: list[DomainEvent] = []

    async def pump():
        async for event in bus.subscribe(replay=False):
            captured.append(event)
            if event.kind == "training.completed":
                return

    pump_task = asyncio.create_task(pump())
    await asyncio.sleep(0)

    record = await svc.start_training(
        TrainingRequest(
            sample_run_ids=["run-1", "run-2"],
            hyperparams={"steps": 4},
        ),
    )
    await asyncio.wait_for(pump_task, timeout=3)

    kinds = [e.kind for e in captured]
    assert kinds[0] == "training.started"
    assert "training.metric" in kinds
    assert "adapter.published" in kinds
    assert kinds[-1] == "training.completed"

    # Persisted state matches what the trainer produced.
    persisted = training_store.get_training_run(record.id)
    assert persisted.status == TrainingStatus.completed
    assert persisted.adapter_out is not None
    assert len(persisted.metrics) == 4

    adapters = registry.list_adapters()
    assert len(adapters) == 1
    assert adapters[0].training_run_id == record.id


@pytest.mark.asyncio
async def test_training_pipeline_marks_failure_on_trainer_exception(tmp_path: Path) -> None:
    bus = EventBus()
    rollouts = InMemoryArtifactStore()
    training_store = InMemoryTrainingStore()
    registry = LocalAdapterRegistry(root=tmp_path / "adapters")

    class ExplodingTrainer(StubTrainer):
        def name(self) -> str:
            return "boom"

        def train(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("boom")

    svc = TrainingService(
        trainer=ExplodingTrainer(),
        artifact_store=rollouts,
        training_store=training_store,
        adapter_registry=registry,
        event_bus=bus,
    )

    captured: list = []

    async def pump():
        async for event in bus.subscribe(replay=False):
            captured.append(event)
            if event.kind in {"training.failed", "training.completed"}:
                return

    pump_task = asyncio.create_task(pump())
    await asyncio.sleep(0)

    record = await svc.start_training(TrainingRequest(sample_run_ids=[]))
    await asyncio.wait_for(pump_task, timeout=2)

    kinds = [e.kind for e in captured]
    assert "training.failed" in kinds
    persisted = training_store.get_training_run(record.id)
    assert persisted.status == TrainingStatus.failed
    assert "boom" in (persisted.error or "")
    assert registry.list_adapters() == []
