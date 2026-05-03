from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from application.coordinator import LocalRolloutCoordinator
from application.eval import DEFAULT_EVAL_TASKS, EvalHarness, EvalTask
from application.event_bus import EventBus
from domain.models import AdapterRecord
from infrastructure.adapters.local import LocalAdapterRegistry
from infrastructure.policy.static import StaticPolicyServer
from infrastructure.rewards.composite import CompositeRewardPipeline
from infrastructure.store.memory import InMemoryArtifactStore
from infrastructure.tools.local import LocalToolHarness
from infrastructure.training.memory_store import InMemoryTrainingStore
from tests.conftest import _StubRepoRunner


@pytest.fixture
def harness(tmp_path: Path):
    bus = EventBus()
    coord = LocalRolloutCoordinator(
        tool_harness=LocalToolHarness(root=tmp_path),
        policy_server=StaticPolicyServer(),
        reward_pipeline=CompositeRewardPipeline(),
        artifact_store=InMemoryArtifactStore(),
        event_bus=bus,
        workspace_root=tmp_path,
        max_parallel=1,
        repo_runner=_StubRepoRunner(),  # type: ignore[arg-type]
    )
    registry = LocalAdapterRegistry(root=tmp_path / "adapters")
    registry.register(
        AdapterRecord(id="adapter-x", base_model="vllm:Qwen/Qwen3-0.6B", path="(filled)")
    )
    training_store = InMemoryTrainingStore()
    return EvalHarness(
        coordinator=coord,
        training_store=training_store,
        adapter_registry=registry,
        event_bus=bus,
    ), bus, registry, training_store, coord


@pytest.mark.asyncio
async def test_eval_harness_runs_default_tasks_and_records_score(harness) -> None:
    h, bus, registry, training_store, _ = harness
    captured = []

    async def pump():
        async for e in bus.subscribe(replay=False):
            captured.append(e)
            if e.kind == "eval.completed":
                return

    pump_task = asyncio.create_task(pump())
    await asyncio.sleep(0)

    report = await h.run("adapter-x")
    await asyncio.wait_for(pump_task, timeout=3)

    assert report.adapter_id == "adapter-x"
    assert report.task_set == "builtin"
    assert len(report.per_task) == len(DEFAULT_EVAL_TASKS)
    assert 0.0 <= report.mean_reward <= 1.0
    # Adapter eval_score updated to match.
    assert registry.get("adapter-x").eval_score == report.mean_reward
    # Persisted report and event emitted.
    assert training_store.list_eval_reports("adapter-x") == [report]
    kinds = [e.kind for e in captured]
    assert kinds[-1] == "eval.completed"


@pytest.mark.asyncio
async def test_eval_harness_uses_custom_task_set(harness) -> None:
    h, *_ = harness
    custom = [
        EvalTask(id="solo", prompt="alone", success_criteria=["lonely"], horizon=2),
    ]
    report = await h.run("adapter-x", tasks=custom, task_set="custom-v1")
    assert report.task_set == "custom-v1"
    assert [p.task_id for p in report.per_task] == ["solo"]


@pytest.mark.asyncio
async def test_eval_harness_rejects_unknown_adapter(harness) -> None:
    h, *_ = harness
    with pytest.raises(ValueError):
        await h.run("adapter-not-real")


@pytest.mark.asyncio
async def test_rollout_request_adapter_id_propagates_to_manifest(harness) -> None:
    h, _, _, _, coord = harness
    await h.run("adapter-x", tasks=[EvalTask(id="x", prompt="p", success_criteria=["s"], horizon=2)])
    runs = coord.artifact_store.list_runs()
    # All eval rollouts should be stamped with adapter_id="adapter-x".
    assert all(r.adapter_id == "adapter-x" for r in runs)
