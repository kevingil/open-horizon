from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from application.coordinator import LocalRolloutCoordinator
from application.event_bus import EventBus
from domain.events import DomainEvent
from domain.models import TaskSpec
from infrastructure.policy.static import StaticPolicyServer
from infrastructure.rewards.heuristic import HeuristicRewardPipeline
from infrastructure.store.memory import InMemoryArtifactStore
from infrastructure.tools.local import LocalToolHarness


@dataclass
class _StubRepoRunner:
    """Test double for the repo_runner slot.

    Phase A removed SimulatedEnvironmentRunner + the EnvironmentRunner ABC,
    so the coordinator now requires either a verifiers external runner or
    a repo runner. Most coordinator tests exercise the in-house loop's
    lifecycle/budget/cancellation logic and don't need a real sandboxed
    repo - this stub is the minimum surface (create_task + step) the
    coordinator pokes at.
    """

    tasks: dict[str, TaskSpec] = field(default_factory=dict)

    def create_task(self, task: TaskSpec) -> TaskSpec:
        self.tasks[task.id] = task
        return task

    def step(self, task_id: str, action: str) -> str:
        return f"env:{task_id}:{action}"


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def coordinator(tmp_path: Path, event_bus: EventBus) -> LocalRolloutCoordinator:
    return LocalRolloutCoordinator(
        tool_harness=LocalToolHarness(root=tmp_path),
        policy_server=StaticPolicyServer(),
        reward_pipeline=HeuristicRewardPipeline(),
        artifact_store=InMemoryArtifactStore(),
        event_bus=event_bus,
        workspace_root=tmp_path,
        max_parallel=2,
        repo_runner=_StubRepoRunner(),  # type: ignore[arg-type]
    )


@pytest.fixture
async def drain(event_bus: EventBus) -> list[DomainEvent]:
    import asyncio
    import contextlib

    captured: list[DomainEvent] = []

    async def pump():
        async for event in event_bus.subscribe(replay=False):
            captured.append(event)

    task = asyncio.create_task(pump())
    yield captured
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
