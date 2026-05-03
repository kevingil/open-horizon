from __future__ import annotations

from pathlib import Path

import pytest

from application.coordinator import LocalRolloutCoordinator
from application.event_bus import EventBus
from domain.events import DomainEvent
from infrastructure.environment.simulated import SimulatedEnvironmentRunner
from infrastructure.policy.static import StaticPolicyServer
from infrastructure.rewards.heuristic import HeuristicRewardPipeline
from infrastructure.store.memory import InMemoryArtifactStore
from infrastructure.tools.local import LocalToolHarness


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def coordinator(tmp_path: Path, event_bus: EventBus) -> LocalRolloutCoordinator:
    return LocalRolloutCoordinator(
        environment_runner=SimulatedEnvironmentRunner(),
        tool_harness=LocalToolHarness(root=tmp_path),
        policy_server=StaticPolicyServer(),
        reward_pipeline=HeuristicRewardPipeline(),
        artifact_store=InMemoryArtifactStore(),
        event_bus=event_bus,
        workspace_root=tmp_path,
        max_parallel=2,
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
