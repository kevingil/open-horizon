from __future__ import annotations

import asyncio

import pytest

from rl_stack.application.coordinator import LocalRolloutCoordinator
from rl_stack.domain.events import DomainEvent
from rl_stack.domain.models import RolloutRequest, RunStatus


async def _collect_until_completed(bus) -> tuple[list[DomainEvent], asyncio.Task]:
    got: list[DomainEvent] = []

    async def pump() -> None:
        async for event in bus.subscribe(replay=False):
            got.append(event)
            if event.kind in {"rollout.completed", "rollout.failed"}:
                return

    task = asyncio.create_task(pump())
    await asyncio.sleep(0)
    return got, task


@pytest.mark.asyncio
async def test_start_rollout_emits_lifecycle_events(
    coordinator: LocalRolloutCoordinator,
) -> None:
    got, pump_task = await _collect_until_completed(coordinator.event_bus)
    detail = await coordinator.start_rollout(
        RolloutRequest(prompt="p", repo_snapshot=".", horizon=2),
    )
    await asyncio.wait_for(pump_task, timeout=2)

    assert detail.manifest.status == RunStatus.completed
    kinds = [e.kind for e in got]
    assert "rollout.started" in kinds
    assert "step.recorded" in kinds
    assert "reward.computed" in kinds
    assert "rollout.completed" in kinds


@pytest.mark.asyncio
async def test_parallel_rollouts_are_isolated(
    coordinator: LocalRolloutCoordinator,
) -> None:
    requests = [RolloutRequest(prompt=f"p{i}", horizon=2) for i in range(3)]
    details = await asyncio.gather(*(coordinator.start_rollout(r) for r in requests))
    ids = [d.manifest.id for d in details]
    assert len(set(ids)) == 3
    # Each trajectory belongs to its own task.
    assert all(d.trajectory.task_id == d.task.id for d in details)


@pytest.mark.asyncio
async def test_bootstrap_seeds_runs_only_once(
    coordinator: LocalRolloutCoordinator,
) -> None:
    await coordinator.bootstrap()
    after_first = len(coordinator.artifact_store.list_runs())
    await coordinator.bootstrap()
    after_second = len(coordinator.artifact_store.list_runs())
    assert after_first == after_second
    assert after_first >= 2
