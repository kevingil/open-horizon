from __future__ import annotations

import asyncio

import pytest

from application.coordinator import LocalRolloutCoordinator
from domain.events import DomainEvent
from domain.models import RolloutRequest, RunStatus


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
async def test_progress_ticks_include_cumulative_step_index(
    coordinator: LocalRolloutCoordinator,
) -> None:
    got, pump_task = await _collect_until_completed(coordinator.event_bus)
    await coordinator.start_rollout(RolloutRequest(prompt="p", horizon=3))
    await asyncio.wait_for(pump_task, timeout=2)

    progress = [e for e in got if e.kind == "progress.ticked"]
    assert len(progress) == 3
    # Step indices must be 0, 1, 2 in order and share the run_id.
    assert [e.step_index for e in progress] == [0, 1, 2]
    run_ids = {e.run_id for e in progress}
    assert len(run_ids) == 1


@pytest.mark.asyncio
async def test_bootstrap_returns_dashboard_without_seeding(
    coordinator: LocalRolloutCoordinator,
) -> None:
    # Phase A removed seed rollouts from bootstrap(); it's a no-op that
    # returns the (empty) dashboard. Demo content now comes from real
    # rollouts against a configured backend, not from a startup hook.
    snapshot = await coordinator.bootstrap()
    assert snapshot.runs == []
    assert coordinator.artifact_store.list_runs() == []
