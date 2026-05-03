from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from application.coordinator import LocalRolloutCoordinator
from application.event_bus import EventBus
from domain.models import (
    ArtifactRecord,
    RewardRecord,
    RolloutRequest,
    RunDetail,
    RunManifest,
    RunStatus,
    TaskSpec,
    ToolPermission,
    TrajectoryRecord,
)
from tests.conftest import _StubRepoRunner
from infrastructure.policy.static import StaticPolicyServer
from infrastructure.rewards.composite import CompositeRewardPipeline
from infrastructure.store.memory import InMemoryArtifactStore
from infrastructure.tools.local import LocalToolHarness
from interface.api.app import create_app
from settings import Settings


def _seed_detail(cost: float, at: datetime) -> RunDetail:
    task = TaskSpec(
        id=f"t-{cost}",
        prompt="seed",
        repo_snapshot=".",
        tool_permissions=[ToolPermission.read],
        horizon=1,
        success_criteria=["seed"],
    )
    traj = TrajectoryRecord(id=f"tr-{cost}", task_id=task.id, steps=[])
    reward = RewardRecord(trajectory_id=traj.id, terminal_reward=0.1, step_rewards=[], provenance="seed")
    manifest = RunManifest(
        id=f"seed-{int(cost * 1000)}",
        model_id="seed", dataset_slice="s",
        infra_target="mac-local", seed=1,
        status=RunStatus.completed,
        created_at=at, updated_at=at,
        estimated_cost_usd=cost,
    )
    return RunDetail(
        manifest=manifest, task=task, trajectory=traj, reward=reward,
        artifacts=[ArtifactRecord(name="a", kind="manifest", path="p")],
    )


@pytest.mark.asyncio
async def test_coordinator_rejects_rollout_when_cap_reached(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    bus = EventBus()
    now = datetime.now(UTC)
    store.save_run(_seed_detail(0.40, now - timedelta(hours=2)))
    store.save_run(_seed_detail(0.35, now - timedelta(hours=1)))

    coord = LocalRolloutCoordinator(
        tool_harness=LocalToolHarness(root=tmp_path),
        policy_server=StaticPolicyServer(),
        reward_pipeline=CompositeRewardPipeline(),
        artifact_store=store,
        event_bus=bus,
        workspace_root=tmp_path,
        max_parallel=1,
        daily_budget_usd=0.5,
        budget_window_hours=24,
        repo_runner=_StubRepoRunner(),  # type: ignore[arg-type]
    )

    captured: list = []

    async def pump():
        async for event in bus.subscribe(replay=False):
            captured.append(event)
            if event.kind in {"rollout.completed", "rollout.failed"}:
                return

    task = asyncio.create_task(pump())
    await asyncio.sleep(0)

    detail = await coord.start_rollout(
        RolloutRequest(prompt="x", horizon=2), run_id="run-budget",
    )
    await asyncio.wait_for(task, timeout=2)

    assert detail.manifest.status == RunStatus.failed
    assert "budget exceeded" in detail.trajectory.errors[0]
    kinds = [e.kind for e in captured]
    assert "budget.exceeded" in kinds
    assert "rollout.failed" in kinds


@pytest.mark.asyncio
async def test_coordinator_allows_rollout_when_under_cap(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    coord = LocalRolloutCoordinator(
        tool_harness=LocalToolHarness(root=tmp_path),
        policy_server=StaticPolicyServer(),
        reward_pipeline=CompositeRewardPipeline(),
        artifact_store=store,
        event_bus=EventBus(),
        workspace_root=tmp_path,
        max_parallel=1,
        daily_budget_usd=10.0,  # well above anything the stub repo runner produces
        budget_window_hours=24,
        repo_runner=_StubRepoRunner(),  # type: ignore[arg-type]
    )
    detail = await coord.start_rollout(
        RolloutRequest(prompt="fine", horizon=2), run_id="run-fine",
    )
    assert detail.manifest.status == RunStatus.completed


@pytest.mark.asyncio
async def test_api_budget_endpoint_reports_status(tmp_path: Path) -> None:
    app = create_app(Settings(workspace_root=tmp_path, daily_budget_usd=1.0))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/budget")
        assert r.status_code == 200
        body = r.json()
        assert body["cap_usd"] == 1.0
        assert body["window_hours"] == 24.0
        assert body["exceeded"] is False
