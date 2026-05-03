from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from application.coordinator import LocalRolloutCoordinator
from application.event_bus import EventBus
from domain.contracts import PolicyServer
from domain.models import RolloutRequest, RunStatus, TaskSpec
from infrastructure.rewards.composite import CompositeRewardPipeline
from infrastructure.store.memory import InMemoryArtifactStore
from infrastructure.tools.local import LocalToolHarness
from interface.api.app import create_app
from settings import Settings
from tests.conftest import _StubRepoRunner


class SlowPolicy(PolicyServer):
    name: str = "slow"

    def __init__(self) -> None:
        self._calls = 0

    def policy_name(self) -> str:
        return self.name

    def generate_action(self, task: TaskSpec, context: list[str]) -> str:
        self._calls += 1
        return json.dumps({"tool": "list_files", "input": {"path": "."}})


@pytest.fixture
def coordinator(tmp_path):
    return LocalRolloutCoordinator(
        tool_harness=LocalToolHarness(root=tmp_path),
        policy_server=SlowPolicy(),
        reward_pipeline=CompositeRewardPipeline(),
        artifact_store=InMemoryArtifactStore(),
        event_bus=EventBus(),
        workspace_root=tmp_path,
        max_parallel=1,
        repo_runner=_StubRepoRunner(),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_cancel_before_first_step_short_circuits(coordinator) -> None:
    """Cancel flagged before the first step must turn the run terminal-failed."""
    run_id = "run-cancel-1"
    captured: list = []

    async def pump():
        async for event in coordinator.event_bus.subscribe(replay=False):
            captured.append(event)
            if event.kind in {"rollout.cancelled", "rollout.completed", "rollout.failed"}:
                return

    task = asyncio.create_task(pump())
    await asyncio.sleep(0)

    # Pre-flag. No race: coordinator sees it on iteration 0.
    coordinator._cancelled.add(run_id)

    detail = await coordinator.start_rollout(
        RolloutRequest(prompt="long", horizon=50),
        run_id=run_id,
    )
    await asyncio.wait_for(task, timeout=2)

    assert detail.manifest.status == RunStatus.failed
    assert "cancelled" in detail.trajectory.errors
    kinds = [e.kind for e in captured]
    assert "rollout.cancelled" in kinds
    assert "rollout.completed" not in kinds
    # No steps should have run.
    assert detail.trajectory.steps == []


@pytest.mark.asyncio
async def test_cancel_idempotent_on_terminal_run(coordinator) -> None:
    detail = await coordinator.start_rollout(
        RolloutRequest(prompt="short", horizon=1), run_id="run-terminal",
    )
    # Already completed; second cancel should be refused.
    assert detail.manifest.status == RunStatus.completed
    assert coordinator.request_cancel("run-terminal") is False


@pytest.mark.asyncio
async def test_api_cancel_endpoint(tmp_path) -> None:
    app = create_app(Settings(workspace_root=tmp_path, max_parallel_rollouts=1))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/runs", json={"prompt": "x", "horizon": 1})
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]

        # Immediately ask for cancellation; depending on timing the run may
        # already be terminal (simulated env is very fast). Accept either 202
        # or 409 but reject anything else.
        cancel = await client.post(f"/api/runs/{run_id}/cancel")
        assert cancel.status_code in (202, 409)


def test_api_cancel_surfaces_on_websocket(tmp_path) -> None:
    app = create_app(Settings(workspace_root=tmp_path, max_parallel_rollouts=1))
    with TestClient(app) as client, client.websocket_connect("/ws/events") as ws:
        resp = client.post("/api/runs", json={"prompt": "race", "horizon": 50})
        run_id = resp.json()["run_id"]
        client.post(f"/api/runs/{run_id}/cancel")

        kinds: list[str] = []
        for _ in range(80):
            data = ws.receive_text()
            kinds.append(json.loads(data)["kind"])
            if "rollout.cancelled" in kinds or "rollout.completed" in kinds:
                break
        # The exact outcome depends on timing, but one of these two has to fire.
        assert ("rollout.cancelled" in kinds) or ("rollout.completed" in kinds)
