from __future__ import annotations

import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from rl_stack.interface.api.app import create_app
from rl_stack.settings import Settings


@pytest.fixture
def app(tmp_path):
    return create_app(Settings(workspace_root=tmp_path, max_parallel_rollouts=2))


@pytest.mark.asyncio
async def test_health_endpoint(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_create_run_accepts_and_emits_events(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with client.stream("GET", "/api/dashboard") as r:
            assert r.status_code == 200

        services = app.state.services
        captured: list = []

        async def pump():
            async for event in services.event_bus.subscribe(replay=False):
                captured.append(event)
                if any(e.kind == "rollout.completed" for e in captured):
                    return

        task = asyncio.create_task(pump())
        await asyncio.sleep(0)

        resp = await client.post(
            "/api/runs",
            json={"prompt": "hello", "repo_snapshot": ".", "horizon": 2},
        )
        assert resp.status_code == 202

        await asyncio.wait_for(task, timeout=3)
        kinds = [e.kind for e in captured]
        assert "rollout.started" in kinds
        assert "rollout.completed" in kinds


@pytest.mark.asyncio
async def test_websocket_streams_domain_events(app) -> None:
    from fastapi.testclient import TestClient

    with TestClient(app) as client, client.websocket_connect("/ws/events") as ws:
        client.post("/api/runs", json={"prompt": "ws-test", "horizon": 1})
        kinds: list[str] = []
        for _ in range(12):
            data = ws.receive_text()
            kinds.append(json.loads(data)["kind"])
            if "rollout.completed" in kinds:
                break
        assert "rollout.started" in kinds
