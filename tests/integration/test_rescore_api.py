from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from rl_stack.interface.api.app import create_app
from rl_stack.settings import Settings


@pytest.fixture
def app(tmp_path):
    return create_app(Settings(workspace_root=tmp_path))


@pytest.mark.asyncio
async def test_list_rubrics(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/rubrics")
        assert r.status_code == 200
        names = {entry["name"] for entry in r.json()}
        assert {"heuristic-v1", "coding-v1", "strict-finish-v1"}.issubset(names)


@pytest.mark.asyncio
async def test_rescore_changes_terminal_reward(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Kick off a rollout so there's something in the store.
        resp = await client.post("/api/runs", json={"prompt": "x", "horizon": 2})
        run_id = resp.json()["run_id"]

        # Wait for completion via repeated GET; simulated env is fast.
        for _ in range(50):
            got = await client.get(f"/api/runs/{run_id}")
            if got.status_code == 200 and got.json()["manifest"]["status"] == "completed":
                break
        else:
            pytest.fail("rollout did not complete")

        before = got.json()["reward"]["terminal_reward"]
        rescored = await client.post(f"/api/runs/{run_id}/rescore?rubric=strict-finish-v1")
        assert rescored.status_code == 200
        body = rescored.json()
        assert body["rubric"] == "strict-finish-v1"
        assert body["previous_terminal_reward"] == before

        refetched = await client.get(f"/api/runs/{run_id}")
        assert refetched.json()["reward"]["terminal_reward"] == body["new_terminal_reward"]


@pytest.mark.asyncio
async def test_rescore_unknown_rubric_returns_400(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/runs", json={"prompt": "x", "horizon": 1})
        run_id = resp.json()["run_id"]
        for _ in range(50):
            got = await client.get(f"/api/runs/{run_id}")
            if got.status_code == 200 and got.json()["manifest"]["status"] != "pending":
                break
        r = await client.post(f"/api/runs/{run_id}/rescore?rubric=not-a-rubric")
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_rescore_unknown_run_returns_404(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/runs/run-missing/rescore?rubric=heuristic-v1")
        assert r.status_code == 404


def test_replay_cli_list(capsys) -> None:
    from rl_stack.interface.cli.replay import main

    code = main(["--list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "heuristic-v1" in out
    assert "strict-finish-v1" in out
