from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from rl_stack.domain.models import AdapterRecord
from rl_stack.interface.api.app import create_app
from rl_stack.settings import Settings


@pytest.fixture
def app(tmp_path):
    return create_app(
        Settings(
            workspace_root=tmp_path,
            adapters_dir=tmp_path / "adapters",
            artifacts_dir=tmp_path / "artifacts",
        )
    )


async def _seed_run(client: AsyncClient) -> str:
    resp = await client.post("/api/runs", json={"prompt": "x", "horizon": 1})
    run_id = resp.json()["run_id"]
    for _ in range(50):
        got = await client.get(f"/api/runs/{run_id}")
        if got.status_code == 200 and got.json()["manifest"]["status"] == "completed":
            return run_id
        await asyncio.sleep(0.02)
    raise RuntimeError("seed run did not complete")


@pytest.mark.asyncio
async def test_list_training_runs_empty(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/training-runs")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


@pytest.mark.asyncio
async def test_create_training_run_completes_and_publishes_adapter(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        run_id = await _seed_run(client)

        resp = await client.post(
            "/api/training-runs",
            json={"sample_run_ids": [run_id], "hyperparams": {"steps": 3}},
        )
        assert resp.status_code == 202
        trun_id = resp.json()["training_run_id"]

        # Wait for completion.
        for _ in range(80):
            got = await client.get(f"/api/training-runs/{trun_id}")
            if got.status_code == 200 and got.json()["status"] == "completed":
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("training run did not complete")

        body = got.json()
        assert body["status"] == "completed"
        assert body["adapter_out"]
        assert len(body["metrics"]) == 3

        # The new adapter is registered and listable.
        adapters = (await client.get("/api/adapters")).json()
        assert any(a["id"] == body["adapter_out"] for a in adapters)


@pytest.mark.asyncio
async def test_get_adapter_returns_lineage_and_eval_reports(app) -> None:
    services = app.state.services
    services.adapter_registry.register(
        AdapterRecord(id="root", base_model="vllm:Qwen/Qwen2.5-0.5B", path="(filled)")
    )
    services.adapter_registry.register(
        AdapterRecord(
            id="child", base_model="vllm:Qwen/Qwen2.5-0.5B", path="(filled)", parent_id="root",
        )
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/adapters/root")
        assert r.status_code == 200
        body = r.json()
        assert body["adapter"]["id"] == "root"
        assert {c["id"] for c in body["children"]} == {"child"}
        assert body["eval_reports"] == []


@pytest.mark.asyncio
async def test_eval_endpoint_runs_harness_and_persists_report(app) -> None:
    services = app.state.services
    services.adapter_registry.register(
        AdapterRecord(id="adapter-eval", base_model="vllm:Qwen/Qwen2.5-0.5B", path="(filled)")
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/adapters/adapter-eval/eval",
            json={
                "tasks": [
                    {
                        "id": "single",
                        "prompt": "p",
                        "success_criteria": ["s"],
                        "horizon": 2,
                    }
                ],
                "task_set": "custom",
            },
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["adapter_id"] == "adapter-eval"
        assert body["task_set"] == "custom"
        # Adapter eval_score now reflects the report.
        adapter = (await client.get("/api/adapters/adapter-eval")).json()["adapter"]
        assert adapter["eval_score"] == body["mean_reward"]


@pytest.mark.asyncio
async def test_eval_endpoint_returns_404_for_unknown_adapter(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/adapters/missing/eval")
        assert resp.status_code == 404
