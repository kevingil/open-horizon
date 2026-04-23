from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..bootstrap import build_application_services
from ..models import RolloutRequest


def create_app(workspace_root: str | Path = ".") -> FastAPI:
    coordinator = build_application_services(workspace_root)

    app = FastAPI(
        title="RL Stack API",
        version="0.1.0",
        description="Thin observability API for the docs-first RL stack scaffold.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/dashboard")
    def dashboard():
        return coordinator.artifact_store.dashboard()

    @app.get("/api/runs")
    def list_runs():
        return coordinator.artifact_store.list_runs()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        run = coordinator.artifact_store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.post("/api/runs")
    def create_run(request: RolloutRequest):
        return coordinator.start_rollout(request)

    return app


app = create_app(".")
