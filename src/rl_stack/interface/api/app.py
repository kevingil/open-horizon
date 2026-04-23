from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ...bootstrap import ApplicationServices, build_application_services
from ...domain.models import RolloutRequest
from ...logging import configure_logging, install_event_bus_handler
from ...settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level, json=settings.log_json)
    services: ApplicationServices = build_application_services(settings)

    background_tasks_ref: set[asyncio.Task] = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        install_event_bus_handler(services.event_bus, loop)
        # Seed bootstrap runs in the background so startup stays fast.
        boot = asyncio.create_task(services.coordinator.bootstrap())
        background_tasks_ref.add(boot)
        boot.add_done_callback(background_tasks_ref.discard)
        yield

    app = FastAPI(
        title="RL Stack API",
        version="0.2.0",
        description="Observability + orchestration API for the RL stack.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/dashboard")
    def dashboard():
        return services.artifact_store.dashboard()

    @app.get("/api/runs")
    def list_runs():
        return services.artifact_store.list_runs()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        run = services.artifact_store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.post("/api/runs", status_code=202)
    async def create_run(request: RolloutRequest, background_tasks: BackgroundTasks):
        # Fire-and-forget: the coordinator emits events; clients watch /ws/events.
        # Allocate run_id here so the caller can cancel before the first event.
        run_id = f"run-{uuid4().hex[:8]}"
        background_tasks.add_task(_safe_start, services, request, run_id)
        return {"status": "accepted", "run_id": run_id, "prompt": request.prompt}

    @app.post("/api/runs/{run_id}/cancel", status_code=202)
    async def cancel_run(run_id: str):
        accepted = services.coordinator.request_cancel(run_id)
        if not accepted:
            raise HTTPException(status_code=409, detail="Run is already terminal")
        return {"status": "cancelling", "run_id": run_id}

    @app.websocket("/ws/events")
    async def events_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        bus = services.event_bus
        try:
            async for event in bus.subscribe():
                await websocket.send_text(event.model_dump_json())
        except WebSocketDisconnect:
            return
        except Exception:
            # Best-effort close; the subscriber iterator cleans itself up.
            await websocket.close(code=1011)

    app.state.services = services
    return app


async def _safe_start(
    services: ApplicationServices, request: RolloutRequest, run_id: str,
) -> None:
    # Coordinator already publishes a RolloutFailed event; swallow here so
    # the BackgroundTask doesn't raise into uvicorn.
    with contextlib.suppress(Exception):
        await services.coordinator.start_rollout(request, run_id=run_id)


app = create_app()
