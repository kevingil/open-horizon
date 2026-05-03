from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from application.eval import EvalTask
from application.rescore import rescore_run
from application.training import TrainingRequest
from bootstrap import ApplicationServices, build_application_services
from domain.events import RewardComputed
from domain.models import RolloutRequest
from domain.rewards import RUBRICS
from infrastructure.policy.sglang_admin import (
    SglangLoraReloader,
    register_sglang_lora_autoreload,
)
from runtime_logging import configure_logging, install_event_bus_handler
from settings import Settings


class CreateTrainingRunBody(BaseModel):
    sample_run_ids: list[str] = Field(default_factory=list)
    parent_adapter_id: str | None = None
    hyperparams: dict[str, float | int | str | bool] = Field(default_factory=dict)


class RunEvalBody(BaseModel):
    tasks: list[EvalTask] | None = None
    task_set: str | None = None


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
        if settings.sglang_autoload_lora and settings.sglang_admin_url:
            reloader = SglangLoraReloader(admin_url=settings.sglang_admin_url)
            autoreload = register_sglang_lora_autoreload(
                bus=services.event_bus, reloader=reloader,
            )
            background_tasks_ref.add(autoreload)
            autoreload.add_done_callback(background_tasks_ref.discard)
        yield

    app = FastAPI(
        title="Long-Horizon Distributed RL API",
        version="0.2.0",
        description="Observability + orchestration API for long-horizon distributed RL.",
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

    @app.get("/api/config")
    def runtime_config():
        runner = services.coordinator.external_rollout_runner
        policy_name = (
            runner.policy_name() if runner is not None
            else services.coordinator.policy_server.policy_name()
        )
        return {
            "env_backend": settings.env_backend,
            "policy_backend": settings.policy_backend,
            "policy_name": policy_name,
            "trainer_backend": settings.trainer_backend,
            "verifiers_env_id": (
                settings.verifiers_env_id if settings.env_backend == "verifiers" else None
            ),
        }

    @app.get("/api/budget")
    def budget_status():
        window = timedelta(hours=settings.budget_window_hours)
        since = datetime.now(UTC) - window
        spent = services.artifact_store.total_cost_since(since)
        cap = settings.daily_budget_usd
        return {
            "window_hours": settings.budget_window_hours,
            "cap_usd": cap,
            "spent_usd": spent,
            "remaining_usd": max(cap - spent, 0.0) if cap > 0 else None,
            "exceeded": bool(cap > 0 and spent >= cap),
        }

    @app.get("/api/rubrics")
    def list_rubrics():
        return [
            {"name": name, "signals": [fn.__name__ for fn in RUBRICS[name].signals]}
            for name in sorted(RUBRICS)
        ]

    @app.post("/api/runs/{run_id}/rescore")
    async def rescore(run_id: str, rubric: str, dry_run: bool = False):
        try:
            result = rescore_run(
                services.artifact_store, run_id, rubric, persist=not dry_run,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not dry_run:
            await services.event_bus.publish(
                RewardComputed(
                    run_id=run_id,
                    terminal_reward=result.new_reward.terminal_reward,
                    provenance=result.new_reward.provenance,
                    audit_flags=result.new_reward.audit_flags,
                )
            )
        return {
            "run_id": result.run_id,
            "rubric": result.rubric,
            "previous_terminal_reward": result.previous_reward.terminal_reward,
            "new_terminal_reward": result.new_reward.terminal_reward,
            "delta": result.delta,
            "persisted": not dry_run,
            "new_reward": result.new_reward.model_dump(),
        }

    @app.get("/api/adapters")
    def list_adapters():
        return services.adapter_registry.list_adapters()

    @app.get("/api/adapters/{adapter_id}")
    def get_adapter(adapter_id: str):
        adapter = services.adapter_registry.get(adapter_id)
        if adapter is None:
            raise HTTPException(status_code=404, detail="Adapter not found")
        children = services.adapter_registry.children_of(adapter_id)
        eval_reports = services.training_store.list_eval_reports(adapter_id)
        return {
            "adapter": adapter.model_dump(),
            "children": [c.model_dump() for c in children],
            "eval_reports": [r.model_dump() for r in eval_reports],
        }

    @app.post("/api/adapters/{adapter_id}/eval", status_code=202)
    async def run_eval(adapter_id: str, body: RunEvalBody | None = None):
        try:
            report = await services.eval_harness.run(
                adapter_id,
                tasks=body.tasks if body else None,
                task_set=body.task_set if body else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return report

    @app.get("/api/training-runs")
    def list_training_runs():
        return services.training_store.list_training_runs()

    @app.get("/api/training-runs/{training_run_id}")
    def get_training_run(training_run_id: str):
        record = services.training_store.get_training_run(training_run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Training run not found")
        return record

    @app.post("/api/training-runs", status_code=202)
    async def create_training_run(body: CreateTrainingRunBody):
        record = await services.training_service.start_training(
            TrainingRequest(
                sample_run_ids=body.sample_run_ids,
                parent_adapter_id=body.parent_adapter_id,
                hyperparams=body.hyperparams,
            ),
        )
        return {"status": "accepted", "training_run_id": record.id}

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
