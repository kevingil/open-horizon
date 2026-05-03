"""Training orchestration.

`start_training()` is the one-shot entry point used by both the API and the
CLI. It persists a TrainingRunRecord in `pending` state, kicks the synchronous
Trainer.train() into a worker thread (so the event loop stays free for the
WebSocket), forwards each metric callback onto the EventBus, and on completion
registers the new adapter and emits AdapterPublished.

Failures: any exception raised inside the trainer turns the run into
`failed` status with the exception message, persists, emits TrainingFailed.
The function never raises into its caller (the API endpoint stays a clean
202 + run id).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

import structlog

from domain.contracts import (
    AdapterRegistry,
    ArtifactStore,
    Trainer,
    TrainingStore,
)
from domain.events import (
    AdapterPublished,
    TrainingCompleted,
    TrainingFailed,
    TrainingMetric,
    TrainingStarted,
)
from domain.models import (
    AdapterRecord,
    RunDetail,
    TrainingMetricPoint,
    TrainingRunRecord,
    TrainingStatus,
)

from .event_bus import EventBus

log = structlog.get_logger(__name__)


@dataclass
class TrainingRequest:
    sample_run_ids: list[str]
    parent_adapter_id: str | None = None
    hyperparams: dict[str, float | int | str | bool] | None = None


@dataclass
class TrainingService:
    trainer: Trainer
    artifact_store: ArtifactStore
    training_store: TrainingStore
    adapter_registry: AdapterRegistry
    event_bus: EventBus
    _tasks: set[asyncio.Task] = field(default_factory=set, init=False, repr=False)

    async def start_training(self, request: TrainingRequest) -> TrainingRunRecord:
        run_id = f"trun-{uuid4().hex[:8]}"
        now = datetime.now(UTC)
        record = TrainingRunRecord(
            id=run_id,
            status=TrainingStatus.pending,
            adapter_in=request.parent_adapter_id,
            sample_run_ids=list(request.sample_run_ids),
            hyperparams=dict(request.hyperparams or {}),
            created_at=now,
            updated_at=now,
        )
        self.training_store.save_training_run(record)
        # Hold a strong ref so the trainer task isn't GC'd mid-flight.
        task = asyncio.create_task(self._drive(record))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return record

    async def _drive(self, record: TrainingRunRecord) -> None:
        bind = structlog.contextvars.bound_contextvars(training_run_id=record.id)
        with bind:
            samples = _gather_samples(self.artifact_store, record.sample_run_ids)
            parent = (
                self.adapter_registry.get(record.adapter_in)
                if record.adapter_in
                else None
            )
            record = record.model_copy(
                update={"status": TrainingStatus.running, "updated_at": datetime.now(UTC)},
            )
            self.training_store.save_training_run(record)
            await self.event_bus.publish(
                TrainingStarted(training_run_id=record.id, record=record)
            )

            loop = asyncio.get_running_loop()
            metrics_acc: list[TrainingMetricPoint] = []

            def on_metric(point: TrainingMetricPoint) -> None:
                metrics_acc.append(point)
                # Forward from the worker thread back to the loop. We don't
                # block on the future; metric publishing is best-effort.
                asyncio.run_coroutine_threadsafe(
                    self.event_bus.publish(
                        TrainingMetric(training_run_id=record.id, metric=point)
                    ),
                    loop,
                )

            try:
                adapter = await asyncio.to_thread(
                    self.trainer.train,
                    record,
                    samples,
                    parent,
                    on_metric,
                    self.adapter_registry,
                )
            except Exception as exc:
                log.exception("training.failed")
                failed = record.model_copy(
                    update={
                        "status": TrainingStatus.failed,
                        "metrics": metrics_acc,
                        "error": str(exc),
                        "updated_at": datetime.now(UTC),
                    }
                )
                self.training_store.save_training_run(failed)
                await self.event_bus.publish(
                    TrainingFailed(training_run_id=record.id, error=str(exc))
                )
                return

            completed = record.model_copy(
                update={
                    "status": TrainingStatus.completed,
                    "adapter_out": adapter.id,
                    "metrics": metrics_acc,
                    "updated_at": datetime.now(UTC),
                }
            )
            self.training_store.save_training_run(completed)
            await self.event_bus.publish(AdapterPublished(adapter=adapter))
            await self.event_bus.publish(
                TrainingCompleted(training_run_id=record.id, record=completed)
            )
            log.info(
                "training.completed",
                adapter=adapter.id,
                steps=len(metrics_acc),
                samples=len(samples),
            )


def _gather_samples(store: ArtifactStore, run_ids: list[str]) -> list[RunDetail]:
    out: list[RunDetail] = []
    for rid in run_ids:
        detail = store.get_run(rid)
        if detail is None:
            log.warning("training.missing_sample", run_id=rid)
            continue
        out.append(detail)
    return out


def _adapter_path_or_none(adapter: AdapterRecord | None) -> str | None:
    return adapter.path if adapter else None
