from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .models import RunDetail, RunManifest, TrajectoryStep, WorkerRecord, utc_now


class _EventBase(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    at: datetime = Field(default_factory=utc_now)
    run_id: str | None = None


class RolloutStarted(_EventBase):
    kind: Literal["rollout.started"] = "rollout.started"
    manifest: RunManifest


class StepRecorded(_EventBase):
    kind: Literal["step.recorded"] = "step.recorded"
    step: TrajectoryStep


class RewardComputed(_EventBase):
    kind: Literal["reward.computed"] = "reward.computed"
    terminal_reward: float
    provenance: str
    audit_flags: list[str] = Field(default_factory=list)


class RolloutCompleted(_EventBase):
    kind: Literal["rollout.completed"] = "rollout.completed"
    detail: RunDetail


class RolloutFailed(_EventBase):
    kind: Literal["rollout.failed"] = "rollout.failed"
    error: str


class WorkerUpdated(_EventBase):
    kind: Literal["worker.updated"] = "worker.updated"
    worker: WorkerRecord


class LogLine(_EventBase):
    kind: Literal["log.line"] = "log.line"
    level: str
    logger: str
    message: str
    context: dict[str, str] = Field(default_factory=dict)


DomainEvent = Annotated[
    RolloutStarted
    | StepRecorded
    | RewardComputed
    | RolloutCompleted
    | RolloutFailed
    | WorkerUpdated
    | LogLine,
    Field(discriminator="kind"),
]
