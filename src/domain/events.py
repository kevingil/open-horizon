from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .models import (
    AdapterRecord,
    EvalReport,
    RunDetail,
    RunManifest,
    TrainingMetricPoint,
    TrainingRunRecord,
    TrajectoryStep,
    WorkerRecord,
    utc_now,
)


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


class RolloutCancelled(_EventBase):
    kind: Literal["rollout.cancelled"] = "rollout.cancelled"
    reason: str = "cancelled"


class ProgressTicked(_EventBase):
    """Per-step progress snapshot: cumulative tokens, cost, latest tool.
    Separate from StepRecorded so dashboards can render running tallies
    without re-parsing trajectory content."""

    kind: Literal["progress.ticked"] = "progress.ticked"
    step_index: int
    tool: str | None = None
    tokens: int = 0
    cost_usd: float = 0.0


class BudgetExceeded(_EventBase):
    kind: Literal["budget.exceeded"] = "budget.exceeded"
    spent_usd: float
    cap_usd: float
    window_hours: float


class WorkerUpdated(_EventBase):
    kind: Literal["worker.updated"] = "worker.updated"
    worker: WorkerRecord


class LogLine(_EventBase):
    kind: Literal["log.line"] = "log.line"
    level: str
    logger: str
    message: str
    context: dict[str, str] = Field(default_factory=dict)


# --- Training / adapter / eval events --------------------------------------


class _TrainingEventBase(_EventBase):
    """Training events ride the same event bus as rollout events; we tag them
    with `training_run_id` so the UI can route them by training run instead of
    by rollout `run_id` (which they don't have)."""

    training_run_id: str


class TrainingStarted(_TrainingEventBase):
    kind: Literal["training.started"] = "training.started"
    record: TrainingRunRecord


class TrainingMetric(_TrainingEventBase):
    kind: Literal["training.metric"] = "training.metric"
    metric: TrainingMetricPoint


class TrainingCompleted(_TrainingEventBase):
    kind: Literal["training.completed"] = "training.completed"
    record: TrainingRunRecord


class TrainingFailed(_TrainingEventBase):
    kind: Literal["training.failed"] = "training.failed"
    error: str


class AdapterPublished(_EventBase):
    kind: Literal["adapter.published"] = "adapter.published"
    adapter: AdapterRecord


class EvalCompleted(_EventBase):
    kind: Literal["eval.completed"] = "eval.completed"
    report: EvalReport


DomainEvent = Annotated[
    RolloutStarted
    | StepRecorded
    | RewardComputed
    | RolloutCompleted
    | RolloutFailed
    | RolloutCancelled
    | ProgressTicked
    | BudgetExceeded
    | WorkerUpdated
    | LogLine
    | TrainingStarted
    | TrainingMetric
    | TrainingCompleted
    | TrainingFailed
    | AdapterPublished
    | EvalCompleted,
    Field(discriminator="kind"),
]
