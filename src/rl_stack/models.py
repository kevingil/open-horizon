from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class WorkerStatus(str, Enum):
    idle = "idle"
    running = "running"
    failed = "failed"


class ToolPermission(str, Enum):
    read = "read"
    edit = "edit"
    terminal = "terminal"
    search = "search"


class TaskSpec(BaseModel):
    id: str
    prompt: str
    repo_snapshot: str
    tool_permissions: list[ToolPermission]
    horizon: int = Field(default=8, ge=1)
    success_criteria: list[str]


class TrajectoryStep(BaseModel):
    index: int
    actor: str
    kind: str
    content: str
    timestamp: datetime = Field(default_factory=utc_now)


class TrajectoryRecord(BaseModel):
    id: str
    task_id: str
    steps: list[TrajectoryStep]
    summaries: list[str] = Field(default_factory=list)
    timings_ms: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class RewardPenalty(BaseModel):
    code: str
    value: float
    reason: str


class RewardRecord(BaseModel):
    trajectory_id: str
    terminal_reward: float
    step_rewards: list[float]
    penalties: list[RewardPenalty] = Field(default_factory=list)
    audit_flags: list[str] = Field(default_factory=list)
    provenance: str


class ArtifactRecord(BaseModel):
    name: str
    kind: str
    path: str


class RunManifest(BaseModel):
    id: str
    model_id: str
    adapter_id: str | None = None
    dataset_slice: str
    infra_target: str
    seed: int
    status: RunStatus
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    estimated_cost_usd: float = 0.0


class RunDetail(BaseModel):
    manifest: RunManifest
    task: TaskSpec
    trajectory: TrajectoryRecord
    reward: RewardRecord
    artifacts: list[ArtifactRecord]


class WorkerRecord(BaseModel):
    id: str
    role: str
    status: WorkerStatus
    run_id: str | None = None
    detail: str


class DashboardSnapshot(BaseModel):
    generated_at: datetime = Field(default_factory=utc_now)
    runs: list[RunManifest]
    workers: list[WorkerRecord]
    recent_artifacts: list[ArtifactRecord]


class RolloutRequest(BaseModel):
    prompt: str
    repo_snapshot: str = "."
    infra_target: str = "mac-local"
    horizon: int = Field(default=6, ge=1)
    success_criteria: list[str] = Field(default_factory=list)
