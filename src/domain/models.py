from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

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
    # Phase E: flips True when the coordinator's training recorder
    # writes a TurnTrainingRecord row joined to this step. Cheap signal
    # for "is there training metadata to fetch" without paying the
    # bytes-payload cost in /api/runs/{id}.
    has_training_metadata: bool = False


class TrajectoryRecord(BaseModel):
    id: str
    task_id: str
    steps: list[TrajectoryStep]
    summaries: list[str] = Field(default_factory=list)
    timings_ms: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class TurnTrainingRecord(BaseModel):
    """Per-turn token-level metadata, stored alongside the trajectory but
    in its own table.

    Long-horizon rollouts can balloon prompt_ids/completion_ids/masks
    to hundreds of KB per turn. Keeping that out of `RunDetail`
    payloads means the dashboard never pays the cost; trainers fetch
    these rows on demand via /api/runs/{run_id}/turns.

    Lists carry token ids as plain ints rather than numpy arrays so
    the model can serialize over JSON; storage backends are free to
    pack them as bytes internally.
    """

    id: str
    run_id: str
    step_index: int
    prompt_ids: list[int] = Field(default_factory=list)
    completion_ids: list[int] = Field(default_factory=list)
    attention_mask: list[int] = Field(default_factory=list)
    loss_mask: list[int] = Field(default_factory=list)
    sampling_args: dict[str, Any] = Field(default_factory=dict)
    model_name: str = ""
    token_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)


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
    # When None, the coordinator's RoundScheduler decides the horizon.
    # Passing an explicit int overrides the scheduler for this rollout.
    horizon: int | None = Field(default=None, ge=1)
    success_criteria: list[str] = Field(default_factory=list)
    # Optional adapter id; the policy server may use it (vLLM mounts LoRAs by
    # name, OpenAI proper ignores it). The coordinator stamps it on the
    # resulting RunManifest so eval harnesses can group rollouts by adapter.
    adapter_id: str | None = None
    # Named profile to drive this rollout. Profiles are defined at
    # startup (Settings.policy_profiles) and pick the (base_url, model,
    # routes_to) triple for the run. Unknown profile names are rejected
    # at request time. Empty = use Settings.default_policy_profile.
    policy_profile: str | None = None


# --- Training / adapters / eval --------------------------------------------


class TrainingStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TrainingMetricPoint(BaseModel):
    """One point on a training run's metric series."""

    step: int = Field(ge=0)
    loss: float
    mean_reward: float | None = None
    kl: float | None = None
    extra: dict[str, float] = Field(default_factory=dict)


class AdapterRecord(BaseModel):
    """A trained adapter (or stub) registered with the AdapterRegistry."""

    id: str
    parent_id: str | None = None
    base_model: str
    training_run_id: str | None = None
    eval_score: float | None = None
    path: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class TrainingRunRecord(BaseModel):
    """One training run from a parent adapter to a (proposed) child adapter."""

    id: str
    status: TrainingStatus = TrainingStatus.pending
    adapter_in: str | None = None
    adapter_out: str | None = None
    sample_run_ids: list[str] = Field(default_factory=list)
    hyperparams: dict[str, float | int | str | bool] = Field(default_factory=dict)
    metrics: list[TrainingMetricPoint] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class EvalTaskScore(BaseModel):
    task_id: str
    terminal_reward: float


class EvalReport(BaseModel):
    """An evaluation pass: an adapter scored on a fixed task set."""

    id: str
    adapter_id: str
    task_set: str
    mean_reward: float
    per_task: list[EvalTaskScore] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
