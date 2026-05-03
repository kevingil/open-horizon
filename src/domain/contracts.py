from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .models import (
    AdapterRecord,
    DashboardSnapshot,
    EvalReport,
    RewardRecord,
    RolloutRequest,
    RunDetail,
    RunManifest,
    TaskSpec,
    TrainingMetricPoint,
    TrainingRunRecord,
    TrajectoryRecord,
)


class EnvironmentRunner(ABC):
    @abstractmethod
    def create_task(self, task: TaskSpec) -> TaskSpec: ...

    @abstractmethod
    def step(self, task_id: str, action: str) -> str: ...

    @abstractmethod
    def reset(self, task_id: str) -> None: ...


class ToolHarness(ABC):
    @abstractmethod
    def read_file(self, path: str) -> str: ...

    @abstractmethod
    def list_files(self, path: str) -> list[str]: ...

    @abstractmethod
    def record_command(self, command: str) -> str: ...


class PolicyServer(ABC):
    @abstractmethod
    def policy_name(self) -> str: ...

    @abstractmethod
    def generate_action(self, task: TaskSpec, context: list[str]) -> str: ...


class RewardPipeline(ABC):
    @abstractmethod
    def score_trajectory(self, task: TaskSpec, trajectory: TrajectoryRecord) -> RewardRecord: ...


class ArtifactStore(ABC):
    @abstractmethod
    def list_runs(self) -> list[RunManifest]: ...

    @abstractmethod
    def get_run(self, run_id: str) -> RunDetail | None: ...

    @abstractmethod
    def save_run(self, run: RunDetail) -> RunDetail: ...

    @abstractmethod
    def dashboard(self) -> DashboardSnapshot: ...

    @abstractmethod
    def total_cost_since(self, since: datetime) -> float:
        """Sum of estimated_cost_usd across runs created at or after `since`."""


class RolloutCoordinator(ABC):
    @abstractmethod
    async def bootstrap(self) -> DashboardSnapshot: ...

    @abstractmethod
    async def start_rollout(self, request: RolloutRequest) -> RunDetail: ...


# --- Training / adapters / eval --------------------------------------------


class AdapterRegistry(ABC):
    """Persistence + filesystem layout for trained adapters."""

    @abstractmethod
    def list_adapters(self) -> list[AdapterRecord]: ...

    @abstractmethod
    def get(self, adapter_id: str) -> AdapterRecord | None: ...

    @abstractmethod
    def register(self, record: AdapterRecord) -> AdapterRecord: ...

    @abstractmethod
    def path_for(self, adapter_id: str) -> Path:
        """Where the adapter's weights / manifest live on disk."""

    @abstractmethod
    def children_of(self, adapter_id: str | None) -> list[AdapterRecord]:
        """Direct children in the adapter lineage tree."""


class TrainingStore(ABC):
    """Persistence for TrainingRunRecord and EvalReport. Separate from
    ArtifactStore (rollouts) so the two responsibilities don't conflate."""

    @abstractmethod
    def list_training_runs(self) -> list[TrainingRunRecord]: ...

    @abstractmethod
    def get_training_run(self, training_run_id: str) -> TrainingRunRecord | None: ...

    @abstractmethod
    def save_training_run(self, record: TrainingRunRecord) -> TrainingRunRecord: ...

    @abstractmethod
    def list_eval_reports(self, adapter_id: str | None = None) -> list[EvalReport]: ...

    @abstractmethod
    def save_eval_report(self, report: EvalReport) -> EvalReport: ...


# Trainers are synchronous from their own POV (one heavy step). The
# orchestration layer wraps them in an asyncio.to_thread call and forwards
# `on_metric` callbacks to the event bus.
MetricCallback = Callable[[TrainingMetricPoint], None]


class Trainer(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def train(
        self,
        run: TrainingRunRecord,
        samples: list[RunDetail],
        parent: AdapterRecord | None,
        on_metric: MetricCallback,
        adapters: AdapterRegistry,
    ) -> AdapterRecord: ...
