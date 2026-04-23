from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from .models import (
    DashboardSnapshot,
    RewardRecord,
    RolloutRequest,
    RunDetail,
    RunManifest,
    TaskSpec,
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
