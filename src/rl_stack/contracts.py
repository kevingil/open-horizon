from __future__ import annotations

from abc import ABC, abstractmethod

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
    def create_task(self, task: TaskSpec) -> TaskSpec:
        raise NotImplementedError

    @abstractmethod
    def step(self, task_id: str, action: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def reset(self, task_id: str) -> None:
        raise NotImplementedError


class ToolHarness(ABC):
    @abstractmethod
    def read_file(self, path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def list_files(self, path: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def record_command(self, command: str) -> str:
        raise NotImplementedError


class PolicyServer(ABC):
    @abstractmethod
    def policy_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_action(self, task: TaskSpec, context: list[str]) -> str:
        raise NotImplementedError


class RewardPipeline(ABC):
    @abstractmethod
    def score_trajectory(self, task: TaskSpec, trajectory: TrajectoryRecord) -> RewardRecord:
        raise NotImplementedError


class ArtifactStore(ABC):
    @abstractmethod
    def list_runs(self) -> list[RunManifest]:
        raise NotImplementedError

    @abstractmethod
    def get_run(self, run_id: str) -> RunDetail | None:
        raise NotImplementedError

    @abstractmethod
    def save_run(self, run: RunDetail) -> RunDetail:
        raise NotImplementedError

    @abstractmethod
    def dashboard(self) -> DashboardSnapshot:
        raise NotImplementedError


class RolloutCoordinator(ABC):
    @abstractmethod
    def bootstrap(self) -> DashboardSnapshot:
        raise NotImplementedError

    @abstractmethod
    def start_rollout(self, request: RolloutRequest) -> RunDetail:
        raise NotImplementedError
