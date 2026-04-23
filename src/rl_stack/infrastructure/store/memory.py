from __future__ import annotations

from dataclasses import dataclass, field

from ...domain.contracts import ArtifactStore
from ...domain.models import (
    ArtifactRecord,
    DashboardSnapshot,
    RunDetail,
    RunManifest,
    WorkerRecord,
    WorkerStatus,
)


@dataclass
class InMemoryArtifactStore(ArtifactStore):
    runs_by_id: dict[str, RunDetail] = field(default_factory=dict)
    workers: list[WorkerRecord] = field(default_factory=list)

    def list_runs(self) -> list[RunManifest]:
        runs = [detail.manifest for detail in self.runs_by_id.values()]
        return sorted(runs, key=lambda run: run.created_at, reverse=True)

    def get_run(self, run_id: str) -> RunDetail | None:
        return self.runs_by_id.get(run_id)

    def save_run(self, run: RunDetail) -> RunDetail:
        self.runs_by_id[run.manifest.id] = run
        return run

    def set_workers(self, workers: list[WorkerRecord]) -> None:
        self.workers = workers

    def dashboard(self) -> DashboardSnapshot:
        recent_artifacts: list[ArtifactRecord] = []
        for detail in self.runs_by_id.values():
            recent_artifacts.extend(detail.artifacts)
        recent_artifacts = recent_artifacts[-10:]
        workers = self.workers or [
            WorkerRecord(
                id="worker-rollout-local",
                role="rollout",
                status=WorkerStatus.idle,
                detail="Local bootstrap worker",
            )
        ]
        return DashboardSnapshot(
            runs=self.list_runs(),
            workers=workers,
            recent_artifacts=recent_artifacts,
        )
