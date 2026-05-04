from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from domain.contracts import ArtifactStore
from domain.models import (
    ArtifactRecord,
    DashboardSnapshot,
    RunDetail,
    RunManifest,
    TurnTrainingRecord,
    WorkerRecord,
    WorkerStatus,
)


@dataclass
class InMemoryArtifactStore(ArtifactStore):
    runs_by_id: dict[str, RunDetail] = field(default_factory=dict)
    workers: list[WorkerRecord] = field(default_factory=list)
    # Phase E: keyed by (run_id, step_index) so the trainer can fetch a
    # specific turn without dragging the whole rollout payload.
    turn_training: dict[tuple[str, int], TurnTrainingRecord] = field(default_factory=dict)

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

    def total_cost_since(self, since: datetime) -> float:
        return round(
            sum(
                detail.manifest.estimated_cost_usd
                for detail in self.runs_by_id.values()
                if detail.manifest.created_at >= since
            ),
            6,
        )

    def save_turn_training(self, record: TurnTrainingRecord) -> TurnTrainingRecord:
        self.turn_training[(record.run_id, record.step_index)] = record
        return record

    def get_turn_training(
        self, run_id: str, step_index: int,
    ) -> TurnTrainingRecord | None:
        return self.turn_training.get((run_id, step_index))

    def list_turn_training(self, run_id: str) -> list[TurnTrainingRecord]:
        rows = [r for (rid, _), r in self.turn_training.items() if rid == run_id]
        return sorted(rows, key=lambda r: r.step_index)

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
