from __future__ import annotations

from dataclasses import dataclass, field

from ...domain.contracts import TrainingStore
from ...domain.models import EvalReport, TrainingRunRecord


@dataclass
class InMemoryTrainingStore(TrainingStore):
    runs: dict[str, TrainingRunRecord] = field(default_factory=dict)
    eval_reports: dict[str, EvalReport] = field(default_factory=dict)

    def list_training_runs(self) -> list[TrainingRunRecord]:
        return sorted(self.runs.values(), key=lambda r: r.created_at, reverse=True)

    def get_training_run(self, training_run_id: str) -> TrainingRunRecord | None:
        return self.runs.get(training_run_id)

    def save_training_run(self, record: TrainingRunRecord) -> TrainingRunRecord:
        self.runs[record.id] = record
        return record

    def list_eval_reports(self, adapter_id: str | None = None) -> list[EvalReport]:
        reports = list(self.eval_reports.values())
        if adapter_id is not None:
            reports = [r for r in reports if r.adapter_id == adapter_id]
        reports.sort(key=lambda r: r.created_at, reverse=True)
        return reports

    def save_eval_report(self, report: EvalReport) -> EvalReport:
        self.eval_reports[report.id] = report
        return report
