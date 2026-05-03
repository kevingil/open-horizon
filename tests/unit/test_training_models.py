from __future__ import annotations

import json

from domain.events import (
    AdapterPublished,
    EvalCompleted,
    TrainingCompleted,
    TrainingMetric,
    TrainingStarted,
)
from domain.models import (
    AdapterRecord,
    EvalReport,
    EvalTaskScore,
    TrainingMetricPoint,
    TrainingRunRecord,
    TrainingStatus,
)


def _adapter(id: str = "adapter-x") -> AdapterRecord:
    return AdapterRecord(id=id, base_model="vllm:Qwen/Qwen2.5-0.5B", path="/tmp/x")


def test_training_run_record_default_status_is_pending() -> None:
    rec = TrainingRunRecord(id="run-1")
    assert rec.status == TrainingStatus.pending
    assert rec.metrics == []
    assert rec.sample_run_ids == []


def test_training_run_record_json_roundtrip() -> None:
    rec = TrainingRunRecord(
        id="run-1",
        status=TrainingStatus.running,
        sample_run_ids=["a", "b"],
        hyperparams={"lr": 1e-5, "batch": 8},
        metrics=[TrainingMetricPoint(step=0, loss=2.5, mean_reward=0.1, kl=0.01)],
    )
    same = TrainingRunRecord.model_validate_json(rec.model_dump_json())
    assert same == rec
    # Hyperparams support floats and ints together.
    assert same.hyperparams["lr"] == 1e-5


def test_eval_report_aggregates_per_task_scores() -> None:
    report = EvalReport(
        id="eval-1", adapter_id="adapter-x", task_set="builtin",
        mean_reward=0.6,
        per_task=[
            EvalTaskScore(task_id="t1", terminal_reward=0.5),
            EvalTaskScore(task_id="t2", terminal_reward=0.7),
        ],
    )
    raw = report.model_dump_json()
    decoded = json.loads(raw)
    assert decoded["mean_reward"] == 0.6
    assert {p["task_id"] for p in decoded["per_task"]} == {"t1", "t2"}


def test_training_events_carry_training_run_id() -> None:
    rec = TrainingRunRecord(id="run-7")
    started = TrainingStarted(training_run_id=rec.id, record=rec)
    payload = json.loads(started.model_dump_json())
    assert payload["kind"] == "training.started"
    assert payload["training_run_id"] == "run-7"

    metric = TrainingMetric(
        training_run_id=rec.id,
        metric=TrainingMetricPoint(step=3, loss=1.4, mean_reward=0.2),
    )
    assert metric.training_run_id == "run-7"
    assert metric.metric.step == 3

    finished = TrainingCompleted(training_run_id=rec.id, record=rec)
    assert finished.kind == "training.completed"


def test_adapter_published_event_carries_record() -> None:
    pub = AdapterPublished(adapter=_adapter())
    assert pub.kind == "adapter.published"
    assert pub.adapter.base_model.startswith("vllm:")


def test_eval_completed_event_includes_report() -> None:
    report = EvalReport(
        id="eval-1", adapter_id="adapter-x", task_set="builtin",
        mean_reward=0.4,
    )
    ev = EvalCompleted(report=report)
    assert ev.kind == "eval.completed"
    assert ev.report.adapter_id == "adapter-x"
