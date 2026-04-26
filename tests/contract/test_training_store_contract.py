"""Contract tests every TrainingStore impl must satisfy."""
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from rl_stack.domain.contracts import TrainingStore
from rl_stack.domain.models import (
    EvalReport,
    EvalTaskScore,
    TrainingMetricPoint,
    TrainingRunRecord,
    TrainingStatus,
)
from rl_stack.infrastructure.training.memory_store import InMemoryTrainingStore
from rl_stack.infrastructure.training.sqlite_store import SqliteTrainingStore


def _run(run_id: str, *, created_at: datetime | None = None) -> TrainingRunRecord:
    return TrainingRunRecord(
        id=run_id,
        sample_run_ids=["a", "b"],
        hyperparams={"lr": 1e-5},
        created_at=created_at or datetime.now(UTC),
    )


def _report(adapter_id: str, *, mean: float = 0.5) -> EvalReport:
    return EvalReport(
        id=f"eval-{adapter_id}",
        adapter_id=adapter_id,
        task_set="builtin",
        mean_reward=mean,
        per_task=[EvalTaskScore(task_id="t1", terminal_reward=mean)],
    )


def _sqlite_factory(tmp_path_factory: pytest.TempPathFactory) -> Callable[[], TrainingStore]:
    base = tmp_path_factory.mktemp("training-sqlite")
    counter = {"n": 0}

    def make() -> TrainingStore:
        counter["n"] += 1
        return SqliteTrainingStore(path=base / f"runs-{counter['n']}.db")

    return make


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> TrainingStore:
    if request.param == "memory":
        return InMemoryTrainingStore()
    return _sqlite_factory(tmp_path_factory)()


def test_save_and_get_training_run(store: TrainingStore) -> None:
    saved = store.save_training_run(_run("trun-1"))
    assert store.get_training_run("trun-1") == saved
    assert store.get_training_run("missing") is None


def test_list_training_runs_most_recent_first(store: TrainingStore) -> None:
    now = datetime.now(UTC)
    store.save_training_run(_run("trun-old", created_at=now - timedelta(hours=2)))
    store.save_training_run(_run("trun-new", created_at=now))
    listed = store.list_training_runs()
    assert [r.id for r in listed] == ["trun-new", "trun-old"]


def test_save_training_run_upserts_status_and_metrics(store: TrainingStore) -> None:
    record = _run("trun-1")
    store.save_training_run(record)
    updated = record.model_copy(
        update={
            "status": TrainingStatus.completed,
            "metrics": [TrainingMetricPoint(step=0, loss=0.5)],
            "adapter_out": "adapter-z",
        }
    )
    store.save_training_run(updated)
    fetched = store.get_training_run("trun-1")
    assert fetched.status == TrainingStatus.completed
    assert fetched.adapter_out == "adapter-z"
    assert len(fetched.metrics) == 1


def test_eval_reports_filter_by_adapter(store: TrainingStore) -> None:
    store.save_eval_report(_report("adapter-a", mean=0.4))
    store.save_eval_report(_report("adapter-b", mean=0.7))
    all_reports = store.list_eval_reports()
    assert {r.adapter_id for r in all_reports} == {"adapter-a", "adapter-b"}
    only_a = store.list_eval_reports(adapter_id="adapter-a")
    assert len(only_a) == 1 and only_a[0].adapter_id == "adapter-a"
