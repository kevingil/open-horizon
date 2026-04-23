from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ...domain.contracts import ArtifactStore
from ...domain.models import (
    ArtifactRecord,
    DashboardSnapshot,
    RewardPenalty,
    RewardRecord,
    RunDetail,
    RunManifest,
    RunStatus,
    TaskSpec,
    ToolPermission,
    TrajectoryRecord,
    TrajectoryStep,
    WorkerRecord,
    WorkerStatus,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id              TEXT PRIMARY KEY,
  model_id        TEXT NOT NULL,
  adapter_id      TEXT,
  dataset_slice   TEXT NOT NULL,
  infra_target    TEXT NOT NULL,
  seed            INTEGER NOT NULL,
  status          TEXT NOT NULL,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  estimated_cost_usd REAL NOT NULL,
  task_json       TEXT NOT NULL,
  trajectory_json TEXT NOT NULL,
  reward_json     TEXT NOT NULL,
  artifacts_json  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_created_at ON runs (created_at DESC);

CREATE TABLE IF NOT EXISTS workers (
  id       TEXT PRIMARY KEY,
  role     TEXT NOT NULL,
  status   TEXT NOT NULL,
  run_id   TEXT,
  detail   TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""


@dataclass
class SqliteArtifactStore(ArtifactStore):
    """Persistent ArtifactStore backed by a single SQLite file.

    Thread-safe via a per-instance lock; we serialise writes because trajectories
    fit comfortably in one transaction and rollouts only produce one RunDetail
    per run on completion.
    """

    path: Path
    workers: list[WorkerRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def list_runs(self) -> list[RunManifest]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_manifest(row) for row in rows]

    def get_run(self, run_id: str) -> RunDetail | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return _row_to_detail(row)

    def save_run(self, run: RunDetail) -> RunDetail:
        m = run.manifest
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    id, model_id, adapter_id, dataset_slice, infra_target, seed,
                    status, created_at, updated_at, estimated_cost_usd,
                    task_json, trajectory_json, reward_json, artifacts_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    model_id=excluded.model_id,
                    adapter_id=excluded.adapter_id,
                    dataset_slice=excluded.dataset_slice,
                    infra_target=excluded.infra_target,
                    seed=excluded.seed,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    estimated_cost_usd=excluded.estimated_cost_usd,
                    task_json=excluded.task_json,
                    trajectory_json=excluded.trajectory_json,
                    reward_json=excluded.reward_json,
                    artifacts_json=excluded.artifacts_json
                """,
                (
                    m.id, m.model_id, m.adapter_id, m.dataset_slice, m.infra_target,
                    m.seed, m.status.value,
                    m.created_at.isoformat(), m.updated_at.isoformat(),
                    m.estimated_cost_usd,
                    run.task.model_dump_json(),
                    run.trajectory.model_dump_json(),
                    run.reward.model_dump_json(),
                    json.dumps([a.model_dump() for a in run.artifacts]),
                ),
            )
        return run

    def set_workers(self, workers: list[WorkerRecord]) -> None:
        from datetime import UTC, datetime

        ts = datetime.now(UTC).isoformat()
        self.workers = workers
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM workers")
            conn.executemany(
                """
                INSERT INTO workers (id, role, status, run_id, detail, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (w.id, w.role, w.status.value, w.run_id, w.detail, ts)
                    for w in workers
                ],
            )

    def dashboard(self) -> DashboardSnapshot:
        runs = self.list_runs()
        with self._lock, self._connect() as conn:
            worker_rows = conn.execute("SELECT * FROM workers").fetchall()
        workers = [_row_to_worker(r) for r in worker_rows] or [
            WorkerRecord(
                id="worker-rollout-local",
                role="rollout",
                status=WorkerStatus.idle,
                detail="Local bootstrap worker",
            )
        ]
        recent_artifacts = _recent_artifacts(self, limit=10)
        return DashboardSnapshot(
            runs=runs,
            workers=workers,
            recent_artifacts=recent_artifacts,
        )


def _recent_artifacts(store: SqliteArtifactStore, *, limit: int) -> list[ArtifactRecord]:
    with store._lock, store._connect() as conn:
        rows = conn.execute(
            "SELECT artifacts_json FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    flat: list[ArtifactRecord] = []
    for row in rows:
        for entry in json.loads(row["artifacts_json"]):
            flat.append(ArtifactRecord.model_validate(entry))
        if len(flat) >= limit:
            break
    return flat[:limit]


def _row_to_manifest(row: sqlite3.Row) -> RunManifest:
    from datetime import datetime

    return RunManifest(
        id=row["id"],
        model_id=row["model_id"],
        adapter_id=row["adapter_id"],
        dataset_slice=row["dataset_slice"],
        infra_target=row["infra_target"],
        seed=row["seed"],
        status=RunStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        estimated_cost_usd=row["estimated_cost_usd"],
    )


def _row_to_detail(row: sqlite3.Row) -> RunDetail:
    task = TaskSpec.model_validate_json(row["task_json"])
    # Rehydrate tool permissions as enums (Pydantic does this automatically).
    assert all(isinstance(p, ToolPermission) for p in task.tool_permissions)
    trajectory = TrajectoryRecord.model_validate_json(row["trajectory_json"])
    reward = RewardRecord.model_validate_json(row["reward_json"])
    artifacts = [
        ArtifactRecord.model_validate(entry)
        for entry in json.loads(row["artifacts_json"])
    ]
    return RunDetail(
        manifest=_row_to_manifest(row),
        task=task,
        trajectory=trajectory,
        reward=reward,
        artifacts=artifacts,
    )


def _row_to_worker(row: sqlite3.Row) -> WorkerRecord:
    return WorkerRecord(
        id=row["id"],
        role=row["role"],
        status=WorkerStatus(row["status"]),
        run_id=row["run_id"],
        detail=row["detail"],
    )


# Silence linter: these are referenced but Pydantic's rehydration may warn.
_ = (TrajectoryStep, RewardPenalty)
