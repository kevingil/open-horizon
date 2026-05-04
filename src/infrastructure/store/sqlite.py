from __future__ import annotations

import array
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from domain.contracts import ArtifactStore
from domain.models import (
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
    TurnTrainingRecord,
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

CREATE TABLE IF NOT EXISTS turn_training (
  id              TEXT NOT NULL,
  run_id          TEXT NOT NULL,
  step_index      INTEGER NOT NULL,
  model_name      TEXT NOT NULL,
  token_count     INTEGER NOT NULL,
  prompt_ids      BLOB NOT NULL,
  completion_ids  BLOB NOT NULL,
  attention_mask  BLOB NOT NULL,
  loss_mask       BLOB NOT NULL,
  sampling_args   TEXT NOT NULL,
  created_at      TEXT NOT NULL,
  PRIMARY KEY (run_id, step_index)
);
CREATE INDEX IF NOT EXISTS turn_training_run_id ON turn_training (run_id);
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

    def save_turn_training(self, record: TurnTrainingRecord) -> TurnTrainingRecord:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO turn_training (
                    id, run_id, step_index, model_name, token_count,
                    prompt_ids, completion_ids, attention_mask, loss_mask,
                    sampling_args, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, step_index) DO UPDATE SET
                    id=excluded.id,
                    model_name=excluded.model_name,
                    token_count=excluded.token_count,
                    prompt_ids=excluded.prompt_ids,
                    completion_ids=excluded.completion_ids,
                    attention_mask=excluded.attention_mask,
                    loss_mask=excluded.loss_mask,
                    sampling_args=excluded.sampling_args,
                    created_at=excluded.created_at
                """,
                (
                    record.id, record.run_id, record.step_index,
                    record.model_name, record.token_count,
                    _pack_ints(record.prompt_ids),
                    _pack_ints(record.completion_ids),
                    _pack_ints(record.attention_mask),
                    _pack_ints(record.loss_mask),
                    json.dumps(record.sampling_args),
                    record.created_at.isoformat(),
                ),
            )
        return record

    def get_turn_training(
        self, run_id: str, step_index: int,
    ) -> TurnTrainingRecord | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM turn_training WHERE run_id = ? AND step_index = ?",
                (run_id, step_index),
            ).fetchone()
        return _row_to_turn_training(row) if row else None

    def list_turn_training(self, run_id: str) -> list[TurnTrainingRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM turn_training WHERE run_id = ? "
                "ORDER BY step_index ASC",
                (run_id,),
            ).fetchall()
        return [_row_to_turn_training(r) for r in rows]

    def total_cost_since(self, since: datetime) -> float:
        iso = since.isoformat()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM runs WHERE created_at >= ?",
                (iso,),
            ).fetchone()
        return round(float(row[0]) if row else 0.0, 6)

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


def _pack_ints(values: list[int]) -> bytes:
    """Pack a token-id / mask list as 4-byte signed ints. Way smaller on
    disk than the JSON shape Pydantic would produce by default."""
    return array.array("i", values).tobytes()


def _unpack_ints(blob: bytes) -> list[int]:
    arr = array.array("i")
    arr.frombytes(blob)
    return arr.tolist()


def _row_to_turn_training(row: sqlite3.Row) -> TurnTrainingRecord:
    return TurnTrainingRecord(
        id=row["id"],
        run_id=row["run_id"],
        step_index=row["step_index"],
        model_name=row["model_name"],
        token_count=row["token_count"],
        prompt_ids=_unpack_ints(row["prompt_ids"]),
        completion_ids=_unpack_ints(row["completion_ids"]),
        attention_mask=_unpack_ints(row["attention_mask"]),
        loss_mask=_unpack_ints(row["loss_mask"]),
        sampling_args=json.loads(row["sampling_args"]),
        created_at=datetime.fromisoformat(row["created_at"]),
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
