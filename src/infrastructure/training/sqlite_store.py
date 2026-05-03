from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

from domain.contracts import TrainingStore
from domain.models import EvalReport, TrainingRunRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS training_runs (
  id           TEXT PRIMARY KEY,
  created_at   TEXT NOT NULL,
  updated_at   TEXT NOT NULL,
  status       TEXT NOT NULL,
  payload      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS training_runs_created_at ON training_runs (created_at DESC);

CREATE TABLE IF NOT EXISTS eval_reports (
  id           TEXT PRIMARY KEY,
  adapter_id   TEXT NOT NULL,
  created_at   TEXT NOT NULL,
  payload      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS eval_reports_adapter ON eval_reports (adapter_id, created_at DESC);
"""


@dataclass
class SqliteTrainingStore(TrainingStore):
    """Persistent TrainingStore backed by SQLite. JSON blobs keep the schema
    slim; queries we actually need (list, get, filter by adapter) only need
    a handful of indexed columns."""

    path: Path
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

    def list_training_runs(self) -> list[TrainingRunRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM training_runs ORDER BY created_at DESC"
            ).fetchall()
        return [TrainingRunRecord.model_validate_json(r["payload"]) for r in rows]

    def get_training_run(self, training_run_id: str) -> TrainingRunRecord | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM training_runs WHERE id = ?", (training_run_id,),
            ).fetchone()
        if row is None:
            return None
        return TrainingRunRecord.model_validate_json(row["payload"])

    def save_training_run(self, record: TrainingRunRecord) -> TrainingRunRecord:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO training_runs (id, created_at, updated_at, status, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    status=excluded.status,
                    payload=excluded.payload
                """,
                (
                    record.id,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.status.value,
                    record.model_dump_json(),
                ),
            )
        return record

    def list_eval_reports(self, adapter_id: str | None = None) -> list[EvalReport]:
        with self._lock, self._connect() as conn:
            if adapter_id is None:
                rows = conn.execute(
                    "SELECT payload FROM eval_reports ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT payload FROM eval_reports WHERE adapter_id = ? "
                    "ORDER BY created_at DESC",
                    (adapter_id,),
                ).fetchall()
        return [EvalReport.model_validate_json(r["payload"]) for r in rows]

    def save_eval_report(self, report: EvalReport) -> EvalReport:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO eval_reports (id, adapter_id, created_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    adapter_id=excluded.adapter_id,
                    payload=excluded.payload
                """,
                (
                    report.id,
                    report.adapter_id,
                    report.created_at.isoformat(),
                    report.model_dump_json(),
                ),
            )
        return report
