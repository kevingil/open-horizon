use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, SecondsFormat, Utc};
use rusqlite::{params, Connection, OptionalExtension, Row};

use horizon_core::events::DomainEvent;
use horizon_core::models::{
    ArtifactRecord, DashboardSnapshot, EvalReport, EventEnvelope, JobKind, JobRecord, JobStatus,
    RewardRecord, RunDetail, RunManifest, RunStatus, TaskSpec, TrainingRunRecord, TrajectoryRecord,
    TurnTrainingRecord, WorkerRecord, WorkerStatus,
};
use horizon_core::utc_now;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("serialize: {0}")]
    Json(#[from] serde_json::Error),
    #[error("corrupt row: {0}")]
    Corrupt(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

type Result<T> = std::result::Result<T, StoreError>;

const SCHEMA: &str = r#"
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

CREATE TABLE IF NOT EXISTS jobs (
  id               TEXT PRIMARY KEY,
  kind             TEXT NOT NULL,
  status           TEXT NOT NULL,
  payload          TEXT NOT NULL,
  lease_owner      TEXT,
  lease_until      TEXT,
  attempts         INTEGER NOT NULL DEFAULT 0,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  error            TEXT,
  created_at       TEXT NOT NULL,
  updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at);

CREATE TABLE IF NOT EXISTS events (
  seq        INTEGER PRIMARY KEY AUTOINCREMENT,
  kind       TEXT NOT NULL,
  run_id     TEXT,
  at         TEXT NOT NULL,
  payload    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_run_id ON events (run_id, seq);
"#;

/// Thread-safe store over one SQLite connection. Every method takes the
/// connection lock for the duration of a single statement or transaction,
/// which is the same serialisation the Python store enforced.
pub struct Store {
    conn: Mutex<Connection>,
    path: Option<PathBuf>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store").field("path", &self.path).finish()
    }
}

fn fmt_ts(ts: &DateTime<Utc>) -> String {
    ts.to_rfc3339_opts(SecondsFormat::Micros, false)
}

fn parse_ts(text: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(text)
        .map(|d| d.with_timezone(&Utc))
        .or_else(|_| {
            // Naive timestamps (no offset) are treated as UTC.
            chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f").map(|n| n.and_utc())
        })
        .map_err(|e| StoreError::Corrupt(format!("timestamp {text:?}: {e}")))
}

fn pack_ints(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn unpack_ints(blob: &[u8]) -> Vec<i32> {
    blob.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

impl Store {
    /// Private in-memory database. Used for the "memory" backend and tests.
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init(conn, None)
    }

    /// File-backed database in WAL mode. Creates parent directories.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(&path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Self::init(conn, Some(path))
    }

    fn init(conn: Connection, path: Option<PathBuf>) -> Result<Self> {
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            conn: Mutex::new(conn),
            path,
        })
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    fn with_conn<T>(&self, f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
        let guard = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        f(&guard)
    }

    // --- runs ---------------------------------------------------------------

    pub fn list_runs(&self) -> Result<Vec<RunManifest>> {
        self.with_conn(|c| {
            let mut stmt = c.prepare("SELECT * FROM runs ORDER BY created_at DESC")?;
            let rows = stmt.query_map([], row_to_manifest)?;
            rows.map(|r| r?).collect()
        })
    }

    pub fn get_run(&self, run_id: &str) -> Result<Option<RunDetail>> {
        self.with_conn(|c| {
            c.query_row(
                "SELECT * FROM runs WHERE id = ?1",
                params![run_id],
                row_to_detail,
            )
            .optional()?
            .transpose()
        })
    }

    pub fn save_run(&self, run: &RunDetail) -> Result<()> {
        let m = &run.manifest;
        let artifacts = serde_json::to_string(&run.artifacts)?;
        self.with_conn(|c| {
            c.execute(
                r#"INSERT INTO runs (id, model_id, adapter_id, dataset_slice, infra_target, seed, status,
                    created_at, updated_at, estimated_cost_usd, task_json, trajectory_json, reward_json, artifacts_json)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
                   ON CONFLICT(id) DO UPDATE SET
                    model_id=excluded.model_id, adapter_id=excluded.adapter_id, dataset_slice=excluded.dataset_slice,
                    infra_target=excluded.infra_target, seed=excluded.seed, status=excluded.status,
                    updated_at=excluded.updated_at, estimated_cost_usd=excluded.estimated_cost_usd,
                    task_json=excluded.task_json, trajectory_json=excluded.trajectory_json,
                    reward_json=excluded.reward_json, artifacts_json=excluded.artifacts_json"#,
                params![
                    m.id, m.model_id, m.adapter_id, m.dataset_slice, m.infra_target, m.seed, m.status.as_str(),
                    fmt_ts(&m.created_at), fmt_ts(&m.updated_at), m.estimated_cost_usd,
                    serde_json::to_string(&run.task)?, serde_json::to_string(&run.trajectory)?,
                    serde_json::to_string(&run.reward)?, artifacts,
                ],
            )?;
            Ok(())
        })
    }

    pub fn total_cost_since(&self, since: DateTime<Utc>) -> Result<f64> {
        self.with_conn(|c| {
            let total: f64 = c.query_row(
                "SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM runs WHERE created_at >= ?1",
                params![fmt_ts(&since)],
                |r| r.get(0),
            )?;
            Ok(horizon_core::round_to(total, 6))
        })
    }

    // --- workers ------------------------------------------------------------

    pub fn list_workers(&self) -> Result<Vec<WorkerRecord>> {
        self.with_conn(|c| {
            let mut stmt =
                c.prepare("SELECT id, role, status, run_id, detail FROM workers ORDER BY id")?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, String>(4)?,
                ))
            })?;
            rows.map(|r| {
                let (id, role, status, run_id, detail) = r?;
                let status = WorkerStatus::parse(&status)
                    .ok_or_else(|| StoreError::Corrupt(format!("worker status {status}")))?;
                Ok(WorkerRecord {
                    id,
                    role,
                    status,
                    run_id,
                    detail,
                })
            })
            .collect()
        })
    }

    pub fn upsert_worker(&self, worker: &WorkerRecord) -> Result<()> {
        self.with_conn(|c| {
            c.execute(
                r#"INSERT INTO workers (id, role, status, run_id, detail, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                   ON CONFLICT(id) DO UPDATE SET role=excluded.role, status=excluded.status, run_id=excluded.run_id,
                   detail=excluded.detail, updated_at=excluded.updated_at"#,
                params![worker.id, worker.role, worker.status.as_str(), worker.run_id, worker.detail, fmt_ts(&utc_now())],
            )?;
            Ok(())
        })
    }

    pub fn dashboard(&self) -> Result<DashboardSnapshot> {
        let runs = self.list_runs()?;
        let mut workers = self.list_workers()?;
        if workers.is_empty() {
            workers.push(WorkerRecord {
                id: "worker-rollout-local".into(),
                role: "rollout".into(),
                status: WorkerStatus::Idle,
                run_id: None,
                detail: "Local bootstrap worker".into(),
            });
        }
        let recent_artifacts = self.recent_artifacts(10)?;
        Ok(DashboardSnapshot {
            generated_at: utc_now(),
            runs,
            workers,
            recent_artifacts,
        })
    }

    fn recent_artifacts(&self, limit: usize) -> Result<Vec<ArtifactRecord>> {
        self.with_conn(|c| {
            let mut stmt =
                c.prepare("SELECT artifacts_json FROM runs ORDER BY created_at DESC LIMIT ?1")?;
            let rows = stmt.query_map(params![limit as i64], |r| r.get::<_, String>(0))?;
            let mut flat = Vec::new();
            for row in rows {
                let entries: Vec<ArtifactRecord> = serde_json::from_str(&row?)?;
                flat.extend(entries);
                if flat.len() >= limit {
                    break;
                }
            }
            flat.truncate(limit);
            Ok(flat)
        })
    }

    // --- turn training ------------------------------------------------------

    pub fn save_turn_training(&self, record: &TurnTrainingRecord) -> Result<()> {
        self.with_conn(|c| {
            c.execute(
                r#"INSERT INTO turn_training (id, run_id, step_index, model_name, token_count, prompt_ids,
                    completion_ids, attention_mask, loss_mask, sampling_args, created_at)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
                   ON CONFLICT(run_id, step_index) DO UPDATE SET id=excluded.id, model_name=excluded.model_name,
                    token_count=excluded.token_count, prompt_ids=excluded.prompt_ids, completion_ids=excluded.completion_ids,
                    attention_mask=excluded.attention_mask, loss_mask=excluded.loss_mask,
                    sampling_args=excluded.sampling_args, created_at=excluded.created_at"#,
                params![
                    record.id, record.run_id, record.step_index, record.model_name, record.token_count,
                    pack_ints(&record.prompt_ids), pack_ints(&record.completion_ids),
                    pack_ints(&record.attention_mask), pack_ints(&record.loss_mask),
                    serde_json::to_string(&record.sampling_args)?, fmt_ts(&record.created_at),
                ],
            )?;
            Ok(())
        })
    }

    pub fn get_turn_training(
        &self,
        run_id: &str,
        step_index: u32,
    ) -> Result<Option<TurnTrainingRecord>> {
        self.with_conn(|c| {
            c.query_row(
                "SELECT * FROM turn_training WHERE run_id = ?1 AND step_index = ?2",
                params![run_id, step_index],
                row_to_turn_training,
            )
            .optional()?
            .transpose()
        })
    }

    pub fn list_turn_training(&self, run_id: &str) -> Result<Vec<TurnTrainingRecord>> {
        self.with_conn(|c| {
            let mut stmt =
                c.prepare("SELECT * FROM turn_training WHERE run_id = ?1 ORDER BY step_index ASC")?;
            let rows = stmt.query_map(params![run_id], row_to_turn_training)?;
            rows.map(|r| r?).collect()
        })
    }

    // --- training runs / eval reports --------------------------------------

    pub fn list_training_runs(&self) -> Result<Vec<TrainingRunRecord>> {
        self.with_conn(|c| {
            let mut stmt =
                c.prepare("SELECT payload FROM training_runs ORDER BY created_at DESC")?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            rows.map(|r| Ok(serde_json::from_str(&r?)?)).collect()
        })
    }

    pub fn get_training_run(&self, id: &str) -> Result<Option<TrainingRunRecord>> {
        self.with_conn(|c| {
            let payload: Option<String> = c
                .query_row(
                    "SELECT payload FROM training_runs WHERE id = ?1",
                    params![id],
                    |r| r.get(0),
                )
                .optional()?;
            payload.map(|p| Ok(serde_json::from_str(&p)?)).transpose()
        })
    }

    pub fn save_training_run(&self, record: &TrainingRunRecord) -> Result<()> {
        self.with_conn(|c| {
            c.execute(
                r#"INSERT INTO training_runs (id, created_at, updated_at, status, payload) VALUES (?1, ?2, ?3, ?4, ?5)
                   ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at, status=excluded.status, payload=excluded.payload"#,
                params![record.id, fmt_ts(&record.created_at), fmt_ts(&record.updated_at), record.status.as_str(), serde_json::to_string(record)?],
            )?;
            Ok(())
        })
    }

    pub fn list_eval_reports(&self, adapter_id: Option<&str>) -> Result<Vec<EvalReport>> {
        self.with_conn(|c| {
            let mut out = Vec::new();
            match adapter_id {
                None => {
                    let mut stmt = c.prepare("SELECT payload FROM eval_reports ORDER BY created_at DESC")?;
                    for row in stmt.query_map([], |r| r.get::<_, String>(0))? {
                        out.push(serde_json::from_str(&row?)?);
                    }
                }
                Some(id) => {
                    let mut stmt = c.prepare("SELECT payload FROM eval_reports WHERE adapter_id = ?1 ORDER BY created_at DESC")?;
                    for row in stmt.query_map(params![id], |r| r.get::<_, String>(0))? {
                        out.push(serde_json::from_str(&row?)?);
                    }
                }
            }
            Ok(out)
        })
    }

    pub fn save_eval_report(&self, report: &EvalReport) -> Result<()> {
        self.with_conn(|c| {
            c.execute(
                r#"INSERT INTO eval_reports (id, adapter_id, created_at, payload) VALUES (?1, ?2, ?3, ?4)
                   ON CONFLICT(id) DO UPDATE SET adapter_id=excluded.adapter_id, payload=excluded.payload"#,
                params![report.id, report.adapter_id, fmt_ts(&report.created_at), serde_json::to_string(report)?],
            )?;
            Ok(())
        })
    }

    // --- jobs ---------------------------------------------------------------

    pub fn enqueue_job(
        &self,
        id: &str,
        kind: JobKind,
        payload: &serde_json::Value,
    ) -> Result<JobRecord> {
        let now = utc_now();
        self.with_conn(|c| {
            c.execute(
                "INSERT INTO jobs (id, kind, status, payload, attempts, cancel_requested, created_at, updated_at)
                 VALUES (?1, ?2, 'queued', ?3, 0, 0, ?4, ?4)",
                params![id, kind.as_str(), serde_json::to_string(payload)?, fmt_ts(&now)],
            )?;
            Ok(())
        })?;
        Ok(JobRecord {
            id: id.to_string(),
            kind,
            status: JobStatus::Queued,
            payload: payload.clone(),
            lease_owner: None,
            lease_until: None,
            attempts: 0,
            cancel_requested: false,
            error: None,
            created_at: now,
            updated_at: now,
        })
    }

    pub fn get_job(&self, id: &str) -> Result<Option<JobRecord>> {
        self.with_conn(|c| {
            c.query_row("SELECT * FROM jobs WHERE id = ?1", params![id], row_to_job)
                .optional()?
                .transpose()
        })
    }

    /// Atomically claim the oldest runnable job of `kind`: queued, or
    /// running with an expired lease (a crashed worker). Returns the job
    /// with its lease stamped.
    pub fn lease_job(
        &self,
        kind: JobKind,
        owner: &str,
        lease_secs: i64,
    ) -> Result<Option<JobRecord>> {
        let now = utc_now();
        let until = now + chrono::Duration::seconds(lease_secs);
        self.with_conn(|c| {
            let tx = c.unchecked_transaction()?;
            let candidate: Option<String> = tx
                .query_row(
                    "SELECT id FROM jobs WHERE kind = ?1 AND cancel_requested = 0 AND (status = 'queued'
                       OR (status = 'running' AND lease_until IS NOT NULL AND lease_until < ?2))
                     ORDER BY created_at ASC LIMIT 1",
                    params![kind.as_str(), fmt_ts(&now)],
                    |r| r.get(0),
                )
                .optional()?;
            let Some(id) = candidate else {
                tx.commit()?;
                return Ok(None);
            };
            tx.execute(
                "UPDATE jobs SET status = 'running', lease_owner = ?2, lease_until = ?3, attempts = attempts + 1,
                 updated_at = ?4 WHERE id = ?1",
                params![id, owner, fmt_ts(&until), fmt_ts(&now)],
            )?;
            let job = tx.query_row("SELECT * FROM jobs WHERE id = ?1", params![id], row_to_job)??;
            tx.commit()?;
            Ok(Some(job))
        })
    }

    pub fn heartbeat_job(&self, id: &str, owner: &str, lease_secs: i64) -> Result<bool> {
        let now = utc_now();
        let until = now + chrono::Duration::seconds(lease_secs);
        self.with_conn(|c| {
            let n = c.execute(
                "UPDATE jobs SET lease_until = ?3, updated_at = ?4 WHERE id = ?1 AND lease_owner = ?2 AND status = 'running'",
                params![id, owner, fmt_ts(&until), fmt_ts(&now)],
            )?;
            Ok(n == 1)
        })
    }

    pub fn finish_job(&self, id: &str, status: JobStatus, error: Option<&str>) -> Result<()> {
        self.with_conn(|c| {
            c.execute(
                "UPDATE jobs SET status = ?2, error = ?3, lease_until = NULL, updated_at = ?4 WHERE id = ?1",
                params![id, status.as_str(), error, fmt_ts(&utc_now())],
            )?;
            Ok(())
        })
    }

    /// Flag a job for cancellation. Returns false when the job is already
    /// terminal. Unknown ids return true so callers racing job creation
    /// don't need to care.
    pub fn request_cancel(&self, id: &str) -> Result<bool> {
        self.with_conn(|c| {
            let status: Option<String> =
                c.query_row("SELECT status FROM jobs WHERE id = ?1", params![id], |r| r.get(0)).optional()?;
            match status.as_deref().and_then(JobStatus::parse) {
                Some(s) if s.is_terminal() => Ok(false),
                Some(JobStatus::Queued) => {
                    c.execute(
                        "UPDATE jobs SET cancel_requested = 1, status = 'cancelled', updated_at = ?2 WHERE id = ?1",
                        params![id, fmt_ts(&utc_now())],
                    )?;
                    Ok(true)
                }
                _ => {
                    c.execute(
                        "UPDATE jobs SET cancel_requested = 1, updated_at = ?2 WHERE id = ?1",
                        params![id, fmt_ts(&utc_now())],
                    )?;
                    Ok(true)
                }
            }
        })
    }

    pub fn is_cancel_requested(&self, id: &str) -> Result<bool> {
        self.with_conn(|c| {
            let flag: Option<i64> = c
                .query_row(
                    "SELECT cancel_requested FROM jobs WHERE id = ?1",
                    params![id],
                    |r| r.get(0),
                )
                .optional()?;
            Ok(flag.unwrap_or(0) != 0)
        })
    }

    pub fn list_jobs(&self, limit: usize) -> Result<Vec<JobRecord>> {
        self.with_conn(|c| {
            let mut stmt = c.prepare("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?1")?;
            let rows = stmt.query_map(params![limit as i64], row_to_job)?;
            rows.map(|r| r?).collect()
        })
    }

    // --- events -------------------------------------------------------------

    pub fn append_event(&self, event: &DomainEvent) -> Result<i64> {
        self.with_conn(|c| {
            c.execute(
                "INSERT INTO events (kind, run_id, at, payload) VALUES (?1, ?2, ?3, ?4)",
                params![
                    event.kind(),
                    event.run_id(),
                    fmt_ts(&event.meta().at),
                    serde_json::to_string(event)?
                ],
            )?;
            Ok(c.last_insert_rowid())
        })
    }

    pub fn events_since(&self, since_seq: i64, limit: usize) -> Result<Vec<EventEnvelope>> {
        self.with_conn(|c| {
            let mut stmt = c.prepare(
                "SELECT seq, payload FROM events WHERE seq > ?1 ORDER BY seq ASC LIMIT ?2",
            )?;
            let rows = stmt.query_map(params![since_seq, limit as i64], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
            })?;
            rows.map(|r| {
                let (seq, payload) = r?;
                Ok(EventEnvelope {
                    seq,
                    event: serde_json::from_str(&payload)?,
                })
            })
            .collect()
        })
    }

    pub fn latest_events(&self, limit: usize) -> Result<Vec<EventEnvelope>> {
        self.with_conn(|c| {
            let mut stmt =
                c.prepare("SELECT seq, payload FROM events ORDER BY seq DESC LIMIT ?1")?;
            let rows = stmt.query_map(params![limit as i64], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
            })?;
            let mut out: Vec<EventEnvelope> = rows
                .map(|r| {
                    let (seq, payload) = r?;
                    Ok(EventEnvelope {
                        seq,
                        event: serde_json::from_str(&payload)?,
                    })
                })
                .collect::<Result<_>>()?;
            out.reverse();
            Ok(out)
        })
    }

    pub fn last_event_seq(&self) -> Result<i64> {
        self.with_conn(|c| {
            Ok(c.query_row("SELECT COALESCE(MAX(seq), 0) FROM events", [], |r| r.get(0))?)
        })
    }
}

fn row_to_manifest(row: &Row<'_>) -> rusqlite::Result<Result<RunManifest>> {
    let status: String = row.get("status")?;
    let created: String = row.get("created_at")?;
    let updated: String = row.get("updated_at")?;
    Ok((|| {
        Ok(RunManifest {
            id: row.get("id")?,
            model_id: row.get("model_id")?,
            adapter_id: row.get("adapter_id")?,
            dataset_slice: row.get("dataset_slice")?,
            infra_target: row.get("infra_target")?,
            seed: row.get("seed")?,
            status: RunStatus::parse(&status)
                .ok_or_else(|| StoreError::Corrupt(format!("run status {status}")))?,
            created_at: parse_ts(&created)?,
            updated_at: parse_ts(&updated)?,
            estimated_cost_usd: row.get("estimated_cost_usd")?,
        })
    })())
}

fn row_to_detail(row: &Row<'_>) -> rusqlite::Result<Result<RunDetail>> {
    let manifest = row_to_manifest(row)?;
    let task: String = row.get("task_json")?;
    let trajectory: String = row.get("trajectory_json")?;
    let reward: String = row.get("reward_json")?;
    let artifacts: String = row.get("artifacts_json")?;
    Ok((|| {
        let task: TaskSpec = serde_json::from_str(&task)?;
        let trajectory: TrajectoryRecord = serde_json::from_str(&trajectory)?;
        let reward: RewardRecord = serde_json::from_str(&reward)?;
        let artifacts: Vec<ArtifactRecord> = serde_json::from_str(&artifacts)?;
        Ok(RunDetail {
            manifest: manifest?,
            task,
            trajectory,
            reward,
            artifacts,
        })
    })())
}

fn row_to_turn_training(row: &Row<'_>) -> rusqlite::Result<Result<TurnTrainingRecord>> {
    let sampling: String = row.get("sampling_args")?;
    let created: String = row.get("created_at")?;
    let prompt: Vec<u8> = row.get("prompt_ids")?;
    let completion: Vec<u8> = row.get("completion_ids")?;
    let attention: Vec<u8> = row.get("attention_mask")?;
    let loss: Vec<u8> = row.get("loss_mask")?;
    let id: String = row.get("id")?;
    let run_id: String = row.get("run_id")?;
    let step_index: u32 = row.get("step_index")?;
    let model_name: String = row.get("model_name")?;
    let token_count: u32 = row.get("token_count")?;
    Ok((|| {
        Ok(TurnTrainingRecord {
            id,
            run_id,
            step_index,
            prompt_ids: unpack_ints(&prompt),
            completion_ids: unpack_ints(&completion),
            attention_mask: unpack_ints(&attention),
            loss_mask: unpack_ints(&loss),
            sampling_args: serde_json::from_str(&sampling)?,
            model_name,
            token_count,
            created_at: parse_ts(&created)?,
        })
    })())
}

fn row_to_job(row: &Row<'_>) -> rusqlite::Result<Result<JobRecord>> {
    let kind: String = row.get("kind")?;
    let status: String = row.get("status")?;
    let payload: String = row.get("payload")?;
    let lease_until: Option<String> = row.get("lease_until")?;
    let created: String = row.get("created_at")?;
    let updated: String = row.get("updated_at")?;
    let id: String = row.get("id")?;
    let lease_owner: Option<String> = row.get("lease_owner")?;
    let attempts: u32 = row.get("attempts")?;
    let cancel_requested: i64 = row.get("cancel_requested")?;
    let error: Option<String> = row.get("error")?;
    Ok((|| {
        Ok(JobRecord {
            id,
            kind: JobKind::parse(&kind)
                .ok_or_else(|| StoreError::Corrupt(format!("job kind {kind}")))?,
            status: JobStatus::parse(&status)
                .ok_or_else(|| StoreError::Corrupt(format!("job status {status}")))?,
            payload: serde_json::from_str(&payload)?,
            lease_owner,
            lease_until: lease_until.as_deref().map(parse_ts).transpose()?,
            attempts,
            cancel_requested: cancel_requested != 0,
            error,
            created_at: parse_ts(&created)?,
            updated_at: parse_ts(&updated)?,
        })
    })())
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizon_core::models::{ToolPermission, TrajectoryStep};

    fn detail(id: &str, cost: f64) -> RunDetail {
        let task = TaskSpec {
            id: "task-1".into(),
            prompt: "p".into(),
            repo_snapshot: ".".into(),
            tool_permissions: vec![ToolPermission::Read],
            horizon: 4,
            success_criteria: vec![],
        };
        let trajectory = TrajectoryRecord::new(
            "task-1",
            vec![TrajectoryStep::new(0, "policy", "action", "{}")],
            vec![],
        );
        let reward = RewardRecord {
            trajectory_id: trajectory.id.clone(),
            terminal_reward: 0.5,
            step_rewards: vec![0.5],
            penalties: vec![],
            audit_flags: vec![],
            provenance: "test".into(),
        };
        RunDetail {
            manifest: RunManifest {
                id: id.into(),
                model_id: "openai:x".into(),
                adapter_id: None,
                dataset_slice: "bootstrap".into(),
                infra_target: "mac-local".into(),
                seed: 7,
                status: RunStatus::Completed,
                created_at: utc_now(),
                updated_at: utc_now(),
                estimated_cost_usd: cost,
            },
            task,
            trajectory,
            reward,
            artifacts: vec![ArtifactRecord {
                name: "a".into(),
                kind: "manifest".into(),
                path: "p".into(),
            }],
        }
    }

    #[test]
    fn runs_round_trip() {
        let store = Store::in_memory().unwrap();
        let d = detail("run-1", 0.25);
        store.save_run(&d).unwrap();
        let back = store.get_run("run-1").unwrap().unwrap();
        assert_eq!(back.task, d.task);
        assert_eq!(back.trajectory.steps.len(), 1);
        assert_eq!(store.list_runs().unwrap().len(), 1);
        let since = utc_now() - chrono::Duration::hours(1);
        assert_eq!(store.total_cost_since(since).unwrap(), 0.25);
        assert_eq!(store.dashboard().unwrap().recent_artifacts.len(), 1);
        assert!(store.get_run("nope").unwrap().is_none());
    }

    #[test]
    fn turn_training_packs_ints() {
        let store = Store::in_memory().unwrap();
        let rec = TurnTrainingRecord {
            id: "turn-1".into(),
            run_id: "run-1".into(),
            step_index: 2,
            prompt_ids: vec![1, -2, 300000],
            completion_ids: vec![],
            attention_mask: vec![1, 1, 1],
            loss_mask: vec![0, 1, 1],
            sampling_args: serde_json::json!({"max_tokens": 5})
                .as_object()
                .unwrap()
                .clone(),
            model_name: "m".into(),
            token_count: 3,
            created_at: utc_now(),
        };
        store.save_turn_training(&rec).unwrap();
        let back = store.get_turn_training("run-1", 2).unwrap().unwrap();
        assert_eq!(back.prompt_ids, vec![1, -2, 300000]);
        assert_eq!(store.list_turn_training("run-1").unwrap().len(), 1);
    }

    #[test]
    fn jobs_lease_cancel_and_expire() {
        let store = Store::in_memory().unwrap();
        store
            .enqueue_job("job-1", JobKind::Rollout, &serde_json::json!({"x": 1}))
            .unwrap();
        assert!(store
            .lease_job(JobKind::Training, "w", 30)
            .unwrap()
            .is_none());
        let job = store
            .lease_job(JobKind::Rollout, "w1", 30)
            .unwrap()
            .unwrap();
        assert_eq!(job.status, JobStatus::Running);
        assert_eq!(job.attempts, 1);
        // Held lease: nobody else can take it.
        assert!(store
            .lease_job(JobKind::Rollout, "w2", 30)
            .unwrap()
            .is_none());
        // Expired lease: recoverable by another worker.
        store.heartbeat_job("job-1", "w1", -5).unwrap();
        let job = store
            .lease_job(JobKind::Rollout, "w2", 30)
            .unwrap()
            .unwrap();
        assert_eq!(job.lease_owner.as_deref(), Some("w2"));
        assert_eq!(job.attempts, 2);
        assert!(store.request_cancel("job-1").unwrap());
        assert!(store.is_cancel_requested("job-1").unwrap());
        store
            .finish_job("job-1", JobStatus::Cancelled, None)
            .unwrap();
        assert!(!store.request_cancel("job-1").unwrap());
        // Queued jobs cancel immediately.
        store
            .enqueue_job("job-2", JobKind::Rollout, &serde_json::json!({}))
            .unwrap();
        assert!(store.request_cancel("job-2").unwrap());
        assert_eq!(
            store.get_job("job-2").unwrap().unwrap().status,
            JobStatus::Cancelled
        );
        assert!(store
            .lease_job(JobKind::Rollout, "w3", 30)
            .unwrap()
            .is_none());
    }

    #[test]
    fn event_log_sequences() {
        let store = Store::in_memory().unwrap();
        let s1 = store
            .append_event(&DomainEvent::rollout_failed("run-1", "a"))
            .unwrap();
        let s2 = store
            .append_event(&DomainEvent::rollout_failed("run-1", "b"))
            .unwrap();
        assert!(s2 > s1);
        let after = store.events_since(s1, 10).unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].seq, s2);
        assert_eq!(store.latest_events(1).unwrap()[0].seq, s2);
        assert_eq!(store.last_event_seq().unwrap(), s2);
    }

    #[test]
    fn file_backed_reopens() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("runs.db");
        {
            let store = Store::open(&path).unwrap();
            store.save_run(&detail("run-1", 0.0)).unwrap();
        }
        let store = Store::open(&path).unwrap();
        assert!(store.get_run("run-1").unwrap().is_some());
    }
}
