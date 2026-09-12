use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension, Row};

use horizon_core::events::{Event, EventEnvelope};
use horizon_core::models::{
    DashboardSnapshot, EvalReport, EvalTaskScore, JobCounts, JobKind, JobRecord, JobStatus,
    LatencyStats, NodeRecord, NodeStatus, RewardBin, RewardRecord, RewardSignal, RunDetail,
    RunManifest, RunStatus, Stats, StatsBucket, TaskSpec, TrainingMetricPoint, TrainingRunRecord,
    TrainingStatus, Trajectory, TrajectoryStep, TurnTrainingRecord,
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

/// Timestamps are stored as microseconds since the Unix epoch.
const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS runs (
  id               TEXT PRIMARY KEY,
  profile          TEXT NOT NULL,
  model_id         TEXT NOT NULL,
  adapter_id       TEXT,
  infra_target     TEXT NOT NULL,
  status           TEXT NOT NULL,
  horizon          INTEGER NOT NULL,
  tokens           INTEGER NOT NULL DEFAULT 0,
  cost_usd         REAL NOT NULL DEFAULT 0,
  terminal_reward  REAL,
  step_count       INTEGER NOT NULL DEFAULT 0,
  error            TEXT,
  task_id          TEXT NOT NULL,
  prompt           TEXT NOT NULL,
  repo_snapshot    TEXT NOT NULL,
  success_criteria TEXT NOT NULL,
  errors           TEXT NOT NULL DEFAULT '[]',
  created_at       INTEGER NOT NULL,
  started_at       INTEGER,
  finished_at      INTEGER,
  updated_at       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_created_at ON runs (created_at DESC);
CREATE INDEX IF NOT EXISTS runs_finished_at ON runs (finished_at);
CREATE INDEX IF NOT EXISTS runs_status ON runs (status);

CREATE TABLE IF NOT EXISTS run_steps (
  run_id                TEXT NOT NULL,
  idx                   INTEGER NOT NULL,
  actor                 TEXT NOT NULL,
  kind                  TEXT NOT NULL,
  content               TEXT NOT NULL,
  at                    INTEGER NOT NULL,
  duration_ms           INTEGER,
  has_training_metadata INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (run_id, idx)
);
CREATE INDEX IF NOT EXISTS run_steps_at ON run_steps (at);

CREATE TABLE IF NOT EXISTS rewards (
  run_id          TEXT PRIMARY KEY,
  terminal_reward REAL NOT NULL,
  rubric          TEXT NOT NULL,
  source          TEXT NOT NULL,
  audit_flags     TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS reward_signals (
  run_id  TEXT NOT NULL,
  ord     INTEGER NOT NULL,
  name    TEXT NOT NULL,
  value   REAL NOT NULL,
  weight  REAL NOT NULL,
  reason  TEXT NOT NULL,
  PRIMARY KEY (run_id, ord)
);

CREATE TABLE IF NOT EXISTS nodes (
  id        TEXT PRIMARY KEY,
  role      TEXT NOT NULL,
  status    TEXT NOT NULL,
  detail    TEXT NOT NULL,
  meta      TEXT NOT NULL DEFAULT '{}',
  last_seen INTEGER NOT NULL
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
  created_at      INTEGER NOT NULL,
  PRIMARY KEY (run_id, step_index)
);

CREATE TABLE IF NOT EXISTS training_runs (
  id             TEXT PRIMARY KEY,
  status         TEXT NOT NULL,
  trainer        TEXT NOT NULL,
  adapter_in     TEXT,
  adapter_out    TEXT,
  sample_run_ids TEXT NOT NULL,
  hyperparams    TEXT NOT NULL,
  error          TEXT,
  created_at     INTEGER NOT NULL,
  updated_at     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS training_runs_created_at ON training_runs (created_at DESC);

CREATE TABLE IF NOT EXISTS training_metrics (
  training_run_id TEXT NOT NULL,
  step            INTEGER NOT NULL,
  loss            REAL NOT NULL,
  mean_reward     REAL,
  kl              REAL,
  extra           TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (training_run_id, step)
);

CREATE TABLE IF NOT EXISTS eval_reports (
  id          TEXT PRIMARY KEY,
  adapter_id  TEXT NOT NULL,
  task_set    TEXT NOT NULL,
  mean_reward REAL NOT NULL,
  per_task    TEXT NOT NULL,
  created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS eval_reports_adapter ON eval_reports (adapter_id, created_at DESC);

CREATE TABLE IF NOT EXISTS jobs (
  id               TEXT PRIMARY KEY,
  kind             TEXT NOT NULL,
  status           TEXT NOT NULL,
  payload          TEXT NOT NULL,
  lease_owner      TEXT,
  lease_until      INTEGER,
  attempts         INTEGER NOT NULL DEFAULT 0,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  error            TEXT,
  created_at       INTEGER NOT NULL,
  updated_at       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at);

CREATE TABLE IF NOT EXISTS events (
  seq          INTEGER PRIMARY KEY AUTOINCREMENT,
  kind         TEXT NOT NULL,
  subject_kind TEXT,
  subject_id   TEXT,
  at           INTEGER NOT NULL,
  payload      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_subject ON events (subject_kind, subject_id, seq);
"#;

/// Thread-safe store over one SQLite connection.
pub struct Store {
    conn: Mutex<Connection>,
    path: Option<PathBuf>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store").field("path", &self.path).finish()
    }
}

fn micros(ts: &DateTime<Utc>) -> i64 {
    ts.timestamp_micros()
}

fn from_micros(us: i64) -> Result<DateTime<Utc>> {
    DateTime::from_timestamp_micros(us)
        .ok_or_else(|| StoreError::Corrupt(format!("timestamp {us}")))
}

fn pack_ints(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn unpack_ints(blob: &[u8]) -> Vec<i32> {
    blob.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn corrupt<T>(what: &str, value: &str) -> Result<T> {
    Err(StoreError::Corrupt(format!("{what} {value:?}")))
}

impl Store {
    /// Private in-memory database. Used for the "memory" backend and tests.
    pub fn in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?, None)
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

    pub fn get_manifest(&self, run_id: &str) -> Result<Option<RunManifest>> {
        self.with_conn(|c| {
            c.query_row(
                "SELECT * FROM runs WHERE id = ?1",
                params![run_id],
                row_to_manifest,
            )
            .optional()?
            .transpose()
        })
    }

    pub fn get_run(&self, run_id: &str) -> Result<Option<RunDetail>> {
        self.with_conn(|c| {
            let Some((manifest, task, errors)) = c
                .query_row("SELECT * FROM runs WHERE id = ?1", params![run_id], |row| {
                    let task = TaskSpec {
                        id: row.get("task_id")?,
                        prompt: row.get("prompt")?,
                        repo_snapshot: row.get("repo_snapshot")?,
                        horizon: row.get("horizon")?,
                        success_criteria: Vec::new(),
                    };
                    let criteria: String = row.get("success_criteria")?;
                    let errors: String = row.get("errors")?;
                    Ok((row_to_manifest(row)?, (task, criteria), errors))
                })
                .optional()?
            else {
                return Ok(None);
            };
            let manifest = manifest?;
            let (mut task, criteria) = task;
            task.success_criteria = serde_json::from_str(&criteria)?;
            let errors: Vec<String> = serde_json::from_str(&errors)?;
            let mut stmt = c.prepare(
                "SELECT idx, actor, kind, content, at, duration_ms, has_training_metadata FROM run_steps WHERE run_id = ?1 ORDER BY idx",
            )?;
            let steps = stmt
                .query_map(params![run_id], |r| {
                    Ok((
                        r.get::<_, u32>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, i64>(4)?,
                        r.get::<_, Option<i64>>(5)?,
                        r.get::<_, i64>(6)?,
                    ))
                })?
                .map(|r| {
                    let (index, actor, kind, content, at, duration_ms, flag) = r?;
                    Ok(TrajectoryStep {
                        index,
                        actor,
                        kind,
                        content,
                        at: from_micros(at)?,
                        duration_ms: duration_ms.map(|d| d.max(0) as u64),
                        has_training_metadata: flag != 0,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let reward = load_reward(c, run_id)?;
            Ok(Some(RunDetail { manifest, task, trajectory: Trajectory { steps, errors }, reward }))
        })
    }

    /// Upsert the whole run: manifest, task, steps, reward.
    pub fn save_run(&self, run: &RunDetail) -> Result<()> {
        let m = &run.manifest;
        self.with_conn(|c| {
            let tx = c.unchecked_transaction()?;
            tx.execute(
                r#"INSERT INTO runs (id, profile, model_id, adapter_id, infra_target, status, horizon, tokens, cost_usd,
                    terminal_reward, step_count, error, task_id, prompt, repo_snapshot, success_criteria, errors,
                    created_at, started_at, finished_at, updated_at)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)
                   ON CONFLICT(id) DO UPDATE SET profile=excluded.profile, model_id=excluded.model_id, adapter_id=excluded.adapter_id,
                    infra_target=excluded.infra_target, status=excluded.status, horizon=excluded.horizon, tokens=excluded.tokens,
                    cost_usd=excluded.cost_usd, terminal_reward=excluded.terminal_reward, step_count=excluded.step_count,
                    error=excluded.error, task_id=excluded.task_id, prompt=excluded.prompt, repo_snapshot=excluded.repo_snapshot,
                    success_criteria=excluded.success_criteria, errors=excluded.errors, started_at=excluded.started_at,
                    finished_at=excluded.finished_at, updated_at=excluded.updated_at"#,
                params![
                    m.id, m.profile, m.model_id, m.adapter_id, m.infra_target, m.status.as_str(), m.horizon, m.tokens as i64,
                    m.cost_usd, m.terminal_reward, m.step_count, m.error, run.task.id, run.task.prompt, run.task.repo_snapshot,
                    serde_json::to_string(&run.task.success_criteria)?, serde_json::to_string(&run.trajectory.errors)?,
                    micros(&m.created_at), m.started_at.as_ref().map(micros), m.finished_at.as_ref().map(micros), micros(&m.updated_at),
                ],
            )?;
            tx.execute("DELETE FROM run_steps WHERE run_id = ?1", params![m.id])?;
            {
                let mut stmt = tx.prepare(
                    "INSERT INTO run_steps (run_id, idx, actor, kind, content, at, duration_ms, has_training_metadata) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                )?;
                for s in &run.trajectory.steps {
                    stmt.execute(params![
                        m.id, s.index, s.actor, s.kind, s.content, micros(&s.at),
                        s.duration_ms.map(|d| d as i64), s.has_training_metadata as i64
                    ])?;
                }
            }
            save_reward(&tx, &m.id, run.reward.as_ref())?;
            tx.commit()?;
            Ok(())
        })
    }

    /// Append one step without rewriting the run (used while a rollout is live).
    pub fn append_step(&self, run_id: &str, step: &TrajectoryStep) -> Result<()> {
        self.with_conn(|c| {
            c.execute(
                "INSERT OR REPLACE INTO run_steps (run_id, idx, actor, kind, content, at, duration_ms, has_training_metadata) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    run_id, step.index, step.actor, step.kind, step.content, micros(&step.at),
                    step.duration_ms.map(|d| d as i64), step.has_training_metadata as i64
                ],
            )?;
            c.execute(
                "UPDATE runs SET step_count = (SELECT COUNT(*) FROM run_steps WHERE run_id = ?1), updated_at = ?2 WHERE id = ?1",
                params![run_id, micros(&utc_now())],
            )?;
            Ok(())
        })
    }

    pub fn save_reward(&self, run_id: &str, reward: &RewardRecord) -> Result<()> {
        self.with_conn(|c| {
            let tx = c.unchecked_transaction()?;
            save_reward(&tx, run_id, Some(reward))?;
            tx.execute(
                "UPDATE runs SET terminal_reward = ?2, updated_at = ?3 WHERE id = ?1",
                params![run_id, reward.terminal_reward, micros(&utc_now())],
            )?;
            tx.commit()?;
            Ok(())
        })
    }

    pub fn total_cost_since(&self, since: DateTime<Utc>) -> Result<f64> {
        self.with_conn(|c| {
            let total: f64 = c.query_row(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM runs WHERE created_at >= ?1",
                params![micros(&since)],
                |r| r.get(0),
            )?;
            Ok(horizon_core::round_to(total, 6))
        })
    }

    // --- fleet ------------------------------------------------------------

    pub fn list_nodes(&self) -> Result<Vec<NodeRecord>> {
        self.with_conn(|c| {
            let mut stmt = c.prepare(
                "SELECT id, role, status, detail, meta, last_seen FROM nodes ORDER BY role, id",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, i64>(5)?,
                ))
            })?;
            rows.map(|r| {
                let (id, role, status, detail, meta, last_seen) = r?;
                let Some(status) = NodeStatus::parse(&status) else {
                    return corrupt("node status", &status);
                };
                Ok(NodeRecord {
                    id,
                    role,
                    status,
                    detail,
                    meta: serde_json::from_str(&meta)?,
                    last_seen: from_micros(last_seen)?,
                })
            })
            .collect()
        })
    }

    pub fn upsert_node(&self, node: &NodeRecord) -> Result<()> {
        self.with_conn(|c| {
            c.execute(
                r#"INSERT INTO nodes (id, role, status, detail, meta, last_seen) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                   ON CONFLICT(id) DO UPDATE SET role=excluded.role, status=excluded.status, detail=excluded.detail,
                   meta=excluded.meta, last_seen=excluded.last_seen"#,
                params![node.id, node.role, node.status.as_str(), node.detail, serde_json::to_string(&node.meta)?, micros(&node.last_seen)],
            )?;
            Ok(())
        })
    }

    /// Mark nodes whose heartbeat is older than `stale` as down. Returns
    /// the nodes that changed so callers can emit events.
    pub fn expire_nodes(&self, stale: chrono::Duration) -> Result<Vec<NodeRecord>> {
        let cutoff = micros(&(utc_now() - stale));
        let changed: Vec<NodeRecord> = self
            .list_nodes()?
            .into_iter()
            .filter(|n| n.status != NodeStatus::Down && micros(&n.last_seen) < cutoff)
            .collect();
        self.with_conn(|c| {
            for n in &changed {
                c.execute(
                    "UPDATE nodes SET status = 'down' WHERE id = ?1",
                    params![n.id],
                )?;
            }
            Ok(())
        })?;
        Ok(changed
            .into_iter()
            .map(|n| NodeRecord {
                status: NodeStatus::Down,
                ..n
            })
            .collect())
    }

    // --- run queries ----------------------------------------------------

    /// Filtered run listing, newest first.
    pub fn query_runs(
        &self,
        status: Option<RunStatus>,
        profile: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<RunManifest>> {
        self.with_conn(|c| {
            let mut sql = String::from("SELECT * FROM runs WHERE 1=1");
            let mut args: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
            if let Some(s) = status {
                sql.push_str(" AND status = ?");
                args.push(Box::new(s.as_str().to_string()));
            }
            if let Some(p) = profile {
                sql.push_str(" AND profile = ?");
                args.push(Box::new(p.to_string()));
            }
            sql.push_str(" ORDER BY created_at DESC LIMIT ? OFFSET ?");
            args.push(Box::new(limit as i64));
            args.push(Box::new(offset as i64));
            let mut stmt = c.prepare(&sql)?;
            let rows = stmt.query_map(
                rusqlite::params_from_iter(args.iter().map(|a| a.as_ref())),
                row_to_manifest,
            )?;
            rows.map(|r| r?).collect()
        })
    }

    pub fn dashboard(&self, window: chrono::Duration) -> Result<DashboardSnapshot> {
        Ok(DashboardSnapshot {
            generated_at: utc_now(),
            runs: self.query_runs(None, None, 200, 0)?,
            nodes: self.list_nodes()?,
            jobs: self.job_counts()?,
            stats: self.stats(window)?,
        })
    }

    // --- stats ------------------------------------------------------------

    /// Rolled-up health over the trailing `window`.
    pub fn stats(&self, window: chrono::Duration) -> Result<Stats> {
        let now = utc_now();
        let since = micros(&(now - window));
        let window_s = window.num_seconds().max(1) as u64;
        self.with_conn(|c| {
            let count = |status: &str| -> rusqlite::Result<u32> {
                c.query_row("SELECT COUNT(*) FROM runs WHERE status = ?1", params![status], |r| r.get(0))
            };
            let in_flight = count("running")?;
            let queued = count("queued")?;
            let (completed, failed, tokens, cost): (u32, u32, i64, f64) = c.query_row(
                "SELECT COALESCE(SUM(status = 'completed'), 0), COALESCE(SUM(status = 'failed'), 0),
                        COALESCE(SUM(tokens), 0), COALESCE(SUM(cost_usd), 0)
                 FROM runs WHERE finished_at IS NOT NULL AND finished_at >= ?1",
                params![since],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )?;
            let mean_reward: Option<f64> = c.query_row(
                "SELECT AVG(terminal_reward) FROM (SELECT terminal_reward FROM runs WHERE status = 'completed'
                   AND terminal_reward IS NOT NULL ORDER BY finished_at DESC LIMIT 50)",
                [],
                |r| r.get(0),
            )?;
            let latencies = |kind: &str| -> Result<LatencyStats> {
                let mut stmt = c.prepare(
                    "SELECT duration_ms FROM run_steps WHERE kind = ?1 AND duration_ms IS NOT NULL AND at >= ?2 ORDER BY at DESC LIMIT 5000",
                )?;
                let mut v: Vec<f64> = stmt.query_map(params![kind, since], |r| r.get::<_, i64>(0))?.map(|r| r.map(|x| x as f64)).collect::<rusqlite::Result<_>>()?;
                Ok(percentiles(&mut v))
            };
            let terminal = completed + failed;
            Ok(Stats {
                window_s,
                in_flight,
                queued,
                rollouts_completed: completed,
                rollouts_failed: failed,
                rollouts_per_min: completed as f64 / (window_s as f64 / 60.0),
                tokens_per_s: tokens.max(0) as f64 / window_s as f64,
                cost_per_hour: cost / (window_s as f64 / 3600.0),
                success_rate: if terminal > 0 { Some(completed as f64 / terminal as f64) } else { None },
                mean_reward,
                policy_latency: latencies("action")?,
                tool_latency: latencies("observation")?,
            })
        })
    }

    /// Throughput time series: fixed buckets over the trailing window.
    pub fn timeseries(
        &self,
        window: chrono::Duration,
        bucket: chrono::Duration,
    ) -> Result<Vec<StatsBucket>> {
        let now = utc_now();
        let bucket_s = bucket.num_seconds().max(1);
        let start_s = (now - window).timestamp() / bucket_s * bucket_s;
        let n = ((now.timestamp() - start_s) / bucket_s + 1).max(1) as usize;
        let mut out: Vec<StatsBucket> = (0..n)
            .map(|i| StatsBucket {
                ts: start_s + i as i64 * bucket_s,
                ..Default::default()
            })
            .collect();
        let idx = |t_us: i64| -> Option<usize> {
            let i = (t_us / 1_000_000 - start_s) / bucket_s;
            if i >= 0 && (i as usize) < n {
                Some(i as usize)
            } else {
                None
            }
        };
        let since = start_s * 1_000_000;
        self.with_conn(|c| {
            let mut stmt = c.prepare(
                "SELECT status, tokens, cost_usd, terminal_reward, finished_at FROM runs WHERE finished_at IS NOT NULL AND finished_at >= ?1",
            )?;
            let mut reward_acc: Vec<(f64, u32)> = vec![(0.0, 0); n];
            for row in stmt.query_map(params![since], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, f64>(2)?, r.get::<_, Option<f64>>(3)?, r.get::<_, i64>(4)?))
            })? {
                let (status, tokens, cost, reward, finished) = row?;
                let Some(i) = idx(finished) else { continue };
                let b = &mut out[i];
                match status.as_str() {
                    "completed" => b.completed += 1,
                    "failed" => b.failed += 1,
                    _ => {}
                }
                b.tokens += tokens.max(0) as u64;
                b.cost_usd += cost;
                if let Some(r) = reward {
                    reward_acc[i].0 += r;
                    reward_acc[i].1 += 1;
                }
            }
            for (i, (sum, cnt)) in reward_acc.into_iter().enumerate() {
                if cnt > 0 {
                    out[i].mean_reward = Some(sum / cnt as f64);
                }
            }
            // In-flight sampled as runs that had started and not finished at bucket start.
            let mut stmt = c.prepare("SELECT started_at, finished_at FROM runs WHERE started_at IS NOT NULL AND (finished_at IS NULL OR finished_at >= ?1)")?;
            let spans: Vec<(i64, Option<i64>)> = stmt.query_map(params![since], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<rusqlite::Result<_>>()?;
            // Sample in-flight at the end of each bucket (or now for the open one).
            let now_us = micros(&now);
            for b in out.iter_mut() {
                let t = ((b.ts + bucket_s) * 1_000_000).min(now_us);
                b.in_flight = spans.iter().filter(|(s, f)| *s <= t && f.is_none_or(|f| f > t)).count() as u32;
            }
            let mut stmt = c.prepare("SELECT at, duration_ms FROM run_steps WHERE kind = 'action' AND duration_ms IS NOT NULL AND at >= ?1")?;
            let mut lat: Vec<Vec<f64>> = vec![Vec::new(); n];
            for row in stmt.query_map(params![since], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))? {
                let (at, d) = row?;
                if let Some(i) = idx(at) {
                    lat[i].push(d as f64);
                }
            }
            for (i, v) in lat.iter_mut().enumerate() {
                if !v.is_empty() {
                    out[i].p95_policy_ms = Some(percentiles(v).p95_ms);
                }
            }
            Ok(out)
        })
    }

    /// Histogram of terminal rewards over the trailing window, `bins` wide across [-1, 1].
    pub fn reward_histogram(
        &self,
        window: chrono::Duration,
        bins: usize,
    ) -> Result<Vec<RewardBin>> {
        let since = micros(&(utc_now() - window));
        let bins = bins.clamp(2, 100);
        let width = 2.0 / bins as f64;
        let mut out: Vec<RewardBin> = (0..bins)
            .map(|i| RewardBin {
                lo: -1.0 + i as f64 * width,
                hi: -1.0 + (i + 1) as f64 * width,
                count: 0,
            })
            .collect();
        self.with_conn(|c| {
            let mut stmt = c.prepare("SELECT terminal_reward FROM runs WHERE terminal_reward IS NOT NULL AND finished_at >= ?1")?;
            for row in stmt.query_map(params![since], |r| r.get::<_, f64>(0))? {
                let r = row?.clamp(-1.0, 1.0);
                let i = (((r + 1.0) / width) as usize).min(bins - 1);
                out[i].count += 1;
            }
            Ok(out)
        })
    }

    // --- turn training ------------------------------------------------------

    pub fn save_turn_training(&self, record: &TurnTrainingRecord) -> Result<()> {
        self.with_conn(|c| {
            c.execute(
                r#"INSERT OR REPLACE INTO turn_training (id, run_id, step_index, model_name, token_count, prompt_ids,
                    completion_ids, attention_mask, loss_mask, sampling_args, created_at)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)"#,
                params![
                    record.id, record.run_id, record.step_index, record.model_name, record.token_count,
                    pack_ints(&record.prompt_ids), pack_ints(&record.completion_ids), pack_ints(&record.attention_mask),
                    pack_ints(&record.loss_mask), serde_json::to_string(&record.sampling_args)?, micros(&record.created_at),
                ],
            )?;
            c.execute("UPDATE run_steps SET has_training_metadata = 1 WHERE run_id = ?1 AND idx = ?2", params![record.run_id, record.step_index])?;
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
            let mut stmt = c.prepare("SELECT * FROM training_runs ORDER BY created_at DESC")?;
            let rows = stmt.query_map([], row_to_training_run)?;
            let mut out: Vec<TrainingRunRecord> = rows.map(|r| r?).collect::<Result<_>>()?;
            for record in &mut out {
                record.metrics = load_metrics(c, &record.id)?;
            }
            Ok(out)
        })
    }

    pub fn get_training_run(&self, id: &str) -> Result<Option<TrainingRunRecord>> {
        self.with_conn(|c| {
            let Some(record) = c
                .query_row(
                    "SELECT * FROM training_runs WHERE id = ?1",
                    params![id],
                    row_to_training_run,
                )
                .optional()?
            else {
                return Ok(None);
            };
            let mut record = record?;
            record.metrics = load_metrics(c, id)?;
            Ok(Some(record))
        })
    }

    pub fn save_training_run(&self, record: &TrainingRunRecord) -> Result<()> {
        self.with_conn(|c| {
            let tx = c.unchecked_transaction()?;
            tx.execute(
                r#"INSERT INTO training_runs (id, status, trainer, adapter_in, adapter_out, sample_run_ids, hyperparams, error, created_at, updated_at)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                   ON CONFLICT(id) DO UPDATE SET status=excluded.status, trainer=excluded.trainer, adapter_in=excluded.adapter_in,
                    adapter_out=excluded.adapter_out, sample_run_ids=excluded.sample_run_ids, hyperparams=excluded.hyperparams,
                    error=excluded.error, updated_at=excluded.updated_at"#,
                params![
                    record.id, record.status.as_str(), record.trainer, record.adapter_in, record.adapter_out,
                    serde_json::to_string(&record.sample_run_ids)?, serde_json::to_string(&record.hyperparams)?, record.error,
                    micros(&record.created_at), micros(&record.updated_at),
                ],
            )?;
            for m in &record.metrics {
                insert_metric(&tx, &record.id, m)?;
            }
            tx.commit()?;
            Ok(())
        })
    }

    pub fn append_training_metric(
        &self,
        training_run_id: &str,
        metric: &TrainingMetricPoint,
    ) -> Result<()> {
        self.with_conn(|c| insert_metric(c, training_run_id, metric))
    }

    pub fn list_eval_reports(&self, adapter_id: Option<&str>) -> Result<Vec<EvalReport>> {
        self.with_conn(|c| {
            let sql = match adapter_id {
                None => "SELECT * FROM eval_reports ORDER BY created_at DESC".to_string(),
                Some(_) => {
                    "SELECT * FROM eval_reports WHERE adapter_id = ?1 ORDER BY created_at DESC"
                        .to_string()
                }
            };
            let mut stmt = c.prepare(&sql)?;
            let rows = match adapter_id {
                None => stmt.query_map([], row_to_eval)?,
                Some(id) => stmt.query_map(params![id], row_to_eval)?,
            };
            rows.map(|r| r?).collect()
        })
    }

    pub fn save_eval_report(&self, report: &EvalReport) -> Result<()> {
        self.with_conn(|c| {
            c.execute(
                "INSERT OR REPLACE INTO eval_reports (id, adapter_id, task_set, mean_reward, per_task, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![report.id, report.adapter_id, report.task_set, report.mean_reward, serde_json::to_string(&report.per_task)?, micros(&report.created_at)],
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
                "INSERT INTO jobs (id, kind, status, payload, attempts, cancel_requested, created_at, updated_at) VALUES (?1, ?2, 'queued', ?3, 0, 0, ?4, ?4)",
                params![id, kind.as_str(), serde_json::to_string(payload)?, micros(&now)],
            )?;
            Ok(())
        })?;
        Ok(JobRecord {
            id: id.into(),
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
    /// running with an expired lease (a crashed worker).
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
                    params![kind.as_str(), micros(&now)],
                    |r| r.get(0),
                )
                .optional()?;
            let Some(id) = candidate else {
                tx.commit()?;
                return Ok(None);
            };
            tx.execute(
                "UPDATE jobs SET status = 'running', lease_owner = ?2, lease_until = ?3, attempts = attempts + 1, updated_at = ?4 WHERE id = ?1",
                params![id, owner, micros(&until), micros(&now)],
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
                params![id, owner, micros(&until), micros(&now)],
            )?;
            Ok(n == 1)
        })
    }

    pub fn finish_job(&self, id: &str, status: JobStatus, error: Option<&str>) -> Result<()> {
        self.with_conn(|c| {
            c.execute("UPDATE jobs SET status = ?2, error = ?3, lease_until = NULL, updated_at = ?4 WHERE id = ?1", params![id, status.as_str(), error, micros(&utc_now())])?;
            Ok(())
        })
    }

    /// Flag a job for cancellation. Returns false when the job is already
    /// terminal. Queued jobs cancel immediately.
    pub fn request_cancel(&self, id: &str) -> Result<bool> {
        self.with_conn(|c| {
            let status: Option<String> = c.query_row("SELECT status FROM jobs WHERE id = ?1", params![id], |r| r.get(0)).optional()?;
            let now = micros(&utc_now());
            match status.as_deref().and_then(JobStatus::parse) {
                Some(s) if s.is_terminal() => Ok(false),
                Some(JobStatus::Queued) => {
                    c.execute("UPDATE jobs SET cancel_requested = 1, status = 'cancelled', updated_at = ?2 WHERE id = ?1", params![id, now])?;
                    Ok(true)
                }
                _ => {
                    c.execute("UPDATE jobs SET cancel_requested = 1, updated_at = ?2 WHERE id = ?1", params![id, now])?;
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

    pub fn job_counts(&self) -> Result<JobCounts> {
        self.with_conn(|c| {
            let mut stmt = c.prepare("SELECT status, COUNT(*) FROM jobs GROUP BY status")?;
            let mut counts = JobCounts::default();
            for row in stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, u32>(1)?)))? {
                let (status, n) = row?;
                match JobStatus::parse(&status) {
                    Some(JobStatus::Queued) => counts.queued = n,
                    Some(JobStatus::Running) => counts.running = n,
                    Some(JobStatus::Completed) => counts.completed = n,
                    Some(JobStatus::Failed) => counts.failed = n,
                    Some(JobStatus::Cancelled) => counts.cancelled = n,
                    None => {}
                }
            }
            Ok(counts)
        })
    }

    // --- events -------------------------------------------------------------

    pub fn append_event(&self, event: &Event) -> Result<i64> {
        self.with_conn(|c| {
            c.execute(
                "INSERT INTO events (kind, subject_kind, subject_id, at, payload) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    event.kind(),
                    event.subject.as_ref().map(|s| s.kind()),
                    event.subject.as_ref().map(|s| s.id().to_string()),
                    micros(&event.at),
                    serde_json::to_string(event)?,
                ],
            )?;
            Ok(c.last_insert_rowid())
        })
    }

    pub fn events_since(&self, since_seq: i64, limit: usize) -> Result<Vec<EventEnvelope>> {
        self.query_events(since_seq, limit, None, None, None)
    }

    /// Events after `since_seq`, optionally filtered by kind prefix and subject.
    pub fn query_events(
        &self,
        since_seq: i64,
        limit: usize,
        kind: Option<&str>,
        subject_kind: Option<&str>,
        subject_id: Option<&str>,
    ) -> Result<Vec<EventEnvelope>> {
        self.with_conn(|c| {
            let mut sql = String::from("SELECT seq, payload FROM events WHERE seq > ?");
            let mut args: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(since_seq)];
            if let Some(k) = kind {
                sql.push_str(" AND kind LIKE ?");
                args.push(Box::new(format!("{k}%")));
            }
            if let Some(sk) = subject_kind {
                sql.push_str(" AND subject_kind = ?");
                args.push(Box::new(sk.to_string()));
            }
            if let Some(sid) = subject_id {
                sql.push_str(" AND subject_id = ?");
                args.push(Box::new(sid.to_string()));
            }
            sql.push_str(" ORDER BY seq ASC LIMIT ?");
            args.push(Box::new(limit as i64));
            let mut stmt = c.prepare(&sql)?;
            let rows = stmt.query_map(
                rusqlite::params_from_iter(args.iter().map(|a| a.as_ref())),
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)),
            )?;
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

fn percentiles(v: &mut [f64]) -> LatencyStats {
    if v.is_empty() {
        return LatencyStats::default();
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Nearest-rank: the smallest value with at least p of the samples at or below it.
    let pick = |p: f64| v[((p * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1];
    LatencyStats {
        samples: v.len() as u32,
        p50_ms: pick(0.5),
        p95_ms: pick(0.95),
        p99_ms: pick(0.99),
    }
}

fn save_reward(c: &Connection, run_id: &str, reward: Option<&RewardRecord>) -> Result<()> {
    c.execute("DELETE FROM rewards WHERE run_id = ?1", params![run_id])?;
    c.execute(
        "DELETE FROM reward_signals WHERE run_id = ?1",
        params![run_id],
    )?;
    let Some(reward) = reward else { return Ok(()) };
    c.execute(
        "INSERT INTO rewards (run_id, terminal_reward, rubric, source, audit_flags) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![run_id, reward.terminal_reward, reward.rubric, reward.source, serde_json::to_string(&reward.audit_flags)?],
    )?;
    let mut stmt = c.prepare("INSERT INTO reward_signals (run_id, ord, name, value, weight, reason) VALUES (?1, ?2, ?3, ?4, ?5, ?6)")?;
    for (i, s) in reward.signals.iter().enumerate() {
        stmt.execute(params![
            run_id, i as i64, s.name, s.value, s.weight, s.reason
        ])?;
    }
    Ok(())
}

fn load_reward(c: &Connection, run_id: &str) -> Result<Option<RewardRecord>> {
    let Some((terminal, rubric, source, flags)) = c
        .query_row(
            "SELECT terminal_reward, rubric, source, audit_flags FROM rewards WHERE run_id = ?1",
            params![run_id],
            |r| {
                Ok((
                    r.get::<_, f64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                ))
            },
        )
        .optional()?
    else {
        return Ok(None);
    };
    let mut stmt = c.prepare(
        "SELECT name, value, weight, reason FROM reward_signals WHERE run_id = ?1 ORDER BY ord",
    )?;
    let signals = stmt
        .query_map(params![run_id], |r| {
            Ok(RewardSignal {
                name: r.get(0)?,
                value: r.get(1)?,
                weight: r.get(2)?,
                reason: r.get(3)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(Some(RewardRecord {
        terminal_reward: terminal,
        rubric,
        source,
        signals,
        audit_flags: serde_json::from_str(&flags)?,
    }))
}

fn insert_metric(c: &Connection, training_run_id: &str, m: &TrainingMetricPoint) -> Result<()> {
    c.execute(
        "INSERT OR REPLACE INTO training_metrics (training_run_id, step, loss, mean_reward, kl, extra) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![training_run_id, m.step, m.loss, m.mean_reward, m.kl, serde_json::to_string(&m.extra)?],
    )?;
    Ok(())
}

fn load_metrics(c: &Connection, training_run_id: &str) -> Result<Vec<TrainingMetricPoint>> {
    let mut stmt = c.prepare("SELECT step, loss, mean_reward, kl, extra FROM training_metrics WHERE training_run_id = ?1 ORDER BY step")?;
    let rows = stmt.query_map(params![training_run_id], |r| {
        Ok((
            r.get::<_, u32>(0)?,
            r.get::<_, f64>(1)?,
            r.get::<_, Option<f64>>(2)?,
            r.get::<_, Option<f64>>(3)?,
            r.get::<_, String>(4)?,
        ))
    })?;
    rows.map(|r| {
        let (step, loss, mean_reward, kl, extra) = r?;
        Ok(TrainingMetricPoint {
            step,
            loss,
            mean_reward,
            kl,
            extra: serde_json::from_str(&extra)?,
        })
    })
    .collect()
}

fn row_to_manifest(row: &Row<'_>) -> rusqlite::Result<Result<RunManifest>> {
    let status: String = row.get("status")?;
    let created: i64 = row.get("created_at")?;
    let started: Option<i64> = row.get("started_at")?;
    let finished: Option<i64> = row.get("finished_at")?;
    let updated: i64 = row.get("updated_at")?;
    let tokens: i64 = row.get("tokens")?;
    let id: String = row.get("id")?;
    let profile: String = row.get("profile")?;
    let model_id: String = row.get("model_id")?;
    let adapter_id: Option<String> = row.get("adapter_id")?;
    let infra_target: String = row.get("infra_target")?;
    let horizon: u32 = row.get("horizon")?;
    let cost_usd: f64 = row.get("cost_usd")?;
    let terminal_reward: Option<f64> = row.get("terminal_reward")?;
    let step_count: u32 = row.get("step_count")?;
    let error: Option<String> = row.get("error")?;
    Ok((|| {
        let Some(status) = RunStatus::parse(&status) else {
            return corrupt("run status", &status);
        };
        Ok(RunManifest {
            id,
            profile,
            model_id,
            adapter_id,
            infra_target,
            status,
            horizon,
            tokens: tokens.max(0) as u64,
            cost_usd,
            terminal_reward,
            step_count,
            error,
            created_at: from_micros(created)?,
            started_at: started.map(from_micros).transpose()?,
            finished_at: finished.map(from_micros).transpose()?,
            updated_at: from_micros(updated)?,
        })
    })())
}

fn row_to_turn_training(row: &Row<'_>) -> rusqlite::Result<Result<TurnTrainingRecord>> {
    let sampling: String = row.get("sampling_args")?;
    let created: i64 = row.get("created_at")?;
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
            created_at: from_micros(created)?,
        })
    })())
}

fn row_to_training_run(row: &Row<'_>) -> rusqlite::Result<Result<TrainingRunRecord>> {
    let status: String = row.get("status")?;
    let id: String = row.get("id")?;
    let trainer: String = row.get("trainer")?;
    let adapter_in: Option<String> = row.get("adapter_in")?;
    let adapter_out: Option<String> = row.get("adapter_out")?;
    let samples: String = row.get("sample_run_ids")?;
    let hyper: String = row.get("hyperparams")?;
    let error: Option<String> = row.get("error")?;
    let created: i64 = row.get("created_at")?;
    let updated: i64 = row.get("updated_at")?;
    Ok((|| {
        let Some(status) = TrainingStatus::parse(&status) else {
            return corrupt("training status", &status);
        };
        Ok(TrainingRunRecord {
            id,
            status,
            trainer,
            adapter_in,
            adapter_out,
            sample_run_ids: serde_json::from_str(&samples)?,
            hyperparams: serde_json::from_str(&hyper)?,
            metrics: Vec::new(),
            error,
            created_at: from_micros(created)?,
            updated_at: from_micros(updated)?,
        })
    })())
}

fn row_to_eval(row: &Row<'_>) -> rusqlite::Result<Result<EvalReport>> {
    let id: String = row.get("id")?;
    let adapter_id: String = row.get("adapter_id")?;
    let task_set: String = row.get("task_set")?;
    let mean_reward: f64 = row.get("mean_reward")?;
    let per_task: String = row.get("per_task")?;
    let created: i64 = row.get("created_at")?;
    Ok((|| {
        let per_task: Vec<EvalTaskScore> = serde_json::from_str(&per_task)?;
        Ok(EvalReport {
            id,
            adapter_id,
            task_set,
            mean_reward,
            per_task,
            created_at: from_micros(created)?,
        })
    })())
}

fn row_to_job(row: &Row<'_>) -> rusqlite::Result<Result<JobRecord>> {
    let kind: String = row.get("kind")?;
    let status: String = row.get("status")?;
    let payload: String = row.get("payload")?;
    let lease_until: Option<i64> = row.get("lease_until")?;
    let created: i64 = row.get("created_at")?;
    let updated: i64 = row.get("updated_at")?;
    let id: String = row.get("id")?;
    let lease_owner: Option<String> = row.get("lease_owner")?;
    let attempts: u32 = row.get("attempts")?;
    let cancel_requested: i64 = row.get("cancel_requested")?;
    let error: Option<String> = row.get("error")?;
    Ok((|| {
        let Some(kind) = JobKind::parse(&kind) else {
            return corrupt("job kind", &kind);
        };
        let Some(status) = JobStatus::parse(&status) else {
            return corrupt("job status", &status);
        };
        Ok(JobRecord {
            id,
            kind,
            status,
            payload: serde_json::from_str(&payload)?,
            lease_owner,
            lease_until: lease_until.map(from_micros).transpose()?,
            attempts,
            cancel_requested: cancel_requested != 0,
            error,
            created_at: from_micros(created)?,
            updated_at: from_micros(updated)?,
        })
    })())
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizon_core::events::DomainEvent;

    fn detail(id: &str, cost: f64) -> RunDetail {
        let task = TaskSpec {
            id: "task-1".into(),
            prompt: "p".into(),
            repo_snapshot: ".".into(),
            horizon: 4,
            success_criteria: vec!["x".into()],
        };
        let trajectory = Trajectory {
            steps: vec![TrajectoryStep::new(0, "policy", "action", "{}")],
            errors: vec![],
        };
        let reward = RewardRecord {
            terminal_reward: 0.5,
            rubric: "heuristic-v1".into(),
            source: "rubric".into(),
            signals: vec![RewardSignal {
                name: "finish".into(),
                value: 1.0,
                weight: 0.3,
                reason: "ok".into(),
            }],
            audit_flags: vec![],
        };
        RunDetail {
            manifest: RunManifest {
                id: id.into(),
                profile: "default".into(),
                model_id: "openai:x".into(),
                adapter_id: None,
                infra_target: "local".into(),
                status: RunStatus::Completed,
                horizon: 4,
                tokens: 12,
                cost_usd: cost,
                terminal_reward: Some(0.5),
                step_count: 1,
                error: None,
                created_at: utc_now(),
                started_at: Some(utc_now() - chrono::Duration::seconds(5)),
                finished_at: Some(utc_now()),
                updated_at: utc_now(),
            },
            task,
            trajectory,
            reward: Some(reward),
        }
    }

    #[test]
    fn stats_timeseries_histogram_and_nodes() {
        let store = Store::in_memory().unwrap();
        let mut d = detail("run-1", 0.5);
        d.trajectory.steps = vec![
            TrajectoryStep::new(0, "policy", "action", "{}").with_duration(400),
            TrajectoryStep::new(1, "environment", "observation", "{}").with_duration(20),
            TrajectoryStep::new(2, "policy", "action", "{}").with_duration(600),
        ];
        store.save_run(&d).unwrap();
        let mut failed = detail("run-2", 0.1);
        failed.manifest.status = RunStatus::Failed;
        failed.manifest.terminal_reward = Some(-0.5);
        store.save_run(&failed).unwrap();
        let w = chrono::Duration::minutes(10);
        let s = store.stats(w).unwrap();
        assert_eq!((s.rollouts_completed, s.rollouts_failed), (1, 1));
        assert_eq!(s.success_rate, Some(0.5));
        assert_eq!(s.policy_latency.samples, 2);
        assert_eq!(s.policy_latency.p50_ms, 400.0);
        assert!(s.tokens_per_s > 0.0);
        let ts = store.timeseries(w, chrono::Duration::seconds(60)).unwrap();
        assert_eq!(ts.iter().map(|b| b.completed + b.failed).sum::<u32>(), 2);
        assert!(ts.iter().any(|b| b.p95_policy_ms == Some(600.0)));
        let hist = store.reward_histogram(w, 4).unwrap();
        assert_eq!(hist.iter().map(|b| b.count).sum::<u32>(), 2);
        assert_eq!(
            store
                .query_runs(Some(RunStatus::Failed), None, 10, 0)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            store.query_runs(None, Some("nope"), 10, 0).unwrap().len(),
            0
        );

        let node = NodeRecord {
            id: "cp-1".into(),
            role: "control-plane".into(),
            status: NodeStatus::Up,
            detail: "ok".into(),
            meta: Default::default(),
            last_seen: utc_now() - chrono::Duration::minutes(5),
        };
        store.upsert_node(&node).unwrap();
        assert_eq!(store.list_nodes().unwrap()[0].status, NodeStatus::Up);
        let expired = store.expire_nodes(chrono::Duration::seconds(30)).unwrap();
        assert_eq!(expired.len(), 1);
        assert_eq!(store.list_nodes().unwrap()[0].status, NodeStatus::Down);
    }

    #[test]
    fn runs_round_trip() {
        let store = Store::in_memory().unwrap();
        let d = detail("run-1", 0.25);
        store.save_run(&d).unwrap();
        let back = store.get_run("run-1").unwrap().unwrap();
        assert_eq!(back.task, d.task);
        assert_eq!(back.trajectory.steps.len(), 1);
        assert_eq!(back.reward.as_ref().unwrap().signals[0].name, "finish");
        assert_eq!(back.manifest.tokens, 12);
        assert_eq!(store.list_runs().unwrap().len(), 1);
        let since = utc_now() - chrono::Duration::hours(1);
        assert_eq!(store.total_cost_since(since).unwrap(), 0.25);
        assert!(store.get_run("nope").unwrap().is_none());

        store
            .append_step(
                "run-1",
                &TrajectoryStep::new(1, "environment", "observation", "{}"),
            )
            .unwrap();
        assert_eq!(store.get_manifest("run-1").unwrap().unwrap().step_count, 2);
        assert_eq!(
            store
                .dashboard(chrono::Duration::minutes(15))
                .unwrap()
                .runs
                .len(),
            1
        );
    }

    #[test]
    fn turn_training_packs_ints_and_flags_step() {
        let store = Store::in_memory().unwrap();
        store.save_run(&detail("run-1", 0.0)).unwrap();
        let rec = TurnTrainingRecord {
            id: "turn-1".into(),
            run_id: "run-1".into(),
            step_index: 0,
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
        let back = store.get_turn_training("run-1", 0).unwrap().unwrap();
        assert_eq!(back.prompt_ids, vec![1, -2, 300000]);
        assert_eq!(store.list_turn_training("run-1").unwrap().len(), 1);
        assert!(store.get_run("run-1").unwrap().unwrap().trajectory.steps[0].has_training_metadata);
    }

    #[test]
    fn training_runs_with_metrics() {
        let store = Store::in_memory().unwrap();
        let record = TrainingRunRecord {
            id: "trun-1".into(),
            status: TrainingStatus::Queued,
            trainer: "stub".into(),
            adapter_in: None,
            adapter_out: None,
            sample_run_ids: vec!["run-1".into()],
            hyperparams: Default::default(),
            metrics: vec![],
            error: None,
            created_at: utc_now(),
            updated_at: utc_now(),
        };
        store.save_training_run(&record).unwrap();
        store
            .append_training_metric(
                "trun-1",
                &TrainingMetricPoint {
                    step: 0,
                    loss: 1.0,
                    mean_reward: None,
                    kl: None,
                    extra: Default::default(),
                },
            )
            .unwrap();
        store
            .append_training_metric(
                "trun-1",
                &TrainingMetricPoint {
                    step: 1,
                    loss: 0.5,
                    mean_reward: Some(0.1),
                    kl: None,
                    extra: Default::default(),
                },
            )
            .unwrap();
        let back = store.get_training_run("trun-1").unwrap().unwrap();
        assert_eq!(back.metrics.len(), 2);
        assert_eq!(back.metrics[1].loss, 0.5);
        assert_eq!(store.list_training_runs().unwrap()[0].metrics.len(), 2);
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
        assert!(store
            .lease_job(JobKind::Rollout, "w2", 30)
            .unwrap()
            .is_none());
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
        store
            .enqueue_job("job-2", JobKind::Rollout, &serde_json::json!({}))
            .unwrap();
        assert!(store.request_cancel("job-2").unwrap());
        assert_eq!(
            store.get_job("job-2").unwrap().unwrap().status,
            JobStatus::Cancelled
        );
        assert_eq!(store.job_counts().unwrap().cancelled, 2);
    }

    #[test]
    fn event_log_sequences() {
        let store = Store::in_memory().unwrap();
        let ev = |m: &str| {
            Event::for_run(
                "run-1",
                DomainEvent::LogLine {
                    level: "INFO".into(),
                    logger: "t".into(),
                    message: m.into(),
                    context: Default::default(),
                },
            )
        };
        let s1 = store.append_event(&ev("a")).unwrap();
        let s2 = store.append_event(&ev("b")).unwrap();
        assert!(s2 > s1);
        assert_eq!(
            store
                .query_events(0, 10, Some("log."), Some("run"), Some("run-1"))
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            store
                .query_events(0, 10, Some("rollout."), None, None)
                .unwrap()
                .len(),
            0
        );
        let after = store.events_since(s1, 10).unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].seq, s2);
        assert_eq!(after[0].event.run_id(), Some("run-1"));
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
