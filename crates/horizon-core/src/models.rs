use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::utc_now;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

impl RunStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            RunStatus::Pending => "pending",
            RunStatus::Running => "running",
            RunStatus::Completed => "completed",
            RunStatus::Failed => "failed",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "pending" => Some(RunStatus::Pending),
            "running" => Some(RunStatus::Running),
            "completed" => Some(RunStatus::Completed),
            "failed" => Some(RunStatus::Failed),
            _ => None,
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, RunStatus::Completed | RunStatus::Failed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum WorkerStatus {
    Idle,
    Running,
    Failed,
}

impl WorkerStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            WorkerStatus::Idle => "idle",
            WorkerStatus::Running => "running",
            WorkerStatus::Failed => "failed",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "idle" => Some(WorkerStatus::Idle),
            "running" => Some(WorkerStatus::Running),
            "failed" => Some(WorkerStatus::Failed),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ToolPermission {
    Read,
    Edit,
    Terminal,
    Search,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct TaskSpec {
    pub id: String,
    pub prompt: String,
    pub repo_snapshot: String,
    pub tool_permissions: Vec<ToolPermission>,
    #[serde(default = "default_horizon")]
    pub horizon: u32,
    pub success_criteria: Vec<String>,
}

fn default_horizon() -> u32 {
    8
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct TrajectoryStep {
    pub index: u32,
    pub actor: String,
    pub kind: String,
    pub content: String,
    #[serde(default = "utc_now")]
    pub timestamp: DateTime<Utc>,
    /// Flips true when a `TurnTrainingRecord` row exists for this step.
    #[serde(default)]
    pub has_training_metadata: bool,
}

impl TrajectoryStep {
    pub fn new(index: u32, actor: &str, kind: &str, content: impl Into<String>) -> Self {
        Self {
            index,
            actor: actor.to_string(),
            kind: kind.to_string(),
            content: content.into(),
            timestamp: utc_now(),
            has_training_metadata: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct TrajectoryRecord {
    pub id: String,
    pub task_id: String,
    pub steps: Vec<TrajectoryStep>,
    #[serde(default)]
    pub summaries: Vec<String>,
    #[serde(default)]
    pub timings_ms: BTreeMap<String, i64>,
    #[serde(default)]
    pub errors: Vec<String>,
}

impl TrajectoryRecord {
    pub fn new(task_id: &str, steps: Vec<TrajectoryStep>, errors: Vec<String>) -> Self {
        Self {
            id: crate::short_id("traj"),
            task_id: task_id.to_string(),
            steps,
            summaries: Vec::new(),
            timings_ms: BTreeMap::new(),
            errors,
        }
    }
}

/// Per-turn token-level metadata, stored in its own table so run payloads
/// never carry the bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct TurnTrainingRecord {
    pub id: String,
    pub run_id: String,
    pub step_index: u32,
    #[serde(default)]
    pub prompt_ids: Vec<i32>,
    #[serde(default)]
    pub completion_ids: Vec<i32>,
    #[serde(default)]
    pub attention_mask: Vec<i32>,
    #[serde(default)]
    pub loss_mask: Vec<i32>,
    #[serde(default)]
    pub sampling_args: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub model_name: String,
    #[serde(default)]
    pub token_count: u32,
    #[serde(default = "utc_now")]
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct RewardPenalty {
    pub code: String,
    pub value: f64,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct RewardRecord {
    pub trajectory_id: String,
    pub terminal_reward: f64,
    pub step_rewards: Vec<f64>,
    #[serde(default)]
    pub penalties: Vec<RewardPenalty>,
    #[serde(default)]
    pub audit_flags: Vec<String>,
    pub provenance: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct ArtifactRecord {
    pub name: String,
    pub kind: String,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct RunManifest {
    pub id: String,
    pub model_id: String,
    #[serde(default)]
    pub adapter_id: Option<String>,
    pub dataset_slice: String,
    pub infra_target: String,
    pub seed: i64,
    pub status: RunStatus,
    #[serde(default = "utc_now")]
    pub created_at: DateTime<Utc>,
    #[serde(default = "utc_now")]
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub estimated_cost_usd: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct RunDetail {
    pub manifest: RunManifest,
    pub task: TaskSpec,
    pub trajectory: TrajectoryRecord,
    pub reward: RewardRecord,
    pub artifacts: Vec<ArtifactRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct WorkerRecord {
    pub id: String,
    pub role: String,
    pub status: WorkerStatus,
    #[serde(default)]
    pub run_id: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct DashboardSnapshot {
    #[serde(default = "utc_now")]
    pub generated_at: DateTime<Utc>,
    pub runs: Vec<RunManifest>,
    pub workers: Vec<WorkerRecord>,
    pub recent_artifacts: Vec<ArtifactRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct RolloutRequest {
    pub prompt: String,
    #[serde(default = "default_repo_snapshot")]
    pub repo_snapshot: String,
    #[serde(default = "default_infra_target")]
    pub infra_target: String,
    /// When absent, the configured round scheduler decides the horizon.
    #[serde(default)]
    pub horizon: Option<u32>,
    #[serde(default)]
    pub success_criteria: Vec<String>,
    #[serde(default)]
    pub adapter_id: Option<String>,
    /// Named policy profile. Absent = the configured default profile.
    #[serde(default)]
    pub policy_profile: Option<String>,
}

fn default_repo_snapshot() -> String {
    ".".to_string()
}

fn default_infra_target() -> String {
    "mac-local".to_string()
}

impl RolloutRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            repo_snapshot: default_repo_snapshot(),
            infra_target: default_infra_target(),
            horizon: None,
            success_criteria: Vec::new(),
            adapter_id: None,
            policy_profile: None,
        }
    }
}

// --- Training / adapters / eval --------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum TrainingStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl TrainingStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            TrainingStatus::Pending => "pending",
            TrainingStatus::Running => "running",
            TrainingStatus::Completed => "completed",
            TrainingStatus::Failed => "failed",
            TrainingStatus::Cancelled => "cancelled",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            TrainingStatus::Completed | TrainingStatus::Failed | TrainingStatus::Cancelled
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct TrainingMetricPoint {
    pub step: u32,
    pub loss: f64,
    #[serde(default)]
    pub mean_reward: Option<f64>,
    #[serde(default)]
    pub kl: Option<f64>,
    #[serde(default)]
    pub extra: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct AdapterRecord {
    pub id: String,
    #[serde(default)]
    pub parent_id: Option<String>,
    pub base_model: String,
    #[serde(default)]
    pub training_run_id: Option<String>,
    #[serde(default)]
    pub eval_score: Option<f64>,
    pub path: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    #[serde(default = "utc_now")]
    pub created_at: DateTime<Utc>,
}

/// Hyperparameter values: JSON scalars only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum HyperValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl HyperValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            HyperValue::Int(i) => Some(*i as f64),
            HyperValue::Float(f) => Some(*f),
            HyperValue::Str(s) => s.parse().ok(),
            HyperValue::Bool(_) => None,
        }
    }
}

pub type Hyperparams = BTreeMap<String, HyperValue>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct TrainingRunRecord {
    pub id: String,
    #[serde(default = "default_training_status")]
    pub status: TrainingStatus,
    #[serde(default)]
    pub adapter_in: Option<String>,
    #[serde(default)]
    pub adapter_out: Option<String>,
    #[serde(default)]
    pub sample_run_ids: Vec<String>,
    #[serde(default)]
    pub hyperparams: Hyperparams,
    #[serde(default)]
    pub metrics: Vec<TrainingMetricPoint>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default = "utc_now")]
    pub created_at: DateTime<Utc>,
    #[serde(default = "utc_now")]
    pub updated_at: DateTime<Utc>,
}

fn default_training_status() -> TrainingStatus {
    TrainingStatus::Pending
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct EvalTaskScore {
    pub task_id: String,
    pub terminal_reward: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct EvalReport {
    pub id: String,
    pub adapter_id: String,
    pub task_set: String,
    pub mean_reward: f64,
    #[serde(default)]
    pub per_task: Vec<EvalTaskScore>,
    #[serde(default = "utc_now")]
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct EvalTask {
    pub id: String,
    pub prompt: String,
    #[serde(default)]
    pub success_criteria: Vec<String>,
    #[serde(default = "default_eval_horizon")]
    pub horizon: u32,
}

fn default_eval_horizon() -> u32 {
    4
}

/// Built-in eval tasks; small, repo-agnostic.
pub fn default_eval_tasks() -> Vec<EvalTask> {
    vec![
        EvalTask {
            id: "eval-readme".into(),
            prompt: "Read README.md and summarise the project.".into(),
            success_criteria: vec!["readme".into()],
            horizon: 4,
        },
        EvalTask {
            id: "eval-list-files".into(),
            prompt: "List the files in the repository root.".into(),
            success_criteria: vec!["files".into()],
            horizon: 3,
        },
        EvalTask {
            id: "eval-search-imports".into(),
            prompt: "Find Python files that import pydantic.".into(),
            success_criteria: vec!["pydantic".into()],
            horizon: 4,
        },
    ]
}

// --- Jobs (durable orchestration) ------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Rollout,
    Training,
}

impl JobKind {
    pub fn as_str(self) -> &'static str {
        match self {
            JobKind::Rollout => "rollout",
            JobKind::Training => "training",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "rollout" => Some(JobKind::Rollout),
            "training" => Some(JobKind::Training),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            JobStatus::Queued => "queued",
            JobStatus::Running => "running",
            JobStatus::Completed => "completed",
            JobStatus::Failed => "failed",
            JobStatus::Cancelled => "cancelled",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "queued" => Some(JobStatus::Queued),
            "running" => Some(JobStatus::Running),
            "completed" => Some(JobStatus::Completed),
            "failed" => Some(JobStatus::Failed),
            "cancelled" => Some(JobStatus::Cancelled),
            _ => None,
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
        )
    }
}

/// A leased unit of work. Rollouts and training runs are both jobs so a
/// crash mid-run leaves a row another worker can pick up after the lease
/// expires, instead of an orphaned in-memory task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct JobRecord {
    pub id: String,
    pub kind: JobKind,
    pub status: JobStatus,
    pub payload: serde_json::Value,
    #[serde(default)]
    pub lease_owner: Option<String>,
    #[serde(default)]
    pub lease_until: Option<DateTime<Utc>>,
    #[serde(default)]
    pub attempts: u32,
    #[serde(default)]
    pub cancel_requested: bool,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default = "utc_now")]
    pub created_at: DateTime<Utc>,
    #[serde(default = "utc_now")]
    pub updated_at: DateTime<Utc>,
}

/// One row of the durable event log. `seq` is monotonic per store so
/// clients can resume a WebSocket with `?since=<seq>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct EventEnvelope {
    pub seq: i64,
    pub event: crate::events::DomainEvent,
}
