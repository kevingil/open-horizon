use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::utc_now;

macro_rules! str_enum {
    ($name:ident { $($variant:ident => $text:literal),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
        #[serde(rename_all = "snake_case")]
        pub enum $name { $($variant),+ }

        impl $name {
            pub fn as_str(self) -> &'static str {
                match self { $($name::$variant => $text),+ }
            }
            pub fn parse(value: &str) -> Option<Self> {
                match value { $($text => Some($name::$variant),)+ _ => None }
            }
        }
    };
}

str_enum!(RunStatus { Queued => "queued", Running => "running", Completed => "completed", Failed => "failed", Cancelled => "cancelled" });
str_enum!(WorkerStatus { Idle => "idle", Running => "running", Failed => "failed" });
str_enum!(TrainingStatus { Queued => "queued", Running => "running", Completed => "completed", Failed => "failed", Cancelled => "cancelled" });
str_enum!(JobKind { Rollout => "rollout", Training => "training" });
str_enum!(JobStatus { Queued => "queued", Running => "running", Completed => "completed", Failed => "failed", Cancelled => "cancelled" });

impl RunStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled
        )
    }
}

impl TrainingStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            TrainingStatus::Completed | TrainingStatus::Failed | TrainingStatus::Cancelled
        )
    }
}

impl JobStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
        )
    }
}

/// What a rollout is asked to do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct TaskSpec {
    pub id: String,
    pub prompt: String,
    pub repo_snapshot: String,
    pub horizon: u32,
    pub success_criteria: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct TrajectoryStep {
    pub index: u32,
    pub actor: String,
    pub kind: String,
    pub content: String,
    pub at: DateTime<Utc>,
    /// True when a `TurnTrainingRecord` row exists for this step.
    #[serde(default)]
    pub has_training_metadata: bool,
}

impl TrajectoryStep {
    pub fn new(index: u32, actor: &str, kind: &str, content: impl Into<String>) -> Self {
        Self {
            index,
            actor: actor.into(),
            kind: kind.into(),
            content: content.into(),
            at: utc_now(),
            has_training_metadata: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, ToSchema)]
pub struct Trajectory {
    pub steps: Vec<TrajectoryStep>,
    pub errors: Vec<String>,
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

/// One named contribution to a reward.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct RewardSignal {
    pub name: String,
    pub value: f64,
    pub weight: f64,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct RewardRecord {
    pub terminal_reward: f64,
    /// `heuristic-v1`, `coding-v1`, `verifiers-vf-math`, ...
    pub rubric: String,
    /// `rubric` (in-house signals) or `verifiers` (env-owned scoring).
    pub source: String,
    pub signals: Vec<RewardSignal>,
    pub audit_flags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct RunManifest {
    pub id: String,
    /// Policy profile name the run was dispatched with.
    pub profile: String,
    /// `openai:<model>` or `verifiers:<env>:<model>`.
    pub model_id: String,
    pub adapter_id: Option<String>,
    pub infra_target: String,
    pub status: RunStatus,
    pub horizon: u32,
    pub tokens: u64,
    pub cost_usd: f64,
    pub terminal_reward: Option<f64>,
    pub step_count: u32,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct RunDetail {
    pub manifest: RunManifest,
    pub task: TaskSpec,
    pub trajectory: Trajectory,
    pub reward: Option<RewardRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct WorkerRecord {
    pub id: String,
    pub role: String,
    pub status: WorkerStatus,
    pub run_id: Option<String>,
    pub detail: String,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, ToSchema)]
pub struct JobCounts {
    pub queued: u32,
    pub running: u32,
    pub completed: u32,
    pub failed: u32,
    pub cancelled: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct DashboardSnapshot {
    pub generated_at: DateTime<Utc>,
    pub runs: Vec<RunManifest>,
    pub workers: Vec<WorkerRecord>,
    pub jobs: JobCounts,
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
    "local".to_string()
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct TrainingMetricPoint {
    pub step: u32,
    pub loss: f64,
    pub mean_reward: Option<f64>,
    pub kl: Option<f64>,
    pub extra: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct AdapterRecord {
    pub id: String,
    pub parent_id: Option<String>,
    pub base_model: String,
    pub training_run_id: Option<String>,
    pub eval_score: Option<f64>,
    pub path: String,
    pub tags: Vec<String>,
    pub metadata: BTreeMap<String, String>,
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
    pub status: TrainingStatus,
    pub trainer: String,
    pub adapter_in: Option<String>,
    pub adapter_out: Option<String>,
    pub sample_run_ids: Vec<String>,
    pub hyperparams: Hyperparams,
    pub metrics: Vec<TrainingMetricPoint>,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct EvalTaskScore {
    pub task_id: String,
    pub run_id: String,
    pub terminal_reward: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct EvalReport {
    pub id: String,
    pub adapter_id: String,
    pub task_set: String,
    pub mean_reward: f64,
    pub per_task: Vec<EvalTaskScore>,
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
            prompt: "Find files that mention pydantic.".into(),
            success_criteria: vec!["pydantic".into()],
            horizon: 4,
        },
    ]
}

// --- Jobs (durable orchestration) ------------------------------------------

/// A leased unit of work. Rollouts and training runs are both jobs so a
/// crash mid-run leaves a row another worker can pick up after the lease
/// expires, instead of an orphaned in-memory task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct JobRecord {
    pub id: String,
    pub kind: JobKind,
    pub status: JobStatus,
    pub payload: serde_json::Value,
    pub lease_owner: Option<String>,
    pub lease_until: Option<DateTime<Utc>>,
    pub attempts: u32,
    pub cancel_requested: bool,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
