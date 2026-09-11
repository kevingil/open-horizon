use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::models::{
    AdapterRecord, EvalReport, RunDetail, RunManifest, TrainingMetricPoint, TrainingRunRecord,
    TrajectoryStep, WorkerRecord,
};
use crate::utc_now;

/// Fields every event carries. Flattened into each variant so the wire
/// shape is `{kind, event_id, at, run_id, ...payload}`, matching what the
/// dashboard already consumes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct EventMeta {
    #[serde(default = "new_event_id")]
    pub event_id: String,
    #[serde(default = "utc_now")]
    pub at: DateTime<Utc>,
    #[serde(default)]
    pub run_id: Option<String>,
}

fn new_event_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

impl EventMeta {
    pub fn for_run(run_id: Option<&str>) -> Self {
        Self {
            event_id: new_event_id(),
            at: utc_now(),
            run_id: run_id.map(|s| s.to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(tag = "kind")]
#[allow(clippy::large_enum_variant)]
pub enum DomainEvent {
    #[serde(rename = "rollout.started")]
    RolloutStarted {
        #[serde(flatten)]
        meta: EventMeta,
        manifest: RunManifest,
    },
    #[serde(rename = "step.recorded")]
    StepRecorded {
        #[serde(flatten)]
        meta: EventMeta,
        step: TrajectoryStep,
    },
    #[serde(rename = "reward.computed")]
    RewardComputed {
        #[serde(flatten)]
        meta: EventMeta,
        terminal_reward: f64,
        provenance: String,
        #[serde(default)]
        audit_flags: Vec<String>,
    },
    #[serde(rename = "rollout.completed")]
    RolloutCompleted {
        #[serde(flatten)]
        meta: EventMeta,
        detail: RunDetail,
    },
    #[serde(rename = "rollout.failed")]
    RolloutFailed {
        #[serde(flatten)]
        meta: EventMeta,
        error: String,
    },
    #[serde(rename = "rollout.cancelled")]
    RolloutCancelled {
        #[serde(flatten)]
        meta: EventMeta,
        #[serde(default = "default_cancel_reason")]
        reason: String,
    },
    #[serde(rename = "progress.ticked")]
    ProgressTicked {
        #[serde(flatten)]
        meta: EventMeta,
        step_index: u32,
        #[serde(default)]
        tool: Option<String>,
        #[serde(default)]
        tokens: u64,
        #[serde(default)]
        cost_usd: f64,
    },
    #[serde(rename = "budget.exceeded")]
    BudgetExceeded {
        #[serde(flatten)]
        meta: EventMeta,
        spent_usd: f64,
        cap_usd: f64,
        window_hours: f64,
    },
    #[serde(rename = "worker.updated")]
    WorkerUpdated {
        #[serde(flatten)]
        meta: EventMeta,
        worker: WorkerRecord,
    },
    #[serde(rename = "log.line")]
    LogLine {
        #[serde(flatten)]
        meta: EventMeta,
        level: String,
        logger: String,
        message: String,
        #[serde(default)]
        context: std::collections::BTreeMap<String, String>,
    },
    #[serde(rename = "training.started")]
    TrainingStarted {
        #[serde(flatten)]
        meta: EventMeta,
        training_run_id: String,
        record: TrainingRunRecord,
    },
    #[serde(rename = "training.metric")]
    TrainingMetric {
        #[serde(flatten)]
        meta: EventMeta,
        training_run_id: String,
        metric: TrainingMetricPoint,
    },
    #[serde(rename = "training.completed")]
    TrainingCompleted {
        #[serde(flatten)]
        meta: EventMeta,
        training_run_id: String,
        record: TrainingRunRecord,
    },
    #[serde(rename = "training.failed")]
    TrainingFailed {
        #[serde(flatten)]
        meta: EventMeta,
        training_run_id: String,
        error: String,
    },
    #[serde(rename = "adapter.published")]
    AdapterPublished {
        #[serde(flatten)]
        meta: EventMeta,
        adapter: AdapterRecord,
    },
    #[serde(rename = "eval.completed")]
    EvalCompleted {
        #[serde(flatten)]
        meta: EventMeta,
        report: EvalReport,
    },
}

fn default_cancel_reason() -> String {
    "cancelled".to_string()
}

impl DomainEvent {
    pub fn kind(&self) -> &'static str {
        match self {
            DomainEvent::RolloutStarted { .. } => "rollout.started",
            DomainEvent::StepRecorded { .. } => "step.recorded",
            DomainEvent::RewardComputed { .. } => "reward.computed",
            DomainEvent::RolloutCompleted { .. } => "rollout.completed",
            DomainEvent::RolloutFailed { .. } => "rollout.failed",
            DomainEvent::RolloutCancelled { .. } => "rollout.cancelled",
            DomainEvent::ProgressTicked { .. } => "progress.ticked",
            DomainEvent::BudgetExceeded { .. } => "budget.exceeded",
            DomainEvent::WorkerUpdated { .. } => "worker.updated",
            DomainEvent::LogLine { .. } => "log.line",
            DomainEvent::TrainingStarted { .. } => "training.started",
            DomainEvent::TrainingMetric { .. } => "training.metric",
            DomainEvent::TrainingCompleted { .. } => "training.completed",
            DomainEvent::TrainingFailed { .. } => "training.failed",
            DomainEvent::AdapterPublished { .. } => "adapter.published",
            DomainEvent::EvalCompleted { .. } => "eval.completed",
        }
    }

    pub fn meta(&self) -> &EventMeta {
        match self {
            DomainEvent::RolloutStarted { meta, .. }
            | DomainEvent::StepRecorded { meta, .. }
            | DomainEvent::RewardComputed { meta, .. }
            | DomainEvent::RolloutCompleted { meta, .. }
            | DomainEvent::RolloutFailed { meta, .. }
            | DomainEvent::RolloutCancelled { meta, .. }
            | DomainEvent::ProgressTicked { meta, .. }
            | DomainEvent::BudgetExceeded { meta, .. }
            | DomainEvent::WorkerUpdated { meta, .. }
            | DomainEvent::LogLine { meta, .. }
            | DomainEvent::TrainingStarted { meta, .. }
            | DomainEvent::TrainingMetric { meta, .. }
            | DomainEvent::TrainingCompleted { meta, .. }
            | DomainEvent::TrainingFailed { meta, .. }
            | DomainEvent::AdapterPublished { meta, .. }
            | DomainEvent::EvalCompleted { meta, .. } => meta,
        }
    }

    pub fn run_id(&self) -> Option<&str> {
        self.meta().run_id.as_deref()
    }

    // Convenience constructors keep call sites short and consistent.

    pub fn rollout_started(run_id: &str, manifest: RunManifest) -> Self {
        Self::RolloutStarted {
            meta: EventMeta::for_run(Some(run_id)),
            manifest,
        }
    }

    pub fn step_recorded(run_id: &str, step: TrajectoryStep) -> Self {
        Self::StepRecorded {
            meta: EventMeta::for_run(Some(run_id)),
            step,
        }
    }

    pub fn reward_computed(run_id: &str, reward: &crate::models::RewardRecord) -> Self {
        Self::RewardComputed {
            meta: EventMeta::for_run(Some(run_id)),
            terminal_reward: reward.terminal_reward,
            provenance: reward.provenance.clone(),
            audit_flags: reward.audit_flags.clone(),
        }
    }

    pub fn rollout_completed(run_id: &str, detail: RunDetail) -> Self {
        Self::RolloutCompleted {
            meta: EventMeta::for_run(Some(run_id)),
            detail,
        }
    }

    pub fn rollout_failed(run_id: &str, error: impl Into<String>) -> Self {
        Self::RolloutFailed {
            meta: EventMeta::for_run(Some(run_id)),
            error: error.into(),
        }
    }

    pub fn rollout_cancelled(run_id: &str) -> Self {
        Self::RolloutCancelled {
            meta: EventMeta::for_run(Some(run_id)),
            reason: default_cancel_reason(),
        }
    }

    pub fn progress_ticked(
        run_id: &str,
        step_index: u32,
        tool: Option<String>,
        tokens: u64,
        cost_usd: f64,
    ) -> Self {
        Self::ProgressTicked {
            meta: EventMeta::for_run(Some(run_id)),
            step_index,
            tool,
            tokens,
            cost_usd,
        }
    }

    pub fn budget_exceeded(run_id: &str, spent_usd: f64, cap_usd: f64, window_hours: f64) -> Self {
        Self::BudgetExceeded {
            meta: EventMeta::for_run(Some(run_id)),
            spent_usd,
            cap_usd,
            window_hours,
        }
    }

    pub fn worker_updated(run_id: Option<&str>, worker: WorkerRecord) -> Self {
        Self::WorkerUpdated {
            meta: EventMeta::for_run(run_id),
            worker,
        }
    }

    pub fn training_started(record: TrainingRunRecord) -> Self {
        Self::TrainingStarted {
            meta: EventMeta::for_run(None),
            training_run_id: record.id.clone(),
            record,
        }
    }

    pub fn training_metric(training_run_id: &str, metric: TrainingMetricPoint) -> Self {
        Self::TrainingMetric {
            meta: EventMeta::for_run(None),
            training_run_id: training_run_id.to_string(),
            metric,
        }
    }

    pub fn training_completed(record: TrainingRunRecord) -> Self {
        Self::TrainingCompleted {
            meta: EventMeta::for_run(None),
            training_run_id: record.id.clone(),
            record,
        }
    }

    pub fn training_failed(training_run_id: &str, error: impl Into<String>) -> Self {
        Self::TrainingFailed {
            meta: EventMeta::for_run(None),
            training_run_id: training_run_id.to_string(),
            error: error.into(),
        }
    }

    pub fn adapter_published(adapter: AdapterRecord) -> Self {
        Self::AdapterPublished {
            meta: EventMeta::for_run(None),
            adapter,
        }
    }

    pub fn eval_completed(report: EvalReport) -> Self {
        Self::EvalCompleted {
            meta: EventMeta::for_run(None),
            report,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_serialize_with_flat_kind_tag() {
        let ev = DomainEvent::rollout_failed("run-1", "boom");
        let json = serde_json::to_value(&ev).unwrap();
        assert_eq!(json["kind"], "rollout.failed");
        assert_eq!(json["run_id"], "run-1");
        assert_eq!(json["error"], "boom");
        assert!(json["event_id"].is_string());
        let back: DomainEvent = serde_json::from_value(json).unwrap();
        assert_eq!(back, ev);
    }
}
