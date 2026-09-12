//! Event stream. Every event is one envelope on the durable log:
//!
//! ```json
//! {"seq": 42, "at": "...", "subject": {"kind": "run", "id": "run-1"},
//!  "kind": "step.recorded", "payload": {"step": {...}}}
//! ```
//!
//! `subject` says what the event is about so clients route without
//! inspecting payloads; `kind` + `payload` carry the typed body.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::models::{
    AdapterRecord, EvalReport, NodeRecord, RewardRecord, RunManifest, TrainingMetricPoint,
    TrainingRunRecord, TrajectoryStep,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Subject {
    Run { id: String },
    TrainingRun { id: String },
    Adapter { id: String },
    Node { id: String },
}

impl Subject {
    pub fn kind(&self) -> &'static str {
        match self {
            Subject::Run { .. } => "run",
            Subject::TrainingRun { .. } => "training_run",
            Subject::Adapter { .. } => "adapter",
            Subject::Node { .. } => "node",
        }
    }

    pub fn id(&self) -> &str {
        match self {
            Subject::Run { id }
            | Subject::TrainingRun { id }
            | Subject::Adapter { id }
            | Subject::Node { id } => id,
        }
    }

    pub fn run_id(&self) -> Option<&str> {
        match self {
            Subject::Run { id } => Some(id),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(tag = "kind", content = "payload")]
#[allow(clippy::large_enum_variant)]
pub enum DomainEvent {
    #[serde(rename = "rollout.queued")]
    RolloutQueued { manifest: RunManifest },
    #[serde(rename = "rollout.started")]
    RolloutStarted { manifest: RunManifest },
    #[serde(rename = "step.recorded")]
    StepRecorded { step: TrajectoryStep },
    #[serde(rename = "progress.ticked")]
    ProgressTicked {
        turn: u32,
        tool: Option<String>,
        tokens: u64,
        cost_usd: f64,
    },
    #[serde(rename = "reward.computed")]
    RewardComputed { reward: RewardRecord },
    #[serde(rename = "rollout.completed")]
    RolloutCompleted { manifest: RunManifest },
    #[serde(rename = "rollout.failed")]
    RolloutFailed {
        manifest: RunManifest,
        error: String,
    },
    #[serde(rename = "rollout.cancelled")]
    RolloutCancelled { manifest: RunManifest },
    #[serde(rename = "budget.exceeded")]
    BudgetExceeded {
        spent_usd: f64,
        cap_usd: f64,
        window_hours: f64,
    },
    #[serde(rename = "node.updated")]
    NodeUpdated { node: NodeRecord },
    #[serde(rename = "log.line")]
    LogLine {
        level: String,
        logger: String,
        message: String,
        context: std::collections::BTreeMap<String, String>,
    },
    #[serde(rename = "training.queued")]
    TrainingQueued { record: TrainingRunRecord },
    #[serde(rename = "training.started")]
    TrainingStarted { record: TrainingRunRecord },
    #[serde(rename = "training.metric")]
    TrainingMetric { metric: TrainingMetricPoint },
    #[serde(rename = "training.completed")]
    TrainingCompleted { record: TrainingRunRecord },
    #[serde(rename = "training.failed")]
    TrainingFailed {
        record: TrainingRunRecord,
        error: String,
    },
    #[serde(rename = "adapter.published")]
    AdapterPublished { adapter: AdapterRecord },
    #[serde(rename = "eval.completed")]
    EvalCompleted { report: EvalReport },
}

impl DomainEvent {
    pub fn kind(&self) -> &'static str {
        match self {
            DomainEvent::RolloutQueued { .. } => "rollout.queued",
            DomainEvent::RolloutStarted { .. } => "rollout.started",
            DomainEvent::StepRecorded { .. } => "step.recorded",
            DomainEvent::ProgressTicked { .. } => "progress.ticked",
            DomainEvent::RewardComputed { .. } => "reward.computed",
            DomainEvent::RolloutCompleted { .. } => "rollout.completed",
            DomainEvent::RolloutFailed { .. } => "rollout.failed",
            DomainEvent::RolloutCancelled { .. } => "rollout.cancelled",
            DomainEvent::BudgetExceeded { .. } => "budget.exceeded",
            DomainEvent::NodeUpdated { .. } => "node.updated",
            DomainEvent::LogLine { .. } => "log.line",
            DomainEvent::TrainingQueued { .. } => "training.queued",
            DomainEvent::TrainingStarted { .. } => "training.started",
            DomainEvent::TrainingMetric { .. } => "training.metric",
            DomainEvent::TrainingCompleted { .. } => "training.completed",
            DomainEvent::TrainingFailed { .. } => "training.failed",
            DomainEvent::AdapterPublished { .. } => "adapter.published",
            DomainEvent::EvalCompleted { .. } => "eval.completed",
        }
    }
}

/// An event with its subject, before it is assigned a sequence number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct Event {
    pub at: DateTime<Utc>,
    #[serde(default)]
    pub subject: Option<Subject>,
    #[serde(flatten)]
    pub event: DomainEvent,
}

impl Event {
    pub fn new(subject: Option<Subject>, event: DomainEvent) -> Self {
        Self {
            at: crate::utc_now(),
            subject,
            event,
        }
    }

    pub fn for_run(run_id: &str, event: DomainEvent) -> Self {
        Self::new(Some(Subject::Run { id: run_id.into() }), event)
    }

    pub fn for_training(training_run_id: &str, event: DomainEvent) -> Self {
        Self::new(
            Some(Subject::TrainingRun {
                id: training_run_id.into(),
            }),
            event,
        )
    }

    pub fn for_adapter(adapter_id: &str, event: DomainEvent) -> Self {
        Self::new(
            Some(Subject::Adapter {
                id: adapter_id.into(),
            }),
            event,
        )
    }

    pub fn global(event: DomainEvent) -> Self {
        Self::new(None, event)
    }

    pub fn kind(&self) -> &'static str {
        self.event.kind()
    }

    pub fn run_id(&self) -> Option<&str> {
        self.subject.as_ref().and_then(Subject::run_id)
    }
}

/// One row of the durable event log. `seq` is monotonic per store so
/// clients can resume a WebSocket with `?since=<seq>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct EventEnvelope {
    pub seq: i64,
    #[serde(flatten)]
    pub event: Event,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_wire_shape() {
        let env = EventEnvelope {
            seq: 7,
            event: Event::for_run(
                "run-1",
                DomainEvent::BudgetExceeded {
                    spent_usd: 1.0,
                    cap_usd: 0.5,
                    window_hours: 24.0,
                },
            ),
        };
        let json = serde_json::to_value(&env).unwrap();
        assert_eq!(json["seq"], 7);
        assert_eq!(json["kind"], "budget.exceeded");
        assert_eq!(json["subject"]["kind"], "run");
        assert_eq!(json["subject"]["id"], "run-1");
        assert_eq!(json["payload"]["cap_usd"], 0.5);
        assert!(json["at"].is_string());
        let back: EventEnvelope = serde_json::from_value(json).unwrap();
        assert_eq!(back, env);
    }
}
