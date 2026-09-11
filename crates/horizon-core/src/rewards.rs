//! Pure reward signal primitives. No I/O, deterministic, replayable.
//!
//! These heuristic rubrics exist for replay and for the in-house repo
//! path. Production reward for long-horizon agents comes from verifiers
//! rubrics or judge models; do not extend these with more string matching.

use serde_json::{json, Value};

use crate::models::{RewardPenalty, RewardRecord, TaskSpec, TrajectoryRecord};
use crate::round_to;

#[derive(Debug, Clone, PartialEq)]
pub struct RewardSignal {
    pub name: &'static str,
    pub value: f64,
    pub reason: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalKind {
    StepCount,
    ErrorPenalty,
    SuccessCriteria,
    Finish,
    TestsPass,
}

impl SignalKind {
    pub fn fn_name(self) -> &'static str {
        match self {
            SignalKind::StepCount => "step_count_signal",
            SignalKind::ErrorPenalty => "error_penalty_signal",
            SignalKind::SuccessCriteria => "success_criteria_signal",
            SignalKind::Finish => "finish_signal",
            SignalKind::TestsPass => "tests_pass_signal",
        }
    }

    pub fn default_weight(self) -> f64 {
        match self {
            SignalKind::StepCount => 0.3,
            SignalKind::ErrorPenalty => 1.0,
            SignalKind::SuccessCriteria => 0.7,
            SignalKind::Finish => 0.3,
            SignalKind::TestsPass => 0.8,
        }
    }

    pub fn evaluate(
        self,
        task: &TaskSpec,
        trajectory: &TrajectoryRecord,
        weight: f64,
    ) -> RewardSignal {
        match self {
            SignalKind::StepCount => step_count_signal(task, trajectory, weight),
            SignalKind::ErrorPenalty => error_penalty_signal(trajectory, weight),
            SignalKind::SuccessCriteria => success_criteria_signal(task, trajectory, weight),
            SignalKind::Finish => finish_signal(trajectory, weight),
            SignalKind::TestsPass => tests_pass_signal(trajectory, weight),
        }
    }
}

fn step_count_signal(task: &TaskSpec, trajectory: &TrajectoryRecord, weight: f64) -> RewardSignal {
    let step_count = trajectory.steps.len();
    let denom = (task.horizon as usize * 2).max(1);
    let value = (step_count as f64 / denom as f64).min(1.0);
    RewardSignal {
        name: "step_count",
        value,
        reason: format!(
            "{step_count} trajectory steps against horizon={}",
            task.horizon
        ),
        weight,
    }
}

fn error_penalty_signal(trajectory: &TrajectoryRecord, weight: f64) -> RewardSignal {
    if trajectory.errors.is_empty() {
        return RewardSignal {
            name: "error_penalty",
            value: 0.0,
            reason: "no errors".into(),
            weight,
        };
    }
    let first: String = trajectory.errors[0].chars().take(120).collect();
    RewardSignal {
        name: "error_penalty",
        value: -1.0,
        reason: format!("{} errors: {first}", trajectory.errors.len()),
        weight,
    }
}

fn success_criteria_signal(
    task: &TaskSpec,
    trajectory: &TrajectoryRecord,
    weight: f64,
) -> RewardSignal {
    if task.success_criteria.is_empty() {
        return RewardSignal {
            name: "success_criteria",
            value: 0.0,
            reason: "no criteria specified".into(),
            weight,
        };
    }
    let haystack = trajectory_text(trajectory).to_lowercase();
    let matches: Vec<&String> = task
        .success_criteria
        .iter()
        .filter(|c| haystack.contains(&c.trim().to_lowercase()))
        .collect();
    let ratio = matches.len() as f64 / task.success_criteria.len() as f64;
    let rendered: Vec<String> = matches.iter().map(|m| format!("'{m}'")).collect();
    RewardSignal {
        name: "success_criteria",
        value: ratio,
        reason: format!(
            "matched {}/{}: [{}]",
            matches.len(),
            task.success_criteria.len(),
            rendered.join(", ")
        ),
        weight,
    }
}

fn finish_signal(trajectory: &TrajectoryRecord, weight: f64) -> RewardSignal {
    for step in trajectory.steps.iter().rev() {
        if step.actor != "policy" {
            continue;
        }
        if crate::tools::is_finish(&step.content) {
            return RewardSignal {
                name: "finish",
                value: 1.0,
                reason: "policy called finish".into(),
                weight,
            };
        }
        break;
    }
    RewardSignal {
        name: "finish",
        value: 0.0,
        reason: "no finish call".into(),
        weight,
    }
}

fn tests_pass_signal(trajectory: &TrajectoryRecord, weight: f64) -> RewardSignal {
    let mut last: Option<Value> = None;
    for step in &trajectory.steps {
        if step.actor != "environment" {
            continue;
        }
        let Some(payload) = maybe_json(&step.content) else {
            continue;
        };
        if payload.get("tool").and_then(Value::as_str) != Some("run_command") {
            continue;
        }
        let command = payload.get("command").and_then(Value::as_str).unwrap_or("");
        let first = command.split(' ').next().unwrap_or("");
        if first == "pytest"
            || command.starts_with("python -m pytest")
            || command.starts_with("python3 -m pytest")
        {
            last = Some(payload);
        }
    }
    let Some(last) = last else {
        return RewardSignal {
            name: "tests_pass",
            value: 0.0,
            reason: "no test run observed".into(),
            weight,
        };
    };
    let rc = last.get("returncode").and_then(Value::as_i64).unwrap_or(1);
    if rc == 0 {
        let sandbox = last
            .get("sandbox")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        return RewardSignal {
            name: "tests_pass",
            value: 1.0,
            reason: format!("pytest exit 0 (sandbox={sandbox})"),
            weight,
        };
    }
    RewardSignal {
        name: "tests_pass",
        value: -1.0,
        reason: format!("pytest exit {rc}"),
        weight,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RubricSpec {
    pub name: &'static str,
    pub version: &'static str,
    pub signals: Vec<(SignalKind, f64)>,
}

impl RubricSpec {
    pub fn provenance(&self) -> String {
        format!("{}-{}", self.name, self.version)
    }

    pub fn signal_names(&self) -> Vec<&'static str> {
        self.signals.iter().map(|(k, _)| k.fn_name()).collect()
    }
}

fn with_default(kind: SignalKind) -> (SignalKind, f64) {
    (kind, kind.default_weight())
}

pub fn heuristic_v1() -> RubricSpec {
    RubricSpec {
        name: "heuristic",
        version: "v1",
        signals: vec![
            with_default(SignalKind::StepCount),
            with_default(SignalKind::ErrorPenalty),
            with_default(SignalKind::SuccessCriteria),
            with_default(SignalKind::Finish),
        ],
    }
}

pub fn coding_v1() -> RubricSpec {
    RubricSpec {
        name: "coding",
        version: "v1",
        signals: vec![
            with_default(SignalKind::ErrorPenalty),
            with_default(SignalKind::SuccessCriteria),
            with_default(SignalKind::Finish),
            with_default(SignalKind::TestsPass),
        ],
    }
}

pub fn strict_finish_v1() -> RubricSpec {
    RubricSpec {
        name: "strict-finish",
        version: "v1",
        signals: vec![
            (SignalKind::ErrorPenalty, 1.0),
            (SignalKind::Finish, 1.5),
            (SignalKind::SuccessCriteria, 0.5),
        ],
    }
}

/// Central rubric registry, keyed by provenance (`heuristic-v1`).
pub fn rubrics() -> Vec<RubricSpec> {
    vec![heuristic_v1(), coding_v1(), strict_finish_v1()]
}

pub fn get_rubric(name: &str) -> Option<RubricSpec> {
    rubrics().into_iter().find(|r| r.provenance() == name)
}

pub fn rubric_names() -> Vec<String> {
    let mut names: Vec<String> = rubrics().iter().map(RubricSpec::provenance).collect();
    names.sort();
    names
}

/// Run a rubric over a trajectory. Pure; safe to call in replay.
pub fn score(task: &TaskSpec, trajectory: &TrajectoryRecord, rubric: &RubricSpec) -> RewardRecord {
    let signals: Vec<RewardSignal> = rubric
        .signals
        .iter()
        .map(|(kind, weight)| kind.evaluate(task, trajectory, *weight))
        .collect();
    let weighted: f64 = signals.iter().map(|s| s.value * s.weight).sum();
    let total_weight: f64 = {
        let w: f64 = signals.iter().map(|s| s.weight.abs()).sum();
        if w == 0.0 {
            1.0
        } else {
            w
        }
    };
    let terminal = (weighted / total_weight).clamp(-1.0, 1.0);

    let penalties = signals
        .iter()
        .filter(|s| s.value < 0.0)
        .map(|s| RewardPenalty {
            code: s.name.to_string(),
            value: s.value * s.weight,
            reason: s.reason.clone(),
        })
        .collect();
    let mut audit_flags = Vec::new();
    if !trajectory.errors.is_empty() {
        audit_flags.push("requires-manual-audit".to_string());
    }
    if signals
        .iter()
        .any(|s| s.name == "success_criteria" && s.value == 0.0)
    {
        audit_flags.push("no-success-criteria-match".to_string());
    }
    let step_count = trajectory.steps.len().max(1);
    RewardRecord {
        trajectory_id: trajectory.id.clone(),
        terminal_reward: round_to(terminal, 4),
        step_rewards: vec![round_to(terminal / step_count as f64, 4); step_count],
        penalties,
        audit_flags,
        provenance: encode_provenance(rubric, &signals),
    }
}

fn encode_provenance(rubric: &RubricSpec, signals: &[RewardSignal]) -> String {
    let payload = json!({
        "rubric": rubric.provenance(),
        "signals": signals.iter().map(|s| json!({
            "name": s.name, "value": round_to(s.value, 4), "weight": s.weight, "reason": s.reason,
        })).collect::<Vec<_>>(),
    });
    payload.to_string()
}

fn trajectory_text(trajectory: &TrajectoryRecord) -> String {
    trajectory
        .steps
        .iter()
        .map(|s| s.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn maybe_json(text: &str) -> Option<Value> {
    let v: Value = serde_json::from_str(text).ok()?;
    if v.is_object() {
        Some(v)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ToolPermission, TrajectoryStep};

    fn task() -> TaskSpec {
        TaskSpec {
            id: "task-1".into(),
            prompt: "do it".into(),
            repo_snapshot: ".".into(),
            tool_permissions: vec![ToolPermission::Read],
            horizon: 4,
            success_criteria: vec!["README".into()],
        }
    }

    #[test]
    fn finish_and_criteria_score_positive() {
        let steps = vec![
            TrajectoryStep::new(
                0,
                "policy",
                "action",
                r#"{"tool":"read_file","input":{"path":"README.md"}}"#,
            ),
            TrajectoryStep::new(
                1,
                "environment",
                "observation",
                r#"{"tool":"read_file","content":"readme text"}"#,
            ),
            TrajectoryStep::new(
                2,
                "policy",
                "action",
                r#"{"tool":"finish","input":{"summary":"done"}}"#,
            ),
        ];
        let traj = TrajectoryRecord::new("task-1", steps, vec![]);
        let reward = score(&task(), &traj, &heuristic_v1());
        assert!(
            reward.terminal_reward > 0.4,
            "got {}",
            reward.terminal_reward
        );
        assert!(reward.penalties.is_empty());
        assert!(reward.provenance.contains("heuristic-v1"));
    }

    #[test]
    fn errors_penalize_and_flag() {
        let traj = TrajectoryRecord::new("task-1", vec![], vec!["boom".into()]);
        let reward = score(&task(), &traj, &heuristic_v1());
        assert!(reward.terminal_reward < 0.0);
        assert_eq!(reward.penalties[0].code, "error_penalty");
        assert!(reward
            .audit_flags
            .contains(&"requires-manual-audit".to_string()));
    }

    #[test]
    fn registry_lookup() {
        assert!(get_rubric("coding-v1").is_some());
        assert!(get_rubric("nope").is_none());
        assert_eq!(
            rubric_names(),
            vec!["coding-v1", "heuristic-v1", "strict-finish-v1"]
        );
    }
}
