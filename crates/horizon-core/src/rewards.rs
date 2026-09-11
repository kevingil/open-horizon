//! Pure reward signal primitives. No I/O, deterministic, replayable.
//!
//! These heuristic rubrics exist for replay and for the in-house repo
//! path. Production reward for long-horizon agents comes from verifiers
//! rubrics or judge models; do not extend these with more string matching.

use serde_json::Value;

use crate::models::{RewardRecord, RewardSignal, TaskSpec, Trajectory};
use crate::round_to;

fn signal(name: &str, value: f64, reason: impl Into<String>, weight: f64) -> RewardSignal {
    RewardSignal {
        name: name.into(),
        value,
        weight,
        reason: reason.into(),
    }
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
    pub fn name(self) -> &'static str {
        match self {
            SignalKind::StepCount => "step_count",
            SignalKind::ErrorPenalty => "error_penalty",
            SignalKind::SuccessCriteria => "success_criteria",
            SignalKind::Finish => "finish",
            SignalKind::TestsPass => "tests_pass",
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

    pub fn evaluate(self, task: &TaskSpec, trajectory: &Trajectory, weight: f64) -> RewardSignal {
        match self {
            SignalKind::StepCount => {
                let step_count = trajectory.steps.len();
                let denom = (task.horizon as usize * 2).max(1);
                let value = (step_count as f64 / denom as f64).min(1.0);
                signal(
                    "step_count",
                    value,
                    format!("{step_count} steps against horizon {}", task.horizon),
                    weight,
                )
            }
            SignalKind::ErrorPenalty => {
                if trajectory.errors.is_empty() {
                    return signal("error_penalty", 0.0, "no errors", weight);
                }
                let first: String = trajectory.errors[0].chars().take(120).collect();
                signal(
                    "error_penalty",
                    -1.0,
                    format!("{} errors: {first}", trajectory.errors.len()),
                    weight,
                )
            }
            SignalKind::SuccessCriteria => {
                if task.success_criteria.is_empty() {
                    return signal("success_criteria", 0.0, "no criteria specified", weight);
                }
                let haystack = trajectory_text(trajectory).to_lowercase();
                let matches: Vec<&str> = task
                    .success_criteria
                    .iter()
                    .filter(|c| haystack.contains(&c.trim().to_lowercase()))
                    .map(String::as_str)
                    .collect();
                let ratio = matches.len() as f64 / task.success_criteria.len() as f64;
                let reason = format!(
                    "matched {}/{}: {}",
                    matches.len(),
                    task.success_criteria.len(),
                    matches.join(", ")
                );
                signal("success_criteria", ratio, reason, weight)
            }
            SignalKind::Finish => {
                for step in trajectory.steps.iter().rev() {
                    if step.actor != "policy" {
                        continue;
                    }
                    if crate::tools::is_finish(&step.content) {
                        return signal("finish", 1.0, "policy called finish", weight);
                    }
                    break;
                }
                signal("finish", 0.0, "no finish call", weight)
            }
            SignalKind::TestsPass => tests_pass(trajectory, weight),
        }
    }
}

fn tests_pass(trajectory: &Trajectory, weight: f64) -> RewardSignal {
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
        return signal("tests_pass", 0.0, "no test run observed", weight);
    };
    let rc = last.get("returncode").and_then(Value::as_i64).unwrap_or(1);
    if rc == 0 {
        let sandbox = last
            .get("sandbox")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        return signal(
            "tests_pass",
            1.0,
            format!("pytest exit 0 (sandbox={sandbox})"),
            weight,
        );
    }
    signal("tests_pass", -1.0, format!("pytest exit {rc}"), weight)
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
        self.signals.iter().map(|(k, _)| k.name()).collect()
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
pub fn score(task: &TaskSpec, trajectory: &Trajectory, rubric: &RubricSpec) -> RewardRecord {
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
    RewardRecord {
        terminal_reward: round_to(terminal, 4),
        rubric: rubric.provenance(),
        source: "rubric".into(),
        signals: signals
            .into_iter()
            .map(|s| RewardSignal {
                value: round_to(s.value, 4),
                ..s
            })
            .collect(),
        audit_flags,
    }
}

fn trajectory_text(trajectory: &Trajectory) -> String {
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
    use crate::models::TrajectoryStep;

    fn task() -> TaskSpec {
        TaskSpec {
            id: "task-1".into(),
            prompt: "do it".into(),
            repo_snapshot: ".".into(),
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
        let traj = Trajectory {
            steps,
            errors: vec![],
        };
        let reward = score(&task(), &traj, &heuristic_v1());
        assert!(
            reward.terminal_reward > 0.4,
            "got {}",
            reward.terminal_reward
        );
        assert!(reward.signals.iter().all(|s| s.value >= 0.0));
        assert_eq!(reward.rubric, "heuristic-v1");
        assert_eq!(reward.source, "rubric");
    }

    #[test]
    fn errors_penalize_and_flag() {
        let traj = Trajectory {
            steps: vec![],
            errors: vec!["boom".into()],
        };
        let reward = score(&task(), &traj, &heuristic_v1());
        assert!(reward.terminal_reward < 0.0);
        assert!(reward
            .signals
            .iter()
            .any(|s| s.name == "error_penalty" && s.value < 0.0));
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
