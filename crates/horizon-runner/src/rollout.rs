//! Per-turn OpenAI tool-call loop for the in-house repo path.
//!
//! Wire-level flow per turn:
//!   1. Send accumulated messages + tool definitions to the model.
//!   2. A `tool_calls` entry becomes our `{"tool", "input"}` action; bare
//!      text is an implicit `finish` with the text as summary.
//!   3. Hand the action to the repo runner; thread the observation back
//!      as a tool message and loop.
//!
//! Cancellation is honoured mid-generation: the HTTP call races the
//! cancellation token, so a cancelled rollout never waits for the
//! provider to finish producing tokens it will throw away.

use std::sync::Arc;

use serde_json::{json, Map, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use horizon_core::models::{TaskSpec, TrajectoryRecord, TrajectoryStep};
use horizon_core::pricing::{estimate_cost_usd, TokenUsage};
use horizon_core::tools::{is_finish, openai_tool_definitions, tool_name_of};

use crate::openai::{ChatMessage, OpenAiClient, OpenAiError};
use crate::repo_runner::RepoRunner;

pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a software-engineering agent working inside a sandboxed \
repository snapshot. Use the provided tools to inspect the repo, run read-only commands, and report \
findings. Call `finish` when the task's success criteria are met. Prefer small, observable steps.";

/// Progress the loop reports while it runs. Consumers turn these into
/// domain events; the loop itself knows nothing about the event bus.
#[derive(Debug, Clone)]
pub enum RolloutEvent {
    Step(TrajectoryStep),
    Progress {
        turn: u32,
        tokens: u64,
        cost_usd: f64,
        tool: Option<String>,
    },
    /// Trainer-ready pair for the turn: the prompt as sent and the action
    /// produced. `step_index` matches the action step in the trajectory.
    Turn {
        step_index: u32,
        prompt_snapshot: String,
        completion: String,
        sampling_args: Map<String, Value>,
    },
}

#[derive(Debug, Clone)]
pub struct RolloutParams {
    pub model: String,
    pub task: TaskSpec,
    pub horizon: u32,
    pub max_tokens_per_run: u64,
    pub max_output_tokens: u32,
    pub max_retries: u32,
    pub system_prompt: Option<String>,
    pub extra_body: Map<String, Value>,
    /// Emit `RolloutEvent::Turn` records (costs a JSON snapshot per turn).
    pub record_turns: bool,
}

#[derive(Debug, Clone)]
pub struct RolloutOutcome {
    pub trajectory: TrajectoryRecord,
    pub usage: TokenUsage,
    pub reasoning_tokens: u64,
    pub cost_usd: f64,
    pub policy_name: String,
    pub cancelled: bool,
    pub token_overflow: bool,
}

impl RolloutOutcome {
    pub fn total_tokens(&self) -> u64 {
        self.usage.total()
    }
}

/// Optional out-of-process cancellation probe (e.g. the job row's
/// `cancel_requested` flag) polled once per turn alongside the in-process
/// token, so a cancel issued to another replica still lands.
pub type CancelProbe = Arc<dyn Fn() -> bool + Send + Sync>;

#[derive(Debug, thiserror::Error)]
pub enum RolloutError {
    #[error("workspace: {0}")]
    Workspace(#[from] std::io::Error),
    #[error("policy: {0}")]
    Policy(#[from] OpenAiError),
}

pub async fn run_repo_rollout(
    client: &OpenAiClient,
    runner: &RepoRunner,
    params: RolloutParams,
    events: mpsc::Sender<RolloutEvent>,
    cancel: CancellationToken,
    cancel_probe: Option<CancelProbe>,
) -> Result<RolloutOutcome, RolloutError> {
    runner.create_task(&params.task).await?;
    let result = drive(
        client,
        runner,
        &params,
        &events,
        &cancel,
        cancel_probe.as_ref(),
    )
    .await;
    runner.cleanup_task(&params.task.id).await;
    result
}

async fn drive(
    client: &OpenAiClient,
    runner: &RepoRunner,
    params: &RolloutParams,
    events: &mpsc::Sender<RolloutEvent>,
    cancel: &CancellationToken,
    cancel_probe: Option<&CancelProbe>,
) -> Result<RolloutOutcome, RolloutError> {
    let tools = openai_tool_definitions();
    let mut messages = vec![
        ChatMessage::text(
            "system",
            params
                .system_prompt
                .as_deref()
                .unwrap_or(DEFAULT_SYSTEM_PROMPT),
        ),
        ChatMessage::text("user", initial_user_prompt(&params.task)),
    ];
    let mut steps: Vec<TrajectoryStep> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    let mut usage = TokenUsage::default();
    let mut reasoning_tokens = 0u64;
    let mut cancelled = false;
    let mut token_overflow = false;

    let mut sampling_args = Map::new();
    sampling_args.insert("max_tokens".into(), json!(params.max_output_tokens));
    for (k, v) in &params.extra_body {
        sampling_args.insert(k.clone(), v.clone());
    }

    let is_cancelled = || cancel.is_cancelled() || cancel_probe.is_some_and(|p| p());

    for turn in 0..params.horizon {
        if is_cancelled() {
            errors.push("cancelled".into());
            cancelled = true;
            break;
        }
        let prompt_snapshot = if params.record_turns {
            serde_json::to_string(&messages).unwrap_or_default()
        } else {
            String::new()
        };

        let completion = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                errors.push("cancelled".into());
                cancelled = true;
                break;
            }
            res = client.chat_with_retry(
                &params.model, &messages, &tools, params.max_output_tokens, &params.extra_body, params.max_retries,
            ) => res?,
        };
        usage.input_tokens += completion.usage.input_tokens;
        usage.output_tokens += completion.usage.output_tokens;
        usage.cache_read_tokens += completion.usage.cache_read_tokens;
        usage.cache_write_tokens += completion.usage.cache_write_tokens;
        reasoning_tokens += completion.reasoning_tokens;

        let (action, tool_call_id) = match &completion.tool_call {
            None => {
                let summary = if completion.text.is_empty() {
                    "done".to_string()
                } else {
                    completion.text.clone()
                };
                (
                    json!({"tool": "finish", "input": {"summary": summary}}).to_string(),
                    None,
                )
            }
            Some(call) => {
                messages.push(ChatMessage {
                    role: "assistant".into(),
                    content: if completion.text.is_empty() { None } else { Some(Value::String(completion.text.clone())) },
                    tool_calls: Some(vec![json!({
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": Value::Object(call.input.clone()).to_string()},
                    })]),
                    tool_call_id: None,
                });
                (
                    json!({"tool": call.name, "input": call.input}).to_string(),
                    Some(call.id.clone()),
                )
            }
        };

        let observation = runner.step(&params.task.id, &action).await;
        let mut action_step = TrajectoryStep::new(turn * 2, "policy", "action", action.clone());
        action_step.has_training_metadata = params.record_turns;
        let obs_step = TrajectoryStep::new(
            turn * 2 + 1,
            "environment",
            "observation",
            observation.clone(),
        );
        steps.push(action_step.clone());
        steps.push(obs_step.clone());
        let _ = events.send(RolloutEvent::Step(action_step)).await;
        let _ = events.send(RolloutEvent::Step(obs_step)).await;
        if params.record_turns {
            let _ = events
                .send(RolloutEvent::Turn {
                    step_index: turn * 2,
                    prompt_snapshot,
                    completion: action.clone(),
                    sampling_args: sampling_args.clone(),
                })
                .await;
        }

        let tokens = usage.total();
        let cost = estimate_cost_usd(&params.model, usage);
        let _ = events
            .send(RolloutEvent::Progress {
                turn,
                tokens,
                cost_usd: cost,
                tool: tool_name_of(&action),
            })
            .await;

        if let Some(id) = tool_call_id {
            messages.push(ChatMessage::tool_result(&id, observation));
        }

        if is_cancelled() {
            errors.push("cancelled".into());
            cancelled = true;
            break;
        }
        if is_finish(&action) {
            break;
        }
        if tokens > params.max_tokens_per_run {
            errors.push(format!(
                "token budget exceeded: {tokens} > {}",
                params.max_tokens_per_run
            ));
            token_overflow = true;
            break;
        }
    }

    Ok(RolloutOutcome {
        trajectory: TrajectoryRecord::new(&params.task.id, steps, errors),
        usage,
        reasoning_tokens,
        cost_usd: estimate_cost_usd(&params.model, usage),
        policy_name: format!("openai:{}", params.model),
        cancelled,
        token_overflow,
    })
}

fn initial_user_prompt(task: &TaskSpec) -> String {
    let criteria = if task.success_criteria.is_empty() {
        "- (no explicit criteria)".to_string()
    } else {
        task.success_criteria
            .iter()
            .map(|c| format!("- {c}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "Task: {}\n\nSuccess criteria:\n{criteria}\n\nHorizon: up to {} tool calls. Call `finish` when done.",
        task.prompt, task.horizon
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::OpenAiConfig;
    use crate::repo_runner::RepoRunnerConfig;
    use crate::sandbox::Sandbox;
    use axum::{routing::post, Json, Router};
    use horizon_core::models::ToolPermission;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Fake OpenAI-compatible server: first call reads README, second finishes.
    async fn fake_server(delay: Duration) -> (String, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(body): Json<Value>| {
                let counter = counter.clone();
                async move {
                    tokio::time::sleep(delay).await;
                    let n = counter.fetch_add(1, Ordering::SeqCst);
                    assert!(body["tools"].is_array());
                    let msg = if n == 0 {
                        json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "read_file", "arguments": "{\"path\": \"README.md\"}"}}]})
                    } else {
                        // Second turn must carry the tool result threaded on call_1.
                        let last = body["messages"].as_array().unwrap().last().unwrap().clone();
                        assert_eq!(last["role"], "tool");
                        assert_eq!(last["tool_call_id"], "call_1");
                        json!({"role": "assistant", "content": "all good"})
                    };
                    Json(json!({"choices": [{"message": msg}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}/v1"), calls)
    }

    fn fixtures() -> (tempfile::TempDir, tempfile::TempDir, RepoRunner) {
        let src = tempfile::tempdir().unwrap();
        std::fs::write(src.path().join("README.md"), "the readme").unwrap();
        let scratch = tempfile::tempdir().unwrap();
        let runner = RepoRunner::new(
            RepoRunnerConfig {
                source_root: src.path().to_path_buf(),
                scratch_root: scratch.path().to_path_buf(),
                command_timeout: Duration::from_secs(5),
                max_output_bytes: 16_384,
            },
            Sandbox::host(),
        )
        .unwrap();
        (src, scratch, runner)
    }

    fn params(model: &str) -> RolloutParams {
        RolloutParams {
            model: model.into(),
            task: TaskSpec {
                id: "task-1".into(),
                prompt: "summarise".into(),
                repo_snapshot: ".".into(),
                tool_permissions: vec![ToolPermission::Read],
                horizon: 4,
                success_criteria: vec!["readme".into()],
            },
            horizon: 4,
            max_tokens_per_run: 100_000,
            max_output_tokens: 256,
            max_retries: 0,
            system_prompt: None,
            extra_body: Map::new(),
            record_turns: true,
        }
    }

    #[tokio::test]
    async fn loop_threads_tool_results_and_finishes() {
        let (base_url, calls) = fake_server(Duration::ZERO).await;
        let (_s, _c, runner) = fixtures();
        let client = OpenAiClient::new(OpenAiConfig {
            base_url,
            api_key: "x".into(),
            timeout: Duration::from_secs(5),
        });
        let (tx, mut rx) = mpsc::channel(64);
        let outcome = run_repo_rollout(
            &client,
            &runner,
            params("gpt-4.1-mini"),
            tx,
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(outcome.trajectory.steps.len(), 4);
        assert!(!outcome.cancelled);
        assert!(is_finish(&outcome.trajectory.steps[2].content));
        assert_eq!(outcome.usage.input_tokens, 20);
        assert!(outcome.cost_usd > 0.0);
        let mut turns = 0;
        let mut steps = 0;
        rx.close();
        while let Some(ev) = rx.recv().await {
            match ev {
                RolloutEvent::Turn { .. } => turns += 1,
                RolloutEvent::Step(_) => steps += 1,
                RolloutEvent::Progress { .. } => {}
            }
        }
        assert_eq!((turns, steps), (2, 4));
    }

    #[tokio::test]
    async fn cancel_interrupts_generation() {
        let (base_url, _calls) = fake_server(Duration::from_secs(5)).await;
        let (_s, _c, runner) = fixtures();
        let client = OpenAiClient::new(OpenAiConfig {
            base_url,
            api_key: "x".into(),
            timeout: Duration::from_secs(30),
        });
        let (tx, _rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let c2 = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            c2.cancel();
        });
        let started = std::time::Instant::now();
        let outcome = run_repo_rollout(&client, &runner, params("sglang:local"), tx, cancel, None)
            .await
            .unwrap();
        assert!(outcome.cancelled);
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "cancel should not wait for the provider"
        );
        assert_eq!(outcome.trajectory.errors, vec!["cancelled"]);
    }
}
