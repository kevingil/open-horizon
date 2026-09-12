//! Rollout coordinator: resolves profiles and horizons, enforces the
//! budget, drives one of two execution paths, persists progress as it
//! happens, and emits the lifecycle events the dashboard consumes.
//!
//! - repo path: the Rust rollout loop against a sandboxed snapshot.
//! - verifiers path: the Python bridge runs `Environment.run_rollout`
//!   and returns a scored trajectory; Rust owns cost and persistence.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use tokio::sync::{mpsc, Mutex, Semaphore};
use tokio_util::sync::CancellationToken;

use horizon_bridge::PythonBridge;
use horizon_core::events::DomainEvent;
use horizon_core::models::{
    RewardRecord, RewardSignal, RolloutRequest, RunDetail, RunManifest, RunStatus, TaskSpec,
    Trajectory, TrajectoryStep, TurnTrainingRecord,
};
use horizon_core::pricing::{estimate_cost_usd, TokenUsage};
use horizon_core::rewards::{self, RubricSpec};
use horizon_core::scheduling::RoundScheduler;
use horizon_core::{round_to, short_id, utc_now};
use horizon_runner::rollout::CancelProbe;
use horizon_runner::{
    run_repo_rollout, OpenAiClient, OpenAiConfig, RepoRunner, RolloutEvent, RolloutParams,
};
use horizon_store::Store;

use crate::event_bus::EventBus;
use crate::settings::{PolicyProfile, RoutesTo, Settings};

#[derive(Debug, thiserror::Error)]
pub enum CoordinatorError {
    #[error("unknown policy profile: {0:?}. Configured: {1:?}")]
    UnknownProfile(String, Vec<String>),
    #[error(
        "profile {0:?} routes to repo but no repo runner is configured (set RL_ENV_BACKEND=repo)"
    )]
    NoRepoRunner(String),
    #[error("{0}")]
    Store(#[from] horizon_store::StoreError),
    #[error("{0}")]
    Rollout(#[from] horizon_runner::rollout::RolloutError),
    #[error("verifiers bridge: {0}")]
    Bridge(#[from] horizon_bridge::BridgeError),
    #[error("{0}")]
    Other(String),
}

/// Payload stored on a rollout job row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutJob {
    pub run_id: String,
    pub request: RolloutRequest,
}

/// How a finished rollout ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ending {
    Completed,
    Cancelled,
    Failed,
}

pub struct Coordinator {
    pub settings: Arc<Settings>,
    pub store: Arc<Store>,
    pub bus: EventBus,
    pub profiles: BTreeMap<String, PolicyProfile>,
    pub default_profile: String,
    pub round_scheduler: RoundScheduler,
    pub rubric: RubricSpec,
    repo_runner: Option<RepoRunner>,
    bridge: Arc<PythonBridge>,
    clients: Mutex<HashMap<String, OpenAiClient>>,
    verifiers_limits: Mutex<HashMap<String, Arc<Semaphore>>>,
    cancels: Mutex<HashMap<String, CancellationToken>>,
}

impl Coordinator {
    pub fn new(
        settings: Arc<Settings>,
        store: Arc<Store>,
        bus: EventBus,
        repo_runner: Option<RepoRunner>,
        bridge: Arc<PythonBridge>,
    ) -> Self {
        Self {
            profiles: settings.effective_profiles(),
            default_profile: settings.default_policy_profile.clone(),
            round_scheduler: settings.round_scheduler,
            rubric: rewards::heuristic_v1(),
            settings,
            store,
            bus,
            repo_runner,
            bridge,
            clients: Mutex::new(HashMap::new()),
            verifiers_limits: Mutex::new(HashMap::new()),
            cancels: Mutex::new(HashMap::new()),
        }
    }

    pub fn profile_name(&self, request: &RolloutRequest) -> String {
        request
            .policy_profile
            .clone()
            .unwrap_or_else(|| self.default_profile.clone())
    }

    pub fn resolve_profile(
        &self,
        request: &RolloutRequest,
    ) -> Result<(String, &PolicyProfile), CoordinatorError> {
        let name = self.profile_name(request);
        match self.profiles.get(&name) {
            Some(p) => Ok((name, p)),
            None => Err(CoordinatorError::UnknownProfile(
                name,
                self.profiles.keys().cloned().collect(),
            )),
        }
    }

    pub fn resolve_horizon(&self, request: &RolloutRequest) -> u32 {
        request
            .horizon
            .unwrap_or_else(|| self.round_scheduler.current_horizon(0))
    }

    pub fn model_id(&self, profile: &PolicyProfile) -> String {
        match profile.routes_to {
            RoutesTo::Verifiers => format!("verifiers:{}:{}", profile.env_id, profile.model),
            RoutesTo::Repo => format!("openai:{}", profile.model),
        }
    }

    pub fn policy_name(&self) -> String {
        match self.profiles.get(&self.default_profile) {
            Some(p) => self.model_id(p),
            None => format!("openai:{}", self.settings.llm_model),
        }
    }

    fn build_task(&self, request: &RolloutRequest) -> TaskSpec {
        TaskSpec {
            id: short_id("task"),
            prompt: request.prompt.clone(),
            repo_snapshot: request.repo_snapshot.clone(),
            horizon: self.resolve_horizon(request),
            success_criteria: request.success_criteria.clone(),
        }
    }

    fn queued_manifest(
        &self,
        run_id: &str,
        request: &RolloutRequest,
        profile_name: &str,
        profile: &PolicyProfile,
        horizon: u32,
    ) -> RunManifest {
        let now = utc_now();
        RunManifest {
            id: run_id.into(),
            profile: profile_name.into(),
            model_id: self.model_id(profile),
            adapter_id: request.adapter_id.clone(),
            infra_target: request.infra_target.clone(),
            status: RunStatus::Queued,
            horizon,
            tokens: 0,
            cost_usd: 0.0,
            terminal_reward: None,
            step_count: 0,
            error: None,
            created_at: now,
            started_at: None,
            finished_at: None,
            updated_at: now,
        }
    }

    /// Validate the request, persist a queued run, and enqueue the job.
    /// Returns the run id so the caller can cancel before it starts.
    pub fn submit(&self, request: RolloutRequest) -> Result<String, CoordinatorError> {
        let (profile_name, profile) = self.resolve_profile(&request)?;
        let run_id = short_id("run");
        let task = self.build_task(&request);
        let manifest =
            self.queued_manifest(&run_id, &request, &profile_name, profile, task.horizon);
        self.store.save_run(&RunDetail {
            manifest: manifest.clone(),
            task,
            trajectory: Trajectory::default(),
            reward: None,
        })?;
        let payload = serde_json::to_value(RolloutJob {
            run_id: run_id.clone(),
            request,
        })
        .map_err(|e| CoordinatorError::Other(e.to_string()))?;
        self.store
            .enqueue_job(&run_id, horizon_core::models::JobKind::Rollout, &payload)?;
        self.bus
            .run(&run_id, DomainEvent::RolloutQueued { manifest });
        Ok(run_id)
    }

    /// Persist a queued run for a caller that will `execute` it directly
    /// (the eval harness). No job row is created.
    pub fn submit_inline(&self, request: &RolloutRequest) -> Result<String, CoordinatorError> {
        let (profile_name, profile) = self.resolve_profile(request)?;
        let run_id = short_id("run");
        let task = self.build_task(request);
        let manifest = self.queued_manifest(&run_id, request, &profile_name, profile, task.horizon);
        self.store.save_run(&RunDetail {
            manifest: manifest.clone(),
            task,
            trajectory: Trajectory::default(),
            reward: None,
        })?;
        self.bus
            .run(&run_id, DomainEvent::RolloutQueued { manifest });
        Ok(run_id)
    }

    /// Flag a run for cancellation: the job row (visible to every
    /// replica) and the in-process token (interrupts generation now).
    /// A queued run is cancelled immediately.
    pub async fn request_cancel(&self, run_id: &str) -> Result<bool, CoordinatorError> {
        let Some(mut manifest) = self.store.get_manifest(run_id)? else {
            return Ok(false);
        };
        if manifest.status.is_terminal() {
            return Ok(false);
        }
        let accepted = self.store.request_cancel(run_id)?;
        if let Some(token) = self.cancels.lock().await.get(run_id) {
            token.cancel();
        } else if manifest.status == RunStatus::Queued {
            manifest.status = RunStatus::Cancelled;
            manifest.updated_at = utc_now();
            if let Some(detail) = self.store.get_run(run_id)? {
                self.store.save_run(&RunDetail {
                    manifest: manifest.clone(),
                    ..detail
                })?;
            }
            self.bus
                .run(run_id, DomainEvent::RolloutCancelled { manifest });
        }
        Ok(accepted)
    }

    pub async fn register_cancel(&self, run_id: &str) -> CancellationToken {
        let token = CancellationToken::new();
        self.cancels
            .lock()
            .await
            .insert(run_id.to_string(), token.clone());
        token
    }

    pub async fn unregister_cancel(&self, run_id: &str) {
        self.cancels.lock().await.remove(run_id);
    }

    /// Run one rollout to completion. Used by the job runner and the eval
    /// harness. Policy and environment failures become a failed run plus
    /// `rollout.failed`, never an Err.
    pub async fn execute(
        &self,
        request: &RolloutRequest,
        run_id: &str,
        cancel: CancellationToken,
    ) -> Result<RunDetail, CoordinatorError> {
        let (profile_name, profile) = self.resolve_profile(request)?;
        let profile = profile.clone();
        // Reuse the task persisted at submit time so ids stay stable across retries.
        let existing = self.store.get_run(run_id)?;
        let task = existing
            .as_ref()
            .map(|d| d.task.clone())
            .unwrap_or_else(|| self.build_task(request));
        let mut manifest = existing.map(|d| d.manifest).unwrap_or_else(|| {
            self.queued_manifest(run_id, request, &profile_name, &profile, task.horizon)
        });

        if let Some(detail) = self.reject_if_over_budget(&task, &mut manifest)? {
            return Ok(detail);
        }
        manifest.status = RunStatus::Running;
        manifest.started_at = Some(utc_now());
        manifest.finished_at = None;
        manifest.updated_at = utc_now();
        self.store.save_run(&RunDetail {
            manifest: manifest.clone(),
            task: task.clone(),
            trajectory: Trajectory::default(),
            reward: None,
        })?;
        self.bus.run(
            run_id,
            DomainEvent::RolloutStarted {
                manifest: manifest.clone(),
            },
        );

        let result = match profile.routes_to {
            RoutesTo::Verifiers => {
                self.execute_verifiers(&profile_name, &profile, &task, run_id, &manifest, &cancel)
                    .await
            }
            RoutesTo::Repo => {
                self.execute_repo(&profile_name, &profile, &task, run_id, &manifest, &cancel)
                    .await
            }
        };
        match result {
            Ok(detail) => Ok(detail),
            Err(err) => {
                let message = err.to_string();
                tracing::error!(run_id, error = %message, "rollout.failed");
                let trajectory = Trajectory {
                    steps: vec![],
                    errors: vec![message.clone()],
                };
                self.finish(
                    run_id,
                    manifest,
                    task,
                    trajectory,
                    None,
                    Ending::Failed,
                    Some(message),
                )
            }
        }
    }

    fn reject_if_over_budget(
        &self,
        task: &TaskSpec,
        manifest: &mut RunManifest,
    ) -> Result<Option<RunDetail>, CoordinatorError> {
        let cap = self.settings.daily_budget_usd;
        if cap <= 0.0 {
            return Ok(None);
        }
        let window = self.settings.budget_window_hours;
        let since = utc_now() - chrono::Duration::milliseconds((window * 3_600_000.0) as i64);
        let spent = self.store.total_cost_since(since)?;
        if spent < cap {
            return Ok(None);
        }
        let error = format!(
            "daily budget exceeded: ${spent:.4} spent in the last {window}h >= cap ${cap:.4}"
        );
        self.bus.run(
            &manifest.id,
            DomainEvent::BudgetExceeded {
                spent_usd: spent,
                cap_usd: cap,
                window_hours: window,
            },
        );
        tracing::warn!(spent, cap, "rollout.rejected.budget");
        let trajectory = Trajectory {
            steps: vec![],
            errors: vec![error.clone()],
        };
        let run_id = manifest.id.clone();
        self.finish(
            &run_id,
            manifest.clone(),
            task.clone(),
            trajectory,
            None,
            Ending::Failed,
            Some(error),
        )
        .map(Some)
    }

    /// Persist the terminal state and emit the matching event.
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        run_id: &str,
        mut manifest: RunManifest,
        task: TaskSpec,
        trajectory: Trajectory,
        reward: Option<RewardRecord>,
        ending: Ending,
        error: Option<String>,
    ) -> Result<RunDetail, CoordinatorError> {
        manifest.status = match ending {
            Ending::Completed => RunStatus::Completed,
            Ending::Cancelled => RunStatus::Cancelled,
            Ending::Failed => RunStatus::Failed,
        };
        manifest.updated_at = utc_now();
        manifest.finished_at = Some(manifest.updated_at);
        manifest.step_count = trajectory.steps.len() as u32;
        manifest.terminal_reward = reward.as_ref().map(|r| r.terminal_reward);
        manifest.error = error.clone();
        let detail = RunDetail {
            manifest: manifest.clone(),
            task,
            trajectory,
            reward,
        };
        self.store.save_run(&detail)?;
        match ending {
            Ending::Completed => {
                self.bus
                    .run(run_id, DomainEvent::RolloutCompleted { manifest });
            }
            Ending::Cancelled => {
                self.bus
                    .run(run_id, DomainEvent::RolloutCancelled { manifest });
            }
            Ending::Failed => {
                let msg = error.unwrap_or_else(|| "rollout failed".into());
                self.bus.run(
                    run_id,
                    DomainEvent::RolloutFailed {
                        manifest,
                        error: msg,
                    },
                );
            }
        }
        Ok(detail)
    }

    async fn client_for(&self, name: &str, profile: &PolicyProfile) -> OpenAiClient {
        let mut clients = self.clients.lock().await;
        clients
            .entry(name.to_string())
            .or_insert_with(|| {
                OpenAiClient::new(OpenAiConfig {
                    base_url: profile.base_url.clone(),
                    api_key: profile.resolve_api_key(),
                    timeout: Duration::from_secs(600),
                })
            })
            .clone()
    }

    async fn execute_repo(
        &self,
        profile_name: &str,
        profile: &PolicyProfile,
        task: &TaskSpec,
        run_id: &str,
        manifest: &RunManifest,
        cancel: &CancellationToken,
    ) -> Result<RunDetail, CoordinatorError> {
        let runner = self
            .repo_runner
            .clone()
            .ok_or_else(|| CoordinatorError::NoRepoRunner(profile_name.to_string()))?;
        let client = self.client_for(profile_name, profile).await;
        let record_turns = self.settings.training_recorder == "tokenized";
        let params = RolloutParams {
            model: profile.model.clone(),
            task: task.clone(),
            horizon: task.horizon,
            max_tokens_per_run: self.settings.max_tokens_per_run,
            max_output_tokens: profile.max_output_tokens,
            max_retries: profile.max_retries,
            system_prompt: None,
            extra_body: profile.extra_body.clone(),
            record_turns,
        };
        let (tx, mut rx) = mpsc::channel::<RolloutEvent>(256);
        let store = self.store.clone();
        let bus = self.bus.clone();
        let bridge = self.bridge.clone();
        let tokenizer = self.settings.recorder_tokenizer.clone();
        let model_name = profile.model.clone();
        let rid = run_id.to_string();
        let forwarder = tokio::spawn(async move {
            while let Some(ev) = rx.recv().await {
                match ev {
                    RolloutEvent::Step(step) => {
                        if let Err(e) = store.append_step(&rid, &step) {
                            tracing::warn!(run_id = %rid, error = %e, "step.persist_failed");
                        }
                        bus.run(&rid, DomainEvent::StepRecorded { step });
                    }
                    RolloutEvent::Progress {
                        turn,
                        tokens,
                        cost_usd,
                        tool,
                    } => {
                        bus.run(
                            &rid,
                            DomainEvent::ProgressTicked {
                                turn,
                                tool,
                                tokens,
                                cost_usd,
                            },
                        );
                    }
                    RolloutEvent::Turn {
                        step_index,
                        prompt_snapshot,
                        completion,
                        sampling_args,
                    } => {
                        record_turn(
                            &bridge,
                            &store,
                            &tokenizer,
                            &model_name,
                            &rid,
                            step_index,
                            prompt_snapshot,
                            completion,
                            sampling_args,
                        )
                        .await;
                    }
                }
            }
        });
        let probe_store = self.store.clone();
        let probe_id = run_id.to_string();
        let probe: CancelProbe =
            Arc::new(move || probe_store.is_cancel_requested(&probe_id).unwrap_or(false));

        let outcome =
            run_repo_rollout(&client, &runner, params, tx, cancel.clone(), Some(probe)).await;
        let _ = forwarder.await;
        let outcome = outcome?;

        let reward = rewards::score(task, &outcome.trajectory, &self.rubric);
        self.bus.run(
            run_id,
            DomainEvent::RewardComputed {
                reward: reward.clone(),
            },
        );
        let mut manifest = manifest.clone();
        manifest.tokens = outcome.total_tokens();
        manifest.cost_usd = outcome.cost_usd;
        let steps = outcome.trajectory.steps.len();
        if outcome.cancelled {
            tracing::info!(run_id, steps, "rollout.cancelled");
            self.finish(
                run_id,
                manifest,
                task.clone(),
                outcome.trajectory,
                Some(reward),
                Ending::Cancelled,
                None,
            )
        } else if outcome.token_overflow {
            let error = format!(
                "token budget exceeded: {} > {}",
                outcome.total_tokens(),
                self.settings.max_tokens_per_run
            );
            self.finish(
                run_id,
                manifest,
                task.clone(),
                outcome.trajectory,
                Some(reward),
                Ending::Failed,
                Some(error),
            )
        } else {
            tracing::info!(
                run_id,
                terminal_reward = reward.terminal_reward,
                steps,
                "rollout.completed"
            );
            self.finish(
                run_id,
                manifest,
                task.clone(),
                outcome.trajectory,
                Some(reward),
                Ending::Completed,
                None,
            )
        }
    }

    async fn verifiers_limit(&self, name: &str, max: usize) -> Arc<Semaphore> {
        self.verifiers_limits
            .lock()
            .await
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(Semaphore::new(max.max(1))))
            .clone()
    }

    async fn execute_verifiers(
        &self,
        profile_name: &str,
        profile: &PolicyProfile,
        task: &TaskSpec,
        run_id: &str,
        manifest: &RunManifest,
        cancel: &CancellationToken,
    ) -> Result<RunDetail, CoordinatorError> {
        let cancelled_before_start = || {
            let trajectory = Trajectory {
                steps: vec![],
                errors: vec![],
            };
            self.finish(
                run_id,
                manifest.clone(),
                task.clone(),
                trajectory,
                None,
                Ending::Cancelled,
                None,
            )
        };
        if cancel.is_cancelled() || self.store.is_cancel_requested(run_id)? {
            return cancelled_before_start();
        }
        let limit = self
            .verifiers_limit(profile_name, profile.max_concurrent)
            .await;
        let _permit = limit
            .acquire()
            .await
            .map_err(|e| CoordinatorError::Other(e.to_string()))?;
        let served_model = profile
            .model
            .split_once(':')
            .map(|(_, m)| m)
            .unwrap_or(&profile.model)
            .to_string();
        let params = json!({
            "run_id": run_id,
            "task": task,
            "model": served_model,
            "base_url": profile.base_url,
            "api_key": profile.resolve_api_key(),
            "env_id": profile.env_id,
            "env_args": profile.env_args,
            "sampling_args": {"max_tokens": profile.max_output_tokens},
            "timeout_s": profile.rollout_timeout_s,
        });
        let timeout = profile
            .rollout_timeout_s
            .map(|s| Duration::from_secs_f64(s + 30.0));
        let call = self
            .bridge
            .call("verifiers.rollout", params, timeout, |_, _| {});
        let result = tokio::select! {
            biased;
            _ = cancel.cancelled() => return cancelled_before_start(),
            r = call => r?,
        };
        let outcome: VerifiersOutcome = serde_json::from_value(result)
            .map_err(|e| CoordinatorError::Other(format!("bridge result: {e}")))?;

        let steps: Vec<TrajectoryStep> = outcome
            .steps
            .into_iter()
            .map(|s| TrajectoryStep::new(s.index, &s.actor, &s.kind, s.content))
            .collect();
        for step in &steps {
            self.store.append_step(run_id, step)?;
            self.bus
                .run(run_id, DomainEvent::StepRecorded { step: step.clone() });
        }
        let usage = TokenUsage {
            input_tokens: outcome.input_tokens,
            output_tokens: outcome.output_tokens,
            ..Default::default()
        };
        let total_tokens = usage.total();
        let cost_usd = estimate_cost_usd(&profile.model, usage);
        self.bus.run(
            run_id,
            DomainEvent::ProgressTicked {
                turn: steps.len().saturating_sub(1) as u32,
                tool: Some("verifiers".into()),
                tokens: total_tokens,
                cost_usd,
            },
        );

        let token_overflow = total_tokens > self.settings.max_tokens_per_run;
        let trajectory = Trajectory {
            steps,
            errors: outcome.errors,
        };
        let reward = RewardRecord {
            terminal_reward: round_to(outcome.terminal_reward.clamp(-1.0, 1.0), 4),
            rubric: format!("verifiers-{}", profile.env_id),
            source: "verifiers".into(),
            signals: outcome
                .signals
                .into_iter()
                .map(|s| RewardSignal {
                    name: s.name,
                    value: s.value,
                    weight: s.weight,
                    reason: s.reason,
                })
                .collect(),
            audit_flags: vec![],
        };
        self.bus.run(
            run_id,
            DomainEvent::RewardComputed {
                reward: reward.clone(),
            },
        );
        let mut manifest = manifest.clone();
        manifest.tokens = total_tokens;
        manifest.cost_usd = cost_usd;
        tracing::info!(
            run_id,
            terminal_reward = reward.terminal_reward,
            steps = trajectory.steps.len(),
            tokens = total_tokens,
            cost_usd,
            "rollout.completed.verifiers"
        );
        if token_overflow {
            let error = format!(
                "token budget exceeded: {total_tokens} > {}",
                self.settings.max_tokens_per_run
            );
            self.finish(
                run_id,
                manifest,
                task.clone(),
                trajectory,
                Some(reward),
                Ending::Failed,
                Some(error),
            )
        } else {
            self.finish(
                run_id,
                manifest,
                task.clone(),
                trajectory,
                Some(reward),
                Ending::Completed,
                None,
            )
        }
    }
}

#[derive(Debug, Deserialize)]
struct VerifiersStep {
    index: u32,
    actor: String,
    kind: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct VerifiersSignal {
    name: String,
    value: f64,
    #[serde(default = "one")]
    weight: f64,
    #[serde(default)]
    reason: String,
}

fn one() -> f64 {
    1.0
}

#[derive(Debug, Deserialize)]
struct VerifiersOutcome {
    #[serde(default)]
    steps: Vec<VerifiersStep>,
    #[serde(default)]
    errors: Vec<String>,
    #[serde(default)]
    terminal_reward: f64,
    #[serde(default)]
    signals: Vec<VerifiersSignal>,
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
}

#[allow(clippy::too_many_arguments)]
async fn record_turn(
    bridge: &PythonBridge,
    store: &Store,
    tokenizer: &str,
    model_name: &str,
    run_id: &str,
    step_index: u32,
    prompt: String,
    completion: String,
    sampling_args: Map<String, Value>,
) {
    let params = json!({"tokenizer": tokenizer, "prompt": prompt, "completion": completion});
    match bridge
        .call(
            "tokenize",
            params,
            Some(Duration::from_secs(120)),
            |_, _| {},
        )
        .await
    {
        Ok(result) => {
            let ids = |k: &str| -> Vec<i32> {
                result
                    .get(k)
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_i64().map(|i| i as i32))
                            .collect()
                    })
                    .unwrap_or_default()
            };
            let prompt_ids = ids("prompt_ids");
            let record = TurnTrainingRecord {
                id: short_id("turn"),
                run_id: run_id.into(),
                step_index,
                token_count: prompt_ids.len() as u32,
                prompt_ids,
                completion_ids: ids("completion_ids"),
                attention_mask: ids("attention_mask"),
                loss_mask: ids("loss_mask"),
                sampling_args,
                model_name: model_name.into(),
                created_at: utc_now(),
            };
            if let Err(e) = store.save_turn_training(&record) {
                tracing::warn!(run_id, error = %e, "recorder.save_failed");
            }
        }
        Err(e) => tracing::warn!(run_id, error = %e, "recorder.tokenize_failed"),
    }
}

/// Rescore a stored run against a registered rubric. Pure over
/// (task, trajectory); never re-executes anything.
pub struct RescoreResult {
    pub run_id: String,
    pub rubric: String,
    pub previous: Option<RewardRecord>,
    pub new: RewardRecord,
}

impl RescoreResult {
    pub fn delta(&self) -> Option<f64> {
        self.previous
            .as_ref()
            .map(|p| round_to(self.new.terminal_reward - p.terminal_reward, 6))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RescoreError {
    #[error("run not found: {0}")]
    RunNotFound(String),
    #[error("unknown rubric: {0}. Available: {1:?}")]
    UnknownRubric(String, Vec<String>),
    #[error("{0}")]
    Store(#[from] horizon_store::StoreError),
}

pub fn rescore_run(
    store: &Store,
    run_id: &str,
    rubric: &str,
    persist: bool,
) -> Result<RescoreResult, RescoreError> {
    let detail = store
        .get_run(run_id)?
        .ok_or_else(|| RescoreError::RunNotFound(run_id.into()))?;
    let spec = rewards::get_rubric(rubric)
        .ok_or_else(|| RescoreError::UnknownRubric(rubric.into(), rewards::rubric_names()))?;
    let new = rewards::score(&detail.task, &detail.trajectory, &spec);
    if persist {
        store.save_reward(run_id, &new)?;
    }
    Ok(RescoreResult {
        run_id: run_id.into(),
        rubric: spec.provenance(),
        previous: detail.reward,
        new,
    })
}
