//! Rollout coordinator: resolves profiles and horizons, enforces the
//! budget, drives one of two execution paths, persists the result, and
//! emits the lifecycle events the dashboard consumes.
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
    ArtifactRecord, RewardPenalty, RewardRecord, RolloutRequest, RunDetail, RunManifest, RunStatus,
    TaskSpec, ToolPermission, TrajectoryRecord, TrajectoryStep, TurnTrainingRecord, WorkerRecord,
    WorkerStatus,
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

    /// Validate the request and enqueue a rollout job. Returns the run id
    /// so the caller can cancel before the first event lands.
    pub fn submit(&self, request: RolloutRequest) -> Result<String, CoordinatorError> {
        self.resolve_profile(&request)?;
        let run_id = short_id("run");
        let payload = serde_json::to_value(RolloutJob {
            run_id: run_id.clone(),
            request,
        })
        .map_err(|e| CoordinatorError::Other(e.to_string()))?;
        self.store
            .enqueue_job(&run_id, horizon_core::models::JobKind::Rollout, &payload)?;
        Ok(run_id)
    }

    /// Flag a run for cancellation: the job row (visible to every
    /// replica) and the in-process token (interrupts generation now).
    pub async fn request_cancel(&self, run_id: &str) -> bool {
        if let Ok(Some(existing)) = self.store.get_run(run_id) {
            if existing.manifest.status.is_terminal() {
                return false;
            }
        }
        let accepted = self.store.request_cancel(run_id).unwrap_or(true);
        if let Some(token) = self.cancels.lock().await.get(run_id) {
            token.cancel();
        }
        accepted
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
    /// harness. Never returns Err for policy or environment failures:
    /// those become a failed `RunDetail` plus `rollout.failed`.
    pub async fn execute(
        &self,
        request: &RolloutRequest,
        run_id: &str,
        cancel: CancellationToken,
    ) -> Result<RunDetail, CoordinatorError> {
        if let Some(detail) = self.reject_if_over_budget(request, run_id)? {
            return Ok(detail);
        }
        let (profile_name, profile) = self.resolve_profile(request)?;
        let profile = profile.clone();
        let now = utc_now();
        let task = TaskSpec {
            id: short_id("task"),
            prompt: request.prompt.clone(),
            repo_snapshot: request.repo_snapshot.clone(),
            tool_permissions: vec![
                ToolPermission::Read,
                ToolPermission::Search,
                ToolPermission::Terminal,
            ],
            horizon: self.resolve_horizon(request),
            success_criteria: if request.success_criteria.is_empty() {
                vec!["manual review".into()]
            } else {
                request.success_criteria.clone()
            },
        };
        let manifest = RunManifest {
            id: run_id.to_string(),
            model_id: self.model_id(&profile),
            adapter_id: request.adapter_id.clone(),
            dataset_slice: "bootstrap".into(),
            infra_target: request.infra_target.clone(),
            seed: 7,
            status: RunStatus::Running,
            created_at: now,
            updated_at: now,
            estimated_cost_usd: if request.infra_target == "mac-local" {
                0.03
            } else {
                0.72
            },
        };
        // Persist the running row first so a crash leaves a visible,
        // recoverable run instead of nothing.
        let running = RunDetail {
            manifest: manifest.clone(),
            task: task.clone(),
            trajectory: TrajectoryRecord::new(&task.id, vec![], vec![]),
            reward: rewards::score(
                &task,
                &TrajectoryRecord::new(&task.id, vec![], vec![]),
                &self.rubric,
            ),
            artifacts: default_artifacts(run_id),
        };
        self.store.save_run(&running)?;
        self.bus
            .publish(DomainEvent::rollout_started(run_id, manifest.clone()));
        self.publish_worker(
            "worker-rollout-local",
            "rollout",
            WorkerStatus::Running,
            Some(run_id),
            "Local rollout worker is active.",
        )?;

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
                let trajectory = TrajectoryRecord::new(&task.id, vec![], vec![message.clone()]);
                let reward = rewards::score(&task, &trajectory, &self.rubric);
                let detail = RunDetail {
                    manifest: RunManifest {
                        status: RunStatus::Failed,
                        updated_at: utc_now(),
                        ..manifest
                    },
                    task,
                    trajectory,
                    reward,
                    artifacts: default_artifacts(run_id),
                };
                self.store.save_run(&detail)?;
                self.publish_worker(
                    "worker-rollout-local",
                    "rollout",
                    WorkerStatus::Failed,
                    Some(run_id),
                    &format!("Rollout failed: {message}"),
                )?;
                self.bus
                    .publish(DomainEvent::rollout_failed(run_id, message));
                Ok(detail)
            }
        }
    }

    fn reject_if_over_budget(
        &self,
        request: &RolloutRequest,
        run_id: &str,
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
        let now = utc_now();
        let model_id = self
            .resolve_profile(request)
            .map(|(_, p)| self.model_id(p))
            .unwrap_or_else(|_| self.policy_name());
        let task = TaskSpec {
            id: short_id("task"),
            prompt: request.prompt.clone(),
            repo_snapshot: request.repo_snapshot.clone(),
            tool_permissions: vec![ToolPermission::Read],
            horizon: self.resolve_horizon(request),
            success_criteria: if request.success_criteria.is_empty() {
                vec!["(budget-blocked)".into()]
            } else {
                request.success_criteria.clone()
            },
        };
        let trajectory = TrajectoryRecord::new(&task.id, vec![], vec![error.clone()]);
        let reward = rewards::score(&task, &trajectory, &self.rubric);
        let detail = RunDetail {
            manifest: RunManifest {
                id: run_id.into(),
                model_id,
                adapter_id: request.adapter_id.clone(),
                dataset_slice: "bootstrap".into(),
                infra_target: request.infra_target.clone(),
                seed: 7,
                status: RunStatus::Failed,
                created_at: now,
                updated_at: now,
                estimated_cost_usd: 0.0,
            },
            task,
            trajectory,
            reward,
            artifacts: default_artifacts(run_id),
        };
        self.store.save_run(&detail)?;
        self.bus
            .publish(DomainEvent::budget_exceeded(run_id, spent, cap, window));
        self.bus.publish(DomainEvent::rollout_failed(run_id, error));
        tracing::warn!(spent, cap, "rollout.rejected.budget");
        Ok(Some(detail))
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
        let run_id_owned = run_id.to_string();
        let forwarder = tokio::spawn(async move {
            while let Some(ev) = rx.recv().await {
                match ev {
                    RolloutEvent::Step(step) => {
                        bus.publish(DomainEvent::step_recorded(&run_id_owned, step));
                    }
                    RolloutEvent::Progress {
                        turn,
                        tokens,
                        cost_usd,
                        tool,
                    } => {
                        bus.publish(DomainEvent::progress_ticked(
                            &run_id_owned,
                            turn,
                            tool,
                            tokens,
                            cost_usd,
                        ));
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
                            &run_id_owned,
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
        self.bus
            .publish(DomainEvent::reward_computed(run_id, &reward));
        let failed = outcome.cancelled || outcome.token_overflow;
        let detail = RunDetail {
            manifest: RunManifest {
                status: if failed {
                    RunStatus::Failed
                } else {
                    RunStatus::Completed
                },
                updated_at: utc_now(),
                estimated_cost_usd: if outcome.cost_usd > 0.0 {
                    outcome.cost_usd
                } else {
                    manifest.estimated_cost_usd
                },
                ..manifest.clone()
            },
            task: task.clone(),
            trajectory: outcome.trajectory.clone(),
            reward: reward.clone(),
            artifacts: default_artifacts(run_id),
        };
        self.store.save_run(&detail)?;
        let steps = outcome.trajectory.steps.len();
        if outcome.cancelled {
            self.publish_worker(
                "worker-rollout-local",
                "rollout",
                WorkerStatus::Failed,
                Some(run_id),
                "Rollout cancelled.",
            )?;
            self.bus.publish(DomainEvent::rollout_cancelled(run_id));
            tracing::info!(run_id, steps, "rollout.cancelled");
        } else if outcome.token_overflow {
            let tokens = outcome.total_tokens();
            self.publish_worker(
                "worker-rollout-local",
                "rollout",
                WorkerStatus::Failed,
                Some(run_id),
                &format!("repo rollout exceeded token budget ({tokens})."),
            )?;
            self.bus.publish(DomainEvent::rollout_failed(
                run_id,
                format!("token budget exceeded: {tokens}"),
            ));
        } else {
            self.publish_worker(
                "worker-rollout-local",
                "rollout",
                WorkerStatus::Idle,
                Some(run_id),
                "Rollout finished; worker idle.",
            )?;
            self.publish_worker(
                "worker-reward-local",
                "reward",
                WorkerStatus::Idle,
                Some(run_id),
                "Reward pipeline is available for replay.",
            )?;
            self.bus
                .publish(DomainEvent::rollout_completed(run_id, detail.clone()));
            tracing::info!(
                run_id,
                terminal_reward = reward.terminal_reward,
                steps,
                "rollout.completed"
            );
        }
        Ok(detail)
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
        if cancel.is_cancelled() || self.store.is_cancel_requested(run_id)? {
            return self.finish_cancelled_before_start(task, run_id, manifest);
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
            "sampling_args": {"max_tokens": 2048},
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
            _ = cancel.cancelled() => return self.finish_cancelled_before_start(task, run_id, manifest),
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
            self.bus
                .publish(DomainEvent::step_recorded(run_id, step.clone()));
        }
        let usage = TokenUsage {
            input_tokens: outcome.input_tokens,
            output_tokens: outcome.output_tokens,
            ..Default::default()
        };
        let total_tokens = usage.total();
        let cost_usd = estimate_cost_usd(&profile.model, usage);
        self.bus.publish(DomainEvent::progress_ticked(
            run_id,
            steps.len().saturating_sub(1) as u32,
            Some("verifiers".into()),
            total_tokens,
            cost_usd,
        ));

        let token_overflow = total_tokens > self.settings.max_tokens_per_run;
        let mut errors = outcome.errors;
        if token_overflow {
            errors.push(format!(
                "token budget exceeded: {total_tokens} > {}",
                self.settings.max_tokens_per_run
            ));
        }
        let trajectory = TrajectoryRecord::new(&task.id, steps, errors);
        let reward = build_verifiers_reward(
            &trajectory.id,
            outcome.terminal_reward,
            &outcome.signals,
            &profile.env_id,
        );
        self.bus
            .publish(DomainEvent::reward_computed(run_id, &reward));
        let detail = RunDetail {
            manifest: RunManifest {
                status: if token_overflow {
                    RunStatus::Failed
                } else {
                    RunStatus::Completed
                },
                updated_at: utc_now(),
                estimated_cost_usd: if cost_usd > 0.0 {
                    cost_usd
                } else {
                    manifest.estimated_cost_usd
                },
                ..manifest.clone()
            },
            task: task.clone(),
            trajectory,
            reward: reward.clone(),
            artifacts: default_artifacts(run_id),
        };
        self.store.save_run(&detail)?;
        if token_overflow {
            self.publish_worker(
                "worker-rollout-local",
                "rollout",
                WorkerStatus::Failed,
                Some(run_id),
                &format!("verifiers rollout exceeded token budget ({total_tokens})."),
            )?;
            self.bus.publish(DomainEvent::rollout_failed(
                run_id,
                format!("token budget exceeded: {total_tokens}"),
            ));
        } else {
            self.publish_worker(
                "worker-rollout-local",
                "rollout",
                WorkerStatus::Idle,
                Some(run_id),
                "verifiers rollout finished; worker idle.",
            )?;
            self.publish_worker(
                "worker-reward-local",
                "reward",
                WorkerStatus::Idle,
                Some(run_id),
                "Reward pipeline is available for replay.",
            )?;
            self.bus
                .publish(DomainEvent::rollout_completed(run_id, detail.clone()));
        }
        tracing::info!(
            run_id,
            terminal_reward = reward.terminal_reward,
            steps = detail.trajectory.steps.len(),
            tokens = total_tokens,
            cost_usd,
            "rollout.completed.verifiers"
        );
        Ok(detail)
    }

    fn finish_cancelled_before_start(
        &self,
        task: &TaskSpec,
        run_id: &str,
        manifest: &RunManifest,
    ) -> Result<RunDetail, CoordinatorError> {
        let trajectory = TrajectoryRecord::new(&task.id, vec![], vec!["cancelled".into()]);
        let reward = rewards::score(task, &trajectory, &self.rubric);
        let detail = RunDetail {
            manifest: RunManifest {
                status: RunStatus::Failed,
                updated_at: utc_now(),
                ..manifest.clone()
            },
            task: task.clone(),
            trajectory,
            reward,
            artifacts: default_artifacts(run_id),
        };
        self.store.save_run(&detail)?;
        self.bus.publish(DomainEvent::rollout_cancelled(run_id));
        self.publish_worker(
            "worker-rollout-local",
            "rollout",
            WorkerStatus::Failed,
            Some(run_id),
            "Rollout cancelled before verifiers rollout started.",
        )?;
        Ok(detail)
    }

    fn publish_worker(
        &self,
        id: &str,
        role: &str,
        status: WorkerStatus,
        run_id: Option<&str>,
        detail: &str,
    ) -> Result<(), CoordinatorError> {
        let worker = WorkerRecord {
            id: id.into(),
            role: role.into(),
            status,
            run_id: run_id.map(str::to_string),
            detail: detail.into(),
        };
        self.store.upsert_worker(&worker)?;
        self.bus
            .publish(DomainEvent::worker_updated(run_id, worker));
        Ok(())
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

fn build_verifiers_reward(
    trajectory_id: &str,
    terminal: f64,
    signals: &[VerifiersSignal],
    env_id: &str,
) -> RewardRecord {
    let payload = json!({
        "source": "verifiers-rubric",
        "rubric": format!("verifiers-{env_id}"),
        "signals": signals.iter().map(|s| json!({"name": s.name, "value": s.value, "weight": s.weight, "reason": s.reason})).collect::<Vec<_>>(),
    });
    RewardRecord {
        trajectory_id: trajectory_id.into(),
        terminal_reward: round_to(terminal.clamp(-1.0, 1.0), 4),
        step_rewards: vec![],
        penalties: signals
            .iter()
            .filter(|s| s.value < 0.0)
            .map(|s| RewardPenalty {
                code: s.name.clone(),
                value: s.value * s.weight,
                reason: s.reason.clone(),
            })
            .collect(),
        audit_flags: vec![],
        provenance: payload.to_string(),
    }
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
            let attention_mask = ids("attention_mask");
            let loss_mask = ids("loss_mask");
            let record = TurnTrainingRecord {
                id: short_id("turn"),
                run_id: run_id.into(),
                step_index,
                token_count: prompt_ids.len() as u32,
                prompt_ids,
                completion_ids: ids("completion_ids"),
                attention_mask,
                loss_mask,
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

pub fn default_artifacts(run_id: &str) -> Vec<ArtifactRecord> {
    ["manifest", "trajectory", "reward"]
        .iter()
        .map(|kind| ArtifactRecord {
            name: format!("{run_id}-{kind}.json"),
            kind: kind.to_string(),
            path: format!("artifacts/{run_id}/{kind}.json"),
        })
        .collect()
}

/// Rescore a stored run against a registered rubric. Pure over
/// (task, trajectory); never re-executes anything.
pub struct RescoreResult {
    pub run_id: String,
    pub rubric: String,
    pub previous: RewardRecord,
    pub new: RewardRecord,
}

impl RescoreResult {
    pub fn delta(&self) -> f64 {
        round_to(self.new.terminal_reward - self.previous.terminal_reward, 6)
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
        store.save_run(&RunDetail {
            reward: new.clone(),
            ..detail.clone()
        })?;
    }
    Ok(RescoreResult {
        run_id: run_id.into(),
        rubric: spec.provenance(),
        previous: detail.reward,
        new,
    })
}
