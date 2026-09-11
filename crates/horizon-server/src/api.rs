//! HTTP + WebSocket API. `/openapi.json` is generated from the Rust
//! types; the frontend regenerates its bindings from it.

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tower_http::cors::{AllowOrigin, CorsLayer};
use utoipa::{OpenApi, ToSchema};

use horizon_core::events::{DomainEvent, EventEnvelope};
use horizon_core::models::{
    AdapterRecord, DashboardSnapshot, EvalReport, EvalTask, Hyperparams, JobRecord, RewardRecord,
    RolloutRequest, RunDetail, RunManifest, TrainingRunRecord, TurnTrainingRecord,
};
use horizon_core::rewards;

use crate::app::AppState;
use crate::coordinator::{rescore_run, CoordinatorError, RescoreError};
use crate::eval::EvalError;
use crate::settings::RoutesTo;
use crate::training::TrainingRequest;

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorBody {
    pub detail: String,
}

pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(ErrorBody { detail: self.1 })).into_response()
    }
}

impl From<horizon_store::StoreError> for ApiError {
    fn from(e: horizon_store::StoreError) -> Self {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    }
}

impl From<CoordinatorError> for ApiError {
    fn from(e: CoordinatorError) -> Self {
        match e {
            CoordinatorError::UnknownProfile(..) => {
                ApiError(StatusCode::UNPROCESSABLE_ENTITY, e.to_string())
            }
            other => ApiError(StatusCode::INTERNAL_SERVER_ERROR, other.to_string()),
        }
    }
}

fn not_found(msg: &str) -> ApiError {
    ApiError(StatusCode::NOT_FOUND, msg.into())
}

#[derive(Debug, Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
}

#[utoipa::path(get, path = "/health", responses((status = 200, body = HealthResponse)))]
async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
    })
}

#[utoipa::path(get, path = "/api/dashboard", responses((status = 200, body = DashboardSnapshot)))]
async fn dashboard(State(state): State<AppState>) -> Result<Json<DashboardSnapshot>, ApiError> {
    Ok(Json(state.store.dashboard()?))
}

#[utoipa::path(get, path = "/api/runs", responses((status = 200, body = Vec<RunManifest>)))]
async fn list_runs(State(state): State<AppState>) -> Result<Json<Vec<RunManifest>>, ApiError> {
    Ok(Json(state.store.list_runs()?))
}

#[utoipa::path(get, path = "/api/runs/{run_id}", params(("run_id" = String, Path)), responses((status = 200, body = RunDetail), (status = 404, body = ErrorBody)))]
async fn get_run(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<RunDetail>, ApiError> {
    state
        .store
        .get_run(&run_id)?
        .map(Json)
        .ok_or_else(|| not_found("Run not found"))
}

#[derive(Debug, Serialize, ToSchema)]
pub struct TurnIndexEntry {
    pub step_index: u32,
    pub model_name: String,
    pub token_count: u32,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[utoipa::path(get, path = "/api/runs/{run_id}/turns", params(("run_id" = String, Path)), responses((status = 200, body = Vec<TurnIndexEntry>), (status = 404, body = ErrorBody)))]
async fn list_run_turns(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<Vec<TurnIndexEntry>>, ApiError> {
    if state.store.get_manifest(&run_id)?.is_none() {
        return Err(not_found("Run not found"));
    }
    let rows = state.store.list_turn_training(&run_id)?;
    Ok(Json(
        rows.into_iter()
            .map(|r| TurnIndexEntry {
                step_index: r.step_index,
                model_name: r.model_name,
                token_count: r.token_count,
                created_at: r.created_at,
            })
            .collect(),
    ))
}

#[utoipa::path(get, path = "/api/runs/{run_id}/turns/{step_index}/training", params(("run_id" = String, Path), ("step_index" = u32, Path)), responses((status = 200, body = TurnTrainingRecord), (status = 404, body = ErrorBody)))]
async fn get_run_turn_training(
    State(state): State<AppState>,
    Path((run_id, step_index)): Path<(String, u32)>,
) -> Result<Json<TurnTrainingRecord>, ApiError> {
    state
        .store
        .get_turn_training(&run_id, step_index)?
        .map(Json)
        .ok_or_else(|| not_found("No training metadata for this turn"))
}

#[derive(Debug, Serialize, ToSchema)]
pub struct CreateRunResponse {
    pub run_id: String,
}

#[utoipa::path(post, path = "/api/runs", request_body = RolloutRequest, responses((status = 202, body = CreateRunResponse), (status = 422, body = ErrorBody)))]
async fn create_run(
    State(state): State<AppState>,
    Json(request): Json<RolloutRequest>,
) -> Result<(StatusCode, Json<CreateRunResponse>), ApiError> {
    let run_id = state.coordinator.submit(request)?;
    state.jobs_notify.notify_one();
    Ok((StatusCode::ACCEPTED, Json(CreateRunResponse { run_id })))
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ProfileInfo {
    pub name: String,
    pub model: String,
    pub base_url: String,
    pub routes_to: String,
    pub env_id: Option<String>,
    pub api_key_env: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ProfilesResponse {
    pub default: String,
    pub profiles: Vec<ProfileInfo>,
}

#[utoipa::path(get, path = "/api/profiles", responses((status = 200, body = ProfilesResponse)))]
async fn list_profiles(State(state): State<AppState>) -> Json<ProfilesResponse> {
    let coord = &state.coordinator;
    Json(ProfilesResponse {
        default: coord.default_profile.clone(),
        profiles: coord
            .profiles
            .iter()
            .map(|(name, p)| ProfileInfo {
                name: name.clone(),
                model: p.model.clone(),
                base_url: p.base_url.clone(),
                routes_to: p.routes_to.as_str().into(),
                env_id: if p.routes_to == RoutesTo::Verifiers {
                    Some(p.env_id.clone())
                } else {
                    None
                },
                api_key_env: p.api_key_env.clone(),
            })
            .collect(),
    })
}

#[derive(Debug, Serialize, ToSchema)]
pub struct CancelResponse {
    pub run_id: String,
}

#[utoipa::path(post, path = "/api/runs/{run_id}/cancel", params(("run_id" = String, Path)), responses((status = 202, body = CancelResponse), (status = 409, body = ErrorBody)))]
async fn cancel_run(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<(StatusCode, Json<CancelResponse>), ApiError> {
    if !state.coordinator.request_cancel(&run_id).await? {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "Run is already terminal or unknown".into(),
        ));
    }
    Ok((StatusCode::ACCEPTED, Json(CancelResponse { run_id })))
}

#[derive(Debug, Serialize, ToSchema)]
pub struct RuntimeConfig {
    pub env_backend: String,
    pub policy_name: String,
    pub trainer_backend: String,
    pub verifiers_env_id: Option<String>,
    pub default_profile: String,
    pub profile_names: Vec<String>,
    pub store_backend: String,
    pub sandbox: String,
    pub worker_id: String,
    pub max_parallel_rollouts: usize,
}

#[utoipa::path(get, path = "/api/config", responses((status = 200, body = RuntimeConfig)))]
async fn runtime_config(State(state): State<AppState>) -> Json<RuntimeConfig> {
    let coord = &state.coordinator;
    let default = coord.profiles.get(&coord.default_profile);
    Json(RuntimeConfig {
        env_backend: state.settings.env_backend.clone(),
        policy_name: coord.policy_name(),
        trainer_backend: state.settings.trainer_backend.clone(),
        verifiers_env_id: default
            .filter(|p| p.routes_to == RoutesTo::Verifiers)
            .map(|p| p.env_id.clone()),
        default_profile: coord.default_profile.clone(),
        profile_names: coord.profiles.keys().cloned().collect(),
        store_backend: state.settings.store_backend.clone(),
        sandbox: state.settings.env_sandbox.clone(),
        worker_id: state.settings.worker_id.clone(),
        max_parallel_rollouts: state.settings.max_parallel_rollouts,
    })
}

#[derive(Debug, Serialize, ToSchema)]
pub struct BudgetStatus {
    pub window_hours: f64,
    pub cap_usd: f64,
    pub spent_usd: f64,
    pub remaining_usd: Option<f64>,
    pub exceeded: bool,
}

#[utoipa::path(get, path = "/api/budget", responses((status = 200, body = BudgetStatus)))]
async fn budget_status(State(state): State<AppState>) -> Result<Json<BudgetStatus>, ApiError> {
    let window = state.settings.budget_window_hours;
    let since =
        horizon_core::utc_now() - chrono::Duration::milliseconds((window * 3_600_000.0) as i64);
    let spent = state.store.total_cost_since(since)?;
    let cap = state.settings.daily_budget_usd;
    Ok(Json(BudgetStatus {
        window_hours: window,
        cap_usd: cap,
        spent_usd: spent,
        remaining_usd: if cap > 0.0 {
            Some((cap - spent).max(0.0))
        } else {
            None
        },
        exceeded: cap > 0.0 && spent >= cap,
    }))
}

#[derive(Debug, Serialize, ToSchema)]
pub struct RubricInfo {
    pub name: String,
    pub signals: Vec<String>,
}

#[utoipa::path(get, path = "/api/rubrics", responses((status = 200, body = Vec<RubricInfo>)))]
async fn list_rubrics() -> Json<Vec<RubricInfo>> {
    let mut specs = rewards::rubrics();
    specs.sort_by_key(|r| r.provenance());
    Json(
        specs
            .into_iter()
            .map(|r| RubricInfo {
                name: r.provenance(),
                signals: r.signal_names().iter().map(|s| s.to_string()).collect(),
            })
            .collect(),
    )
}

#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct RescoreQuery {
    pub rubric: String,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct RescoreResponse {
    pub run_id: String,
    pub rubric: String,
    pub previous: Option<RewardRecord>,
    pub reward: RewardRecord,
    pub delta: Option<f64>,
    pub persisted: bool,
}

#[utoipa::path(post, path = "/api/runs/{run_id}/rescore", params(("run_id" = String, Path), RescoreQuery), responses((status = 200, body = RescoreResponse), (status = 400, body = ErrorBody), (status = 404, body = ErrorBody)))]
async fn rescore(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    Query(q): Query<RescoreQuery>,
) -> Result<Json<RescoreResponse>, ApiError> {
    let result =
        rescore_run(&state.store, &run_id, &q.rubric, !q.dry_run).map_err(|e| match e {
            RescoreError::RunNotFound(_) => ApiError(StatusCode::NOT_FOUND, e.to_string()),
            RescoreError::UnknownRubric(..) => ApiError(StatusCode::BAD_REQUEST, e.to_string()),
            RescoreError::Store(s) => ApiError(StatusCode::INTERNAL_SERVER_ERROR, s.to_string()),
        })?;
    if !q.dry_run {
        state.bus.run(
            &run_id,
            DomainEvent::RewardComputed {
                reward: result.new.clone(),
            },
        );
    }
    Ok(Json(RescoreResponse {
        run_id: result.run_id.clone(),
        rubric: result.rubric.clone(),
        delta: result.delta(),
        previous: result.previous,
        reward: result.new,
        persisted: !q.dry_run,
    }))
}

#[utoipa::path(get, path = "/api/adapters", responses((status = 200, body = Vec<AdapterRecord>)))]
async fn list_adapters(State(state): State<AppState>) -> Json<Vec<AdapterRecord>> {
    Json(state.adapters.list_adapters())
}

#[derive(Debug, Serialize, ToSchema)]
pub struct AdapterDetail {
    pub adapter: AdapterRecord,
    pub children: Vec<AdapterRecord>,
    pub eval_reports: Vec<EvalReport>,
}

#[utoipa::path(get, path = "/api/adapters/{adapter_id}", params(("adapter_id" = String, Path)), responses((status = 200, body = AdapterDetail), (status = 404, body = ErrorBody)))]
async fn get_adapter(
    State(state): State<AppState>,
    Path(adapter_id): Path<String>,
) -> Result<Json<AdapterDetail>, ApiError> {
    let adapter = state
        .adapters
        .get(&adapter_id)
        .ok_or_else(|| not_found("Adapter not found"))?;
    Ok(Json(AdapterDetail {
        children: state.adapters.children_of(Some(&adapter_id)),
        eval_reports: state.store.list_eval_reports(Some(&adapter_id))?,
        adapter,
    }))
}

#[derive(Debug, Default, Deserialize, ToSchema)]
pub struct RunEvalBody {
    #[serde(default)]
    pub tasks: Option<Vec<EvalTask>>,
    #[serde(default)]
    pub task_set: Option<String>,
}

#[utoipa::path(post, path = "/api/adapters/{adapter_id}/eval", params(("adapter_id" = String, Path)), request_body(content = Option<RunEvalBody>), responses((status = 200, body = EvalReport), (status = 404, body = ErrorBody)))]
async fn run_eval(
    State(state): State<AppState>,
    Path(adapter_id): Path<String>,
    body: Option<Json<RunEvalBody>>,
) -> Result<Json<EvalReport>, ApiError> {
    let body = body.map(|b| b.0).unwrap_or_default();
    let report = state
        .eval
        .run(&adapter_id, body.tasks, body.task_set)
        .await
        .map_err(|e| match e {
            EvalError::AdapterNotFound(_) | EvalError::EmptyTaskSet => {
                ApiError(StatusCode::NOT_FOUND, e.to_string())
            }
            other => ApiError(StatusCode::INTERNAL_SERVER_ERROR, other.to_string()),
        })?;
    Ok(Json(report))
}

#[utoipa::path(get, path = "/api/training-runs", responses((status = 200, body = Vec<TrainingRunRecord>)))]
async fn list_training_runs(
    State(state): State<AppState>,
) -> Result<Json<Vec<TrainingRunRecord>>, ApiError> {
    Ok(Json(state.store.list_training_runs()?))
}

#[utoipa::path(get, path = "/api/training-runs/{id}", params(("id" = String, Path)), responses((status = 200, body = TrainingRunRecord), (status = 404, body = ErrorBody)))]
async fn get_training_run(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<TrainingRunRecord>, ApiError> {
    state
        .store
        .get_training_run(&id)?
        .map(Json)
        .ok_or_else(|| not_found("Training run not found"))
}

#[derive(Debug, Default, Deserialize, ToSchema)]
pub struct CreateTrainingRunBody {
    #[serde(default)]
    pub sample_run_ids: Vec<String>,
    #[serde(default)]
    pub parent_adapter_id: Option<String>,
    #[serde(default)]
    pub hyperparams: Hyperparams,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct CreateTrainingRunResponse {
    pub training_run_id: String,
}

#[utoipa::path(post, path = "/api/training-runs", request_body = CreateTrainingRunBody, responses((status = 202, body = CreateTrainingRunResponse)))]
async fn create_training_run(
    State(state): State<AppState>,
    Json(body): Json<CreateTrainingRunBody>,
) -> Result<(StatusCode, Json<CreateTrainingRunResponse>), ApiError> {
    let record = state.training.start(TrainingRequest {
        sample_run_ids: body.sample_run_ids,
        parent_adapter_id: body.parent_adapter_id,
        hyperparams: body.hyperparams,
    })?;
    state.jobs_notify.notify_one();
    Ok((
        StatusCode::ACCEPTED,
        Json(CreateTrainingRunResponse {
            training_run_id: record.id,
        }),
    ))
}

#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct JobsQuery {
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    50
}

#[utoipa::path(get, path = "/api/jobs", params(JobsQuery), responses((status = 200, body = Vec<JobRecord>)))]
async fn list_jobs(
    State(state): State<AppState>,
    Query(q): Query<JobsQuery>,
) -> Result<Json<Vec<JobRecord>>, ApiError> {
    Ok(Json(state.store.list_jobs(q.limit.min(500))?))
}

#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct EventsQuery {
    #[serde(default)]
    pub since: i64,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

#[utoipa::path(get, path = "/api/events", params(EventsQuery), responses((status = 200, body = Vec<EventEnvelope>)))]
async fn list_events(
    State(state): State<AppState>,
    Query(q): Query<EventsQuery>,
) -> Json<Vec<EventEnvelope>> {
    Json(state.bus.since(q.since, q.limit.min(1000)))
}

#[derive(Debug, Deserialize)]
pub struct WsQuery {
    #[serde(default)]
    pub since: Option<i64>,
}

async fn events_ws(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(q): Query<WsQuery>,
) -> Response {
    ws.on_upgrade(move |socket| handle_ws(socket, state, q.since))
}

async fn handle_ws(mut socket: WebSocket, state: AppState, since: Option<i64>) {
    // Subscribe before replaying so nothing published in between is lost.
    let mut rx = state.bus.subscribe();
    let backlog = match since {
        Some(seq) => state.bus.since(seq, 1000),
        None => state.bus.recent(),
    };
    let mut last_seq = 0;
    for env in &backlog {
        last_seq = env.seq;
        let text = serde_json::to_string(env).unwrap_or_default();
        if socket.send(Message::Text(text.into())).await.is_err() {
            return;
        }
    }
    loop {
        tokio::select! {
            incoming = socket.recv() => {
                match incoming {
                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => return,
                    _ => {}
                }
            }
            next = rx.recv() => {
                match next {
                    Ok(env) => {
                        if env.seq <= last_seq && env.seq > 0 {
                            continue;
                        }
                        last_seq = env.seq;
                        let text = serde_json::to_string(&env).unwrap_or_default();
                        if socket.send(Message::Text(text.into())).await.is_err() {
                            return;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(skipped = n, "ws.subscriber.lagged");
                    }
                    Err(_) => return,
                }
            }
        }
    }
}

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Open Horizon API",
        version = "0.3.0",
        description = "Observability + orchestration API for long-horizon distributed RL."
    ),
    paths(
        health,
        dashboard,
        list_runs,
        get_run,
        list_run_turns,
        get_run_turn_training,
        create_run,
        list_profiles,
        cancel_run,
        runtime_config,
        budget_status,
        list_rubrics,
        rescore,
        list_adapters,
        get_adapter,
        run_eval,
        list_training_runs,
        get_training_run,
        create_training_run,
        list_jobs,
        list_events,
    ),
    components(schemas(
        DomainEvent,
        horizon_core::events::Event,
        horizon_core::events::Subject,
        EventEnvelope
    ))
)]
pub struct ApiDoc;

async fn openapi_json() -> Json<utoipa::openapi::OpenApi> {
    Json(ApiDoc::openapi())
}

pub fn router(state: AppState) -> Router {
    let origins: Vec<HeaderValue> = state
        .settings
        .cors_origins
        .iter()
        .filter_map(|o| o.parse().ok())
        .collect();
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);
    Router::new()
        .route("/health", get(health))
        .route("/openapi.json", get(openapi_json))
        .route("/api/dashboard", get(dashboard))
        .route("/api/runs", get(list_runs).post(create_run))
        .route("/api/runs/{run_id}", get(get_run))
        .route("/api/runs/{run_id}/turns", get(list_run_turns))
        .route(
            "/api/runs/{run_id}/turns/{step_index}/training",
            get(get_run_turn_training),
        )
        .route("/api/runs/{run_id}/cancel", post(cancel_run))
        .route("/api/runs/{run_id}/rescore", post(rescore))
        .route("/api/profiles", get(list_profiles))
        .route("/api/config", get(runtime_config))
        .route("/api/budget", get(budget_status))
        .route("/api/rubrics", get(list_rubrics))
        .route("/api/adapters", get(list_adapters))
        .route("/api/adapters/{adapter_id}", get(get_adapter))
        .route("/api/adapters/{adapter_id}/eval", post(run_eval))
        .route(
            "/api/training-runs",
            get(list_training_runs).post(create_training_run),
        )
        .route("/api/training-runs/{id}", get(get_training_run))
        .route("/api/jobs", get(list_jobs))
        .route("/api/events", get(list_events))
        .route("/ws/events", get(events_ws))
        .layer(cors)
        .with_state(state)
}
