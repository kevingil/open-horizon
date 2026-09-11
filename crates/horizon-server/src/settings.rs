//! Runtime configuration, env-driven with the `RL_` prefix. `.env` in
//! the working directory is loaded first without overriding real env.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use horizon_core::scheduling::{RoundScheduler, ScalingMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RoutesTo {
    Verifiers,
    Repo,
}

impl RoutesTo {
    pub fn as_str(self) -> &'static str {
        match self {
            RoutesTo::Verifiers => "verifiers",
            RoutesTo::Repo => "repo",
        }
    }
}

/// A named (base_url, api_key, model) triple a rollout can request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyProfile {
    pub base_url: String,
    pub model: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default, skip_serializing)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub extra_body: Map<String, Value>,
    #[serde(default = "default_routes_to")]
    pub routes_to: RoutesTo,
    #[serde(default = "d_max_output_tokens")]
    pub max_output_tokens: u32,
    #[serde(default = "d_max_retries")]
    pub max_retries: u32,
    #[serde(default = "d_env_id")]
    pub env_id: String,
    #[serde(default)]
    pub env_args: Map<String, Value>,
    #[serde(default = "d_max_concurrent")]
    pub max_concurrent: usize,
    #[serde(default)]
    pub rollout_timeout_s: Option<f64>,
}

fn default_routes_to() -> RoutesTo {
    RoutesTo::Verifiers
}
fn d_max_output_tokens() -> u32 {
    2048
}
fn d_max_retries() -> u32 {
    3
}
fn d_env_id() -> String {
    "vf-math".into()
}
fn d_max_concurrent() -> usize {
    4
}

impl PolicyProfile {
    /// api_key resolution: literal -> env var named by api_key_env -> "not-needed".
    pub fn resolve_api_key(&self) -> String {
        if let Some(k) = &self.api_key {
            return k.clone();
        }
        if let Some(name) = &self.api_key_env {
            if let Ok(v) = std::env::var(name) {
                if !v.is_empty() {
                    return v;
                }
            }
        }
        "not-needed".into()
    }
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub workspace_root: PathBuf,
    pub artifacts_dir: PathBuf,
    pub host: String,
    pub port: u16,

    pub llm_api_key: Option<String>,
    pub llm_base_url: String,
    pub llm_model: String,
    pub llm_max_output_tokens: u32,
    pub llm_max_retries: u32,
    pub policy_profiles: BTreeMap<String, PolicyProfile>,
    pub default_policy_profile: String,

    pub store_backend: String,
    pub env_backend: String,
    pub env_command_timeout_s: f64,
    pub env_max_output_bytes: usize,
    pub verifiers_env_id: String,
    pub verifiers_env_args: Map<String, Value>,
    pub verifiers_max_concurrent: usize,
    pub verifiers_rollout_timeout_s: Option<f64>,
    pub env_sandbox: String,
    pub sandbox_image: String,

    pub max_parallel_rollouts: usize,
    pub max_tokens_per_run: u64,
    pub daily_budget_usd: f64,
    pub budget_window_hours: f64,

    pub round_scheduler: RoundScheduler,

    pub trainer_backend: String,
    pub adapters_dir: PathBuf,
    pub train_step_delay_s: f64,
    pub grpo_base_model: String,
    pub prime_rl_base_model: String,
    pub prime_rl_cli: String,
    pub prime_rl_config_template: Option<PathBuf>,
    pub training_recorder: String,
    pub recorder_tokenizer: String,

    pub bridge_python: String,
    pub bridge_timeout_s: f64,

    pub sglang_admin_url: Option<String>,
    pub sglang_autoload_lora: bool,

    pub job_lease_secs: i64,
    pub job_max_attempts: u32,
    pub worker_id: String,

    pub log_level: String,
    pub log_json: bool,
    pub cors_origins: Vec<String>,
}

fn env(key: &str) -> Option<String> {
    std::env::var(format!("RL_{key}"))
        .ok()
        .filter(|v| !v.is_empty())
}

fn env_or(key: &str, default: &str) -> String {
    env(key).unwrap_or_else(|| default.to_string())
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    env(key).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_bool(key: &str, default: bool) -> bool {
    env(key)
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(default)
}

fn env_json_map(key: &str) -> Map<String, Value> {
    env(key)
        .and_then(|v| serde_json::from_str::<Value>(&v).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

impl Settings {
    /// Load `.env` (never overriding existing variables) and read settings.
    pub fn from_env() -> Self {
        let _ = dotenvy::dotenv();
        Self::from_current_env()
    }

    pub fn from_current_env() -> Self {
        let scaling_mode = match env_or("SCALING_MODE", "linear").as_str() {
            "step" => ScalingMode::Step,
            _ => ScalingMode::Linear,
        };
        let round_scheduler = if env_or("ROUND_SCHEDULER", "fixed") == "scaling" {
            RoundScheduler::Scaling {
                start: env_parse("SCALING_HORIZON_START", 4),
                end: env_parse("SCALING_HORIZON_END", 16),
                ramp_steps: env_parse("SCALING_RAMP_STEPS", 100),
                mode: scaling_mode,
            }
        } else {
            RoundScheduler::Fixed {
                horizon: env_parse("ROUND_SCHEDULER_DEFAULT_HORIZON", 6),
            }
        };
        let policy_profiles: BTreeMap<String, PolicyProfile> = env("POLICY_PROFILES")
            .and_then(|v| {
                serde_json::from_str(&v)
                    .map_err(
                        |e| tracing::error!(error = %e, "RL_POLICY_PROFILES is not valid JSON"),
                    )
                    .ok()
            })
            .unwrap_or_default();
        let cors_origins = env("CORS_ORIGINS")
            .map(|v| {
                serde_json::from_str::<Vec<String>>(&v).unwrap_or_else(|_| {
                    v.split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect()
                })
            })
            .unwrap_or_else(|| {
                vec![
                    "http://localhost:5173".into(),
                    "http://127.0.0.1:5173".into(),
                ]
            });
        let artifacts_dir = PathBuf::from(env_or("ARTIFACTS_DIR", "./artifacts"));
        let default_adapters = artifacts_dir.join("adapters");
        Self {
            workspace_root: PathBuf::from(env_or("WORKSPACE_ROOT", ".")),
            artifacts_dir: artifacts_dir.clone(),
            host: env_or("HOST", "127.0.0.1"),
            port: env_parse("PORT", 8000),
            llm_api_key: env("LLM_API_KEY"),
            llm_base_url: env_or("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_model: env_or("LLM_MODEL", "gpt-5.4-mini"),
            llm_max_output_tokens: env_parse("LLM_MAX_OUTPUT_TOKENS", 2048),
            llm_max_retries: env_parse("LLM_MAX_RETRIES", 3),
            policy_profiles,
            default_policy_profile: env_or("DEFAULT_POLICY_PROFILE", "default"),
            store_backend: env_or("STORE_BACKEND", "memory"),
            env_backend: env_or("ENV_BACKEND", "verifiers"),
            env_command_timeout_s: env_parse("ENV_COMMAND_TIMEOUT_S", 10.0),
            env_max_output_bytes: env_parse("ENV_MAX_OUTPUT_BYTES", 16_384),
            verifiers_env_id: env_or("VERIFIERS_ENV_ID", "vf-math"),
            verifiers_env_args: env_json_map("VERIFIERS_ENV_ARGS"),
            verifiers_max_concurrent: env_parse("VERIFIERS_MAX_CONCURRENT", 4),
            verifiers_rollout_timeout_s: env("VERIFIERS_ROLLOUT_TIMEOUT_S")
                .and_then(|v| v.parse().ok()),
            env_sandbox: env_or("ENV_SANDBOX", "none"),
            sandbox_image: env_or("SANDBOX_IMAGE", "python:3.11-slim"),
            max_parallel_rollouts: env_parse("MAX_PARALLEL_ROLLOUTS", 4),
            max_tokens_per_run: env_parse("MAX_TOKENS_PER_RUN", 100_000),
            daily_budget_usd: env_parse("DAILY_BUDGET_USD", 5.0),
            budget_window_hours: env_parse("BUDGET_WINDOW_HOURS", 24.0),
            round_scheduler,
            trainer_backend: env_or("TRAINER_BACKEND", "stub"),
            adapters_dir: env("ADAPTERS_DIR")
                .map(PathBuf::from)
                .unwrap_or(default_adapters),
            train_step_delay_s: env_parse("TRAIN_STEP_DELAY_S", 0.0),
            grpo_base_model: env_or("GRPO_BASE_MODEL", "Qwen/Qwen3-0.6B"),
            prime_rl_base_model: env_or("PRIME_RL_BASE_MODEL", "Qwen/Qwen3-8B"),
            prime_rl_cli: env_or("PRIME_RL_CLI", "prime-rl"),
            prime_rl_config_template: env("PRIME_RL_CONFIG_TEMPLATE").map(PathBuf::from),
            training_recorder: env_or("TRAINING_RECORDER", "null"),
            recorder_tokenizer: env_or("RECORDER_TOKENIZER", "Qwen/Qwen3-0.6B"),
            bridge_python: env_or("BRIDGE_PYTHON", "python"),
            bridge_timeout_s: env_parse("BRIDGE_TIMEOUT_S", 0.0),
            sglang_admin_url: env("SGLANG_ADMIN_URL"),
            sglang_autoload_lora: env_bool("SGLANG_AUTOLOAD_LORA", false),
            job_lease_secs: env_parse("JOB_LEASE_SECS", 30),
            job_max_attempts: env_parse("JOB_MAX_ATTEMPTS", 3),
            worker_id: env("WORKER_ID").unwrap_or_else(|| format!("worker-{}", std::process::id())),
            log_level: env_or("LOG_LEVEL", "INFO"),
            log_json: env_bool("LOG_JSON", false),
            cors_origins,
        }
    }

    /// Test-friendly defaults: in-memory store, repo backend, stub trainer,
    /// no budget, artifacts under `dir`.
    pub fn for_tests(dir: &std::path::Path) -> Self {
        let mut s = Self::from_current_env();
        s.workspace_root = dir.to_path_buf();
        s.artifacts_dir = dir.join("artifacts");
        s.adapters_dir = dir.join("artifacts/adapters");
        s.store_backend = "memory".into();
        s.env_backend = "repo".into();
        s.trainer_backend = "stub".into();
        s.daily_budget_usd = 0.0;
        s.policy_profiles.clear();
        s.llm_base_url = "http://127.0.0.1:1/v1".into();
        s.llm_api_key = Some("test".into());
        s.sglang_autoload_lora = false;
        s
    }

    /// The dict of profiles the coordinator dispatches from. Synthesises
    /// a single default profile from the legacy `llm_*` fields when no
    /// explicit profiles are configured.
    pub fn effective_profiles(&self) -> BTreeMap<String, PolicyProfile> {
        if !self.policy_profiles.is_empty() {
            return self.policy_profiles.clone();
        }
        let routes_to = if self.env_backend == "verifiers" {
            RoutesTo::Verifiers
        } else {
            RoutesTo::Repo
        };
        let profile = PolicyProfile {
            base_url: self.llm_base_url.clone(),
            model: self.llm_model.clone(),
            api_key_env: None,
            api_key: self.llm_api_key.clone(),
            extra_body: Map::new(),
            routes_to,
            max_output_tokens: self.llm_max_output_tokens,
            max_retries: self.llm_max_retries,
            env_id: self.verifiers_env_id.clone(),
            env_args: self.verifiers_env_args.clone(),
            max_concurrent: self.verifiers_max_concurrent,
            rollout_timeout_s: self.verifiers_rollout_timeout_s,
        };
        BTreeMap::from([(self.default_policy_profile.clone(), profile)])
    }
}
