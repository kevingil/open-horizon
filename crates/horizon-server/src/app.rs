//! Wiring: build every service from settings.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Notify;

use horizon_bridge::{BridgeConfig, PythonBridge};
use horizon_runner::{RepoRunner, RepoRunnerConfig, Sandbox};
use horizon_store::{LocalAdapterRegistry, Store};

use crate::coordinator::Coordinator;
use crate::eval::EvalHarness;
use crate::event_bus::EventBus;
use crate::settings::Settings;
use crate::training::TrainingService;

#[derive(Clone)]
pub struct AppState {
    pub settings: Arc<Settings>,
    pub store: Arc<Store>,
    pub bus: EventBus,
    pub coordinator: Arc<Coordinator>,
    pub training: Arc<TrainingService>,
    pub eval: Arc<EvalHarness>,
    pub adapters: LocalAdapterRegistry,
    pub bridge: Arc<PythonBridge>,
    /// Woken whenever a job is enqueued so the runner picks it up at once.
    pub jobs_notify: Arc<Notify>,
    /// Live in-flight counters the fleet heartbeat reports.
    pub counters: Arc<crate::fleet::FleetCounters>,
}

pub async fn build_app(settings: Settings) -> anyhow::Result<AppState> {
    let settings = Arc::new(settings);
    let root = std::fs::canonicalize(&settings.workspace_root)
        .unwrap_or_else(|_| settings.workspace_root.clone());
    let store = Arc::new(match settings.store_backend.as_str() {
        "memory" => Store::in_memory()?,
        "sqlite" => Store::open(settings.artifacts_dir.join("runs.db"))?,
        other => anyhow::bail!("Unsupported store backend: {other}"),
    });
    let bus = EventBus::new(store.clone(), 200, 1024);
    let adapters = LocalAdapterRegistry::new(&settings.adapters_dir)?;

    let repo_runner = if settings.env_backend == "repo" {
        let sandbox = Sandbox::from_setting(&settings.env_sandbox, Some(&settings.sandbox_image))
            .await
            .map_err(anyhow::Error::msg)?;
        Some(RepoRunner::new(
            RepoRunnerConfig {
                source_root: root.clone(),
                scratch_root: settings.artifacts_dir.join("workspaces"),
                command_timeout: Duration::from_secs_f64(settings.env_command_timeout_s),
                max_output_bytes: settings.env_max_output_bytes,
            },
            sandbox,
        )?)
    } else {
        None
    };

    let src_dir: PathBuf = locate_python_src(&root);
    let bridge = Arc::new(PythonBridge::new(BridgeConfig::python_module(
        &settings.bridge_python,
        root.clone(),
        src_dir,
    )));

    let coordinator = Arc::new(Coordinator::new(
        settings.clone(),
        store.clone(),
        bus.clone(),
        repo_runner,
        bridge.clone(),
    ));
    let training = Arc::new(TrainingService::new(
        settings.clone(),
        store.clone(),
        adapters.clone(),
        bus.clone(),
        bridge.clone(),
    ));
    let eval = Arc::new(EvalHarness {
        coordinator: coordinator.clone(),
        store: store.clone(),
        adapters: adapters.clone(),
        bus: bus.clone(),
    });

    Ok(AppState {
        settings,
        store,
        bus,
        coordinator,
        training,
        eval,
        adapters,
        bridge,
        jobs_notify: Arc::new(Notify::new()),
        counters: Arc::new(crate::fleet::FleetCounters::default()),
    })
}

/// The Python bridge package lives in `python/` next to the workspace
/// root, or wherever `RL_BRIDGE_SRC` points.
fn locate_python_src(root: &std::path::Path) -> PathBuf {
    if let Ok(v) = std::env::var("RL_BRIDGE_SRC") {
        return PathBuf::from(v);
    }
    let candidates = [
        root.join("python"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python"),
    ];
    for c in candidates {
        if c.join("horizon_bridge").is_dir() {
            return c.canonicalize().unwrap_or(c);
        }
    }
    root.join("python")
}
