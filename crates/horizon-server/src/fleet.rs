//! Fleet registry: who is part of this deployment and whether they are
//! alive. The control plane heartbeats itself, probes each policy
//! endpoint, pings the Python bridge when a profile or trainer needs it,
//! and reports the sandbox. Stale heartbeats flip to `down`.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};
use tokio_util::sync::CancellationToken;

use horizon_core::events::{DomainEvent, Event, Subject};
use horizon_core::models::{NodeRecord, NodeStatus};
use horizon_core::utc_now;

use crate::app::AppState;
use crate::settings::RoutesTo;

pub const HEARTBEAT: Duration = Duration::from_secs(5);
pub const PROBE: Duration = Duration::from_secs(15);
const STALE_AFTER_SECS: i64 = 30;

/// Live counters the job runner updates; the heartbeat reads them.
#[derive(Debug, Default)]
pub struct FleetCounters {
    pub in_flight: AtomicU32,
    pub training_in_flight: AtomicU32,
}

fn meta(pairs: Vec<(&str, Value)>) -> Map<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn publish(state: &AppState, node: NodeRecord) {
    if let Err(e) = state.store.upsert_node(&node) {
        tracing::warn!(node = %node.id, error = %e, "fleet.upsert_failed");
        return;
    }
    state.bus.publish(Event::new(
        Some(Subject::Node {
            id: node.id.clone(),
        }),
        DomainEvent::NodeUpdated { node },
    ));
}

pub fn spawn(
    state: AppState,
    counters: Arc<FleetCounters>,
    shutdown: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let started = Instant::now();
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("client");
        let mut last_probe: Option<Instant> = None;
        loop {
            // Control plane: always fresh.
            let s = &state.settings;
            publish(
                &state,
                NodeRecord {
                    id: s.worker_id.clone(),
                    role: "control-plane".into(),
                    status: if counters.in_flight.load(Ordering::Relaxed) > 0 {
                        NodeStatus::Busy
                    } else {
                        NodeStatus::Up
                    },
                    detail: format!("job runner, {} rollout slots", s.max_parallel_rollouts),
                    meta: meta(vec![
                        (
                            "in_flight",
                            json!(counters.in_flight.load(Ordering::Relaxed)),
                        ),
                        (
                            "training_in_flight",
                            json!(counters.training_in_flight.load(Ordering::Relaxed)),
                        ),
                        ("max_parallel_rollouts", json!(s.max_parallel_rollouts)),
                        ("store", json!(s.store_backend)),
                        ("uptime_s", json!(started.elapsed().as_secs())),
                        ("version", json!(env!("CARGO_PKG_VERSION"))),
                        ("pid", json!(std::process::id())),
                    ]),
                    last_seen: utc_now(),
                },
            );

            let due = last_probe.is_none_or(|t| t.elapsed() >= PROBE);
            if due {
                last_probe = Some(Instant::now());
                probe_policies(&state, &http).await;
                probe_bridge(&state).await;
                publish(
                    &state,
                    NodeRecord {
                        id: "sandbox".into(),
                        role: "sandbox".into(),
                        status: if s.env_backend == "repo" {
                            NodeStatus::Up
                        } else {
                            NodeStatus::Standby
                        },
                        detail: match s.env_sandbox.as_str() {
                            "docker" => format!("docker · {}", s.sandbox_image),
                            other => format!("{other} · host subprocess"),
                        },
                        meta: meta(vec![
                            ("kind", json!(s.env_sandbox)),
                            ("image", json!(s.sandbox_image)),
                            ("command_timeout_s", json!(s.env_command_timeout_s)),
                        ]),
                        last_seen: utc_now(),
                    },
                );
                match state
                    .store
                    .expire_nodes(chrono::Duration::seconds(STALE_AFTER_SECS))
                {
                    Ok(expired) => {
                        for node in expired {
                            tracing::warn!(node = %node.id, "fleet.node_down");
                            state.bus.publish(Event::new(
                                Some(Subject::Node {
                                    id: node.id.clone(),
                                }),
                                DomainEvent::NodeUpdated { node },
                            ));
                        }
                    }
                    Err(e) => tracing::warn!(error = %e, "fleet.expire_failed"),
                }
            }

            tokio::select! {
                _ = shutdown.cancelled() => break,
                _ = tokio::time::sleep(HEARTBEAT) => {}
            }
        }
    })
}

async fn probe_policies(state: &AppState, http: &reqwest::Client) {
    for (name, profile) in &state.coordinator.profiles {
        let url = format!("{}/models", profile.base_url.trim_end_matches('/'));
        let started = Instant::now();
        let result = http
            .get(&url)
            .bearer_auth(profile.resolve_api_key())
            .send()
            .await;
        let latency_ms = started.elapsed().as_millis() as u64;
        let (status, detail) = match result {
            Ok(resp) if resp.status().is_success() => (
                NodeStatus::Up,
                format!("{} · {latency_ms} ms", profile.model),
            ),
            Ok(resp) => (
                NodeStatus::Down,
                format!("{} · HTTP {}", profile.model, resp.status().as_u16()),
            ),
            Err(e) => (
                NodeStatus::Down,
                format!("{} · {}", profile.model, short_err(&e.to_string())),
            ),
        };
        publish(
            state,
            NodeRecord {
                id: format!("policy:{name}"),
                role: "policy".into(),
                status,
                detail,
                meta: meta(vec![
                    ("profile", json!(name)),
                    ("base_url", json!(profile.base_url)),
                    ("model", json!(profile.model)),
                    ("routes_to", json!(profile.routes_to.as_str())),
                    ("latency_ms", json!(latency_ms)),
                ]),
                last_seen: utc_now(),
            },
        );
    }
}

async fn probe_bridge(state: &AppState) {
    let needs_bridge = state
        .coordinator
        .profiles
        .values()
        .any(|p| p.routes_to == RoutesTo::Verifiers)
        || state.settings.trainer_backend != "stub"
        || state.settings.training_recorder == "tokenized";
    let (status, detail, extra) = if !needs_bridge {
        (
            NodeStatus::Standby,
            "not required by the current profiles or trainer".to_string(),
            json!({}),
        )
    } else {
        match state.bridge.ping(Duration::from_secs(10)).await {
            Ok(v) => (
                NodeStatus::Up,
                format!(
                    "python {} · bridge {}",
                    v["python"].as_str().unwrap_or("?"),
                    v["version"].as_str().unwrap_or("?")
                ),
                v,
            ),
            Err(e) => (NodeStatus::Down, short_err(&e.to_string()), json!({})),
        }
    };
    publish(
        state,
        NodeRecord {
            id: "bridge".into(),
            role: "bridge".into(),
            status,
            detail,
            meta: meta(vec![
                ("python", json!(state.settings.bridge_python)),
                ("ping", extra),
            ]),
            last_seen: utc_now(),
        },
    );
}

fn short_err(e: &str) -> String {
    e.chars().take(120).collect()
}
