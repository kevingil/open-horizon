//! Hot-reload trained LoRA adapters into a running SGLang server.
//! Tails the event bus for `adapter.published` and POSTs to
//! `/load_lora_adapter`. Failures are logged, never raised.

use std::time::Duration;

use serde_json::json;

use horizon_core::events::DomainEvent;

use crate::event_bus::EventBus;

#[derive(Debug, Clone)]
pub struct SglangLoraReloader {
    admin_url: String,
    http: reqwest::Client,
}

impl SglangLoraReloader {
    pub fn new(admin_url: &str) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("client");
        Self {
            admin_url: admin_url.trim_end_matches('/').to_string(),
            http,
        }
    }

    pub async fn load(&self, lora_name: &str, lora_path: &str) -> bool {
        self.post(
            "/load_lora_adapter",
            json!({"lora_name": lora_name, "lora_path": lora_path}),
        )
        .await
    }

    pub async fn unload(&self, lora_name: &str) -> bool {
        self.post("/unload_lora_adapter", json!({"lora_name": lora_name}))
            .await
    }

    async fn post(&self, path: &str, body: serde_json::Value) -> bool {
        let url = format!("{}{path}", self.admin_url);
        match self.http.post(&url).json(&body).send().await {
            Ok(resp) if resp.status().is_success() => true,
            Ok(resp) => {
                tracing::warn!(
                    url,
                    status = resp.status().as_u16(),
                    "sglang.lora.bad_status"
                );
                false
            }
            Err(e) => {
                tracing::warn!(url, error = %e, "sglang.lora.network_error");
                false
            }
        }
    }
}

/// Background task: hot-load every newly published adapter.
pub fn spawn_autoreload(
    bus: EventBus,
    reloader: SglangLoraReloader,
) -> tokio::task::JoinHandle<()> {
    let mut rx = bus.subscribe();
    tokio::spawn(async move {
        tracing::info!(admin_url = %reloader.admin_url, "sglang.lora.autoreload.started");
        loop {
            match rx.recv().await {
                Ok(envelope) => {
                    if let DomainEvent::AdapterPublished { adapter, .. } = envelope.event {
                        if reloader.load(&adapter.id, &adapter.path).await {
                            tracing::info!(adapter_id = %adapter.id, path = %adapter.path, "sglang.lora.loaded");
                        } else {
                            tracing::warn!(adapter_id = %adapter.id, hint = "verify SGLang was launched with --enable-lora", "sglang.lora.load_failed");
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            }
        }
    })
}
