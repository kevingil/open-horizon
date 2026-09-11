//! Training orchestration. A training run is a durable job: the record
//! is persisted `pending`, the job runner leases it, and the trainer
//! runs either in Rust (stub) or in the Python bridge (grpo, prime-rl).
//! Metrics stream back as `training.metric` events.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use horizon_bridge::PythonBridge;
use horizon_core::events::DomainEvent;
use horizon_core::models::{
    AdapterRecord, HyperValue, Hyperparams, JobKind, RunDetail, TrainingMetricPoint,
    TrainingRunRecord, TrainingStatus,
};
use horizon_core::{round_to, short_id, utc_now};
use horizon_store::{LocalAdapterRegistry, Store};

use crate::event_bus::EventBus;
use crate::settings::Settings;

#[derive(Debug, Clone)]
pub struct TrainingRequest {
    pub sample_run_ids: Vec<String>,
    pub parent_adapter_id: Option<String>,
    pub hyperparams: Hyperparams,
}

pub struct TrainingService {
    pub settings: Arc<Settings>,
    pub store: Arc<Store>,
    pub adapters: LocalAdapterRegistry,
    pub bus: EventBus,
    bridge: Arc<PythonBridge>,
}

impl TrainingService {
    pub fn new(
        settings: Arc<Settings>,
        store: Arc<Store>,
        adapters: LocalAdapterRegistry,
        bus: EventBus,
        bridge: Arc<PythonBridge>,
    ) -> Self {
        Self {
            settings,
            store,
            adapters,
            bus,
            bridge,
        }
    }

    pub fn trainer_name(&self) -> String {
        match self.settings.trainer_backend.as_str() {
            "stub" => "stub-trainer-v1".into(),
            other => other.to_string(),
        }
    }

    /// Persist a pending record and enqueue the job. Never blocks on training.
    pub fn start(
        &self,
        request: TrainingRequest,
    ) -> Result<TrainingRunRecord, horizon_store::StoreError> {
        let now = utc_now();
        let record = TrainingRunRecord {
            id: short_id("trun"),
            status: TrainingStatus::Pending,
            adapter_in: request.parent_adapter_id.clone(),
            adapter_out: None,
            sample_run_ids: request.sample_run_ids.clone(),
            hyperparams: request.hyperparams.clone(),
            metrics: vec![],
            error: None,
            created_at: now,
            updated_at: now,
        };
        self.store.save_training_run(&record)?;
        self.store.enqueue_job(
            &record.id,
            JobKind::Training,
            &json!({"training_run_id": record.id}),
        )?;
        Ok(record)
    }

    /// Drive one training run to a terminal state. Errors inside the
    /// trainer become a `failed` record plus `training.failed`.
    pub async fn execute(
        &self,
        training_run_id: &str,
    ) -> Result<TrainingRunRecord, horizon_store::StoreError> {
        let Some(mut record) = self.store.get_training_run(training_run_id)? else {
            tracing::warn!(training_run_id, "training.missing_record");
            return Err(horizon_store::StoreError::Corrupt(format!(
                "training run {training_run_id} not found"
            )));
        };
        let samples = self.gather_samples(&record.sample_run_ids)?;
        let parent = record
            .adapter_in
            .as_deref()
            .and_then(|id| self.adapters.get(id));
        record.status = TrainingStatus::Running;
        record.updated_at = utc_now();
        self.store.save_training_run(&record)?;
        self.bus
            .publish(DomainEvent::training_started(record.clone()));

        let metrics: Arc<std::sync::Mutex<Vec<TrainingMetricPoint>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let bus = self.bus.clone();
        let id = record.id.clone();
        let acc = metrics.clone();
        let on_metric = move |point: TrainingMetricPoint| {
            acc.lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(point.clone());
            bus.publish(DomainEvent::training_metric(&id, point));
        };

        let result = match self.settings.trainer_backend.as_str() {
            "stub" => {
                stub_train(
                    &record,
                    &samples,
                    parent.as_ref(),
                    &self.adapters,
                    self.settings.train_step_delay_s,
                    on_metric,
                )
                .await
            }
            other => {
                self.bridge_train(other, &record, &samples, parent.as_ref(), on_metric)
                    .await
            }
        };
        let collected = metrics.lock().unwrap_or_else(|e| e.into_inner()).clone();
        match result {
            Ok(adapter) => {
                let adapter = self
                    .adapters
                    .register(&adapter)
                    .map_err(horizon_store::StoreError::Io)?;
                record.status = TrainingStatus::Completed;
                record.adapter_out = Some(adapter.id.clone());
                record.metrics = collected;
                record.updated_at = utc_now();
                self.store.save_training_run(&record)?;
                self.bus
                    .publish(DomainEvent::adapter_published(adapter.clone()));
                self.bus
                    .publish(DomainEvent::training_completed(record.clone()));
                tracing::info!(training_run_id = %record.id, adapter = %adapter.id, steps = record.metrics.len(), samples = samples.len(), "training.completed");
            }
            Err(error) => {
                tracing::error!(training_run_id = %record.id, error = %error, "training.failed");
                record.status = TrainingStatus::Failed;
                record.metrics = collected;
                record.error = Some(error.clone());
                record.updated_at = utc_now();
                self.store.save_training_run(&record)?;
                self.bus
                    .publish(DomainEvent::training_failed(&record.id, error));
            }
        }
        Ok(record)
    }

    fn gather_samples(
        &self,
        run_ids: &[String],
    ) -> Result<Vec<RunDetail>, horizon_store::StoreError> {
        let mut out = Vec::new();
        for id in run_ids {
            match self.store.get_run(id)? {
                Some(d) => out.push(d),
                None => tracing::warn!(run_id = %id, "training.missing_sample"),
            }
        }
        Ok(out)
    }

    async fn bridge_train(
        &self,
        trainer: &str,
        record: &TrainingRunRecord,
        samples: &[RunDetail],
        parent: Option<&AdapterRecord>,
        mut on_metric: impl FnMut(TrainingMetricPoint),
    ) -> Result<AdapterRecord, String> {
        let new_id = short_id("adapter");
        let target_dir = self.adapters.path_for(&new_id);
        std::fs::create_dir_all(&target_dir).map_err(|e| e.to_string())?;
        let mut turn_training: BTreeMap<String, Value> = BTreeMap::new();
        for sample in samples {
            let rows = self
                .store
                .list_turn_training(&sample.manifest.id)
                .map_err(|e| e.to_string())?;
            turn_training.insert(sample.manifest.id.clone(), json!(rows));
        }
        let params = json!({
            "trainer": trainer,
            "run": record,
            "samples": samples,
            "turn_training": turn_training,
            "parent": parent,
            "new_adapter_id": new_id,
            "adapter_dir": target_dir,
            "options": {
                "grpo_base_model": self.settings.grpo_base_model,
                "prime_rl_base_model": self.settings.prime_rl_base_model,
                "prime_rl_cli": self.settings.prime_rl_cli,
                "prime_rl_config_template": self.settings.prime_rl_config_template,
            },
        });
        let timeout = if self.settings.bridge_timeout_s > 0.0 {
            Some(Duration::from_secs_f64(self.settings.bridge_timeout_s))
        } else {
            None
        };
        let result = self
            .bridge
            .call("train", params, timeout, |name, data| {
                if name == "metric" {
                    if let Ok(point) = serde_json::from_value::<TrainingMetricPoint>(data) {
                        on_metric(point);
                    }
                }
            })
            .await
            .map_err(|e| e.to_string())?;
        let adapter = result
            .get("adapter")
            .cloned()
            .ok_or_else(|| "bridge returned no adapter".to_string())?;
        serde_json::from_value(adapter).map_err(|e| format!("bridge adapter record: {e}"))
    }
}

/// Deterministic stand-in trainer: synthetic loss curve driven by the
/// mean sample reward, copies parent weights, registers the adapter.
async fn stub_train(
    run: &TrainingRunRecord,
    samples: &[RunDetail],
    parent: Option<&AdapterRecord>,
    adapters: &LocalAdapterRegistry,
    default_delay_s: f64,
    mut on_metric: impl FnMut(TrainingMetricPoint),
) -> Result<AdapterRecord, String> {
    let steps = run
        .hyperparams
        .get("steps")
        .and_then(HyperValue::as_f64)
        .map(|v| v as u32)
        .unwrap_or(8)
        .max(1);
    let delay = run
        .hyperparams
        .get("step_delay_s")
        .and_then(HyperValue::as_f64)
        .unwrap_or(default_delay_s);
    let baseline = if samples.is_empty() {
        0.0
    } else {
        samples
            .iter()
            .map(|s| s.reward.terminal_reward)
            .sum::<f64>()
            / samples.len() as f64
    };
    let floor = (1.0 - baseline).max(0.05);
    for step in 0..steps {
        let decay = (steps - step) as f64 / steps as f64;
        on_metric(TrainingMetricPoint {
            step,
            loss: round_to(floor + (1.0 - floor) * decay, 4),
            mean_reward: Some(round_to(baseline + (1.0 - decay) * 0.05, 4)),
            kl: Some(round_to(0.02 * decay, 4)),
            extra: BTreeMap::new(),
        });
        if delay > 0.0 {
            tokio::time::sleep(Duration::from_secs_f64(delay)).await;
        }
    }
    let new_id = short_id("adapter");
    let target = adapters.path_for(&new_id);
    std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
    copy_parent_weights(parent, &target).map_err(|e| e.to_string())?;
    let mut metadata = BTreeMap::new();
    metadata.insert("trainer".into(), "stub-trainer-v1".into());
    metadata.insert("samples".into(), samples.len().to_string());
    metadata.insert("baseline_reward".into(), format!("{baseline:.4}"));
    Ok(AdapterRecord {
        id: new_id,
        parent_id: parent.map(|p| p.id.clone()),
        base_model: parent
            .map(|p| p.base_model.clone())
            .unwrap_or_else(|| "stub:base".into()),
        training_run_id: Some(run.id.clone()),
        eval_score: None,
        path: target.display().to_string(),
        tags: vec!["stub".into()],
        metadata,
        created_at: utc_now(),
    })
}

fn copy_parent_weights(
    parent: Option<&AdapterRecord>,
    target: &std::path::Path,
) -> std::io::Result<()> {
    let Some(parent) = parent else {
        return std::fs::write(target.join("adapter.stub"), "stub-adapter\n");
    };
    let parent_dir = std::path::Path::new(&parent.path);
    if !parent_dir.is_dir() {
        return std::fs::write(target.join("adapter.stub"), "stub-adapter\n");
    }
    for entry in std::fs::read_dir(parent_dir)?.flatten() {
        let src = entry.path();
        if src.file_name().is_some_and(|n| n == "manifest.json") || !src.is_file() {
            continue;
        }
        let dst = target.join(entry.file_name());
        if !dst.exists() {
            std::fs::copy(&src, &dst)?;
        }
    }
    Ok(())
}
