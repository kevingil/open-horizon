//! Eval harness: run a fixed task set against an adapter through the
//! same coordinator path as training rollouts, aggregate terminal
//! rewards, persist an EvalReport, and stamp eval_score on the adapter.

use std::sync::Arc;

use horizon_core::events::DomainEvent;
use horizon_core::models::{
    default_eval_tasks, EvalReport, EvalTask, EvalTaskScore, RolloutRequest,
};
use horizon_core::{round_to, short_id, utc_now};
use horizon_store::{LocalAdapterRegistry, Store};

use crate::coordinator::Coordinator;
use crate::event_bus::EventBus;

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("adapter not found: {0}")]
    AdapterNotFound(String),
    #[error("eval task set is empty")]
    EmptyTaskSet,
    #[error("{0}")]
    Coordinator(#[from] crate::coordinator::CoordinatorError),
    #[error("{0}")]
    Store(#[from] horizon_store::StoreError),
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

pub struct EvalHarness {
    pub coordinator: Arc<Coordinator>,
    pub store: Arc<Store>,
    pub adapters: LocalAdapterRegistry,
    pub bus: EventBus,
}

impl EvalHarness {
    pub async fn run(
        &self,
        adapter_id: &str,
        tasks: Option<Vec<EvalTask>>,
        task_set: Option<String>,
    ) -> Result<EvalReport, EvalError> {
        let adapter = self
            .adapters
            .get(adapter_id)
            .ok_or_else(|| EvalError::AdapterNotFound(adapter_id.into()))?;
        let tasks = tasks.unwrap_or_else(default_eval_tasks);
        if tasks.is_empty() {
            return Err(EvalError::EmptyTaskSet);
        }
        let mut per_task = Vec::with_capacity(tasks.len());
        for task in &tasks {
            let request = RolloutRequest {
                horizon: Some(task.horizon),
                success_criteria: task.success_criteria.clone(),
                adapter_id: Some(adapter_id.into()),
                ..RolloutRequest::new(task.prompt.clone())
            };
            let run_id = short_id("run");
            let cancel = self.coordinator.register_cancel(&run_id).await;
            let detail = self.coordinator.execute(&request, &run_id, cancel).await;
            self.coordinator.unregister_cancel(&run_id).await;
            let detail = detail?;
            per_task.push(EvalTaskScore {
                task_id: task.id.clone(),
                terminal_reward: detail.reward.terminal_reward,
            });
        }
        let mean = round_to(
            per_task.iter().map(|p| p.terminal_reward).sum::<f64>() / per_task.len() as f64,
            4,
        );
        let report = EvalReport {
            id: short_id("eval"),
            adapter_id: adapter_id.into(),
            task_set: task_set.unwrap_or_else(|| "builtin".into()),
            mean_reward: mean,
            per_task,
            created_at: utc_now(),
        };
        self.store.save_eval_report(&report)?;
        let mut updated = adapter;
        updated.eval_score = Some(mean);
        self.adapters.register(&updated)?;
        self.bus
            .publish(DomainEvent::eval_completed(report.clone()));
        tracing::info!(
            adapter = adapter_id,
            mean,
            tasks = report.per_task.len(),
            "eval.completed"
        );
        Ok(report)
    }
}
