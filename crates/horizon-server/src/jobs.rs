//! Job runner: leases rollout and training jobs from the store, keeps
//! their leases alive while they run, and marks them terminal. A crashed
//! process leaves an expired lease that the next runner picks up.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Notify, Semaphore};
use tokio_util::sync::CancellationToken;

use horizon_core::models::{JobKind, JobRecord, JobStatus};

use crate::app::AppState;
use crate::coordinator::RolloutJob;

const POLL_INTERVAL: Duration = Duration::from_millis(250);

pub struct JobRunner {
    state: AppState,
    rollout_slots: Arc<Semaphore>,
    shutdown: CancellationToken,
}

impl JobRunner {
    pub fn new(state: AppState, shutdown: CancellationToken) -> Self {
        let rollout_slots = Arc::new(Semaphore::new(state.settings.max_parallel_rollouts.max(1)));
        Self {
            state,
            rollout_slots,
            shutdown,
        }
    }

    pub fn spawn(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move { self.run().await })
    }

    async fn run(self) {
        let notify: Arc<Notify> = self.state.jobs_notify.clone();
        tracing::info!(worker_id = %self.state.settings.worker_id, "jobs.runner.started");
        loop {
            if self.shutdown.is_cancelled() {
                break;
            }
            let mut leased_any = false;
            // Rollouts: only lease when a concurrency slot is free, so
            // queued work stays visible to other replicas.
            if let Ok(permit) = self.rollout_slots.clone().try_acquire_owned() {
                match self.lease(JobKind::Rollout) {
                    Some(job) => {
                        leased_any = true;
                        let state = self.state.clone();
                        tokio::spawn(async move {
                            run_rollout_job(state, job).await;
                            drop(permit);
                        });
                    }
                    None => drop(permit),
                }
            }
            if let Some(job) = self.lease(JobKind::Training) {
                leased_any = true;
                let state = self.state.clone();
                tokio::spawn(async move { run_training_job(state, job).await });
            }
            if leased_any {
                continue;
            }
            tokio::select! {
                _ = notify.notified() => {}
                _ = tokio::time::sleep(POLL_INTERVAL) => {}
                _ = self.shutdown.cancelled() => break,
            }
        }
        tracing::info!("jobs.runner.stopped");
    }

    fn lease(&self, kind: JobKind) -> Option<JobRecord> {
        match self.state.store.lease_job(
            kind,
            &self.state.settings.worker_id,
            self.state.settings.job_lease_secs,
        ) {
            Ok(job) => job,
            Err(e) => {
                tracing::error!(error = %e, "jobs.lease_failed");
                None
            }
        }
    }
}

fn spawn_heartbeat(state: &AppState, job_id: &str) -> CancellationToken {
    let stop = CancellationToken::new();
    let store = state.store.clone();
    let owner = state.settings.worker_id.clone();
    let lease = state.settings.job_lease_secs;
    let id = job_id.to_string();
    let token = stop.clone();
    tokio::spawn(async move {
        let interval = Duration::from_secs((lease / 3).max(1) as u64);
        loop {
            tokio::select! {
                _ = token.cancelled() => break,
                _ = tokio::time::sleep(interval) => {
                    match store.heartbeat_job(&id, &owner, lease) {
                        Ok(true) => {}
                        Ok(false) => {
                            tracing::warn!(job_id = %id, "jobs.heartbeat.lost_lease");
                            break;
                        }
                        Err(e) => tracing::warn!(job_id = %id, error = %e, "jobs.heartbeat.failed"),
                    }
                }
            }
        }
    });
    stop
}

async fn run_rollout_job(state: AppState, job: JobRecord) {
    let payload: RolloutJob = match serde_json::from_value(job.payload.clone()) {
        Ok(p) => p,
        Err(e) => {
            let _ = state.store.finish_job(
                &job.id,
                JobStatus::Failed,
                Some(&format!("bad payload: {e}")),
            );
            return;
        }
    };
    if job.attempts > state.settings.job_max_attempts {
        let msg = format!("gave up after {} attempts", job.attempts - 1);
        tracing::error!(run_id = %payload.run_id, "jobs.rollout.exhausted");
        let _ = state
            .store
            .finish_job(&job.id, JobStatus::Failed, Some(&msg));
        return;
    }
    if job.attempts > 1 {
        tracing::warn!(run_id = %payload.run_id, attempt = job.attempts, "jobs.rollout.retry_after_lost_lease");
    }
    let heartbeat = spawn_heartbeat(&state, &job.id);
    let cancel = state.coordinator.register_cancel(&payload.run_id).await;
    let result = state
        .coordinator
        .execute(&payload.request, &payload.run_id, cancel)
        .await;
    state.coordinator.unregister_cancel(&payload.run_id).await;
    heartbeat.cancel();
    match result {
        Ok(detail) => {
            let status = match detail.manifest.status {
                horizon_core::models::RunStatus::Completed => JobStatus::Completed,
                horizon_core::models::RunStatus::Cancelled => JobStatus::Cancelled,
                _ => JobStatus::Failed,
            };
            let _ = state
                .store
                .finish_job(&job.id, status, detail.manifest.error.as_deref());
        }
        Err(e) => {
            tracing::error!(run_id = %payload.run_id, error = %e, "jobs.rollout.failed");
            let _ = state
                .store
                .finish_job(&job.id, JobStatus::Failed, Some(&e.to_string()));
        }
    }
}

async fn run_training_job(state: AppState, job: JobRecord) {
    let training_run_id = job
        .payload
        .get("training_run_id")
        .and_then(|v| v.as_str())
        .unwrap_or(&job.id)
        .to_string();
    if job.attempts > state.settings.job_max_attempts {
        let _ = state.store.finish_job(
            &job.id,
            JobStatus::Failed,
            Some("gave up after repeated lost leases"),
        );
        return;
    }
    let heartbeat = spawn_heartbeat(&state, &job.id);
    state
        .counters
        .training_in_flight
        .fetch_add(1, Ordering::Relaxed);
    let result = state.training.execute(&training_run_id).await;
    state
        .counters
        .training_in_flight
        .fetch_sub(1, Ordering::Relaxed);
    heartbeat.cancel();
    match result {
        Ok(record) => {
            let status = match record.status {
                horizon_core::models::TrainingStatus::Completed => JobStatus::Completed,
                horizon_core::models::TrainingStatus::Cancelled => JobStatus::Cancelled,
                _ => JobStatus::Failed,
            };
            let _ = state
                .store
                .finish_job(&job.id, status, record.error.as_deref());
        }
        Err(e) => {
            let _ = state
                .store
                .finish_job(&job.id, JobStatus::Failed, Some(&e.to_string()));
        }
    }
}
