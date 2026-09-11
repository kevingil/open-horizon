//! End-to-end: HTTP API + job runner + fake OpenAI provider.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::{routing::post, Json, Router};
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use horizon_server::jobs::JobRunner;
use horizon_server::{api, build_app, Settings};

struct Harness {
    base: String,

    llm_calls: Arc<AtomicUsize>,
    _dir: tempfile::TempDir,
    shutdown: CancellationToken,
}

/// Fake OpenAI-compatible provider. Turn 1 reads README.md, turn 2 finishes.
/// `slow` makes every completion take 3s so cancellation can be observed.
async fn fake_llm(slow: bool) -> (String, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |Json(body): Json<Value>| {
            let counter = counter.clone();
            async move {
                if slow {
                    tokio::time::sleep(Duration::from_secs(3)).await;
                }
                let n = counter.fetch_add(1, Ordering::SeqCst);
                let tool_results = body["messages"].as_array().unwrap().iter().filter(|m| m["role"] == "tool").count();
                let msg = if tool_results == 0 {
                    json!({"role": "assistant", "content": null, "tool_calls": [{"id": format!("call_{n}"), "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"path\": \"README.md\"}"}}]})
                } else {
                    json!({"role": "assistant", "content": "The readme says hello."})
                };
                Json(json!({"choices": [{"message": msg}], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}/v1"), calls)
}

async fn harness(slow_llm: bool) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("README.md"), "hello from the readme").unwrap();
    let (llm_url, llm_calls) = fake_llm(slow_llm).await;
    let mut settings = Settings::for_tests(dir.path());
    settings.llm_base_url = llm_url;
    settings.llm_model = "gpt-4.1-mini".into();
    settings.max_parallel_rollouts = 2;
    let state = build_app(settings).await.unwrap();
    let shutdown = CancellationToken::new();
    JobRunner::new(state.clone(), shutdown.clone()).spawn();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = api::router(state);
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    Harness {
        base: format!("http://{addr}"),
        llm_calls,
        _dir: dir,
        shutdown,
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

async fn get(base: &str, path: &str) -> (u16, Value) {
    let resp = reqwest::get(format!("{base}{path}")).await.unwrap();
    let status = resp.status().as_u16();
    (status, resp.json().await.unwrap_or(Value::Null))
}

async fn post_json(base: &str, path: &str, body: Value) -> (u16, Value) {
    let resp = reqwest::Client::new()
        .post(format!("{base}{path}"))
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    (status, resp.json().await.unwrap_or(Value::Null))
}

async fn wait_for_run(base: &str, run_id: &str, terminal: &[&str]) -> Value {
    for _ in 0..200 {
        let (status, run) = get(base, &format!("/api/runs/{run_id}")).await;
        if status == 200 && terminal.contains(&run["manifest"]["status"].as_str().unwrap_or("")) {
            return run;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("run {run_id} never reached {terminal:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn health_config_profiles_rubrics() {
    let h = harness(false).await;
    assert_eq!(get(&h.base, "/health").await.1["status"], "ok");
    let (_, config) = get(&h.base, "/api/config").await;
    assert_eq!(config["env_backend"], "repo");
    assert_eq!(config["policy_name"], "openai:gpt-4.1-mini");
    assert_eq!(config["trainer_backend"], "stub");
    let (_, profiles) = get(&h.base, "/api/profiles").await;
    assert_eq!(profiles["default"], "default");
    assert_eq!(profiles["profiles"][0]["routes_to"], "repo");
    let (_, rubrics) = get(&h.base, "/api/rubrics").await;
    assert_eq!(rubrics.as_array().unwrap().len(), 3);
    let (_, openapi) = get(&h.base, "/openapi.json").await;
    assert!(openapi["paths"]["/api/runs"].is_object());
    assert!(openapi["components"]["schemas"]["RunDetail"].is_object());
    let (_, dash) = get(&h.base, "/api/dashboard").await;
    assert_eq!(dash["workers"][0]["id"], "worker-rollout-local");
}

#[tokio::test(flavor = "multi_thread")]
async fn rollout_runs_through_job_queue_and_rescores() {
    let h = harness(false).await;
    let (status, body) = post_json(
        &h.base,
        "/api/runs",
        json!({"prompt": "summarise", "success_criteria": ["hello"], "horizon": 4}),
    )
    .await;
    assert_eq!(status, 202, "{body}");
    let run_id = body["run_id"].as_str().unwrap().to_string();
    let run = wait_for_run(&h.base, &run_id, &["completed", "failed"]).await;
    assert_eq!(run["manifest"]["status"], "completed", "{run}");
    assert_eq!(run["trajectory"]["steps"].as_array().unwrap().len(), 4);
    assert!(run["reward"]["terminal_reward"].as_f64().unwrap() > 0.0);
    assert!(run["manifest"]["estimated_cost_usd"].as_f64().unwrap() > 0.0);
    assert_eq!(h.llm_calls.load(Ordering::SeqCst), 2);

    let (_, jobs) = get(&h.base, "/api/jobs").await;
    assert_eq!(jobs[0]["id"], run_id);
    assert_eq!(jobs[0]["status"], "completed");

    let (_, events) = get(&h.base, "/api/events?since=0&limit=100").await;
    let kinds: Vec<&str> = events
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["event"]["kind"].as_str().unwrap())
        .collect();
    assert!(kinds.contains(&"rollout.started"));
    assert!(kinds.contains(&"step.recorded"));
    assert!(kinds.contains(&"reward.computed"));
    assert!(kinds.contains(&"rollout.completed"));

    let (status, rescored) = post_json(
        &h.base,
        &format!("/api/runs/{run_id}/rescore?rubric=coding-v1"),
        json!({}),
    )
    .await;
    assert_eq!(status, 200, "{rescored}");
    assert_eq!(rescored["rubric"], "coding-v1");
    assert_eq!(rescored["persisted"], true);
    let (status, _) = post_json(
        &h.base,
        &format!("/api/runs/{run_id}/rescore?rubric=nope"),
        json!({}),
    )
    .await;
    assert_eq!(status, 400);
    let (status, _) = post_json(
        &h.base,
        "/api/runs/run-missing/rescore?rubric=coding-v1",
        json!({}),
    )
    .await;
    assert_eq!(status, 404);

    // Terminal runs refuse cancellation.
    let (status, _) = post_json(&h.base, &format!("/api/runs/{run_id}/cancel"), json!({})).await;
    assert_eq!(status, 409);
    let (status, _) = post_json(
        &h.base,
        "/api/runs",
        json!({"prompt": "x", "policy_profile": "ghost"}),
    )
    .await;
    assert_eq!(status, 422);
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_interrupts_a_running_rollout() {
    let h = harness(true).await;
    let (_, body) = post_json(
        &h.base,
        "/api/runs",
        json!({"prompt": "slow one", "horizon": 6}),
    )
    .await;
    let run_id = body["run_id"].as_str().unwrap().to_string();
    // Wait until the job is running, then cancel mid-generation.
    for _ in 0..100 {
        let (status, run) = get(&h.base, &format!("/api/runs/{run_id}")).await;
        if status == 200 && run["manifest"]["status"] == "running" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let started = std::time::Instant::now();
    let (status, _) = post_json(&h.base, &format!("/api/runs/{run_id}/cancel"), json!({})).await;
    assert_eq!(status, 202);
    let run = wait_for_run(&h.base, &run_id, &["failed", "completed"]).await;
    assert_eq!(run["manifest"]["status"], "failed");
    assert_eq!(run["trajectory"]["errors"][0], "cancelled");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "cancel must not wait for the provider"
    );
    let (_, job) = get(&h.base, "/api/jobs").await;
    assert_eq!(job[0]["status"], "cancelled");
    let (_, events) = get(&h.base, "/api/events?since=0&limit=100").await;
    assert!(events
        .as_array()
        .unwrap()
        .iter()
        .any(|e| e["event"]["kind"] == "rollout.cancelled"));
}

#[tokio::test(flavor = "multi_thread")]
async fn training_eval_and_adapters() {
    let h = harness(false).await;
    let (_, body) = post_json(
        &h.base,
        "/api/runs",
        json!({"prompt": "sample", "success_criteria": ["hello"], "horizon": 3}),
    )
    .await;
    let run_id = body["run_id"].as_str().unwrap().to_string();
    wait_for_run(&h.base, &run_id, &["completed"]).await;

    let (status, body) = post_json(
        &h.base,
        "/api/training-runs",
        json!({"sample_run_ids": [run_id], "hyperparams": {"steps": 3}}),
    )
    .await;
    assert_eq!(status, 202, "{body}");
    let trun = body["training_run_id"].as_str().unwrap().to_string();
    let mut record = Value::Null;
    for _ in 0..200 {
        let (_, r) = get(&h.base, &format!("/api/training-runs/{trun}")).await;
        if r["status"] == "completed" || r["status"] == "failed" {
            record = r;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(record["status"], "completed", "{record}");
    assert_eq!(record["metrics"].as_array().unwrap().len(), 3);
    let adapter_id = record["adapter_out"].as_str().unwrap().to_string();

    let (_, adapters) = get(&h.base, "/api/adapters").await;
    assert_eq!(adapters[0]["id"], adapter_id);
    let (_, detail) = get(&h.base, &format!("/api/adapters/{adapter_id}")).await;
    assert_eq!(detail["adapter"]["tags"][0], "stub");
    assert!(detail["adapter"]["path"]
        .as_str()
        .unwrap()
        .ends_with(&adapter_id));

    let (status, report) = post_json(&h.base, &format!("/api/adapters/{adapter_id}/eval"), json!({"tasks": [{"id": "t1", "prompt": "read the readme", "success_criteria": ["hello"], "horizon": 3}]})).await;
    assert_eq!(status, 202, "{report}");
    assert_eq!(report["per_task"][0]["task_id"], "t1");
    assert!(report["mean_reward"].as_f64().unwrap() > 0.0);
    let (_, detail) = get(&h.base, &format!("/api/adapters/{adapter_id}")).await;
    assert!(detail["adapter"]["eval_score"].as_f64().is_some());
    assert_eq!(detail["eval_reports"].as_array().unwrap().len(), 1);
    let (status, _) = post_json(&h.base, "/api/adapters/adapter-missing/eval", json!({})).await;
    assert_eq!(status, 404);

    // Lineage: a child trained from this adapter shows up under it.
    let (_, body) = post_json(
        &h.base,
        "/api/training-runs",
        json!({"sample_run_ids": [], "parent_adapter_id": adapter_id, "hyperparams": {"steps": 1}}),
    )
    .await;
    let child_trun = body["training_run_id"].as_str().unwrap().to_string();
    for _ in 0..200 {
        let (_, r) = get(&h.base, &format!("/api/training-runs/{child_trun}")).await;
        if r["status"] == "completed" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let (_, detail) = get(&h.base, &format!("/api/adapters/{adapter_id}")).await;
    assert_eq!(detail["children"].as_array().unwrap().len(), 1);
    let (_, training_runs) = get(&h.base, "/api/training-runs").await;
    assert_eq!(training_runs.as_array().unwrap().len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn budget_blocks_rollouts() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("README.md"), "x").unwrap();
    let mut settings = Settings::for_tests(dir.path());
    settings.daily_budget_usd = 0.000001;
    let state = build_app(settings).await.unwrap();
    // Seed spend above the cap.
    let mut seeded = horizon_core::models::RolloutRequest::new("seed");
    seeded.horizon = Some(1);
    let cancel = CancellationToken::new();
    let detail = state
        .coordinator
        .execute(&seeded, "run-seed", cancel.clone())
        .await
        .unwrap();
    // Provider is unreachable so this failed, but the cost row exists; bump it directly.
    let mut paid = detail.clone();
    paid.manifest.estimated_cost_usd = 1.0;
    state.store.save_run(&paid).unwrap();
    let blocked = state
        .coordinator
        .execute(&seeded, "run-blocked", cancel)
        .await
        .unwrap();
    assert_eq!(
        blocked.manifest.status,
        horizon_core::models::RunStatus::Failed
    );
    assert!(blocked.trajectory.errors[0].contains("daily budget exceeded"));
    let events = state.bus.recent();
    assert!(events.iter().any(|e| e.event.kind() == "budget.exceeded"));
}

#[tokio::test(flavor = "multi_thread")]
async fn websocket_replays_then_streams() {
    let h = harness(false).await;
    let (_, body) = post_json(
        &h.base,
        "/api/runs",
        json!({"prompt": "first", "horizon": 2}),
    )
    .await;
    let first = body["run_id"].as_str().unwrap().to_string();
    wait_for_run(&h.base, &first, &["completed", "failed"]).await;

    let ws_url = format!("{}/ws/events?since=0", h.base.replace("http://", "ws://"));
    let (mut ws, _) = tokio_tungstenite::connect_async(ws_url).await.unwrap();
    let mut replayed = Vec::new();
    while let Ok(Some(Ok(msg))) = tokio::time::timeout(Duration::from_millis(300), ws.next()).await
    {
        let v: Value = serde_json::from_str(msg.to_text().unwrap()).unwrap();
        replayed.push(v);
    }
    assert!(
        replayed
            .iter()
            .any(|e| e["kind"] == "rollout.completed" && e["run_id"] == first),
        "replay missing: {replayed:?}"
    );
    assert!(replayed.iter().all(|e| e["seq"].is_number()));

    let (_, body) = post_json(
        &h.base,
        "/api/runs",
        json!({"prompt": "second", "horizon": 2}),
    )
    .await;
    let second = body["run_id"].as_str().unwrap().to_string();
    let mut saw_live = false;
    while let Ok(Some(Ok(msg))) = tokio::time::timeout(Duration::from_secs(5), ws.next()).await {
        let v: Value = serde_json::from_str(msg.to_text().unwrap()).unwrap();
        if v["kind"] == "rollout.completed" && v["run_id"] == second {
            saw_live = true;
            break;
        }
    }
    assert!(saw_live);
}
