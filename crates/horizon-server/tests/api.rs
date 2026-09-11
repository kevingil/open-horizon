//! End-to-end: HTTP API + job runner + scripted mock policy.

use std::time::Duration;

use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use horizon_server::jobs::JobRunner;
use horizon_server::mock_policy::{self, MockPolicyConfig};
use horizon_server::{api, build_app, Settings};

struct Harness {
    base: String,
    _dir: tempfile::TempDir,
    shutdown: CancellationToken,
}

async fn mock_llm(latency_ms: u64) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let bind = addr.to_string();
    tokio::spawn(async move {
        mock_policy::serve(MockPolicyConfig { bind, latency_ms })
            .await
            .unwrap()
    });
    // Wait for the port to accept connections.
    for _ in 0..50 {
        if tokio::net::TcpStream::connect(addr).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    format!("http://{addr}/v1")
}

async fn harness(latency_ms: u64) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("README.md"),
        "hello from the readme, an rl platform for long-horizon agents",
    )
    .unwrap();
    let mut settings = Settings::for_tests(dir.path());
    settings.llm_base_url = mock_llm(latency_ms).await;
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

const TERMINAL: &[&str] = &["completed", "failed", "cancelled"];

async fn wait_for_run(base: &str, run_id: &str) -> Value {
    for _ in 0..400 {
        let (status, run) = get(base, &format!("/api/runs/{run_id}")).await;
        if status == 200 && TERMINAL.contains(&run["manifest"]["status"].as_str().unwrap_or("")) {
            return run;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("run {run_id} never became terminal");
}

async fn wait_for_training(base: &str, id: &str) -> Value {
    for _ in 0..400 {
        let (_, r) = get(base, &format!("/api/training-runs/{id}")).await;
        if ["completed", "failed", "cancelled"].contains(&r["status"].as_str().unwrap_or("")) {
            return r;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("training run {id} never became terminal");
}

#[tokio::test(flavor = "multi_thread")]
async fn health_config_profiles_rubrics_openapi() {
    let h = harness(0).await;
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
    assert!(openapi["components"]["schemas"]["EventEnvelope"].is_object());
    let (_, dash) = get(&h.base, "/api/dashboard").await;
    assert!(dash["jobs"]["queued"].is_number());
}

#[tokio::test(flavor = "multi_thread")]
async fn rollout_runs_through_job_queue_and_rescores() {
    let h = harness(0).await;
    let (status, body) = post_json(
        &h.base,
        "/api/runs",
        json!({"prompt": "summarise", "success_criteria": ["readme"], "horizon": 4}),
    )
    .await;
    assert_eq!(status, 202, "{body}");
    let run_id = body["run_id"].as_str().unwrap().to_string();
    let run = wait_for_run(&h.base, &run_id).await;
    assert_eq!(run["manifest"]["status"], "completed", "{run}");
    assert_eq!(run["manifest"]["profile"], "default");
    assert_eq!(run["manifest"]["horizon"], 4);
    assert_eq!(run["manifest"]["step_count"], 8, "{run}");
    assert_eq!(run["trajectory"]["steps"].as_array().unwrap().len(), 8);
    assert!(run["manifest"]["tokens"].as_u64().unwrap() > 0);
    assert!(run["manifest"]["cost_usd"].as_f64().unwrap() > 0.0);
    assert!(run["reward"]["terminal_reward"].as_f64().unwrap() > 0.0);
    assert_eq!(run["reward"]["rubric"], "heuristic-v1");
    assert!(run["reward"]["signals"]
        .as_array()
        .unwrap()
        .iter()
        .any(|s| s["name"] == "finish" && s["value"] == 1.0));
    assert_eq!(
        run["manifest"]["terminal_reward"],
        run["reward"]["terminal_reward"]
    );

    let (_, jobs) = get(&h.base, "/api/jobs").await;
    assert_eq!(jobs[0]["id"], run_id);
    assert_eq!(jobs[0]["status"], "completed");

    let (_, events) = get(&h.base, "/api/events?since=0&limit=200").await;
    let kinds: Vec<&str> = events
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["kind"].as_str().unwrap())
        .collect();
    for expected in [
        "rollout.queued",
        "rollout.started",
        "step.recorded",
        "progress.ticked",
        "reward.computed",
        "rollout.completed",
    ] {
        assert!(kinds.contains(&expected), "missing {expected} in {kinds:?}");
    }
    let step_event = events
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["kind"] == "step.recorded")
        .unwrap();
    assert_eq!(step_event["subject"]["kind"], "run");
    assert_eq!(step_event["subject"]["id"], run_id);
    assert!(step_event["payload"]["step"]["content"].is_string());

    let (status, rescored) = post_json(
        &h.base,
        &format!("/api/runs/{run_id}/rescore?rubric=coding-v1"),
        json!({}),
    )
    .await;
    assert_eq!(status, 200, "{rescored}");
    assert_eq!(rescored["rubric"], "coding-v1");
    assert_eq!(rescored["reward"]["rubric"], "coding-v1");
    assert!(rescored["delta"].is_number());
    assert_eq!(rescored["persisted"], true);
    let (_, run) = get(&h.base, &format!("/api/runs/{run_id}")).await;
    assert_eq!(run["reward"]["rubric"], "coding-v1");
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
    let h = harness(3000).await;
    let (_, body) = post_json(
        &h.base,
        "/api/runs",
        json!({"prompt": "slow one", "horizon": 6}),
    )
    .await;
    let run_id = body["run_id"].as_str().unwrap().to_string();
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
    let run = wait_for_run(&h.base, &run_id).await;
    assert_eq!(run["manifest"]["status"], "cancelled", "{run}");
    assert!(run["manifest"]["error"].is_null());
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "cancel must not wait for the provider"
    );
    let (_, job) = get(&h.base, "/api/jobs").await;
    assert_eq!(job[0]["status"], "cancelled", "{job}");
    let (_, events) = get(&h.base, "/api/events?since=0&limit=200").await;
    assert!(events
        .as_array()
        .unwrap()
        .iter()
        .any(|e| e["kind"] == "rollout.cancelled"));
}

#[tokio::test(flavor = "multi_thread")]
async fn queued_run_cancels_immediately() {
    let h = harness(3000).await;
    // Saturate both rollout slots, then queue a third and cancel it.
    for _ in 0..2 {
        post_json(
            &h.base,
            "/api/runs",
            json!({"prompt": "busy", "horizon": 6}),
        )
        .await;
    }
    let (_, body) = post_json(
        &h.base,
        "/api/runs",
        json!({"prompt": "waiting", "horizon": 6}),
    )
    .await;
    let run_id = body["run_id"].as_str().unwrap().to_string();
    let (_, run) = get(&h.base, &format!("/api/runs/{run_id}")).await;
    assert_eq!(run["manifest"]["status"], "queued");
    let (status, _) = post_json(&h.base, &format!("/api/runs/{run_id}/cancel"), json!({})).await;
    assert_eq!(status, 202);
    let (_, run) = get(&h.base, &format!("/api/runs/{run_id}")).await;
    assert_eq!(run["manifest"]["status"], "cancelled");
    let (_, job) = get(&h.base, "/api/jobs").await;
    let mine = job
        .as_array()
        .unwrap()
        .iter()
        .find(|j| j["id"] == run_id)
        .unwrap();
    assert_eq!(mine["status"], "cancelled");
}

#[tokio::test(flavor = "multi_thread")]
async fn training_eval_and_adapters() {
    let h = harness(0).await;
    let (_, body) = post_json(
        &h.base,
        "/api/runs",
        json!({"prompt": "sample", "success_criteria": ["readme"], "horizon": 3}),
    )
    .await;
    let run_id = body["run_id"].as_str().unwrap().to_string();
    assert_eq!(
        wait_for_run(&h.base, &run_id).await["manifest"]["status"],
        "completed"
    );

    let (status, body) = post_json(
        &h.base,
        "/api/training-runs",
        json!({"sample_run_ids": [run_id], "hyperparams": {"steps": 3}}),
    )
    .await;
    assert_eq!(status, 202, "{body}");
    let trun = body["training_run_id"].as_str().unwrap().to_string();
    let record = wait_for_training(&h.base, &trun).await;
    assert_eq!(record["status"], "completed", "{record}");
    assert_eq!(record["trainer"], "stub-trainer-v1");
    assert_eq!(record["metrics"].as_array().unwrap().len(), 3);
    let adapter_id = record["adapter_out"].as_str().unwrap().to_string();

    let (_, adapters) = get(&h.base, "/api/adapters").await;
    assert_eq!(adapters[0]["id"], adapter_id);
    let (_, detail) = get(&h.base, &format!("/api/adapters/{adapter_id}")).await;
    assert_eq!(detail["adapter"]["tags"][0], "stub");

    let (status, report) = post_json(&h.base, &format!("/api/adapters/{adapter_id}/eval"), json!({"tasks": [{"id": "t1", "prompt": "read the readme", "success_criteria": ["readme"], "horizon": 3}]})).await;
    assert_eq!(status, 200, "{report}");
    assert_eq!(report["per_task"][0]["task_id"], "t1");
    assert!(report["per_task"][0]["run_id"].is_string());
    assert!(report["mean_reward"].as_f64().unwrap() > 0.0);
    let eval_run = report["per_task"][0]["run_id"].as_str().unwrap();
    let (_, run) = get(&h.base, &format!("/api/runs/{eval_run}")).await;
    assert_eq!(run["manifest"]["adapter_id"], adapter_id);
    assert_eq!(run["manifest"]["infra_target"], "eval");
    let (_, detail) = get(&h.base, &format!("/api/adapters/{adapter_id}")).await;
    assert!(detail["adapter"]["eval_score"].as_f64().is_some());
    assert_eq!(detail["eval_reports"].as_array().unwrap().len(), 1);
    let (status, _) = post_json(&h.base, "/api/adapters/adapter-missing/eval", json!({})).await;
    assert_eq!(status, 404);

    let (_, body) = post_json(
        &h.base,
        "/api/training-runs",
        json!({"sample_run_ids": [], "parent_adapter_id": adapter_id, "hyperparams": {"steps": 1}}),
    )
    .await;
    let child = wait_for_training(&h.base, body["training_run_id"].as_str().unwrap()).await;
    assert_eq!(child["status"], "completed");
    let (_, detail) = get(&h.base, &format!("/api/adapters/{adapter_id}")).await;
    assert_eq!(detail["children"].as_array().unwrap().len(), 1);
    let (_, training_runs) = get(&h.base, "/api/training-runs").await;
    assert_eq!(training_runs.as_array().unwrap().len(), 2);
    let (_, events) = get(&h.base, "/api/events?since=0&limit=500").await;
    let kinds: Vec<&str> = events
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["kind"].as_str().unwrap())
        .collect();
    for expected in [
        "training.queued",
        "training.started",
        "training.metric",
        "training.completed",
        "adapter.published",
        "eval.completed",
    ] {
        assert!(kinds.contains(&expected), "missing {expected}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn budget_blocks_rollouts() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("README.md"), "x").unwrap();
    let mut settings = Settings::for_tests(dir.path());
    settings.daily_budget_usd = 0.000001;
    let state = build_app(settings).await.unwrap();
    let mut request = horizon_core::models::RolloutRequest::new("seed");
    request.horizon = Some(1);
    let cancel = CancellationToken::new();
    let seed = state.coordinator.submit_inline(&request).unwrap();
    let detail = state
        .coordinator
        .execute(&request, &seed, cancel.clone())
        .await
        .unwrap();
    // Provider is unreachable so this failed; bump its cost directly to trip the cap.
    let mut paid = detail.clone();
    paid.manifest.cost_usd = 1.0;
    state.store.save_run(&paid).unwrap();
    let blocked_id = state.coordinator.submit_inline(&request).unwrap();
    let blocked = state
        .coordinator
        .execute(&request, &blocked_id, cancel)
        .await
        .unwrap();
    assert_eq!(
        blocked.manifest.status,
        horizon_core::models::RunStatus::Failed
    );
    assert!(blocked
        .manifest
        .error
        .as_deref()
        .unwrap()
        .contains("daily budget exceeded"));
    assert!(state
        .bus
        .recent()
        .iter()
        .any(|e| e.event.kind() == "budget.exceeded"));
}

#[tokio::test(flavor = "multi_thread")]
async fn websocket_replays_then_streams() {
    let h = harness(0).await;
    let (_, body) = post_json(
        &h.base,
        "/api/runs",
        json!({"prompt": "first", "horizon": 2}),
    )
    .await;
    let first = body["run_id"].as_str().unwrap().to_string();
    wait_for_run(&h.base, &first).await;

    let ws_url = format!("{}/ws/events?since=0", h.base.replace("http://", "ws://"));
    let (mut ws, _) = tokio_tungstenite::connect_async(ws_url).await.unwrap();
    let mut replayed = Vec::new();
    while let Ok(Some(Ok(msg))) = tokio::time::timeout(Duration::from_millis(300), ws.next()).await
    {
        replayed.push(serde_json::from_str::<Value>(msg.to_text().unwrap()).unwrap());
    }
    assert!(
        replayed
            .iter()
            .any(|e| e["kind"] == "rollout.completed" && e["subject"]["id"] == first),
        "replay missing: {replayed:?}"
    );
    assert!(replayed
        .iter()
        .all(|e| e["seq"].is_number() && e["at"].is_string()));

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
        if v["kind"] == "rollout.completed" && v["subject"]["id"] == second {
            saw_live = true;
            break;
        }
    }
    assert!(saw_live);
}
