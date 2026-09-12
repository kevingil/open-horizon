//! A scripted OpenAI-compatible policy for smoke tests and demos.
//!
//! It plays a small repo-exploration agent: read the README, list files,
//! search, run an allowlisted command, then finish with a summary. Turn
//! selection is driven by how many tool results are already in the
//! conversation, so it works against any horizon. Latency and token
//! counts are synthetic but plausible so dashboards show real movement.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::{
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct MockPolicyConfig {
    pub bind: String,
    /// Per-completion latency in milliseconds (jittered ±40%).
    pub latency_ms: u64,
}

fn script(turn: usize, horizon_hint: usize) -> Value {
    let plan: [(&str, Value); 5] = [
        ("read_file", json!({"path": "README.md"})),
        ("list_files", json!({"path": "."})),
        ("search", json!({"pattern": "horizon", "path": "."})),
        ("run_command", json!({"command": "ls"})),
        (
            "write_file",
            json!({"path": "notes/summary.md", "content": "# Findings\n\nExplored the repository.\n"}),
        ),
    ];
    // Leave the last turn for `finish` when the horizon is short.
    let last_tool_turn = horizon_hint.saturating_sub(2).min(plan.len() - 1);
    if turn > last_tool_turn {
        return json!({"role": "assistant", "content": "Explored the repository: readme, file listing, a search, and a shell command. The README describes an RL platform for long-horizon agents."});
    }
    let (name, args) = &plan[turn];
    json!({"role": "assistant", "content": null, "tool_calls": [{
        "id": format!("call_{turn}"), "type": "function",
        "function": {"name": name, "arguments": args.to_string()}
    }]})
}

fn horizon_from_prompt(messages: &[Value]) -> usize {
    messages
        .iter()
        .filter_map(|m| m.get("content").and_then(Value::as_str))
        .find_map(|c| {
            c.split("Horizon: up to ")
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|n| n.parse().ok())
        })
        .unwrap_or(6)
}

pub async fn serve(config: MockPolicyConfig) -> anyhow::Result<()> {
    let calls = Arc::new(AtomicU64::new(0));
    let latency = config.latency_ms;
    let app = Router::new().route(
        "/v1/models",
        get(|| async { Json(json!({"object": "list", "data": [{"id": "mock-policy", "object": "model", "owned_by": "horizon"}]})) }),
    ).route(
        "/v1/chat/completions",
        post(move |Json(body): Json<Value>| {
            let calls = calls.clone();
            async move {
                let n = calls.fetch_add(1, Ordering::SeqCst);
                let jitter = 0.6 + ((n * 7919) % 80) as f64 / 100.0;
                tokio::time::sleep(Duration::from_millis((latency as f64 * jitter) as u64)).await;
                let messages = body["messages"].as_array().cloned().unwrap_or_default();
                let tool_results = messages.iter().filter(|m| m["role"] == "tool").count();
                let horizon = horizon_from_prompt(&messages);
                let msg = script(tool_results, horizon);
                let prompt_tokens = 180 + 95 * tool_results as u64 + (n % 13);
                let completion_tokens = if msg["tool_calls"].is_array() { 28 + (n % 9) } else { 64 + (n % 21) };
                Json(json!({
                    "id": format!("chatcmpl-mock-{n}"),
                    "object": "chat.completion",
                    "model": body["model"],
                    "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "prompt_tokens_details": {"cached_tokens": if tool_results > 0 { 120 } else { 0 }}
                    }
                }))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind(&config.bind).await?;
    tracing::info!(bind = %config.bind, "mock-policy.serve");
    axum::serve(listener, app).await?;
    Ok(())
}
