//! Supervised Python worker for the parts of the stack that stay Python:
//! verifiers environments, prime-rl and GRPO trainers, and tokenizer
//! access. Rust owns the process, the protocol, and the restart policy;
//! Python owns nothing but the research code.
//!
//! Protocol: newline-delimited JSON over stdin/stdout.
//!
//! ```text
//! -> {"id": "r1", "op": "verifiers.rollout", "params": {...}}
//! <- {"id": "r1", "event": "metric", "data": {...}}      (zero or more)
//! <- {"id": "r1", "ok": true, "result": {...}}
//! <- {"id": "r1", "ok": false, "error": "..."}
//! ```
//!
//! Requests are multiplexed by id so one worker can run several
//! verifiers rollouts concurrently (verifiers is asyncio-based). When
//! the process dies, every in-flight request fails with `Died` and the
//! next call respawns it.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, Mutex};

#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("failed to spawn python bridge worker: {0}")]
    Spawn(std::io::Error),
    #[error("bridge worker exited while a request was in flight")]
    Died,
    #[error("bridge request failed: {0}")]
    Remote(String),
    #[error("bridge protocol error: {0}")]
    Protocol(String),
    #[error("bridge request timed out after {0:?}")]
    Timeout(Duration),
}

#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Program plus arguments, e.g. `["python", "-m", "horizon_bridge.worker"]`.
    pub command: Vec<String>,
    pub cwd: PathBuf,
    /// Extra environment for the worker (PYTHONPATH and friends).
    pub env: Vec<(String, String)>,
}

impl BridgeConfig {
    pub fn python_module(python: &str, cwd: PathBuf, src_dir: PathBuf) -> Self {
        Self {
            command: vec![
                python.to_string(),
                "-u".into(),
                "-m".into(),
                "horizon_bridge.worker".into(),
            ],
            cwd,
            env: vec![("PYTHONPATH".into(), src_dir.display().to_string())],
        }
    }
}

enum Incoming {
    Event(Value),
    Done(Result<Value, BridgeError>),
}

struct Worker {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    pending: Arc<Mutex<HashMap<String, mpsc::UnboundedSender<Incoming>>>>,
    alive: Arc<AtomicBool>,
}

pub struct PythonBridge {
    config: BridgeConfig,
    worker: Mutex<Option<Arc<Worker>>>,
    next_id: AtomicU64,
}

impl std::fmt::Debug for PythonBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PythonBridge")
            .field("command", &self.config.command)
            .finish()
    }
}

impl PythonBridge {
    pub fn new(config: BridgeConfig) -> Self {
        Self {
            config,
            worker: Mutex::new(None),
            next_id: AtomicU64::new(1),
        }
    }

    /// Run one request to completion, forwarding streamed events to
    /// `on_event`. Spawns the worker on first use or after a crash.
    pub async fn call(
        &self,
        op: &str,
        params: Value,
        timeout: Option<Duration>,
        mut on_event: impl FnMut(&str, Value),
    ) -> Result<Value, BridgeError> {
        let worker = self.ensure_worker().await?;
        let id = format!("r{}", self.next_id.fetch_add(1, Ordering::SeqCst));
        let (tx, mut rx) = mpsc::unbounded_channel();
        worker.pending.lock().await.insert(id.clone(), tx);

        let line = json!({"id": id, "op": op, "params": params}).to_string();
        {
            let mut stdin = worker.stdin.lock().await;
            if let Err(e) = async {
                stdin.write_all(line.as_bytes()).await?;
                stdin.write_all(b"\n").await?;
                stdin.flush().await
            }
            .await
            {
                worker.pending.lock().await.remove(&id);
                tracing::warn!(error = %e, "bridge stdin write failed; restarting worker");
                self.retire(&worker).await;
                return Err(BridgeError::Died);
            }
        }

        let deadline = timeout.map(|t| tokio::time::Instant::now() + t);
        loop {
            let next = match deadline {
                Some(d) => match tokio::time::timeout_at(d, rx.recv()).await {
                    Ok(v) => v,
                    Err(_) => {
                        worker.pending.lock().await.remove(&id);
                        return Err(BridgeError::Timeout(timeout.unwrap_or_default()));
                    }
                },
                None => rx.recv().await,
            };
            match next {
                Some(Incoming::Event(v)) => {
                    let name = v
                        .get("event")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    on_event(&name, v.get("data").cloned().unwrap_or(Value::Null));
                }
                Some(Incoming::Done(result)) => return result,
                None => {
                    self.retire(&worker).await;
                    return Err(BridgeError::Died);
                }
            }
        }
    }

    pub async fn ping(&self, timeout: Duration) -> Result<Value, BridgeError> {
        self.call("ping", json!({}), Some(timeout), |_, _| {}).await
    }

    pub async fn shutdown(&self) {
        // Take the handle out first so the slot lock is released before
        // `retire` acquires it again.
        let taken = self.worker.lock().await.take();
        if let Some(w) = taken {
            self.retire(&w).await;
        }
    }

    async fn ensure_worker(&self) -> Result<Arc<Worker>, BridgeError> {
        let mut slot = self.worker.lock().await;
        if let Some(w) = slot.as_ref() {
            if w.alive.load(Ordering::SeqCst) {
                return Ok(w.clone());
            }
        }
        let worker = self.spawn().await?;
        *slot = Some(worker.clone());
        Ok(worker)
    }

    async fn retire(&self, worker: &Arc<Worker>) {
        let mut slot = self.worker.lock().await;
        if slot.as_ref().is_some_and(|w| Arc::ptr_eq(w, worker)) {
            *slot = None;
        }
        // Fail whoever is still waiting; dropping the senders closes their channels.
        worker.pending.lock().await.clear();
        // The child is killed on drop; make it explicit for clarity.
        let _ = worker.child_kill().await;
    }

    async fn spawn(&self) -> Result<Arc<Worker>, BridgeError> {
        let (program, args) =
            self.config.command.split_first().ok_or_else(|| {
                BridgeError::Spawn(std::io::Error::other("bridge command is empty"))
            })?;
        let mut cmd = Command::new(program);
        cmd.args(args)
            .current_dir(&self.config.cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        for (k, v) in &self.config.env {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().map_err(BridgeError::Spawn)?;
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");
        let pending: Arc<Mutex<HashMap<String, mpsc::UnboundedSender<Incoming>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let reader_pending = pending.clone();
        let alive = Arc::new(AtomicBool::new(true));
        let reader_alive = alive.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            loop {
                match lines.next_line().await {
                    Ok(Some(line)) => {
                        if line.trim().is_empty() {
                            continue;
                        }
                        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                            tracing::warn!(line = %line.chars().take(200).collect::<String>(), "bridge emitted non-JSON line");
                            continue;
                        };
                        let Some(id) = msg.get("id").and_then(Value::as_str).map(str::to_string)
                        else {
                            continue;
                        };
                        let mut map = reader_pending.lock().await;
                        if msg.get("event").is_some() {
                            if let Some(tx) = map.get(&id) {
                                let _ = tx.send(Incoming::Event(msg));
                            }
                        } else if let Some(tx) = map.remove(&id) {
                            let result = if msg.get("ok").and_then(Value::as_bool).unwrap_or(false)
                            {
                                Ok(msg.get("result").cloned().unwrap_or(Value::Null))
                            } else {
                                Err(BridgeError::Remote(
                                    msg.get("error")
                                        .and_then(Value::as_str)
                                        .unwrap_or("unknown error")
                                        .to_string(),
                                ))
                            };
                            let _ = tx.send(Incoming::Done(result));
                        }
                    }
                    Ok(None) | Err(_) => {
                        tracing::warn!("bridge worker stdout closed");
                        reader_alive.store(false, Ordering::SeqCst);
                        reader_pending.lock().await.clear();
                        break;
                    }
                }
            }
        });
        tracing::info!(command = ?self.config.command, "bridge worker started");
        Ok(Arc::new(Worker {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            pending,
            alive,
        }))
    }
}

impl Worker {
    async fn child_kill(&self) -> std::io::Result<()> {
        self.alive.store(false, Ordering::SeqCst);
        let mut child = self.child.lock().await;
        let _ = child.start_kill();
        let _ = child.wait().await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo_worker() -> (tempfile::TempDir, BridgeConfig) {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("w.py");
        std::fs::write(
            &script,
            r#"
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    op, rid, params = req["op"], req["id"], req["params"]
    if op == "ping":
        print(json.dumps({"id": rid, "ok": True, "result": {"pong": True}}), flush=True)
    elif op == "stream":
        for i in range(3):
            print(json.dumps({"id": rid, "event": "metric", "data": {"step": i}}), flush=True)
        print(json.dumps({"id": rid, "ok": True, "result": params}), flush=True)
    elif op == "fail":
        print(json.dumps({"id": rid, "ok": False, "error": "nope"}), flush=True)
    elif op == "die":
        sys.exit(3)
"#,
        )
        .unwrap();
        let cfg = BridgeConfig {
            command: vec!["python3".into(), "-u".into(), script.display().to_string()],
            cwd: dir.path().to_path_buf(),
            env: vec![],
        };
        (dir, cfg)
    }

    #[tokio::test]
    async fn round_trip_stream_error_and_restart() {
        let (_dir, cfg) = echo_worker();
        let bridge = PythonBridge::new(cfg);
        let pong = bridge.ping(Duration::from_secs(5)).await.unwrap();
        assert_eq!(pong["pong"], true);

        let mut events = Vec::new();
        let result = bridge
            .call(
                "stream",
                json!({"x": 1}),
                Some(Duration::from_secs(5)),
                |name, data| {
                    events.push((name.to_string(), data));
                },
            )
            .await
            .unwrap();
        assert_eq!(result["x"], 1);
        assert_eq!(events.len(), 3);
        assert_eq!(events[2].1["step"], 2);

        let err = bridge
            .call("fail", json!({}), Some(Duration::from_secs(5)), |_, _| {})
            .await
            .unwrap_err();
        assert!(matches!(err, BridgeError::Remote(ref m) if m == "nope"));

        let err = bridge
            .call("die", json!({}), Some(Duration::from_secs(5)), |_, _| {})
            .await
            .unwrap_err();
        assert!(matches!(err, BridgeError::Died), "{err:?}");
        // Next call respawns.
        let pong = bridge.ping(Duration::from_secs(5)).await.unwrap();
        assert_eq!(pong["pong"], true);
        bridge.shutdown().await;
    }
}
