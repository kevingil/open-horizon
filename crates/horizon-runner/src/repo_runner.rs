//! Environment runner backed by a real workspace snapshot.
//!
//! The policy emits a JSON tool call; this runner executes it against a
//! per-task sandboxed tempdir and returns the observation as JSON. Path
//! allowlist, command allowlist, wall-clock timeout, byte cap.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Map, Value};
use tokio::sync::Mutex;

use horizon_core::models::TaskSpec;
use horizon_core::tools::{is_allowed_command, is_known_tool, parse_action};

use crate::sandbox::Sandbox;
use crate::snapshot::snapshot_repo;

/// Cap written files so a runaway policy can't fill disk.
const MAX_WRITE_BYTES: usize = 256_000;

#[derive(Debug, Clone)]
pub struct RepoRunnerConfig {
    pub source_root: PathBuf,
    pub scratch_root: PathBuf,
    pub command_timeout: Duration,
    pub max_output_bytes: usize,
}

#[derive(Debug)]
struct TaskState {
    workspace: PathBuf,
    finished: bool,
}

#[derive(Debug, Clone)]
pub struct RepoRunner {
    config: Arc<RepoRunnerConfig>,
    sandbox: Sandbox,
    states: Arc<Mutex<HashMap<String, TaskState>>>,
}

impl RepoRunner {
    pub fn new(config: RepoRunnerConfig, sandbox: Sandbox) -> std::io::Result<Self> {
        std::fs::create_dir_all(&config.scratch_root)?;
        let config = RepoRunnerConfig {
            source_root: config.source_root.canonicalize()?,
            scratch_root: config.scratch_root.canonicalize()?,
            ..config
        };
        Ok(Self {
            config: Arc::new(config),
            sandbox,
            states: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn sandbox_name(&self) -> &'static str {
        self.sandbox.name()
    }

    /// Snapshot the source repo into a fresh workspace for this task.
    pub async fn create_task(&self, task: &TaskSpec) -> std::io::Result<PathBuf> {
        let workspace = tempfile::Builder::new()
            .prefix(&format!("rl-{}-", task.id))
            .tempdir_in(&self.config.scratch_root)?
            .keep();
        snapshot_repo(&self.config.source_root, &workspace).await?;
        self.states.lock().await.insert(
            task.id.clone(),
            TaskState {
                workspace: workspace.clone(),
                finished: false,
            },
        );
        Ok(workspace)
    }

    /// Delete the task's workspace. Best-effort.
    pub async fn cleanup_task(&self, task_id: &str) {
        if let Some(state) = self.states.lock().await.remove(task_id) {
            let _ = tokio::fs::remove_dir_all(&state.workspace).await;
        }
    }

    pub async fn step(&self, task_id: &str, action: &str) -> String {
        let workspace = {
            let states = self.states.lock().await;
            match states.get(task_id) {
                None => return err("unknown task"),
                Some(s) if s.finished => return err("task already finished"),
                Some(s) => s.workspace.clone(),
            }
        };
        let preview: String = action.chars().take(160).collect();
        let Some(call) = parse_action(action) else {
            return err(&format!("could not parse action: {preview}"));
        };
        if !is_known_tool(&call.tool) {
            return err(&format!("unknown tool: {}", call.tool));
        }
        let result = match call.tool.as_str() {
            "read_file" => self.read_file(&workspace, &call.input),
            "list_files" => self.list_files(&workspace, &call.input),
            "search" => self.search(&workspace, &call.input).await,
            "run_command" => self.run_command(&workspace, &call.input).await,
            "write_file" => self.write_file(&workspace, &call.input),
            "finish" => {
                let summary = call
                    .input
                    .get("summary")
                    .map(value_to_string)
                    .unwrap_or_default();
                if let Some(s) = self.states.lock().await.get_mut(task_id) {
                    s.finished = true;
                }
                Ok(json!({"tool": "finish", "summary": summary}))
            }
            other => Ok(json!({"error": format!("unhandled tool: {other}")})),
        };
        match result {
            Ok(v) => truncate_json(&v, self.config.max_output_bytes),
            Err(e) => err(&format!("{} failed: {e}", call.tool)),
        }
    }

    fn resolve(&self, workspace: &Path, rel: &str) -> Result<PathBuf, String> {
        let joined = workspace.join(rel);
        // Resolve symlinks and `..` for the existing prefix; the final
        // component may not exist yet (write_file), so normalise it lexically.
        let target = normalise(&joined, workspace)?;
        if !target.starts_with(workspace) {
            return Err(format!("path escapes workspace: {rel}"));
        }
        Ok(target)
    }

    fn read_file(&self, workspace: &Path, args: &Map<String, Value>) -> Result<Value, String> {
        let path = arg_str(args, "path", "");
        let target = self.resolve(workspace, &path)?;
        if !target.is_file() {
            return Ok(json!({"tool": "read_file", "path": path, "error": "not a file"}));
        }
        let bytes = std::fs::read(&target).map_err(|e| e.to_string())?;
        Ok(json!({"tool": "read_file", "path": path, "content": String::from_utf8_lossy(&bytes)}))
    }

    fn list_files(&self, workspace: &Path, args: &Map<String, Value>) -> Result<Value, String> {
        let path = arg_str(args, "path", ".");
        let target = self.resolve(workspace, &path)?;
        if !target.exists() {
            return Ok(json!({"tool": "list_files", "path": path, "error": "missing"}));
        }
        let mut entries = Vec::new();
        walk(&target, workspace, &mut entries);
        entries.sort();
        entries.truncate(500);
        Ok(json!({"tool": "list_files", "path": path, "entries": entries}))
    }

    async fn search(&self, workspace: &Path, args: &Map<String, Value>) -> Result<Value, String> {
        let pattern = arg_str(args, "pattern", "");
        let path = arg_str(args, "path", ".");
        let target = self.resolve(workspace, &path)?;
        // Run relative to the workspace so matches carry workspace-relative paths.
        let rel = target
            .strip_prefix(workspace)
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let rel = if rel.is_empty() { ".".to_string() } else { rel };
        let argv: Vec<String> = if which("rg") {
            vec![
                "rg".into(),
                "--no-heading".into(),
                "--line-number".into(),
                "--color".into(),
                "never".into(),
                "-e".into(),
                pattern.clone(),
                rel,
            ]
        } else {
            vec![
                "grep".into(),
                "-rn".into(),
                "-e".into(),
                pattern.clone(),
                rel,
            ]
        };
        // Search runs on the host: it is read-only against a path-checked workspace.
        let out = Sandbox::host()
            .run(workspace, &argv, self.config.command_timeout)
            .await;
        let matches: Vec<&str> = out.stdout.lines().take(500).collect();
        Ok(
            json!({"tool": "search", "pattern": pattern, "path": path, "matches": matches, "returncode": out.returncode}),
        )
    }

    async fn run_command(
        &self,
        workspace: &Path,
        args: &Map<String, Value>,
    ) -> Result<Value, String> {
        let command = arg_str(args, "command", "").trim().to_string();
        let parts = match shell_split(&command) {
            Ok(p) => p,
            Err(e) => {
                return Ok(json!({"tool": "run_command", "error": format!("parse error: {e}")}))
            }
        };
        let Some(first) = parts.first() else {
            return Ok(json!({"tool": "run_command", "error": "empty command"}));
        };
        if !is_allowed_command(first) {
            return Ok(json!({
                "tool": "run_command",
                "error": format!("command not allowed: {first}"),
                "allowlist": sorted_allowlist(),
            }));
        }
        let result = self
            .sandbox
            .run(workspace, &parts, self.config.command_timeout)
            .await;
        if result.timed_out {
            return Ok(json!({"tool": "run_command", "error": "timeout", "command": command}));
        }
        Ok(json!({
            "tool": "run_command",
            "command": command,
            "stdout": tail(&result.stdout, self.config.max_output_bytes),
            "stderr": tail(&result.stderr, self.config.max_output_bytes),
            "returncode": result.returncode,
            "sandbox": self.sandbox.name(),
        }))
    }

    fn write_file(&self, workspace: &Path, args: &Map<String, Value>) -> Result<Value, String> {
        let path = arg_str(args, "path", "");
        let Some(Value::String(content)) = args.get("content") else {
            return Ok(json!({"tool": "write_file", "error": "content must be a string"}));
        };
        if content.len() > MAX_WRITE_BYTES {
            return Ok(
                json!({"tool": "write_file", "error": format!("content exceeds {MAX_WRITE_BYTES} bytes")}),
            );
        }
        let target = self.resolve(workspace, &path)?;
        let meta = std::fs::symlink_metadata(&target).ok();
        if meta
            .as_ref()
            .is_some_and(|m| m.file_type().is_symlink() || m.is_dir())
        {
            return Ok(
                json!({"tool": "write_file", "error": "refusing to overwrite symlink or directory"}),
            );
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&target, content).map_err(|e| e.to_string())?;
        Ok(json!({"tool": "write_file", "path": path, "bytes": content.len()}))
    }
}

fn normalise(path: &Path, root: &Path) -> Result<PathBuf, String> {
    // Canonicalise the longest existing prefix so symlinks inside the
    // workspace can't point outside it, then append the rest lexically.
    let mut existing = path.to_path_buf();
    let mut rest: Vec<std::ffi::OsString> = Vec::new();
    while !existing.exists() {
        let Some(name) = existing.file_name().map(|n| n.to_os_string()) else {
            break;
        };
        rest.push(name);
        if !existing.pop() {
            break;
        }
    }
    let mut out = existing.canonicalize().map_err(|e| e.to_string())?;
    for comp in rest.iter().rev() {
        if comp == ".." {
            out.pop();
        } else if comp != "." {
            out.push(comp);
        }
    }
    if !out.starts_with(root) {
        return Err("path escapes workspace".into());
    }
    Ok(out)
}

fn walk(dir: &Path, root: &Path, out: &mut Vec<String>) {
    if dir.is_file() {
        if let Ok(rel) = dir.strip_prefix(root) {
            out.push(rel.display().to_string());
        }
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            walk(&p, root, out);
        } else if p.is_file() {
            if let Ok(rel) = p.strip_prefix(root) {
                out.push(rel.display().to_string());
            }
        }
    }
}

fn which(bin: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|p| p.join(bin).is_file()))
        .unwrap_or(false)
}

fn sorted_allowlist() -> Vec<&'static str> {
    let mut v = horizon_core::tools::COMMAND_ALLOWLIST.to_vec();
    v.sort();
    v
}

fn arg_str(args: &Map<String, Value>, key: &str, default: &str) -> String {
    match args.get(key) {
        None | Some(Value::Null) => default.to_string(),
        Some(v) => value_to_string(v),
    }
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn tail(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut start = text.len() - max_bytes;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    text[start..].to_string()
}

fn err(reason: &str) -> String {
    json!({"error": reason}).to_string()
}

/// Serialize within `max_bytes`, shrinking the payload structurally rather
/// than cutting the JSON text: long strings are clipped and long arrays are
/// shortened, so the policy always receives well-formed, labelled output.
fn truncate_json(data: &Value, max_bytes: usize) -> String {
    let text = data.to_string();
    if text.len() <= max_bytes {
        return text;
    }
    let mut value = data.clone();
    let mut budget = max_bytes;
    for _ in 0..12 {
        shrink(&mut value, budget);
        let text = value.to_string();
        if text.len() <= max_bytes {
            if let Value::Object(map) = &mut value {
                map.insert("_truncated".into(), Value::Bool(true));
            }
            let text = value.to_string();
            if text.len() <= max_bytes {
                return text;
            }
        }
        budget /= 2;
        if budget < 64 {
            break;
        }
    }
    json!({"_truncated": true, "preview": text.chars().take(max_bytes / 4).collect::<String>()})
        .to_string()
}

fn shrink(value: &mut Value, budget: usize) {
    match value {
        Value::String(s) => {
            let keep = budget.saturating_sub(24).max(16);
            if s.len() > keep {
                let mut end = keep;
                while end > 0 && !s.is_char_boundary(end) {
                    end -= 1;
                }
                let dropped = s.len() - end;
                s.truncate(end);
                s.push_str(&format!("…[{dropped} bytes truncated]"));
            }
        }
        Value::Array(items) => {
            // Rough per-item share of the budget; keep a prefix that fits.
            let mut used = 2usize;
            let mut keep = 0usize;
            for item in items.iter() {
                let len = item.to_string().len() + 1;
                if used + len > budget {
                    break;
                }
                used += len;
                keep += 1;
            }
            if keep < items.len() {
                let dropped = items.len() - keep;
                items.truncate(keep);
                items.push(Value::String(format!("…[{dropped} more items truncated]")));
            }
        }
        Value::Object(map) => {
            let per = budget / map.len().max(1);
            for (_, v) in map.iter_mut() {
                shrink(v, per.max(48));
            }
        }
        _ => {}
    }
}

/// POSIX-style shell splitting (quotes and backslashes), no expansion.
pub fn shell_split(input: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_token = false;
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' => {
                in_token = true;
                loop {
                    match chars.next() {
                        Some('\'') => break,
                        Some(ch) => cur.push(ch),
                        None => return Err("No closing quotation".into()),
                    }
                }
            }
            '"' => {
                in_token = true;
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some('\\') => match chars.next() {
                            Some(esc @ ('"' | '\\' | '$' | '`')) => cur.push(esc),
                            Some(other) => {
                                cur.push('\\');
                                cur.push(other);
                            }
                            None => return Err("No escaped character".into()),
                        },
                        Some(ch) => cur.push(ch),
                        None => return Err("No closing quotation".into()),
                    }
                }
            }
            '\\' => match chars.next() {
                Some(ch) => {
                    in_token = true;
                    cur.push(ch);
                }
                None => return Err("No escaped character".into()),
            },
            c if c.is_whitespace() => {
                if in_token {
                    out.push(std::mem::take(&mut cur));
                    in_token = false;
                }
            }
            c => {
                in_token = true;
                cur.push(c);
            }
        }
    }
    if in_token {
        out.push(cur);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn task() -> TaskSpec {
        TaskSpec {
            id: "task-1".into(),
            prompt: "p".into(),
            repo_snapshot: ".".into(),
            horizon: 4,
            success_criteria: vec![],
        }
    }

    async fn runner() -> (tempfile::TempDir, tempfile::TempDir, RepoRunner) {
        let src = tempfile::tempdir().unwrap();
        std::fs::write(src.path().join("README.md"), "hello readme").unwrap();
        std::fs::create_dir_all(src.path().join("pkg")).unwrap();
        std::fs::write(src.path().join("pkg/mod.py"), "import pydantic\n").unwrap();
        let scratch = tempfile::tempdir().unwrap();
        let r = RepoRunner::new(
            RepoRunnerConfig {
                source_root: src.path().to_path_buf(),
                scratch_root: scratch.path().to_path_buf(),
                command_timeout: Duration::from_secs(5),
                max_output_bytes: 16_384,
            },
            Sandbox::host(),
        )
        .unwrap();
        (src, scratch, r)
    }

    #[tokio::test]
    async fn tools_against_snapshot() {
        let (_s, _c, r) = runner().await;
        let t = task();
        r.create_task(&t).await.unwrap();
        let obs = r
            .step(
                "task-1",
                r#"{"tool":"read_file","input":{"path":"README.md"}}"#,
            )
            .await;
        assert!(obs.contains("hello readme"), "{obs}");
        let obs = r
            .step("task-1", r#"{"tool":"list_files","input":{}}"#)
            .await;
        assert!(obs.contains("pkg/mod.py"), "{obs}");
        let obs = r
            .step(
                "task-1",
                r#"{"tool":"search","input":{"pattern":"pydantic"}}"#,
            )
            .await;
        assert!(obs.contains("mod.py"), "{obs}");
        let obs = r
            .step(
                "task-1",
                r#"{"tool":"run_command","input":{"command":"ls pkg"}}"#,
            )
            .await;
        assert!(obs.contains("mod.py"), "{obs}");
        let obs = r
            .step(
                "task-1",
                r#"{"tool":"run_command","input":{"command":"rm -rf /"}}"#,
            )
            .await;
        assert!(obs.contains("command not allowed"), "{obs}");
        let obs = r
            .step(
                "task-1",
                r#"{"tool":"read_file","input":{"path":"../../etc/passwd"}}"#,
            )
            .await;
        assert!(obs.contains("escapes workspace"), "{obs}");
        let obs = r
            .step(
                "task-1",
                r#"{"tool":"write_file","input":{"path":"new/file.txt","content":"x"}}"#,
            )
            .await;
        assert!(obs.contains("\"bytes\":1"), "{obs}");
        let obs = r
            .step("task-1", r#"{"tool":"finish","input":{"summary":"ok"}}"#)
            .await;
        assert!(obs.contains("\"summary\":\"ok\""), "{obs}");
        let obs = r.step("task-1", r#"{"tool":"finish","input":{}}"#).await;
        assert!(obs.contains("already finished"), "{obs}");
        assert!(r.step("nope", "{}").await.contains("unknown task"));
    }

    #[tokio::test]
    async fn rejects_unparseable_actions() {
        let (_s, _c, r) = runner().await;
        r.create_task(&task()).await.unwrap();
        assert!(r
            .step("task-1", "not json")
            .await
            .contains("could not parse"));
        assert!(r
            .step("task-1", r#"{"tool":"teleport","input":{}}"#)
            .await
            .contains("unknown tool"));
    }

    #[test]
    fn truncation_keeps_json_well_formed() {
        let big = json!({"tool": "search", "matches": (0..500).map(|i| format!("file{i}.rs:1: horizon")).collect::<Vec<_>>(), "pattern": "horizon"});
        let out = truncate_json(&big, 2000);
        assert!(out.len() <= 2000);
        let parsed: Value = serde_json::from_str(&out).expect("well-formed JSON");
        assert_eq!(parsed["_truncated"], true);
        assert_eq!(parsed["tool"], "search");
        assert!(parsed["matches"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()
            .as_str()
            .unwrap()
            .contains("truncated"));
        let text = json!({"tool": "read_file", "content": "x".repeat(50_000)});
        let out = truncate_json(&text, 1000);
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert!(parsed["content"]
            .as_str()
            .unwrap()
            .contains("bytes truncated"));
    }

    #[test]
    fn shell_split_handles_quotes() {
        assert_eq!(
            shell_split("ls -la 'a b' \"c d\"").unwrap(),
            vec!["ls", "-la", "a b", "c d"]
        );
        assert!(shell_split("echo 'unterminated").is_err());
        assert_eq!(shell_split("  ").unwrap(), Vec::<String>::new());
    }
}
