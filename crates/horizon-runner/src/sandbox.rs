//! Sandboxes for tool commands.
//!
//! `SandboxKind::None` runs on the host with only the allowlist between
//! the policy and the machine; keep it for laptops and tests.
//! `SandboxKind::Docker` runs each command in a disposable container:
//! no network, cpu and memory caps, non-root, workspace mounted at /ws.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxResult {
    pub returncode: i32,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
}

#[derive(Debug, Clone)]
pub enum SandboxKind {
    None,
    Docker {
        image: String,
        memory: String,
        cpus: String,
        network: String,
        user: String,
    },
}

impl SandboxKind {
    pub fn docker(image: Option<&str>) -> Self {
        SandboxKind::Docker {
            image: image.unwrap_or("python:3.11-slim").to_string(),
            memory: "512m".into(),
            cpus: "1".into(),
            network: "none".into(),
            user: "nobody".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sandbox {
    kind: SandboxKind,
}

impl Sandbox {
    pub fn new(kind: SandboxKind) -> Self {
        Self { kind }
    }

    pub fn host() -> Self {
        Self::new(SandboxKind::None)
    }

    /// Build from a settings string. Docker falls back to the host sandbox
    /// when the daemon is unreachable, mirroring the previous behaviour.
    pub async fn from_setting(kind: &str, image: Option<&str>) -> Result<Self, String> {
        match kind {
            "none" => Ok(Self::host()),
            "docker" => {
                if docker_available().await {
                    Ok(Self::new(SandboxKind::docker(image)))
                } else {
                    tracing::warn!("docker unavailable; falling back to host sandbox");
                    Ok(Self::host())
                }
            }
            other => Err(format!("Unknown sandbox kind: {other}")),
        }
    }

    pub fn name(&self) -> &'static str {
        match self.kind {
            SandboxKind::None => "none",
            SandboxKind::Docker { .. } => "docker",
        }
    }

    pub async fn run(&self, workspace: &Path, argv: &[String], timeout: Duration) -> SandboxResult {
        let (program, args, budget): (String, Vec<String>, Duration) = match &self.kind {
            SandboxKind::None => (argv[0].clone(), argv[1..].to_vec(), timeout),
            SandboxKind::Docker {
                image,
                memory,
                cpus,
                network,
                user,
            } => {
                let mut args = vec![
                    "run".to_string(),
                    "--rm".into(),
                    format!("--network={network}"),
                    format!("--memory={memory}"),
                    format!("--cpus={cpus}"),
                    "--user".into(),
                    user.clone(),
                    "-v".into(),
                    format!("{}:/ws", workspace.display()),
                    "-w".into(),
                    "/ws".into(),
                    image.clone(),
                ];
                args.extend(argv.iter().cloned());
                ("docker".into(), args, timeout + Duration::from_secs(5))
            }
        };
        let mut cmd = Command::new(&program);
        cmd.args(&args)
            .current_dir(workspace)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return SandboxResult {
                    returncode: 127,
                    stdout: String::new(),
                    stderr: format!("spawn failed: {e}"),
                    timed_out: false,
                }
            }
        };
        match tokio::time::timeout(budget, child.wait_with_output()).await {
            Ok(Ok(out)) => SandboxResult {
                returncode: out.status.code().unwrap_or(-1),
                stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
                timed_out: false,
            },
            Ok(Err(e)) => SandboxResult {
                returncode: -1,
                stdout: String::new(),
                stderr: e.to_string(),
                timed_out: false,
            },
            // Dropping the future kills the child (kill_on_drop).
            Err(_) => SandboxResult {
                returncode: 124,
                stdout: String::new(),
                stderr: "timeout".into(),
                timed_out: true,
            },
        }
    }
}

async fn docker_available() -> bool {
    let probe = Command::new("docker")
        .args(["version", "--format", "{{.Server.Version}}"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .status();
    matches!(tokio::time::timeout(Duration::from_secs(3), probe).await, Ok(Ok(s)) if s.success())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn host_sandbox_runs_and_times_out() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::host();
        let ok = sb
            .run(
                dir.path(),
                &["echo".into(), "hi".into()],
                Duration::from_secs(5),
            )
            .await;
        assert_eq!(ok.returncode, 0);
        assert_eq!(ok.stdout.trim(), "hi");
        let slow = sb
            .run(
                dir.path(),
                &["sleep".into(), "5".into()],
                Duration::from_millis(100),
            )
            .await;
        assert!(slow.timed_out);
        assert_eq!(slow.returncode, 124);
    }
}
