//! Rollout execution: sandboxes, repository snapshots, and tool dispatch.

pub mod openai;
pub mod repo_runner;
pub mod sandbox;
pub mod snapshot;

pub use openai::{ChatMessage, OpenAiClient, OpenAiConfig};
pub use repo_runner::{RepoRunner, RepoRunnerConfig};
pub use sandbox::{Sandbox, SandboxKind, SandboxResult};
