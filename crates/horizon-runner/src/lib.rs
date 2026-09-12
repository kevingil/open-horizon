//! Rollout execution: the piece that keeps the inference server busy.
//!
//! - `sandbox`: run policy commands on the host or inside Docker with
//!   hard timeouts and kill-on-drop.
//! - `snapshot`: cheap per-task working copies of a repository.
//! - `repo_runner`: tool dispatch against a snapshot with path and
//!   command allowlists and output caps.
//! - `openai`: a minimal OpenAI-compatible chat client with retries and
//!   token accounting that understands OpenAI and Anthropic-via-compat
//!   usage shapes.
//! - `rollout`: the per-turn tool-call loop, cancellable mid-generation.

pub mod openai;
pub mod repo_runner;
pub mod rollout;
pub mod sandbox;
pub mod snapshot;

pub use openai::{ChatMessage, OpenAiClient, OpenAiConfig};
pub use repo_runner::{RepoRunner, RepoRunnerConfig};
pub use rollout::{run_repo_rollout, RolloutEvent, RolloutOutcome, RolloutParams};
pub use sandbox::{Sandbox, SandboxKind, SandboxResult};
