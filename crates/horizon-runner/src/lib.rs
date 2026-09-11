//! Rollout execution: sandboxes and repository snapshots.

pub mod sandbox;
pub mod snapshot;

pub use sandbox::{Sandbox, SandboxKind, SandboxResult};
