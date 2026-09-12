//! Durable persistence for runs, training, jobs, and the event log.
//!
//! One SQLite implementation serves both the "memory" backend (an
//! in-memory database) and the "sqlite" backend (a file). The schema for
//! `runs`, `workers`, `turn_training`, `training_runs`, and
//! `eval_reports` is byte-compatible with the databases the Python
//! stack wrote, so existing artifacts keep loading. `jobs` and `events`
//! are new: they make orchestration crash-safe and let WebSocket clients
//! resume from a sequence number.

pub mod adapters;
mod sqlite;

pub use adapters::LocalAdapterRegistry;
pub use sqlite::{Store, StoreError};

pub type Result<T> = std::result::Result<T, StoreError>;
