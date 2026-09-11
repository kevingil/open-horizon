//! Durable persistence for runs, training, jobs, and the event log.
//!
//! One SQLite implementation serves both the "memory" backend (an
//! in-memory database) and the "sqlite" backend (a file).

mod sqlite;

pub use sqlite::{Store, StoreError};

pub type Result<T> = std::result::Result<T, StoreError>;
