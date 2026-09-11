//! Open Horizon control plane: durable jobs, event log, HTTP + WebSocket
//! API, and the coordinator that drives rollouts and training.

pub mod api;
pub mod app;
pub mod coordinator;
pub mod eval;
pub mod event_bus;
pub mod jobs;
pub mod logging;
pub mod settings;
pub mod sglang;
pub mod training;

pub use app::{build_app, AppState};
pub use settings::{PolicyProfile, RoutesTo, Settings};
