//! Canonical domain types for the Open Horizon platform.
//!
//! Rust owns the DTOs: every wire shape the API, the event stream, the
//! store, and the Python bridge agree on lives here and is exported to
//! OpenAPI via `utoipa`. Python workers consume the JSON these types
//! produce; they never define their own copies.

pub mod events;
pub mod models;
pub mod pricing;
pub mod rewards;
pub mod scheduling;
pub mod tools;

pub use chrono::{DateTime, Utc};

/// Current UTC timestamp. Single call site so tests can reason about clock use.
pub fn utc_now() -> DateTime<Utc> {
    Utc::now()
}

/// Short random id with a prefix, e.g. `run-1a2b3c4d`. Mirrors the
/// `f"{prefix}-{uuid4().hex[:8]}"` convention the Python stack used so
/// existing databases and dashboards keep reading naturally.
pub fn short_id(prefix: &str) -> String {
    let hex = uuid::Uuid::new_v4().simple().to_string();
    format!("{prefix}-{}", &hex[..8])
}

/// Round to `places` decimal places, half away from zero.
pub fn round_to(value: f64, places: i32) -> f64 {
    let factor = 10f64.powi(places);
    (value * factor).round() / factor
}
