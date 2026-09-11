//! Canonical domain types for the Open Horizon platform.

pub mod events;
pub mod models;
pub mod pricing;
pub mod scheduling;

pub use chrono::{DateTime, Utc};

/// Current UTC timestamp. Single call site so tests can reason about clock use.
pub fn utc_now() -> DateTime<Utc> {
    Utc::now()
}

/// Short random id with a prefix, e.g. `run-1a2b3c4d`.
pub fn short_id(prefix: &str) -> String {
    let hex = uuid::Uuid::new_v4().simple().to_string();
    format!("{prefix}-{}", &hex[..8])
}

/// Round to `places` decimal places, half away from zero.
pub fn round_to(value: f64, places: i32) -> f64 {
    let factor = 10f64.powi(places);
    (value * factor).round() / factor
}
