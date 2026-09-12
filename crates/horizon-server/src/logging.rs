//! tracing setup plus a layer that mirrors log records onto the event
//! bus as `log.line` events, so the dashboard's log pane keeps working.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use tracing::field::{Field, Visit};
use tracing::{Event as TraceEvent, Subscriber};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

use horizon_core::events::{DomainEvent, Event, Subject};

use crate::event_bus::EventBus;

pub struct BusLayer {
    bus: EventBus,
}

#[derive(Default)]
struct FieldCollector {
    message: String,
    fields: BTreeMap<String, String>,
}

impl Visit for FieldCollector {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.message, "{value:?}");
        } else {
            self.fields
                .insert(field.name().to_string(), format!("{value:?}"));
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message.push_str(value);
        } else {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }
    }
}

impl<S: Subscriber> Layer<S> for BusLayer {
    fn on_event(&self, event: &TraceEvent<'_>, _ctx: Context<'_, S>) {
        // Only our own crates; dependency chatter stays off the wire.
        let target = event.metadata().target();
        if !target.starts_with("horizon") {
            return;
        }
        let mut collector = FieldCollector::default();
        event.record(&mut collector);
        let run_id = collector.fields.remove("run_id");
        collector.fields.remove("task_id");
        let message = if collector.message.is_empty() {
            collector.fields.remove("event").unwrap_or_default()
        } else {
            collector.message.clone()
        };
        let subject = run_id.map(|id| Subject::Run { id });
        self.bus.publish(Event::new(
            subject,
            DomainEvent::LogLine {
                level: event.metadata().level().to_string().to_uppercase(),
                logger: target.to_string(),
                message,
                context: collector.fields,
            },
        ));
    }
}

/// Install the global subscriber. Safe to call once per process; later
/// calls are ignored so tests can share a runtime.
pub fn init(level: &str, json: bool, bus: Option<EventBus>) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(format!(
            "{},hyper=warn,reqwest=warn,tower_http=info",
            level.to_lowercase()
        ))
    });
    let bus_layer = bus.map(|bus| BusLayer { bus });
    let registry = tracing_subscriber::registry().with(filter).with(bus_layer);
    let result = if json {
        registry
            .with(tracing_subscriber::fmt::layer().json().with_target(true))
            .try_init()
    } else {
        registry
            .with(tracing_subscriber::fmt::layer().with_target(false))
            .try_init()
    };
    let _ = result;
}
