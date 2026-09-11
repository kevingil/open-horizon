//! Event bus: every domain event is appended to the store's event log
//! (durable, sequenced) and fanned out to in-process subscribers over a
//! broadcast channel. Slow WebSocket clients lag and skip, never block.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;

use horizon_core::events::DomainEvent;
use horizon_core::models::EventEnvelope;
use horizon_store::Store;

#[derive(Clone)]
pub struct EventBus {
    inner: Arc<Inner>,
}

struct Inner {
    store: Arc<Store>,
    tx: broadcast::Sender<EventEnvelope>,
    replay: Mutex<VecDeque<EventEnvelope>>,
    replay_size: usize,
}

impl EventBus {
    pub fn new(store: Arc<Store>, replay_size: usize, capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity);
        // Warm the replay ring from the durable log so a restart still
        // gives fresh dashboards recent context.
        let warm: VecDeque<EventEnvelope> =
            store.latest_events(replay_size).unwrap_or_default().into();
        Self {
            inner: Arc::new(Inner {
                store,
                tx,
                replay: Mutex::new(warm),
                replay_size,
            }),
        }
    }

    /// Append to the durable log, then broadcast. Synchronous on purpose:
    /// it is called from the tracing layer and from job tasks alike.
    pub fn publish(&self, event: DomainEvent) -> i64 {
        let seq = match self.inner.store.append_event(&event) {
            Ok(seq) => seq,
            Err(e) => {
                eprintln!("event log append failed: {e}");
                -1
            }
        };
        let envelope = EventEnvelope { seq, event };
        {
            let mut ring = self.inner.replay.lock().unwrap_or_else(|e| e.into_inner());
            ring.push_back(envelope.clone());
            while ring.len() > self.inner.replay_size {
                ring.pop_front();
            }
        }
        let _ = self.inner.tx.send(envelope);
        seq
    }

    pub fn recent(&self) -> Vec<EventEnvelope> {
        self.inner
            .replay
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    pub fn since(&self, seq: i64, limit: usize) -> Vec<EventEnvelope> {
        self.inner
            .store
            .events_since(seq, limit)
            .unwrap_or_default()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<EventEnvelope> {
        self.inner.tx.subscribe()
    }

    pub fn subscriber_count(&self) -> usize {
        self.inner.tx.receiver_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn publish_is_durable_and_broadcast() {
        let store = Arc::new(Store::in_memory().unwrap());
        let bus = EventBus::new(store.clone(), 2, 16);
        let mut rx = bus.subscribe();
        let s1 = bus.publish(DomainEvent::rollout_failed("run-1", "a"));
        bus.publish(DomainEvent::rollout_failed("run-1", "b"));
        bus.publish(DomainEvent::rollout_failed("run-1", "c"));
        assert_eq!(bus.recent().len(), 2);
        assert_eq!(rx.recv().await.unwrap().seq, s1);
        assert_eq!(bus.since(s1, 10).len(), 2);
        // A new bus over the same store warms its ring from the log.
        let bus2 = EventBus::new(store, 5, 16);
        assert_eq!(bus2.recent().len(), 3);
    }
}
