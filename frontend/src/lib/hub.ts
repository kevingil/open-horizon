/**
 * One WebSocket for the whole app. Subscribers get every envelope; the
 * socket resumes from the last sequence number on reconnect so no event
 * is dropped across a blip.
 */
import { useEffect, useSyncExternalStore } from "react";
import { API_BASE } from "./api";
import type { EventEnvelope } from "./events";

export type SocketStatus = "connecting" | "open" | "closed";

type Listener = (event: EventEnvelope) => void;

const WS_URL = API_BASE.replace(/^http/, "ws") + "/ws/events";
const listeners = new Set<Listener>();
const statusListeners = new Set<() => void>();
let status: SocketStatus = "connecting";
let socket: WebSocket | null = null;
let attempt = 0;
let lastSeq: number | null = null;
let reconnectTimer: number | null = null;
let started = false;

function setStatus(next: SocketStatus) {
  status = next;
  for (const l of statusListeners) l();
}

function connect() {
  setStatus("connecting");
  const url = lastSeq === null ? WS_URL : `${WS_URL}?since=${lastSeq}`;
  socket = new WebSocket(url);
  socket.onopen = () => {
    attempt = 0;
    setStatus("open");
  };
  socket.onmessage = (ev) => {
    let event: EventEnvelope;
    try {
      event = JSON.parse(ev.data) as EventEnvelope;
    } catch {
      return;
    }
    if (typeof event.seq === "number" && event.seq > 0) lastSeq = event.seq;
    for (const l of listeners) l(event);
  };
  socket.onclose = () => {
    setStatus("closed");
    const delay = Math.min(500 * 2 ** attempt, 8000);
    attempt += 1;
    reconnectTimer = window.setTimeout(connect, delay);
  };
  socket.onerror = () => socket?.close();
}

export function ensureHub() {
  if (started) return;
  started = true;
  connect();
}

export function subscribe(listener: Listener): () => void {
  ensureHub();
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function useEvents(listener: Listener) {
  useEffect(() => subscribe(listener), [listener]);
}

export function useSocketStatus(): SocketStatus {
  ensureHub();
  return useSyncExternalStore(
    (cb) => {
      statusListeners.add(cb);
      return () => statusListeners.delete(cb);
    },
    () => status,
  );
}

export function stopHub() {
  if (reconnectTimer) window.clearTimeout(reconnectTimer);
  socket?.close();
}
