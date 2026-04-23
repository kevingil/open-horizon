import { useEffect, useRef, useState } from "react";
import type { DomainEvent } from "./events";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const WS_URL = API_BASE.replace(/^http/, "ws") + "/ws/events";

export type SocketStatus = "connecting" | "open" | "closed";

/**
 * Subscribe to the server event bus. Handlers are called for every event.
 * Auto-reconnects with exponential backoff.
 */
export function useEventStream(onEvent: (event: DomainEvent) => void): SocketStatus {
  const [status, setStatus] = useState<SocketStatus>("connecting");
  const handlerRef = useRef(onEvent);
  handlerRef.current = onEvent;

  useEffect(() => {
    let closed = false;
    let socket: WebSocket | null = null;
    let attempt = 0;
    let reconnectTimer: number | null = null;

    function connect() {
      setStatus("connecting");
      socket = new WebSocket(WS_URL);
      socket.onopen = () => {
        attempt = 0;
        setStatus("open");
      };
      socket.onmessage = (ev) => {
        try {
          const event = JSON.parse(ev.data) as DomainEvent;
          handlerRef.current(event);
        } catch {
          // swallow malformed frames
        }
      };
      socket.onclose = () => {
        setStatus("closed");
        if (closed) return;
        const delay = Math.min(500 * 2 ** attempt, 8000);
        attempt += 1;
        reconnectTimer = window.setTimeout(connect, delay);
      };
      socket.onerror = () => socket?.close();
    }

    connect();

    return () => {
      closed = true;
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, []);

  return status;
}
