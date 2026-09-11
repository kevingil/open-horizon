import { useEffect, useRef, useState } from "react";
import { API_BASE } from "./api";
import type { EventEnvelope } from "./events";

const WS_URL = API_BASE.replace(/^http/, "ws") + "/ws/events";

export type SocketStatus = "connecting" | "open" | "closed";

/**
 * Subscribe to the server event log. On reconnect the socket resumes from
 * the last sequence number it saw, so nothing is missed across a blip.
 */
export function useEventStream(onEvent: (event: EventEnvelope) => void): SocketStatus {
  const [status, setStatus] = useState<SocketStatus>("connecting");
  const handlerRef = useRef(onEvent);
  handlerRef.current = onEvent;

  useEffect(() => {
    let closed = false;
    let socket: WebSocket | null = null;
    let attempt = 0;
    let lastSeq: number | null = null;
    let reconnectTimer: number | null = null;

    function connect() {
      setStatus("connecting");
      const url = lastSeq === null ? WS_URL : `${WS_URL}?since=${lastSeq}`;
      socket = new WebSocket(url);
      socket.onopen = () => {
        attempt = 0;
        setStatus("open");
      };
      socket.onmessage = (ev) => {
        try {
          const event = JSON.parse(ev.data) as EventEnvelope;
          if (typeof event.seq === "number" && event.seq > 0) lastSeq = event.seq;
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
