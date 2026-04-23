import { useEffect, useState } from "react";
import { fetchDashboard } from "./api";
import type { DomainEvent, LogLine } from "./events";
import { useEventStream, type SocketStatus } from "./socket";
import type { DashboardSnapshot, RunManifest, WorkerRecord } from "./types";

export interface LiveState {
  snapshot: DashboardSnapshot | null;
  logs: LogLine[];
  status: SocketStatus;
  error: string | null;
}

const MAX_LOGS = 200;

export function useLiveDashboard(): LiveState {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchDashboard()
      .then((snap) => active && setSnapshot(snap))
      .catch((err) => active && setError(err instanceof Error ? err.message : String(err)));
    return () => {
      active = false;
    };
  }, []);

  const status = useEventStream((event) => {
    setSnapshot((prev) => prev && applyEvent(prev, event));
    if (event.kind === "log.line") {
      setLogs((prev) => {
        const next = [...prev, event];
        return next.length > MAX_LOGS ? next.slice(next.length - MAX_LOGS) : next;
      });
    }
  });

  return { snapshot, logs, status, error };
}

function applyEvent(snapshot: DashboardSnapshot, event: DomainEvent): DashboardSnapshot {
  switch (event.kind) {
    case "rollout.started":
      return { ...snapshot, runs: upsertRun(snapshot.runs, event.manifest) };
    case "rollout.completed": {
      const runs = upsertRun(snapshot.runs, event.detail.manifest);
      const recent_artifacts = [...event.detail.artifacts, ...snapshot.recent_artifacts].slice(0, 10);
      return { ...snapshot, runs, recent_artifacts };
    }
    case "rollout.failed":
    case "rollout.cancelled": {
      const runs = snapshot.runs.map((r) =>
        r.id === event.run_id ? { ...r, status: "failed" as const, updated_at: event.at } : r,
      );
      return { ...snapshot, runs };
    }
    case "worker.updated":
      return { ...snapshot, workers: upsertWorker(snapshot.workers, event.worker) };
    default:
      return snapshot;
  }
}

function upsertRun(runs: RunManifest[], manifest: RunManifest): RunManifest[] {
  const existing = runs.findIndex((r) => r.id === manifest.id);
  if (existing >= 0) {
    const next = [...runs];
    next[existing] = manifest;
    return next;
  }
  return [manifest, ...runs];
}

function upsertWorker(workers: WorkerRecord[], worker: WorkerRecord): WorkerRecord[] {
  const existing = workers.findIndex((w) => w.id === worker.id);
  if (existing >= 0) {
    const next = [...workers];
    next[existing] = worker;
    return next;
  }
  return [...workers, worker];
}
