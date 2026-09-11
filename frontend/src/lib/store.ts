import { useEffect, useState } from "react";
import { fetchDashboard } from "./api";
import { runIdOf, type EventEnvelope, type EventOf } from "./events";
import { useEventStream, type SocketStatus } from "./socket";
import type { DashboardSnapshot, RunManifest, WorkerRecord } from "./types";

export interface RunProgress {
  turn: number;
  tool: string | null;
  tokens: number;
  cost_usd: number;
  at: string;
}

export interface LiveState {
  snapshot: DashboardSnapshot | null;
  logs: EventOf<"log.line">[];
  progress: Record<string, RunProgress>;
  status: SocketStatus;
  error: string | null;
}

const MAX_LOGS = 200;

export function useLiveDashboard(): LiveState {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [logs, setLogs] = useState<EventOf<"log.line">[]>([]);
  const [progress, setProgress] = useState<Record<string, RunProgress>>({});
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
    const runId = runIdOf(event);
    if (!runId) return;
    if (event.kind === "progress.ticked") {
      const p = event.payload;
      setProgress((prev) => ({
        ...prev,
        [runId]: { turn: p.turn, tool: p.tool ?? null, tokens: p.tokens, cost_usd: p.cost_usd, at: event.at },
      }));
    }
    if (
      event.kind === "rollout.completed" ||
      event.kind === "rollout.failed" ||
      event.kind === "rollout.cancelled"
    ) {
      setProgress((prev) => {
        if (!(runId in prev)) return prev;
        const next = { ...prev };
        delete next[runId];
        return next;
      });
    }
  });

  return { snapshot, logs, progress, status, error };
}

function applyEvent(snapshot: DashboardSnapshot, event: EventEnvelope): DashboardSnapshot {
  switch (event.kind) {
    case "rollout.queued":
    case "rollout.started":
    case "rollout.completed":
    case "rollout.cancelled":
      return { ...snapshot, runs: upsertRun(snapshot.runs, event.payload.manifest), jobs: bumpJobs(snapshot, event.kind) };
    case "rollout.failed":
      return { ...snapshot, runs: upsertRun(snapshot.runs, event.payload.manifest), jobs: bumpJobs(snapshot, event.kind) };
    case "worker.updated":
      return { ...snapshot, workers: upsertWorker(snapshot.workers, event.payload.worker) };
    default:
      return snapshot;
  }
}

/** Keep the job counters roughly live between snapshot refreshes. */
function bumpJobs(snapshot: DashboardSnapshot, kind: string) {
  const jobs = { ...snapshot.jobs };
  switch (kind) {
    case "rollout.queued":
      jobs.queued += 1;
      break;
    case "rollout.started":
      jobs.queued = Math.max(0, jobs.queued - 1);
      jobs.running += 1;
      break;
    case "rollout.completed":
      jobs.running = Math.max(0, jobs.running - 1);
      jobs.completed += 1;
      break;
    case "rollout.failed":
      jobs.running = Math.max(0, jobs.running - 1);
      jobs.failed += 1;
      break;
    case "rollout.cancelled":
      jobs.running = Math.max(0, jobs.running - 1);
      jobs.cancelled += 1;
      break;
  }
  return jobs;
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
