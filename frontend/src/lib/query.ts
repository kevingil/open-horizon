/**
 * TanStack Query hooks with WebSocket-driven invalidation. Each hook
 * polls slowly as a fallback and refetches promptly when the event
 * stream says its data changed.
 */
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef } from "react";
import {
  fetchAdapter,
  fetchAdapters,
  fetchBudget,
  fetchDashboard,
  fetchEvents,
  fetchFleet,
  fetchJobs,
  fetchRewardHistogram,
  fetchRun,
  fetchRunEvents,
  fetchRuns,
  fetchRuntimeConfig,
  fetchStats,
  fetchTimeseries,
  fetchTrainingRun,
  fetchTrainingRuns,
} from "./api";
import type { EventEnvelope } from "./events";
import { useEvents } from "./hub";
import type { RunStatus } from "./types";

const SLOW = 15_000;

/** Which query keys an event kind should refresh. */
function affectedKeys(event: EventEnvelope): string[][] {
  const k = event.kind;
  const id = event.subject?.id;
  if (k.startsWith("rollout.") || k === "reward.computed") {
    const keys = [["runs"], ["stats"], ["dashboard"], ["jobs"]];
    if (id) keys.push(["run", id]);
    return keys;
  }
  if (k === "step.recorded" || k === "progress.ticked") return id ? [["run", id]] : [];
  if (k.startsWith("training.")) {
    const keys = [["training-runs"], ["jobs"]];
    if (id) keys.push(["training-run", id]);
    return keys;
  }
  if (k === "adapter.published" || k === "eval.completed") return [["adapters"], ["adapter"]];
  if (k === "node.updated") return [["fleet"], ["dashboard"]];
  if (k === "budget.exceeded") return [["budget"]];
  return [];
}

/** Mount once near the root: turns the event stream into query refreshes. */
export function useLiveInvalidation() {
  const qc = useQueryClient();
  const pending = useRef<Set<string>>(new Set());
  const timer = useRef<number | null>(null);
  const handler = useCallback(
    (event: EventEnvelope) => {
      for (const key of affectedKeys(event)) pending.current.add(JSON.stringify(key));
      if (timer.current === null) {
        timer.current = window.setTimeout(() => {
          timer.current = null;
          for (const key of pending.current) void qc.invalidateQueries({ queryKey: JSON.parse(key) });
          pending.current.clear();
        }, 400);
      }
    },
    [qc],
  );
  useEvents(handler);
  useEffect(() => () => {
    if (timer.current !== null) window.clearTimeout(timer.current);
  }, []);
}

export const useDashboard = (windowS: number) =>
  useQuery({ queryKey: ["dashboard", windowS], queryFn: () => fetchDashboard(windowS), refetchInterval: SLOW });
export const useStats = (windowS: number) =>
  useQuery({ queryKey: ["stats", windowS], queryFn: () => fetchStats(windowS), refetchInterval: SLOW });
export const useTimeseries = (windowS: number, bucketS: number) =>
  useQuery({ queryKey: ["stats", "series", windowS, bucketS], queryFn: () => fetchTimeseries(windowS, bucketS), refetchInterval: SLOW });
export const useRewardHistogram = (windowS: number) =>
  useQuery({ queryKey: ["stats", "hist", windowS], queryFn: () => fetchRewardHistogram(windowS, 20), refetchInterval: SLOW });
export const useFleet = () => useQuery({ queryKey: ["fleet"], queryFn: fetchFleet, refetchInterval: SLOW });
export const useRuns = (status?: RunStatus, profile?: string) =>
  useQuery({ queryKey: ["runs", status ?? "", profile ?? ""], queryFn: () => fetchRuns({ status, profile, limit: 500 }), refetchInterval: SLOW });
export const useRun = (id: string) => useQuery({ queryKey: ["run", id], queryFn: () => fetchRun(id), refetchInterval: SLOW });
export const useRunEvents = (id: string) =>
  useQuery({ queryKey: ["run", id, "events"], queryFn: () => fetchRunEvents(id), refetchInterval: SLOW });
export const useJobs = () => useQuery({ queryKey: ["jobs"], queryFn: () => fetchJobs(300), refetchInterval: SLOW });
export const useBudget = () => useQuery({ queryKey: ["budget"], queryFn: fetchBudget, refetchInterval: SLOW });
export const useRuntimeConfig = () => useQuery({ queryKey: ["config"], queryFn: fetchRuntimeConfig, staleTime: Infinity });
export const useAdapters = () => useQuery({ queryKey: ["adapters"], queryFn: fetchAdapters, refetchInterval: SLOW });
export const useAdapter = (id: string) => useQuery({ queryKey: ["adapter", id], queryFn: () => fetchAdapter(id), refetchInterval: SLOW });
export const useTrainingRuns = () => useQuery({ queryKey: ["training-runs"], queryFn: fetchTrainingRuns, refetchInterval: SLOW });
export const useTrainingRun = (id: string) =>
  useQuery({ queryKey: ["training-run", id], queryFn: () => fetchTrainingRun(id), refetchInterval: SLOW });
export const useRecentEvents = (kind?: string) =>
  useQuery({ queryKey: ["events", kind ?? ""], queryFn: () => fetchEvents({ kind, limit: 500 }), refetchInterval: SLOW });
