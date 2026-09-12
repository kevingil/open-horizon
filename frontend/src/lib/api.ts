import type {
  AdapterDetail,
  AdapterRecord,
  BudgetStatus,
  CreateRunResponse,
  CreateTrainingRunBody,
  CreateTrainingRunResponse,
  DashboardSnapshot,
  EvalReport,
  JobRecord,
  NodeRecord,
  RescoreResponse,
  RewardBin,
  RolloutRequest,
  RubricInfo,
  RunDetail,
  RunManifest,
  RunStatus,
  RuntimeConfig,
  Stats,
  StatsBucket,
  TrainingRunRecord,
} from "./types";
import type { EventEnvelope } from "./events";

export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // keep the status code
    }
    throw new Error(`${init?.method ?? "GET"} ${path} failed: ${detail}`);
  }
  return response.json() as Promise<T>;
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

function qs(params: Record<string, string | number | undefined | null>): string {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") p.set(k, String(v));
  const s = p.toString();
  return s ? `?${s}` : "";
}

export const fetchDashboard = (windowS = 900) => request<DashboardSnapshot>(`/api/dashboard${qs({ window_s: windowS })}`);
export const fetchStats = (windowS = 900) => request<Stats>(`/api/stats${qs({ window_s: windowS })}`);
export const fetchTimeseries = (windowS = 900, bucketS = 30) =>
  request<StatsBucket[]>(`/api/stats/timeseries${qs({ window_s: windowS, bucket_s: bucketS })}`);
export const fetchRewardHistogram = (windowS = 900, bins = 20) =>
  request<RewardBin[]>(`/api/stats/rewards${qs({ window_s: windowS, bins })}`);
export const fetchFleet = () => request<NodeRecord[]>("/api/fleet");
export const fetchRuns = (opts: { status?: RunStatus; profile?: string; limit?: number; offset?: number } = {}) =>
  request<RunManifest[]>(`/api/runs${qs({ status: opts.status, profile: opts.profile, limit: opts.limit, offset: opts.offset })}`);
export const fetchRun = (runId: string) => request<RunDetail>(`/api/runs/${runId}`);
export const fetchRunEvents = (runId: string, kind?: string) =>
  request<EventEnvelope[]>(`/api/runs/${runId}/events${qs({ kind, limit: 1000 })}`);
export const createRun = (body: RolloutRequest) => post<CreateRunResponse>("/api/runs", body);
export const cancelRun = (runId: string) => post<{ run_id: string }>(`/api/runs/${runId}/cancel`);
export const fetchBudget = () => request<BudgetStatus>("/api/budget");
export const fetchRuntimeConfig = () => request<RuntimeConfig>("/api/config");
export const fetchRubrics = () => request<RubricInfo[]>("/api/rubrics");
export const fetchJobs = (limit = 200) => request<JobRecord[]>(`/api/jobs${qs({ limit })}`);
export const fetchEvents = (opts: { since?: number; limit?: number; kind?: string; subject_kind?: string; subject_id?: string } = {}) =>
  request<EventEnvelope[]>(`/api/events${qs({ since: opts.since ?? 0, limit: opts.limit ?? 500, kind: opts.kind, subject_kind: opts.subject_kind, subject_id: opts.subject_id })}`);

export function rescoreRun(runId: string, rubric: string): Promise<RescoreResponse> {
  return post<RescoreResponse>(`/api/runs/${runId}/rescore${qs({ rubric })}`);
}

export const fetchAdapters = () => request<AdapterRecord[]>("/api/adapters");
export const fetchAdapter = (id: string) => request<AdapterDetail>(`/api/adapters/${id}`);
export const runEval = (adapterId: string) => post<EvalReport>(`/api/adapters/${adapterId}/eval`, {});
export const fetchTrainingRuns = () => request<TrainingRunRecord[]>("/api/training-runs");
export const fetchTrainingRun = (id: string) => request<TrainingRunRecord>(`/api/training-runs/${id}`);
export const createTrainingRun = (body: CreateTrainingRunBody) => post<CreateTrainingRunResponse>("/api/training-runs", body);
