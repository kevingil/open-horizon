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
  RescoreResponse,
  RolloutRequest,
  RubricInfo,
  RunDetail,
  RuntimeConfig,
  TrainingRunRecord,
} from "./types";

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

export const fetchDashboard = () => request<DashboardSnapshot>("/api/dashboard");
export const fetchRun = (runId: string) => request<RunDetail>(`/api/runs/${runId}`);
export const createRun = (body: RolloutRequest) => post<CreateRunResponse>("/api/runs", body);
export const cancelRun = (runId: string) => post<{ run_id: string }>(`/api/runs/${runId}/cancel`);
export const fetchBudget = () => request<BudgetStatus>("/api/budget");
export const fetchRuntimeConfig = () => request<RuntimeConfig>("/api/config");
export const fetchRubrics = () => request<RubricInfo[]>("/api/rubrics");
export const fetchJobs = (limit = 50) => request<JobRecord[]>(`/api/jobs?limit=${limit}`);

export function rescoreRun(runId: string, rubric: string): Promise<RescoreResponse> {
  const params = new URLSearchParams({ rubric });
  return post<RescoreResponse>(`/api/runs/${runId}/rescore?${params.toString()}`);
}

export const fetchAdapters = () => request<AdapterRecord[]>("/api/adapters");
export const fetchAdapter = (id: string) => request<AdapterDetail>(`/api/adapters/${id}`);
export const runEval = (adapterId: string) => post<EvalReport>(`/api/adapters/${adapterId}/eval`, {});
export const fetchTrainingRuns = () => request<TrainingRunRecord[]>("/api/training-runs");
export const fetchTrainingRun = (id: string) => request<TrainingRunRecord>(`/api/training-runs/${id}`);
export const createTrainingRun = (body: CreateTrainingRunBody) =>
  post<CreateTrainingRunResponse>("/api/training-runs", body);
