import type {
  AdapterDetail,
  AdapterRecord,
  DashboardSnapshot,
  EvalReport,
  RunDetail,
  TrainingRunRecord,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function fetchDashboard(): Promise<DashboardSnapshot> {
  const response = await fetch(`${API_BASE}/api/dashboard`);
  if (!response.ok) {
    throw new Error("Failed to fetch dashboard");
  }
  return response.json();
}

export async function fetchRun(runId: string): Promise<RunDetail> {
  const response = await fetch(`${API_BASE}/api/runs/${runId}`);
  if (!response.ok) {
    throw new Error("Failed to fetch run");
  }
  return response.json();
}

export async function cancelRun(runId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/runs/${runId}/cancel`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`Cancel failed (${response.status})`);
  }
}

export interface RubricInfo {
  name: string;
  signals: string[];
}

export interface BudgetStatus {
  window_hours: number;
  cap_usd: number;
  spent_usd: number;
  remaining_usd: number | null;
  exceeded: boolean;
}

export async function fetchBudget(): Promise<BudgetStatus> {
  const response = await fetch(`${API_BASE}/api/budget`);
  if (!response.ok) {
    throw new Error("Failed to fetch budget");
  }
  return response.json();
}

export async function fetchRubrics(): Promise<RubricInfo[]> {
  const response = await fetch(`${API_BASE}/api/rubrics`);
  if (!response.ok) {
    throw new Error("Failed to fetch rubrics");
  }
  return response.json();
}

export interface RescoreResponse {
  run_id: string;
  rubric: string;
  previous_terminal_reward: number;
  new_terminal_reward: number;
  delta: number;
  persisted: boolean;
}

export async function rescoreRun(runId: string, rubric: string): Promise<RescoreResponse> {
  const url = new URL(`${API_BASE}/api/runs/${runId}/rescore`);
  url.searchParams.set("rubric", rubric);
  const response = await fetch(url.toString(), { method: "POST" });
  if (!response.ok) {
    throw new Error(`Rescore failed (${response.status})`);
  }
  return response.json();
}

export async function fetchAdapters(): Promise<AdapterRecord[]> {
  const r = await fetch(`${API_BASE}/api/adapters`);
  if (!r.ok) throw new Error("Failed to fetch adapters");
  return r.json();
}

export async function fetchAdapter(id: string): Promise<AdapterDetail> {
  const r = await fetch(`${API_BASE}/api/adapters/${id}`);
  if (!r.ok) throw new Error(`Failed to fetch adapter ${id}`);
  return r.json();
}

export async function runEval(adapterId: string): Promise<EvalReport> {
  const r = await fetch(`${API_BASE}/api/adapters/${adapterId}/eval`, { method: "POST" });
  if (!r.ok) throw new Error(`Eval failed (${r.status})`);
  return r.json();
}

export async function fetchTrainingRuns(): Promise<TrainingRunRecord[]> {
  const r = await fetch(`${API_BASE}/api/training-runs`);
  if (!r.ok) throw new Error("Failed to fetch training runs");
  return r.json();
}

export async function fetchTrainingRun(id: string): Promise<TrainingRunRecord> {
  const r = await fetch(`${API_BASE}/api/training-runs/${id}`);
  if (!r.ok) throw new Error(`Failed to fetch training run ${id}`);
  return r.json();
}

export interface CreateTrainingRunBody {
  sample_run_ids: string[];
  parent_adapter_id?: string | null;
  hyperparams?: Record<string, number | string | boolean>;
}

export async function createTrainingRun(
  body: CreateTrainingRunBody,
): Promise<{ status: string; training_run_id: string }> {
  const r = await fetch(`${API_BASE}/api/training-runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`Create training run failed (${r.status})`);
  return r.json();
}
