import type { DashboardSnapshot, RunDetail } from "./types";

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
