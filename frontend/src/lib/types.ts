export type RunStatus = "pending" | "running" | "completed" | "failed";
export type WorkerStatus = "idle" | "running" | "failed";

export interface RunManifest {
  id: string;
  model_id: string;
  adapter_id: string | null;
  dataset_slice: string;
  infra_target: string;
  seed: number;
  status: RunStatus;
  created_at: string;
  updated_at: string;
  estimated_cost_usd: number;
}

export interface WorkerRecord {
  id: string;
  role: string;
  status: WorkerStatus;
  run_id: string | null;
  detail: string;
}

export interface ArtifactRecord {
  name: string;
  kind: string;
  path: string;
}

export interface DashboardSnapshot {
  generated_at: string;
  runs: RunManifest[];
  workers: WorkerRecord[];
  recent_artifacts: ArtifactRecord[];
}

export interface TaskSpec {
  id: string;
  prompt: string;
  repo_snapshot: string;
  tool_permissions: string[];
  horizon: number;
  success_criteria: string[];
}

export interface TrajectoryStep {
  index: number;
  actor: string;
  kind: string;
  content: string;
  timestamp: string;
}

export interface TrajectoryRecord {
  id: string;
  task_id: string;
  steps: TrajectoryStep[];
  summaries: string[];
  timings_ms: Record<string, number>;
  errors: string[];
}

export interface RewardPenalty {
  code: string;
  value: number;
  reason: string;
}

export interface RewardRecord {
  trajectory_id: string;
  terminal_reward: number;
  step_rewards: number[];
  penalties: RewardPenalty[];
  audit_flags: string[];
  provenance: string;
}

export interface RunDetail {
  manifest: RunManifest;
  task: TaskSpec;
  trajectory: TrajectoryRecord;
  reward: RewardRecord;
  artifacts: ArtifactRecord[];
}
