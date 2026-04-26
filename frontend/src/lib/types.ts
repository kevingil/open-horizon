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

export interface AdapterRecord {
  id: string;
  parent_id: string | null;
  base_model: string;
  training_run_id: string | null;
  eval_score: number | null;
  path: string;
  tags: string[];
  metadata: Record<string, string>;
  created_at: string;
}

export interface TrainingMetricPoint {
  step: number;
  loss: number;
  mean_reward: number | null;
  kl: number | null;
  extra: Record<string, number>;
}

export type TrainingStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

export interface TrainingRunRecord {
  id: string;
  status: TrainingStatus;
  adapter_in: string | null;
  adapter_out: string | null;
  sample_run_ids: string[];
  hyperparams: Record<string, number | string | boolean>;
  metrics: TrainingMetricPoint[];
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface EvalTaskScore {
  task_id: string;
  terminal_reward: number;
}

export interface EvalReport {
  id: string;
  adapter_id: string;
  task_set: string;
  mean_reward: number;
  per_task: EvalTaskScore[];
  created_at: string;
}

export interface AdapterDetail {
  adapter: AdapterRecord;
  children: AdapterRecord[];
  eval_reports: EvalReport[];
}
