import type {
  AdapterRecord,
  EvalReport,
  RunDetail,
  RunManifest,
  TrainingMetricPoint,
  TrainingRunRecord,
  TrajectoryStep,
  WorkerRecord,
} from "./types";

export type RolloutStarted = {
  kind: "rollout.started";
  event_id: string;
  at: string;
  run_id: string;
  manifest: RunManifest;
};

export type StepRecorded = {
  kind: "step.recorded";
  event_id: string;
  at: string;
  run_id: string | null;
  step: TrajectoryStep;
};

export type RewardComputed = {
  kind: "reward.computed";
  event_id: string;
  at: string;
  run_id: string | null;
  terminal_reward: number;
  provenance: string;
  audit_flags: string[];
};

export type RolloutCompleted = {
  kind: "rollout.completed";
  event_id: string;
  at: string;
  run_id: string | null;
  detail: RunDetail;
};

export type RolloutFailed = {
  kind: "rollout.failed";
  event_id: string;
  at: string;
  run_id: string | null;
  error: string;
};

export type RolloutCancelled = {
  kind: "rollout.cancelled";
  event_id: string;
  at: string;
  run_id: string | null;
  reason: string;
};

export type ProgressTicked = {
  kind: "progress.ticked";
  event_id: string;
  at: string;
  run_id: string | null;
  step_index: number;
  tool: string | null;
  tokens: number;
  cost_usd: number;
};

export type WorkerUpdated = {
  kind: "worker.updated";
  event_id: string;
  at: string;
  run_id: string | null;
  worker: WorkerRecord;
};

export type LogLine = {
  kind: "log.line";
  event_id: string;
  at: string;
  run_id: string | null;
  level: string;
  logger: string;
  message: string;
  context: Record<string, string>;
};

export type BudgetExceeded = {
  kind: "budget.exceeded";
  event_id: string;
  at: string;
  run_id: string | null;
  spent_usd: number;
  cap_usd: number;
  window_hours: number;
};

export type TrainingStarted = {
  kind: "training.started";
  event_id: string;
  at: string;
  run_id: string | null;
  training_run_id: string;
  record: TrainingRunRecord;
};

export type TrainingMetric = {
  kind: "training.metric";
  event_id: string;
  at: string;
  run_id: string | null;
  training_run_id: string;
  metric: TrainingMetricPoint;
};

export type TrainingCompleted = {
  kind: "training.completed";
  event_id: string;
  at: string;
  run_id: string | null;
  training_run_id: string;
  record: TrainingRunRecord;
};

export type TrainingFailed = {
  kind: "training.failed";
  event_id: string;
  at: string;
  run_id: string | null;
  training_run_id: string;
  error: string;
};

export type AdapterPublished = {
  kind: "adapter.published";
  event_id: string;
  at: string;
  run_id: string | null;
  adapter: AdapterRecord;
};

export type EvalCompleted = {
  kind: "eval.completed";
  event_id: string;
  at: string;
  run_id: string | null;
  report: EvalReport;
};

/** Durable log sequence number; present on every event the server emits. */
export type Sequenced = { seq?: number };

export type DomainEvent =
  | RolloutStarted
  | StepRecorded
  | RewardComputed
  | RolloutCompleted
  | RolloutFailed
  | RolloutCancelled
  | ProgressTicked
  | BudgetExceeded
  | WorkerUpdated
  | LogLine
  | TrainingStarted
  | TrainingMetric
  | TrainingCompleted
  | TrainingFailed
  | AdapterPublished
  | EvalCompleted;
