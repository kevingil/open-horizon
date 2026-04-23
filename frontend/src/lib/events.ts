import type { RunDetail, RunManifest, TrajectoryStep, WorkerRecord } from "./types";

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

export type DomainEvent =
  | RolloutStarted
  | StepRecorded
  | RewardComputed
  | RolloutCompleted
  | RolloutFailed
  | WorkerUpdated
  | LogLine;
