/**
 * Domain types, re-exported from the OpenAPI bindings generated out of
 * the Rust DTOs (`npm run gen:api`). Nothing here is hand-maintained.
 */
import type { components } from "./api-schema";

type S = components["schemas"];

export type RunStatus = S["RunStatus"];
export type WorkerStatus = S["WorkerStatus"];
export type TrainingStatus = S["TrainingStatus"];
export type JobStatus = S["JobStatus"];

export type RunManifest = S["RunManifest"];
export type RunDetail = S["RunDetail"];
export type TaskSpec = S["TaskSpec"];
export type Trajectory = S["Trajectory"];
export type TrajectoryStep = S["TrajectoryStep"];
export type RewardRecord = S["RewardRecord"];
export type RewardSignal = S["RewardSignal"];
export type WorkerRecord = S["WorkerRecord"];
export type JobCounts = S["JobCounts"];
export type JobRecord = S["JobRecord"];
export type DashboardSnapshot = S["DashboardSnapshot"];
export type RolloutRequest = S["RolloutRequest"];

export type AdapterRecord = S["AdapterRecord"];
export type AdapterDetail = S["AdapterDetail"];
export type TrainingMetricPoint = S["TrainingMetricPoint"];
export type TrainingRunRecord = S["TrainingRunRecord"];
export type EvalReport = S["EvalReport"];
export type EvalTask = S["EvalTask"];

export type BudgetStatus = S["BudgetStatus"];
export type RuntimeConfig = S["RuntimeConfig"];
export type RubricInfo = S["RubricInfo"];
export type RescoreResponse = S["RescoreResponse"];
export type CreateTrainingRunBody = S["CreateTrainingRunBody"];
export type CreateTrainingRunResponse = S["CreateTrainingRunResponse"];
export type CreateRunResponse = S["CreateRunResponse"];
