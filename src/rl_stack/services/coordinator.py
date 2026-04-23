from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ..contracts import ArtifactStore, EnvironmentRunner, PolicyServer, RewardPipeline, RolloutCoordinator, ToolHarness
from ..models import (
    ArtifactRecord,
    RolloutRequest,
    RunDetail,
    RunManifest,
    RunStatus,
    TaskSpec,
    ToolPermission,
    TrajectoryRecord,
    TrajectoryStep,
    WorkerRecord,
    WorkerStatus,
)


@dataclass
class LocalRolloutCoordinator(RolloutCoordinator):
    environment_runner: EnvironmentRunner
    tool_harness: ToolHarness
    policy_server: PolicyServer
    reward_pipeline: RewardPipeline
    artifact_store: ArtifactStore
    workspace_root: Path

    def bootstrap(self):
        if self.artifact_store.list_runs():
            return self.artifact_store.dashboard()
        for prompt in [
            "Bootstrap local debug rollout for coding task replay.",
            "Replay reward computation from stored trajectory artifacts.",
        ]:
            self.start_rollout(
                RolloutRequest(
                    prompt=prompt,
                    repo_snapshot=".",
                    infra_target="mac-local",
                    horizon=4,
                    success_criteria=[
                        "trajectory saved",
                        "reward replayable",
                        "artifacts visible in dashboard",
                    ],
                )
            )
        return self.artifact_store.dashboard()

    def start_rollout(self, request: RolloutRequest) -> RunDetail:
        now = datetime.now(UTC)
        run_id = f"run-{uuid4().hex[:8]}"
        task_id = f"task-{uuid4().hex[:8]}"
        task = TaskSpec(
            id=task_id,
            prompt=request.prompt,
            repo_snapshot=request.repo_snapshot,
            tool_permissions=[
                ToolPermission.read,
                ToolPermission.search,
                ToolPermission.terminal,
            ],
            horizon=request.horizon,
            success_criteria=request.success_criteria or ["manual review"],
        )
        self.environment_runner.create_task(task)

        steps: list[TrajectoryStep] = []
        context: list[str] = []
        for index in range(request.horizon):
            action = self.policy_server.generate_action(task, context)
            observation = self.environment_runner.step(task.id, action)
            self.tool_harness.record_command(f"simulate:{action}")
            steps.append(TrajectoryStep(index=index * 2, actor="policy", kind="action", content=action))
            steps.append(
                TrajectoryStep(index=index * 2 + 1, actor="environment", kind="observation", content=observation)
            )
            context.append(action)

        trajectory = TrajectoryRecord(
            id=f"traj-{uuid4().hex[:8]}",
            task_id=task.id,
            steps=steps,
            summaries=[
                "Local debug summary",
                "Trajectory is replayable through saved records",
            ],
            timings_ms={"rollout": request.horizon * 75, "reward": 40},
        )
        reward = self.reward_pipeline.score_trajectory(task, trajectory)
        manifest = RunManifest(
            id=run_id,
            model_id=self.policy_server.policy_name(),
            dataset_slice="bootstrap",
            infra_target=request.infra_target,
            seed=7,
            status=RunStatus.completed,
            created_at=now,
            updated_at=now,
            estimated_cost_usd=0.03 if request.infra_target == "mac-local" else 0.72,
        )
        artifacts = [
            ArtifactRecord(
                name=f"{run_id}-manifest.json",
                kind="manifest",
                path=f"artifacts/{run_id}/manifest.json",
            ),
            ArtifactRecord(
                name=f"{run_id}-trajectory.json",
                kind="trajectory",
                path=f"artifacts/{run_id}/trajectory.json",
            ),
            ArtifactRecord(
                name=f"{run_id}-reward.json",
                kind="reward",
                path=f"artifacts/{run_id}/reward.json",
            ),
        ]
        detail = RunDetail(
            manifest=manifest,
            task=task,
            trajectory=trajectory,
            reward=reward,
            artifacts=artifacts,
        )
        self.artifact_store.save_run(detail)
        self.artifact_store.set_workers(
            [
                WorkerRecord(
                    id="worker-rollout-local",
                    role="rollout",
                    status=WorkerStatus.running,
                    run_id=run_id,
                    detail="Local rollout worker is active.",
                ),
                WorkerRecord(
                    id="worker-reward-local",
                    role="reward",
                    status=WorkerStatus.idle,
                    run_id=run_id,
                    detail="Reward pipeline is available for replay.",
                ),
                WorkerRecord(
                    id="worker-observe-local",
                    role="observability",
                    status=WorkerStatus.idle,
                    detail="Dashboard reflects artifact-backed state.",
                ),
            ]
        )
        return detail
