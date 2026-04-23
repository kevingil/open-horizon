from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import structlog

from ..domain.contracts import (
    ArtifactStore,
    EnvironmentRunner,
    PolicyServer,
    RewardPipeline,
    RolloutCoordinator,
    ToolHarness,
)
from ..domain.events import (
    RewardComputed,
    RolloutCompleted,
    RolloutFailed,
    RolloutStarted,
    StepRecorded,
    WorkerUpdated,
)
from ..domain.models import (
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
from .event_bus import EventBus

log = structlog.get_logger(__name__)


@dataclass
class LocalRolloutCoordinator(RolloutCoordinator):
    environment_runner: EnvironmentRunner
    tool_harness: ToolHarness
    policy_server: PolicyServer
    reward_pipeline: RewardPipeline
    artifact_store: ArtifactStore
    event_bus: EventBus
    workspace_root: Path
    max_parallel: int = 4
    _semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.max_parallel)

    async def bootstrap(self):
        if self.artifact_store.list_runs():
            return self.artifact_store.dashboard()
        seeds = [
            "Bootstrap local debug rollout for coding task replay.",
            "Replay reward computation from stored trajectory artifacts.",
        ]
        for prompt in seeds:
            await self.start_rollout(
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

    async def start_rollout(self, request: RolloutRequest) -> RunDetail:
        async with self._semaphore:
            return await self._execute(request)

    async def _execute(self, request: RolloutRequest) -> RunDetail:
        now = datetime.now(UTC)
        run_id = f"run-{uuid4().hex[:8]}"
        task_id = f"task-{uuid4().hex[:8]}"
        bind = structlog.contextvars.bound_contextvars(run_id=run_id, task_id=task_id)
        with bind:
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

            manifest = RunManifest(
                id=run_id,
                model_id=self.policy_server.policy_name(),
                dataset_slice="bootstrap",
                infra_target=request.infra_target,
                seed=7,
                status=RunStatus.running,
                created_at=now,
                updated_at=now,
                estimated_cost_usd=0.03 if request.infra_target == "mac-local" else 0.72,
            )
            await self.event_bus.publish(RolloutStarted(run_id=run_id, manifest=manifest))
            await self._publish_worker(
                "worker-rollout-local", "rollout", WorkerStatus.running, run_id,
                "Local rollout worker is active.",
            )

            try:
                steps, trajectory = await self._run_steps(task, request)
                reward = self.reward_pipeline.score_trajectory(task, trajectory)
                await self.event_bus.publish(
                    RewardComputed(
                        run_id=run_id,
                        terminal_reward=reward.terminal_reward,
                        provenance=reward.provenance,
                        audit_flags=reward.audit_flags,
                    )
                )
                manifest = manifest.model_copy(
                    update={"status": RunStatus.completed, "updated_at": datetime.now(UTC)},
                )
                detail = RunDetail(
                    manifest=manifest,
                    task=task,
                    trajectory=trajectory,
                    reward=reward,
                    artifacts=_default_artifacts(run_id),
                )
                self.artifact_store.save_run(detail)
                await self._publish_worker(
                    "worker-rollout-local", "rollout", WorkerStatus.idle, run_id,
                    "Rollout finished; worker idle.",
                )
                await self._publish_worker(
                    "worker-reward-local", "reward", WorkerStatus.idle, run_id,
                    "Reward pipeline is available for replay.",
                )
                await self.event_bus.publish(RolloutCompleted(run_id=run_id, detail=detail))
                log.info("rollout.completed", terminal_reward=reward.terminal_reward, steps=len(steps))
                return detail
            except Exception as exc:
                log.exception("rollout.failed")
                failed_manifest = manifest.model_copy(
                    update={"status": RunStatus.failed, "updated_at": datetime.now(UTC)},
                )
                failed_detail = RunDetail(
                    manifest=failed_manifest,
                    task=task,
                    trajectory=TrajectoryRecord(
                        id=f"traj-{uuid4().hex[:8]}",
                        task_id=task.id,
                        steps=[],
                        errors=[str(exc)],
                    ),
                    reward=self.reward_pipeline.score_trajectory(
                        task,
                        TrajectoryRecord(
                            id=f"traj-{uuid4().hex[:8]}", task_id=task.id, steps=[],
                            errors=[str(exc)],
                        ),
                    ),
                    artifacts=_default_artifacts(run_id),
                )
                self.artifact_store.save_run(failed_detail)
                await self._publish_worker(
                    "worker-rollout-local", "rollout", WorkerStatus.failed, run_id,
                    f"Rollout failed: {exc}",
                )
                await self.event_bus.publish(RolloutFailed(run_id=run_id, error=str(exc)))
                return failed_detail

    async def _run_steps(
        self, task: TaskSpec, request: RolloutRequest
    ) -> tuple[list[TrajectoryStep], TrajectoryRecord]:
        steps: list[TrajectoryStep] = []
        context: list[str] = []
        for index in range(request.horizon):
            action = self.policy_server.generate_action(task, context)
            observation = self.environment_runner.step(task.id, action)
            self.tool_harness.record_command(f"simulate:{action}")

            action_step = TrajectoryStep(
                index=index * 2, actor="policy", kind="action", content=action,
            )
            obs_step = TrajectoryStep(
                index=index * 2 + 1, actor="environment", kind="observation", content=observation,
            )
            steps.extend([action_step, obs_step])
            await self.event_bus.publish(StepRecorded(run_id=task.id, step=action_step))
            await self.event_bus.publish(StepRecorded(run_id=task.id, step=obs_step))
            context.append(action)
            await asyncio.sleep(0)

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
        return steps, trajectory

    async def _publish_worker(
        self, worker_id: str, role: str, status: WorkerStatus, run_id: str | None, detail: str,
    ) -> None:
        worker = WorkerRecord(id=worker_id, role=role, status=status, run_id=run_id, detail=detail)
        existing = {w.id: w for w in getattr(self.artifact_store, "workers", []) or []}
        existing[worker.id] = worker
        if hasattr(self.artifact_store, "set_workers"):
            self.artifact_store.set_workers(list(existing.values()))
        await self.event_bus.publish(WorkerUpdated(run_id=run_id, worker=worker))


def _default_artifacts(run_id: str) -> list[ArtifactRecord]:
    return [
        ArtifactRecord(
            name=f"{run_id}-manifest.json", kind="manifest",
            path=f"artifacts/{run_id}/manifest.json",
        ),
        ArtifactRecord(
            name=f"{run_id}-trajectory.json", kind="trajectory",
            path=f"artifacts/{run_id}/trajectory.json",
        ),
        ArtifactRecord(
            name=f"{run_id}-reward.json", kind="reward",
            path=f"artifacts/{run_id}/reward.json",
        ),
    ]
