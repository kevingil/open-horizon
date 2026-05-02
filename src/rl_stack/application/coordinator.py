from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
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

if TYPE_CHECKING:
    from ..infrastructure.environment.verifiers_runner import (
        VerifiersRolloutRunner,
    )
from ..domain.events import (
    BudgetExceeded,
    ProgressTicked,
    RewardComputed,
    RolloutCancelled,
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
    max_tokens_per_run: int = 100_000
    daily_budget_usd: float = 0.0  # 0 disables the cap
    budget_window_hours: float = 24.0
    # When set, the coordinator delegates the per-step rollout loop to this
    # runner instead of driving policy/env/tools itself. Used by the
    # verifiers backend, where verifiers' env.rollout() owns the loop and
    # produces both trajectory and rubric-scored reward in one shot.
    external_rollout_runner: VerifiersRolloutRunner | None = None
    _semaphore: asyncio.Semaphore = field(init=False)
    _cancelled: set[str] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.max_parallel)

    def request_cancel(self, run_id: str) -> bool:
        """Flag a run for cancellation. Returns False if the run is already
        known to be terminal, True otherwise (including unknown run_ids, so the
        caller doesn't need to care about races)."""
        existing = self.artifact_store.get_run(run_id)
        if existing is not None and existing.manifest.status in {
            RunStatus.completed, RunStatus.failed,
        }:
            return False
        self._cancelled.add(run_id)
        return True

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

    async def start_rollout(
        self, request: RolloutRequest, *, run_id: str | None = None,
    ) -> RunDetail:
        run_id = run_id or f"run-{uuid4().hex[:8]}"
        if await self._reject_if_over_budget(request, run_id):
            return self.artifact_store.get_run(run_id)  # type: ignore[return-value]
        async with self._semaphore:
            return await self._execute(request, run_id)

    async def _reject_if_over_budget(self, request: RolloutRequest, run_id: str) -> bool:
        if self.daily_budget_usd <= 0:
            return False
        since = datetime.now(UTC) - timedelta(hours=self.budget_window_hours)
        spent = self.artifact_store.total_cost_since(since)
        if spent < self.daily_budget_usd:
            return False
        now = datetime.now(UTC)
        error_msg = (
            f"daily budget exceeded: ${spent:.4f} spent in the last "
            f"{self.budget_window_hours:g}h ≥ cap ${self.daily_budget_usd:.4f}"
        )
        manifest = RunManifest(
            id=run_id,
            model_id=self.policy_server.policy_name(),
            adapter_id=request.adapter_id,
            dataset_slice="bootstrap",
            infra_target=request.infra_target,
            seed=7,
            status=RunStatus.failed,
            created_at=now,
            updated_at=now,
        )
        task = TaskSpec(
            id=f"task-{uuid4().hex[:8]}",
            prompt=request.prompt,
            repo_snapshot=request.repo_snapshot,
            tool_permissions=[ToolPermission.read],
            horizon=request.horizon,
            success_criteria=request.success_criteria or ["(budget-blocked)"],
        )
        trajectory = TrajectoryRecord(
            id=f"traj-{uuid4().hex[:8]}", task_id=task.id, steps=[], errors=[error_msg],
        )
        reward = self.reward_pipeline.score_trajectory(task, trajectory)
        detail = RunDetail(
            manifest=manifest, task=task, trajectory=trajectory, reward=reward,
            artifacts=_default_artifacts(run_id),
        )
        self.artifact_store.save_run(detail)
        await self.event_bus.publish(
            BudgetExceeded(
                run_id=run_id, spent_usd=spent,
                cap_usd=self.daily_budget_usd, window_hours=self.budget_window_hours,
            )
        )
        await self.event_bus.publish(RolloutFailed(run_id=run_id, error=error_msg))
        log.warning("rollout.rejected.budget", spent=spent, cap=self.daily_budget_usd)
        return True

    async def _execute(self, request: RolloutRequest, run_id: str) -> RunDetail:
        now = datetime.now(UTC)
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
            external = self.external_rollout_runner
            if external is None:
                self.environment_runner.create_task(task)
                if hasattr(self.policy_server, "begin_task"):
                    self.policy_server.begin_task(task)

            model_id = (
                external.policy_name() if external is not None
                else self.policy_server.policy_name()
            )
            manifest = RunManifest(
                id=run_id,
                model_id=model_id,
                adapter_id=request.adapter_id,
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
                if external is not None:
                    return await self._execute_external(
                        external, task, request, run_id, manifest,
                    )
                steps, trajectory = await self._run_steps(task, request, run_id)
                cancelled = "cancelled" in trajectory.errors
                reward = self.reward_pipeline.score_trajectory(task, trajectory)
                cost = self._cost_for(task.id)
                await self.event_bus.publish(
                    RewardComputed(
                        run_id=run_id,
                        terminal_reward=reward.terminal_reward,
                        provenance=reward.provenance,
                        audit_flags=reward.audit_flags,
                    )
                )
                final_status = RunStatus.failed if cancelled else RunStatus.completed
                manifest = manifest.model_copy(
                    update={
                        "status": final_status,
                        "updated_at": datetime.now(UTC),
                        "estimated_cost_usd": cost if cost > 0 else manifest.estimated_cost_usd,
                    },
                )
                detail = RunDetail(
                    manifest=manifest,
                    task=task,
                    trajectory=trajectory,
                    reward=reward,
                    artifacts=_default_artifacts(run_id),
                )
                self.artifact_store.save_run(detail)
                worker_status = WorkerStatus.failed if cancelled else WorkerStatus.idle
                worker_detail = "Rollout cancelled." if cancelled else "Rollout finished; worker idle."
                await self._publish_worker(
                    "worker-rollout-local", "rollout", worker_status, run_id, worker_detail,
                )
                if cancelled:
                    await self.event_bus.publish(RolloutCancelled(run_id=run_id))
                    log.info("rollout.cancelled", steps=len(steps))
                else:
                    await self._publish_worker(
                        "worker-reward-local", "reward", WorkerStatus.idle, run_id,
                        "Reward pipeline is available for replay.",
                    )
                    await self.event_bus.publish(RolloutCompleted(run_id=run_id, detail=detail))
                    log.info(
                        "rollout.completed",
                        terminal_reward=reward.terminal_reward,
                        steps=len(steps),
                    )
                self._cancelled.discard(run_id)
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
                self._cancelled.discard(run_id)
                return failed_detail

    async def _execute_external(
        self,
        external: VerifiersRolloutRunner,
        task: TaskSpec,
        request: RolloutRequest,
        run_id: str,
        manifest: RunManifest,
    ) -> RunDetail:
        """Run a rollout where an external framework (verifiers) owns the
        per-step loop. Skips CompositeRewardPipeline because verifiers' rubric
        already produced the reward; persists everything via the same store
        and emits the same lifecycle events as the in-house path."""
        if run_id in self._cancelled:
            self._cancelled.discard(run_id)
            cancelled_traj = TrajectoryRecord(
                id=f"traj-{uuid4().hex[:8]}",
                task_id=task.id,
                steps=[],
                errors=["cancelled"],
            )
            cancelled_reward = self.reward_pipeline.score_trajectory(task, cancelled_traj)
            cancelled_detail = RunDetail(
                manifest=manifest.model_copy(
                    update={"status": RunStatus.failed, "updated_at": datetime.now(UTC)},
                ),
                task=task,
                trajectory=cancelled_traj,
                reward=cancelled_reward,
                artifacts=_default_artifacts(run_id),
            )
            self.artifact_store.save_run(cancelled_detail)
            await self.event_bus.publish(RolloutCancelled(run_id=run_id))
            await self._publish_worker(
                "worker-rollout-local", "rollout", WorkerStatus.failed, run_id,
                "Rollout cancelled before verifiers rollout started.",
            )
            return cancelled_detail

        outcome = await external.run(task, run_id)
        for step in outcome.steps:
            await self.event_bus.publish(StepRecorded(run_id=run_id, step=step))
        await self.event_bus.publish(
            ProgressTicked(
                run_id=run_id,
                step_index=max(0, len(outcome.steps) - 1),
                tool="verifiers",
                tokens=outcome.total_tokens,
                cost_usd=outcome.cost_usd,
            )
        )

        token_overflow = outcome.total_tokens > self.max_tokens_per_run
        trajectory = outcome.trajectory
        if token_overflow:
            trajectory = trajectory.model_copy(
                update={
                    "errors": [
                        *trajectory.errors,
                        (
                            f"token budget exceeded: {outcome.total_tokens} > "
                            f"{self.max_tokens_per_run}"
                        ),
                    ],
                },
            )
        await self.event_bus.publish(
            RewardComputed(
                run_id=run_id,
                terminal_reward=outcome.reward.terminal_reward,
                provenance=outcome.reward.provenance,
                audit_flags=outcome.reward.audit_flags,
            )
        )
        final_status = RunStatus.failed if token_overflow else RunStatus.completed
        manifest = manifest.model_copy(
            update={
                "status": final_status,
                "updated_at": datetime.now(UTC),
                "estimated_cost_usd": (
                    outcome.cost_usd if outcome.cost_usd > 0 else manifest.estimated_cost_usd
                ),
            },
        )
        detail = RunDetail(
            manifest=manifest,
            task=task,
            trajectory=trajectory,
            reward=outcome.reward,
            artifacts=_default_artifacts(run_id),
        )
        self.artifact_store.save_run(detail)
        worker_status = WorkerStatus.failed if token_overflow else WorkerStatus.idle
        worker_detail = (
            f"verifiers rollout exceeded token budget ({outcome.total_tokens})."
            if token_overflow
            else "verifiers rollout finished; worker idle."
        )
        await self._publish_worker(
            "worker-rollout-local", "rollout", worker_status, run_id, worker_detail,
        )
        if not token_overflow:
            await self._publish_worker(
                "worker-reward-local", "reward", WorkerStatus.idle, run_id,
                "Reward pipeline is available for replay.",
            )
            await self.event_bus.publish(RolloutCompleted(run_id=run_id, detail=detail))
        else:
            await self.event_bus.publish(
                RolloutFailed(
                    run_id=run_id,
                    error=f"token budget exceeded: {outcome.total_tokens}",
                )
            )
        self._cancelled.discard(run_id)
        log.info(
            "rollout.completed.verifiers",
            terminal_reward=outcome.reward.terminal_reward,
            steps=len(outcome.steps),
            tokens=outcome.total_tokens,
            cost_usd=outcome.cost_usd,
        )
        return detail

    async def _run_steps(
        self, task: TaskSpec, request: RolloutRequest, run_id: str,
    ) -> tuple[list[TrajectoryStep], TrajectoryRecord]:
        steps: list[TrajectoryStep] = []
        context: list[str] = []
        errors: list[str] = []
        for index in range(request.horizon):
            if run_id in self._cancelled:
                errors.append("cancelled")
                break

            action = self.policy_server.generate_action(task, context)
            observation = self.environment_runner.step(task.id, action)
            self.tool_harness.record_command(action)

            action_step = TrajectoryStep(
                index=index * 2, actor="policy", kind="action", content=action,
            )
            obs_step = TrajectoryStep(
                index=index * 2 + 1, actor="environment", kind="observation", content=observation,
            )
            steps.extend([action_step, obs_step])
            await self.event_bus.publish(StepRecorded(run_id=run_id, step=action_step))
            await self.event_bus.publish(StepRecorded(run_id=run_id, step=obs_step))
            await self.event_bus.publish(
                ProgressTicked(
                    run_id=run_id,
                    step_index=index,
                    tool=_tool_from(action),
                    tokens=self._tokens_for(task.id),
                    cost_usd=self._cost_for(task.id),
                )
            )
            context.append(observation)
            await asyncio.sleep(0)

            if run_id in self._cancelled:
                errors.append("cancelled")
                break
            if _is_finish(action):
                break

            tokens = self._tokens_for(task.id)
            if tokens > self.max_tokens_per_run:
                errors.append(f"token budget exceeded: {tokens} > {self.max_tokens_per_run}")
                break

        trajectory = TrajectoryRecord(
            id=f"traj-{uuid4().hex[:8]}",
            task_id=task.id,
            steps=steps,
            summaries=[],
            timings_ms={},
            errors=errors,
        )
        return steps, trajectory

    def _tokens_for(self, task_id: str) -> int:
        fn = getattr(self.policy_server, "total_tokens", None)
        if fn is None:
            return 0
        try:
            return int(fn(task_id))
        except Exception:
            return 0

    def _cost_for(self, task_id: str) -> float:
        fn = getattr(self.policy_server, "cumulative_cost_usd", None)
        if fn is None:
            return 0.0
        try:
            return float(fn(task_id))
        except Exception:
            return 0.0

    async def _publish_worker(
        self, worker_id: str, role: str, status: WorkerStatus, run_id: str | None, detail: str,
    ) -> None:
        worker = WorkerRecord(id=worker_id, role=role, status=status, run_id=run_id, detail=detail)
        existing = {w.id: w for w in getattr(self.artifact_store, "workers", []) or []}
        existing[worker.id] = worker
        if hasattr(self.artifact_store, "set_workers"):
            self.artifact_store.set_workers(list(existing.values()))
        await self.event_bus.publish(WorkerUpdated(run_id=run_id, worker=worker))


def _is_finish(action: str) -> bool:
    payload = _maybe_json(action)
    if payload is None:
        return False
    name = payload.get("tool") or payload.get("name")
    return name == "finish"


def _tool_from(action: str) -> str | None:
    payload = _maybe_json(action)
    if payload is None:
        return None
    value = payload.get("tool") or payload.get("name")
    return value if isinstance(value, str) else None


def _maybe_json(action: str) -> dict | None:
    import json as _json

    try:
        payload = _json.loads(action)
    except _json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


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
