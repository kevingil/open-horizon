from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import structlog

from domain.contracts import (
    ArtifactStore,
    RewardPipeline,
    RolloutCoordinator,
    ToolHarness,
)
from domain.events import (
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
from domain.models import (
    ArtifactRecord,
    RolloutRequest,
    RunDetail,
    RunManifest,
    RunStatus,
    TaskSpec,
    ToolPermission,
    TrajectoryRecord,
    WorkerRecord,
    WorkerStatus,
)
from domain.scheduling import FixedRoundsScheduler, RoundScheduler
from settings import PolicyProfile

from .event_bus import EventBus

if TYPE_CHECKING:
    from infrastructure.environment.repo_runner import RepoEnvironmentRunner
    from infrastructure.environment.verifiers_runner import (
        VerifiersRolloutRunner,
    )
    from infrastructure.policy.repo_loop import OpenAILike
    from infrastructure.training.recorder import TrainingRecorder

log = structlog.get_logger(__name__)


class UnknownPolicyProfile(KeyError):
    """Raised when a rollout requests a profile not in the configured set.

    Surfaces as a 422 when /api/runs validates the request, never falls
    through to a mid-rollout failure.
    """


@dataclass
class LocalRolloutCoordinator(RolloutCoordinator):
    tool_harness: ToolHarness
    reward_pipeline: RewardPipeline
    artifact_store: ArtifactStore
    event_bus: EventBus
    workspace_root: Path
    max_parallel: int = 4
    max_tokens_per_run: int = 100_000
    daily_budget_usd: float = 0.0  # 0 disables the cap
    budget_window_hours: float = 24.0
    # Two execution paths, picked per rollout based on which is wired in:
    # - external_rollout_runner: verifiers owns the multi-turn loop and
    #   returns a fully scored State. Production path for long-horizon
    #   agentic envs (vf-math, rlm, opencode/*).
    # - repo_runner: in-house tool-call loop against a sandboxed tempdir
    #   snapshot of a real git checkout. Driven by run_repo_rollout
    #   against the OpenAI client below.
    # Coordinator raises if neither is set when start_rollout is called.
    external_rollout_runner: VerifiersRolloutRunner | None = None
    repo_runner: RepoEnvironmentRunner | None = None
    # Inputs to run_repo_rollout when the repo path is taken. Only
    # required when repo_runner is set. Kept as a per-coordinator
    # default for callers that don't configure named profiles.
    policy_client: OpenAILike | None = None
    policy_model: str = "gpt-5.4-mini"
    policy_max_output_tokens: int = 2048
    policy_max_retries: int = 3
    policy_extra_body: dict[str, Any] = field(default_factory=dict)
    # Phase C: named policy profiles. Per-rollout selection by name
    # without per-run free-form configs. When `profiles` is empty the
    # coordinator synthesizes a single "default" profile from the legacy
    # fields above so existing callers keep working unchanged.
    profiles: dict[str, PolicyProfile] = field(default_factory=dict)
    default_profile: str = "default"
    # Phase D: per-rollout horizon comes from this scheduler when the
    # request itself omits one. Default mirrors today's
    # `request.horizon=6` behavior; ScalingRoundsScheduler supports the
    # AgentGym-style ramp curriculum.
    round_scheduler: RoundScheduler = field(
        default_factory=lambda: FixedRoundsScheduler(horizon=6)
    )
    # Phase E: optional per-turn training recorder. When set, every
    # repo-path turn produces a TurnTrainingRecord that the coordinator
    # persists alongside the trajectory (separate table/dict, never in
    # RunDetail). Default None = no training metadata captured.
    training_recorder: TrainingRecorder | None = None
    _client_cache: dict[str, Any] = field(init=False, default_factory=dict)
    _verifiers_cache: dict[str, Any] = field(init=False, default_factory=dict)
    _semaphore: asyncio.Semaphore = field(init=False)
    _cancelled: set[str] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.max_parallel)
        self._synthesize_default_profile()

    def _synthesize_default_profile(self) -> None:
        """If no profiles were configured, synthesize a "default" from
        the legacy fields so existing callers keep working unchanged.
        Tests pre-populate _client_cache / _verifiers_cache; we only
        synthesize the *spec* and leave the cache to whoever wired it."""
        if self.profiles:
            return
        if self.external_rollout_runner is not None:
            self.profiles[self.default_profile] = PolicyProfile(
                base_url="stub://legacy",
                model=getattr(self.external_rollout_runner, "model", "stub"),
                routes_to="verifiers",
                env_id=getattr(self.external_rollout_runner, "env_id", "stub"),
            )
            self._verifiers_cache[self.default_profile] = self.external_rollout_runner
            return
        if self.repo_runner is not None:
            self.profiles[self.default_profile] = PolicyProfile(
                base_url="stub://legacy",
                model=self.policy_model,
                routes_to="repo",
                max_output_tokens=self.policy_max_output_tokens,
                max_retries=self.policy_max_retries,
                extra_body=self.policy_extra_body,
            )
            if self.policy_client is not None:
                self._client_cache[self.default_profile] = self.policy_client

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
        """No-op startup hook: returns the current dashboard.

        Phase A removed the seed rollouts that lived here; demo content
        now comes from real rollouts against a configured backend.
        """
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
            model_id=self._resolve_model_id(request),
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
            horizon=self._resolve_horizon(request),
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

    def _resolve_profile_name(self, request: RolloutRequest) -> str:
        return request.policy_profile or self.default_profile

    def _resolve_horizon(
        self, request: RolloutRequest, *, training_step: int = 0,
    ) -> int:
        """Pick the per-rollout horizon. Explicit `request.horizon` wins;
        otherwise the configured RoundScheduler decides."""
        if request.horizon is not None:
            return request.horizon
        return self.round_scheduler.current_horizon(training_step=training_step)

    def _resolve_profile(self, request: RolloutRequest) -> PolicyProfile:
        name = self._resolve_profile_name(request)
        if name not in self.profiles:
            raise UnknownPolicyProfile(
                f"unknown policy profile: {name!r}. "
                f"Configured: {sorted(self.profiles)}",
            )
        return self.profiles[name]

    def _resolve_model_id(self, request: RolloutRequest | None = None) -> str:
        if request is not None:
            try:
                profile = self._resolve_profile(request)
            except UnknownPolicyProfile:
                pass
            else:
                if profile.routes_to == "verifiers":
                    runner = self._get_verifiers_runner(self._resolve_profile_name(request))
                    return runner.policy_name()
                return f"openai:{profile.model}"
        if self.external_rollout_runner is not None:
            return self.external_rollout_runner.policy_name()
        return f"openai:{self.policy_model}"

    def _get_client(self, profile_name: str) -> OpenAILike:
        cached = self._client_cache.get(profile_name)
        if cached is not None:
            return cached
        profile = self.profiles[profile_name]
        from openai import OpenAI

        api_key = self._resolve_api_key(profile)
        client = OpenAI(api_key=api_key, base_url=profile.base_url)
        self._client_cache[profile_name] = client
        return client

    def _get_verifiers_runner(self, profile_name: str) -> VerifiersRolloutRunner:
        cached = self._verifiers_cache.get(profile_name)
        if cached is not None:
            return cached
        profile = self.profiles[profile_name]
        from openai import OpenAI

        from infrastructure.environment.verifiers_runner import (
            VerifiersRolloutRunner,
        )

        api_key = self._resolve_api_key(profile)
        client = OpenAI(api_key=api_key, base_url=profile.base_url)
        runner = VerifiersRolloutRunner(
            client=client,
            model=profile.model,
            env_id=profile.env_id,
            env_args=profile.env_args,
            max_concurrent=profile.max_concurrent,
            rollout_timeout_s=profile.rollout_timeout_s,
        )
        self._verifiers_cache[profile_name] = runner
        return runner

    @staticmethod
    def _resolve_api_key(profile: PolicyProfile) -> str:
        import os

        if profile.api_key is not None:
            return profile.api_key.get_secret_value()
        if profile.api_key_env:
            value = os.environ.get(profile.api_key_env)
            if value:
                return value
        return "not-needed"

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
                horizon=self._resolve_horizon(request),
                success_criteria=request.success_criteria or ["manual review"],
            )
            profile = self._resolve_profile(request)
            profile_name = self._resolve_profile_name(request)

            manifest = RunManifest(
                id=run_id,
                model_id=self._resolve_model_id(request),
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
                if profile.routes_to == "verifiers":
                    runner = self._get_verifiers_runner(profile_name)
                    return await self._execute_external(
                        runner, task, request, run_id, manifest,
                    )
                if self.repo_runner is None:
                    raise RuntimeError(
                        f"profile {profile_name!r} routes to repo but no "
                        "repo_runner is configured on the coordinator."
                    )
                return await self._execute_repo(
                    self.repo_runner, task, request, run_id, manifest, profile,
                    profile_name,
                )
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
        """Run a rollout where verifiers owns the multi-turn loop. Skips
        CompositeRewardPipeline because verifiers' rubric already produced
        the reward; persists everything via the same store and emits the
        same lifecycle events as the repo path."""
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

    async def _execute_repo(
        self,
        repo_runner: RepoEnvironmentRunner,
        task: TaskSpec,
        request: RolloutRequest,
        run_id: str,
        manifest: RunManifest,
        profile: PolicyProfile,
        profile_name: str,
    ) -> RunDetail:
        """Drive a per-turn OpenAI tool-call loop against a sandboxed repo
        snapshot via run_repo_rollout. Reward comes from the in-house
        CompositeRewardPipeline (no external rubric on this path)."""
        from infrastructure.policy.repo_loop import run_repo_rollout

        client = self._get_client(profile_name)

        async def _on_step(step) -> None:
            await self.event_bus.publish(StepRecorded(run_id=run_id, step=step))

        async def _on_progress(turn: int, tokens: int, cost: float, tool: str | None) -> None:
            await self.event_bus.publish(
                ProgressTicked(
                    run_id=run_id, step_index=turn, tool=tool,
                    tokens=tokens, cost_usd=cost,
                ),
            )

        recorder = self.training_recorder

        async def _on_turn(
            turn: int, prompt_text: str, completion_text: str,
            sampling_args: dict[str, Any],
        ) -> None:
            if recorder is None:
                return
            recorder.record(
                run_id=run_id,
                step_index=turn * 2,  # match action_step's index in trajectory
                prompt_text=prompt_text,
                completion_text=completion_text,
                sampling_args=sampling_args,
                model_name=profile.model,
                store=self.artifact_store,
            )

        outcome = await run_repo_rollout(
            client=client,
            model=profile.model,
            task=task,
            repo_runner=repo_runner,
            record_command=self.tool_harness.record_command,
            horizon=task.horizon,
            max_tokens_per_run=self.max_tokens_per_run,
            max_output_tokens=profile.max_output_tokens,
            max_retries=profile.max_retries,
            extra_body=profile.extra_body,
            on_step=_on_step,
            on_turn=_on_turn if recorder is not None else None,
            on_progress=_on_progress,
            is_cancelled=lambda: run_id in self._cancelled,
        )

        cancelled = outcome.cancelled
        token_overflow = outcome.token_overflow
        reward = self.reward_pipeline.score_trajectory(task, outcome.trajectory)
        await self.event_bus.publish(
            RewardComputed(
                run_id=run_id,
                terminal_reward=reward.terminal_reward,
                provenance=reward.provenance,
                audit_flags=reward.audit_flags,
            )
        )
        final_status = RunStatus.failed if (cancelled or token_overflow) else RunStatus.completed
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
            trajectory=outcome.trajectory,
            reward=reward,
            artifacts=_default_artifacts(run_id),
        )
        self.artifact_store.save_run(detail)
        if cancelled:
            await self._publish_worker(
                "worker-rollout-local", "rollout", WorkerStatus.failed, run_id,
                "Rollout cancelled.",
            )
            await self.event_bus.publish(RolloutCancelled(run_id=run_id))
            log.info("rollout.cancelled", steps=len(outcome.steps))
        elif token_overflow:
            await self._publish_worker(
                "worker-rollout-local", "rollout", WorkerStatus.failed, run_id,
                f"repo rollout exceeded token budget ({outcome.total_tokens}).",
            )
            await self.event_bus.publish(
                RolloutFailed(
                    run_id=run_id,
                    error=f"token budget exceeded: {outcome.total_tokens}",
                )
            )
        else:
            await self._publish_worker(
                "worker-rollout-local", "rollout", WorkerStatus.idle, run_id,
                "Rollout finished; worker idle.",
            )
            await self._publish_worker(
                "worker-reward-local", "reward", WorkerStatus.idle, run_id,
                "Reward pipeline is available for replay.",
            )
            await self.event_bus.publish(RolloutCompleted(run_id=run_id, detail=detail))
            log.info(
                "rollout.completed",
                terminal_reward=reward.terminal_reward,
                steps=len(outcome.steps),
            )
        self._cancelled.discard(run_id)
        return detail

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
