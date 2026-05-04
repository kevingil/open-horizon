from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from application.coordinator import LocalRolloutCoordinator
from application.eval import EvalHarness
from application.event_bus import EventBus
from application.training import TrainingService
from domain.contracts import (
    AdapterRegistry,
    ArtifactStore,
    Trainer,
    TrainingStore,
)
from domain.scheduling import (
    FixedRoundsScheduler,
    RoundScheduler,
    ScalingRoundsScheduler,
)
from infrastructure.adapters.local import LocalAdapterRegistry
from infrastructure.environment.repo_runner import RepoEnvironmentRunner
from infrastructure.environment.sandbox import build_sandbox
from infrastructure.rewards.composite import CompositeRewardPipeline
from infrastructure.store.memory import InMemoryArtifactStore
from infrastructure.store.sqlite import SqliteArtifactStore
from infrastructure.tools.local import LocalToolHarness
from infrastructure.training.memory_store import InMemoryTrainingStore
from infrastructure.training.sqlite_store import SqliteTrainingStore
from infrastructure.training.stub import StubTrainer
from settings import PolicyProfile, Settings


@dataclass
class ApplicationServices:
    coordinator: LocalRolloutCoordinator
    event_bus: EventBus
    artifact_store: ArtifactStore
    training_service: TrainingService
    training_store: TrainingStore
    adapter_registry: AdapterRegistry
    eval_harness: EvalHarness
    settings: Settings


def build_application_services(
    settings: Settings | None = None,
    *,
    event_bus: EventBus | None = None,
) -> ApplicationServices:
    settings = settings or Settings()
    root = Path(settings.workspace_root).resolve()
    bus = event_bus or EventBus()
    store = _build_store(settings)
    training_store = _build_training_store(settings)
    adapter_registry = LocalAdapterRegistry(root=settings.adapters_dir)
    trainer = _build_trainer(settings)

    repo_runner = _build_repo_runner(settings, root)
    profiles = _build_profiles(settings)
    round_scheduler = _build_round_scheduler(settings)

    coordinator = LocalRolloutCoordinator(
        tool_harness=LocalToolHarness(root=root),
        reward_pipeline=CompositeRewardPipeline(),
        artifact_store=store,
        event_bus=bus,
        workspace_root=root,
        max_parallel=settings.max_parallel_rollouts,
        max_tokens_per_run=settings.max_tokens_per_run,
        daily_budget_usd=settings.daily_budget_usd,
        budget_window_hours=settings.budget_window_hours,
        repo_runner=repo_runner,
        profiles=profiles,
        default_profile=settings.default_policy_profile,
        round_scheduler=round_scheduler,
    )
    training_service = TrainingService(
        trainer=trainer,
        artifact_store=store,
        training_store=training_store,
        adapter_registry=adapter_registry,
        event_bus=bus,
    )
    eval_harness = EvalHarness(
        coordinator=coordinator,
        training_store=training_store,
        adapter_registry=adapter_registry,
        event_bus=bus,
    )
    return ApplicationServices(
        coordinator=coordinator,
        event_bus=bus,
        artifact_store=store,
        training_service=training_service,
        training_store=training_store,
        adapter_registry=adapter_registry,
        eval_harness=eval_harness,
        settings=settings,
    )


def _build_repo_runner(settings: Settings, root: Path) -> RepoEnvironmentRunner | None:
    """Repo path is opt-in: only built when env_backend=repo. The verifiers
    path doesn't need a runner, and we no longer have a no-op simulated
    runner to fall back on."""
    if settings.env_backend != "repo":
        return None
    scratch = settings.artifacts_dir / "workspaces"
    sandbox = build_sandbox(settings.env_sandbox, settings.sandbox_image)
    return RepoEnvironmentRunner(
        source_root=root,
        scratch_root=scratch,
        command_timeout_s=settings.env_command_timeout_s,
        max_output_bytes=settings.env_max_output_bytes,
        sandbox=sandbox,
    )


def _build_round_scheduler(settings: Settings) -> RoundScheduler:
    """Construct the per-rollout horizon scheduler from settings."""
    if settings.round_scheduler == "scaling":
        mode = settings.scaling_mode if settings.scaling_mode in ("linear", "step") else "linear"
        return ScalingRoundsScheduler(
            start=settings.scaling_horizon_start,
            end=settings.scaling_horizon_end,
            ramp_steps=settings.scaling_ramp_steps,
            mode=mode,  # type: ignore[arg-type]
        )
    return FixedRoundsScheduler(horizon=settings.round_scheduler_default_horizon)


def _build_profiles(settings: Settings) -> dict[str, PolicyProfile]:
    """Return the dict of profiles the coordinator will dispatch from.

    If `settings.policy_profiles` is configured, hand it back unchanged.
    Otherwise synthesize a single "default" profile from the legacy
    `llm_*` + `verifiers_*` + `env_backend` fields so existing callers
    keep working without setting up profiles explicitly.
    """
    if settings.policy_profiles:
        return dict(settings.policy_profiles)
    routes_to = "verifiers" if settings.env_backend == "verifiers" else "repo"
    default = PolicyProfile(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        routes_to=routes_to,
        max_output_tokens=settings.llm_max_output_tokens,
        max_retries=settings.llm_max_retries,
        env_id=settings.verifiers_env_id,
        env_args=settings.verifiers_env_args,
        max_concurrent=settings.verifiers_max_concurrent,
        rollout_timeout_s=settings.verifiers_rollout_timeout_s,
    )
    return {settings.default_policy_profile: default}


def _build_store(settings: Settings) -> ArtifactStore:
    match settings.store_backend:
        case "memory":
            return InMemoryArtifactStore()
        case "sqlite":
            return SqliteArtifactStore(path=settings.artifacts_dir / "runs.db")
        case other:
            raise ValueError(f"Unsupported store backend: {other}")


def _build_training_store(settings: Settings) -> TrainingStore:
    match settings.training_store_backend:
        case "memory":
            return InMemoryTrainingStore()
        case "sqlite":
            return SqliteTrainingStore(path=settings.artifacts_dir / "training.db")
        case other:
            raise ValueError(f"Unsupported training store backend: {other}")


def _build_trainer(settings: Settings) -> Trainer:
    match settings.trainer_backend:
        case "stub":
            return StubTrainer(default_step_delay_s=settings.train_step_delay_s)
        case "grpo":
            # Import lazily so users on the stub backend don't need torch.
            from infrastructure.training.grpo import GrpoTrainer

            return GrpoTrainer(base_model=settings.grpo_base_model)
        case other:
            raise ValueError(f"Unsupported trainer backend: {other}")
