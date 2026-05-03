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
    EnvironmentRunner,
    Trainer,
    TrainingStore,
)
from infrastructure.adapters.local import LocalAdapterRegistry
from infrastructure.environment.repo_runner import RepoEnvironmentRunner
from infrastructure.environment.sandbox import build_sandbox
from infrastructure.environment.simulated import SimulatedEnvironmentRunner
from infrastructure.environment.verifiers_runner import VerifiersRolloutRunner
from infrastructure.policy.openai_compat import OpenAICompatPolicyServer
from infrastructure.policy.static import StaticPolicyServer
from infrastructure.rewards.composite import CompositeRewardPipeline
from infrastructure.store.memory import InMemoryArtifactStore
from infrastructure.store.sqlite import SqliteArtifactStore
from infrastructure.tools.local import LocalToolHarness
from infrastructure.training.memory_store import InMemoryTrainingStore
from infrastructure.training.sqlite_store import SqliteTrainingStore
from infrastructure.training.stub import StubTrainer
from settings import Settings


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

    policy = _build_policy(settings)

    environment = _build_environment(settings, root)
    external_runner = _build_external_rollout_runner(settings)

    coordinator = LocalRolloutCoordinator(
        environment_runner=environment,
        tool_harness=LocalToolHarness(root=root),
        policy_server=policy,
        reward_pipeline=CompositeRewardPipeline(),
        artifact_store=store,
        event_bus=bus,
        workspace_root=root,
        max_parallel=settings.max_parallel_rollouts,
        max_tokens_per_run=settings.max_tokens_per_run,
        daily_budget_usd=settings.daily_budget_usd,
        budget_window_hours=settings.budget_window_hours,
        external_rollout_runner=external_runner,
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


def _build_policy(settings: Settings):
    match settings.policy_backend:
        case "static":
            return StaticPolicyServer()
        case "openai":
            from openai import OpenAI

            api_key = (
                settings.llm_api_key.get_secret_value()
                if settings.llm_api_key is not None
                else "not-needed"  # local providers (vLLM, Ollama) don't require a key
            )
            return OpenAICompatPolicyServer(
                client=OpenAI(api_key=api_key, base_url=settings.llm_base_url),
                model=settings.llm_model,
                max_output_tokens=settings.llm_max_output_tokens,
                max_retries=settings.llm_max_retries,
            )
        case other:
            raise ValueError(f"Unsupported policy backend: {other}")


def _build_environment(settings: Settings, root: Path) -> EnvironmentRunner:
    match settings.env_backend:
        case "simulated":
            return SimulatedEnvironmentRunner()
        case "repo":
            scratch = settings.artifacts_dir / "workspaces"
            sandbox = build_sandbox(settings.env_sandbox, settings.sandbox_image)
            return RepoEnvironmentRunner(
                source_root=root,
                scratch_root=scratch,
                command_timeout_s=settings.env_command_timeout_s,
                max_output_bytes=settings.env_max_output_bytes,
                sandbox=sandbox,
            )
        case "verifiers":
            # verifiers owns its own rollout loop; the EnvironmentRunner ABC
            # is unused on this path. Hand back a no-op so the dataclass
            # invariant holds.
            return SimulatedEnvironmentRunner()
        case other:
            raise ValueError(f"Unsupported env backend: {other}")


def _build_external_rollout_runner(settings: Settings) -> VerifiersRolloutRunner | None:
    if settings.env_backend != "verifiers":
        return None
    from openai import OpenAI

    api_key = (
        settings.llm_api_key.get_secret_value()
        if settings.llm_api_key is not None
        else "not-needed"
    )
    client = OpenAI(api_key=api_key, base_url=settings.llm_base_url)
    return VerifiersRolloutRunner(
        client=client,
        model=settings.llm_model,
        env_id=settings.verifiers_env_id,
        env_args=settings.verifiers_env_args,
        max_concurrent=settings.verifiers_max_concurrent,
        rollout_timeout_s=settings.verifiers_rollout_timeout_s,
    )


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
