from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .application.coordinator import LocalRolloutCoordinator
from .application.event_bus import EventBus
from .domain.contracts import ArtifactStore
from .infrastructure.environment.simulated import SimulatedEnvironmentRunner
from .infrastructure.policy.static import StaticPolicyServer
from .infrastructure.rewards.heuristic import HeuristicRewardPipeline
from .infrastructure.store.memory import InMemoryArtifactStore
from .infrastructure.store.sqlite import SqliteArtifactStore
from .infrastructure.tools.local import LocalToolHarness
from .settings import Settings


@dataclass
class ApplicationServices:
    coordinator: LocalRolloutCoordinator
    event_bus: EventBus
    artifact_store: ArtifactStore
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

    policy = _build_policy(settings)

    coordinator = LocalRolloutCoordinator(
        environment_runner=SimulatedEnvironmentRunner(),
        tool_harness=LocalToolHarness(root=root),
        policy_server=policy,
        reward_pipeline=HeuristicRewardPipeline(),
        artifact_store=store,
        event_bus=bus,
        workspace_root=root,
        max_parallel=settings.max_parallel_rollouts,
    )
    return ApplicationServices(
        coordinator=coordinator,
        event_bus=bus,
        artifact_store=store,
        settings=settings,
    )


def _build_policy(settings: Settings):
    match settings.policy_backend:
        case "static":
            return StaticPolicyServer()
        case other:
            raise ValueError(f"Unsupported policy backend: {other}")


def _build_store(settings: Settings) -> ArtifactStore:
    match settings.store_backend:
        case "memory":
            return InMemoryArtifactStore()
        case "sqlite":
            return SqliteArtifactStore(path=settings.artifacts_dir / "runs.db")
        case other:
            raise ValueError(f"Unsupported store backend: {other}")
