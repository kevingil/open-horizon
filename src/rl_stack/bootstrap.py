from __future__ import annotations

from pathlib import Path

from .services.artifact_store import InMemoryArtifactStore
from .services.coordinator import LocalRolloutCoordinator
from .services.environment import SimulatedEnvironmentRunner
from .services.policy import StaticPolicyServer
from .services.rewards import HeuristicRewardPipeline
from .services.tools import LocalToolHarness


def build_application_services(workspace_root: str | Path) -> LocalRolloutCoordinator:
    root = Path(workspace_root).resolve()
    artifact_store = InMemoryArtifactStore()
    coordinator = LocalRolloutCoordinator(
        environment_runner=SimulatedEnvironmentRunner(),
        tool_harness=LocalToolHarness(root=root),
        policy_server=StaticPolicyServer(),
        reward_pipeline=HeuristicRewardPipeline(),
        artifact_store=artifact_store,
        workspace_root=root,
    )
    coordinator.bootstrap()
    return coordinator
