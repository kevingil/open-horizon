"""Contract tests every ArtifactStore impl must satisfy."""
from __future__ import annotations

from collections.abc import Callable

import pytest

from rl_stack.domain.contracts import ArtifactStore
from rl_stack.domain.models import (
    ArtifactRecord,
    RewardRecord,
    RunDetail,
    RunManifest,
    RunStatus,
    TaskSpec,
    ToolPermission,
    TrajectoryRecord,
)
from rl_stack.infrastructure.store.memory import InMemoryArtifactStore


def _detail(run_id: str = "run-1") -> RunDetail:
    task = TaskSpec(
        id=f"{run_id}-task",
        prompt="prompt",
        repo_snapshot=".",
        tool_permissions=[ToolPermission.read],
        horizon=2,
        success_criteria=["ok"],
    )
    traj = TrajectoryRecord(id=f"{run_id}-traj", task_id=task.id, steps=[])
    reward = RewardRecord(
        trajectory_id=traj.id,
        terminal_reward=0.5,
        step_rewards=[],
        provenance="test",
    )
    manifest = RunManifest(
        id=run_id,
        model_id="test-policy",
        dataset_slice="s",
        infra_target="mac-local",
        seed=1,
        status=RunStatus.completed,
    )
    return RunDetail(
        manifest=manifest,
        task=task,
        trajectory=traj,
        reward=reward,
        artifacts=[ArtifactRecord(name=f"{run_id}-m", kind="manifest", path="p")],
    )


STORE_FACTORIES: list[Callable[[], ArtifactStore]] = [InMemoryArtifactStore]


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_save_and_get_run_roundtrip(factory: Callable[[], ArtifactStore]) -> None:
    store = factory()
    saved = store.save_run(_detail("run-a"))
    assert store.get_run("run-a") == saved
    assert store.get_run("missing") is None


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_list_runs_returns_most_recent_first(factory: Callable[[], ArtifactStore]) -> None:
    store = factory()
    store.save_run(_detail("run-old"))
    store.save_run(_detail("run-new"))
    runs = store.list_runs()
    assert {r.id for r in runs} == {"run-old", "run-new"}


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_dashboard_includes_runs_and_recent_artifacts(
    factory: Callable[[], ArtifactStore],
) -> None:
    store = factory()
    store.save_run(_detail("run-1"))
    snap = store.dashboard()
    assert any(r.id == "run-1" for r in snap.runs)
    assert snap.recent_artifacts
