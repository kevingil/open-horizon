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
from rl_stack.infrastructure.store.sqlite import SqliteArtifactStore


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


def _sqlite_factory(tmp_path_factory: pytest.TempPathFactory) -> Callable[[], ArtifactStore]:
    base = tmp_path_factory.mktemp("sqlite-store")
    counter = {"n": 0}

    def make() -> ArtifactStore:
        counter["n"] += 1
        return SqliteArtifactStore(path=base / f"runs-{counter['n']}.db")

    return make


@pytest.fixture(params=["memory", "sqlite"])
def factory(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Callable[[], ArtifactStore]:
    if request.param == "memory":
        return InMemoryArtifactStore
    return _sqlite_factory(tmp_path_factory)


def test_save_and_get_run_roundtrip(factory: Callable[[], ArtifactStore]) -> None:
    store = factory()
    saved = store.save_run(_detail("run-a"))
    assert store.get_run("run-a") == saved
    assert store.get_run("missing") is None


def test_list_runs_returns_most_recent_first(factory: Callable[[], ArtifactStore]) -> None:
    store = factory()
    store.save_run(_detail("run-old"))
    store.save_run(_detail("run-new"))
    runs = store.list_runs()
    assert {r.id for r in runs} == {"run-old", "run-new"}


def test_dashboard_includes_runs_and_recent_artifacts(
    factory: Callable[[], ArtifactStore],
) -> None:
    store = factory()
    store.save_run(_detail("run-1"))
    snap = store.dashboard()
    assert any(r.id == "run-1" for r in snap.runs)
    assert snap.recent_artifacts


def test_total_cost_since_sums_recent_runs(factory: Callable[[], ArtifactStore]) -> None:
    from datetime import UTC, datetime, timedelta

    store = factory()
    now = datetime.now(UTC)

    def _with_cost(run_id: str, at: datetime, cost: float):
        d = _detail(run_id)
        return d.model_copy(
            update={
                "manifest": d.manifest.model_copy(
                    update={"created_at": at, "estimated_cost_usd": cost}
                )
            }
        )

    store.save_run(_with_cost("old", now - timedelta(hours=48), 0.20))
    store.save_run(_with_cost("recent-a", now - timedelta(hours=2), 0.10))
    store.save_run(_with_cost("recent-b", now - timedelta(minutes=5), 0.05))

    assert store.total_cost_since(now - timedelta(hours=24)) == 0.15
    assert store.total_cost_since(now - timedelta(hours=72)) == 0.35
    assert store.total_cost_since(now + timedelta(hours=1)) == 0.0
