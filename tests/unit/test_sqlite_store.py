from __future__ import annotations

from pathlib import Path

from rl_stack.domain.models import (
    ArtifactRecord,
    RewardRecord,
    RunDetail,
    RunManifest,
    RunStatus,
    TaskSpec,
    ToolPermission,
    TrajectoryRecord,
    WorkerRecord,
    WorkerStatus,
)
from rl_stack.infrastructure.store.sqlite import SqliteArtifactStore


def _detail(run_id: str) -> RunDetail:
    task = TaskSpec(
        id=f"{run_id}-t",
        prompt="prompt",
        repo_snapshot=".",
        tool_permissions=[ToolPermission.read, ToolPermission.terminal],
        horizon=3,
        success_criteria=["ok"],
    )
    traj = TrajectoryRecord(id=f"{run_id}-traj", task_id=task.id, steps=[])
    reward = RewardRecord(
        trajectory_id=traj.id,
        terminal_reward=0.7,
        step_rewards=[0.2, 0.3, 0.2],
        provenance="heuristic-local-v0",
    )
    manifest = RunManifest(
        id=run_id,
        model_id="claude-haiku-4-5",
        dataset_slice="bootstrap",
        infra_target="mac-local",
        seed=1,
        status=RunStatus.completed,
    )
    return RunDetail(
        manifest=manifest,
        task=task,
        trajectory=traj,
        reward=reward,
        artifacts=[ArtifactRecord(name="m", kind="manifest", path=f"{run_id}/m.json")],
    )


def test_data_survives_reopen(tmp_path: Path) -> None:
    db = tmp_path / "runs.db"
    store = SqliteArtifactStore(path=db)
    store.save_run(_detail("run-1"))
    store.set_workers([
        WorkerRecord(
            id="worker-a", role="rollout", status=WorkerStatus.running,
            run_id="run-1", detail="active",
        ),
    ])

    reopened = SqliteArtifactStore(path=db)
    got = reopened.get_run("run-1")
    assert got is not None
    assert got.manifest.id == "run-1"
    assert got.task.tool_permissions == [ToolPermission.read, ToolPermission.terminal]
    snap = reopened.dashboard()
    assert any(w.id == "worker-a" for w in snap.workers)


def test_upsert_overwrites_existing_run(tmp_path: Path) -> None:
    store = SqliteArtifactStore(path=tmp_path / "runs.db")
    store.save_run(_detail("run-1"))
    updated = _detail("run-1")
    updated = updated.model_copy(
        update={"manifest": updated.manifest.model_copy(update={"estimated_cost_usd": 9.99})}
    )
    store.save_run(updated)
    assert store.get_run("run-1").manifest.estimated_cost_usd == 9.99
    assert len(store.list_runs()) == 1
