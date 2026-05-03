from __future__ import annotations

import json

import pytest

from application.rescore import rescore_run
from domain.models import (
    ArtifactRecord,
    RewardRecord,
    RunDetail,
    RunManifest,
    RunStatus,
    TaskSpec,
    ToolPermission,
    TrajectoryRecord,
    TrajectoryStep,
)
from domain.rewards import HEURISTIC_V1, RUBRICS, STRICT_FINISH_V1, get_rubric
from infrastructure.store.memory import InMemoryArtifactStore


def _detail(run_id: str, steps: list[TrajectoryStep] | None = None) -> RunDetail:
    task = TaskSpec(
        id=f"{run_id}-t",
        prompt="find README",
        repo_snapshot=".",
        tool_permissions=[ToolPermission.read],
        horizon=3,
        success_criteria=["README surfaced"],
    )
    traj = TrajectoryRecord(id=f"{run_id}-traj", task_id=task.id, steps=steps or [])
    reward = RewardRecord(trajectory_id=traj.id, terminal_reward=0.1, step_rewards=[], provenance="old")
    manifest = RunManifest(
        id=run_id, model_id="test", dataset_slice="s",
        infra_target="mac-local", seed=1, status=RunStatus.completed,
    )
    return RunDetail(
        manifest=manifest, task=task, trajectory=traj, reward=reward,
        artifacts=[ArtifactRecord(name="m", kind="manifest", path="p")],
    )


def test_registry_contains_expected_rubrics() -> None:
    assert "heuristic-v1" in RUBRICS
    assert "coding-v1" in RUBRICS
    assert "strict-finish-v1" in RUBRICS
    assert get_rubric("heuristic-v1") is HEURISTIC_V1


def test_get_rubric_raises_on_unknown() -> None:
    with pytest.raises(KeyError):
        get_rubric("nope")


def test_rescore_updates_stored_reward_by_default() -> None:
    store = InMemoryArtifactStore()
    finish = TrajectoryStep(
        index=0, actor="policy", kind="action",
        content=json.dumps({"tool": "finish", "input": {"summary": "README surfaced"}}),
    )
    store.save_run(_detail("run-1", steps=[finish]))

    result = rescore_run(store, "run-1", HEURISTIC_V1)
    assert result.previous_reward.terminal_reward == 0.1
    assert result.new_reward.terminal_reward != 0.1
    assert store.get_run("run-1").reward == result.new_reward


def test_rescore_dry_run_does_not_persist() -> None:
    store = InMemoryArtifactStore()
    store.save_run(_detail("run-1"))
    before = store.get_run("run-1").reward
    rescore_run(store, "run-1", HEURISTIC_V1, persist=False)
    assert store.get_run("run-1").reward == before


def test_rescore_different_rubrics_yield_different_rewards() -> None:
    store = InMemoryArtifactStore()
    finish = TrajectoryStep(
        index=0, actor="policy", kind="action",
        content=json.dumps({"tool": "finish", "input": {"summary": "README surfaced"}}),
    )
    store.save_run(_detail("run-1", steps=[finish]))

    a = rescore_run(store, "run-1", HEURISTIC_V1, persist=False)
    b = rescore_run(store, "run-1", STRICT_FINISH_V1, persist=False)
    assert a.new_reward.terminal_reward != b.new_reward.terminal_reward
    # strict-finish weights finish more heavily.
    assert b.new_reward.terminal_reward > a.new_reward.terminal_reward


def test_rescore_raises_on_missing_run() -> None:
    store = InMemoryArtifactStore()
    with pytest.raises(ValueError):
        rescore_run(store, "nope", HEURISTIC_V1)
